#include <cuda_runtime.h>

#define divc( a, b ) ((a+b-1)/b)


constexpr int g_tile = 16;

__global__ void compute_qkt( const float* __restrict__ Q,
                             const float* __restrict__ K,
                             float* S,
                             int N, int d_model, int h,
                             int dk, float dk_i )
{
    __shared__ float s_bufQ[g_tile][g_tile];
    __shared__ float s_bufKt[g_tile][g_tile];

    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int head_id = blockIdx.z;

    float acc = 0.0f;
    for( int i = 0; i < divc( dk, g_tile ); ++i )
    {
        int tidx = threadIdx.x;
        int tidy = threadIdx.y;

        int Q_offset = i * g_tile + tidx;
        int Q_idx = row * d_model + Q_offset + head_id * dk;

        s_bufQ[tidy][tidx] = ( row < N && Q_offset < dk )? Q[Q_idx] : 0.0f;

        int Kt_offset = i * g_tile + tidy;
        int Kt_idx = col * d_model + Kt_offset + head_id * dk;

        s_bufKt[tidy][tidx] = ( col < N && Kt_offset < dk )? K[Kt_idx] : 0.0f;

        __syncthreads();

        for( int k = 0; k < g_tile; ++k )
        {
            acc += s_bufQ[tidy][k] * s_bufKt[k][tidx];
        }

        __syncthreads();
    }

    if( row < N && col < N )
    {
        int S_idx = head_id * ( N * N ) + row * N + col;
        S[S_idx] = acc * dk_i;
    }
}

constexpr int g_maxwarps = 1024 / 32;

// con l'online softmax abbiamo trovato
// massimo e denominatore in una sola passata anziché due!
__global__ void online_softmax( float* S, int N, int h )
{
    __shared__ float s_bufMax[g_maxwarps];
    __shared__ float s_bufSum[g_maxwarps];

    int tid = threadIdx.x;
    int head_id = blockIdx.x / N;
    int row = blockIdx.x % N;

    float psum = 0.0f;
    float max_val = -1e20f;
    for( int i = tid; i < N; i += blockDim.x )
    {
        int idx =  head_id * ( N * N ) + row * N + i;
        float x = S[idx];
        float tmp = max_val;
        max_val = fmaxf( tmp, x );
        psum = psum * __expf( tmp - max_val ) + __expf( x - max_val );
    }

    for( int i = 16; i > 0; i >>= 1 )
    {
        float m2 = __shfl_down_sync( 0xffffffff, max_val, i );
        float psum2 = __shfl_down_sync( 0xffffffff, psum, i );
        float new_max = fmaxf( max_val, m2 );
        psum = psum * __expf( max_val - new_max ) + psum2 * __expf( m2 - new_max );
        max_val = new_max;
    }

    int lane_id = tid % 32;
    if( lane_id == 0 )
    {
        int warp_id = tid / 32;
        s_bufMax[warp_id] = max_val;
        s_bufSum[warp_id] = psum;
    }

    __syncthreads();

    int num_warps = blockDim.x / 32;
    if( tid < 32 )
    {
        if( tid < num_warps )
        {
            max_val = s_bufMax[tid];
            psum = s_bufSum[tid];
        }
        else
        {
            max_val = -1e20f;
            psum = 0.0f;
        }

        for( int i = 16; i > 0; i >>= 1 )
        {
            float m2 = __shfl_down_sync( 0xffffffff, max_val, i );
            float psum2 = __shfl_down_sync( 0xffffffff, psum, i );
            float new_max = fmaxf( max_val, m2 );
            psum = psum * __expf( max_val - new_max ) + psum2 * __expf( m2 - new_max );
            max_val = new_max;
        }

        if( tid == 0 )
        {
            s_bufMax[0] = max_val;
            s_bufSum[0] = 1.0f / psum;
        }
    }

    __syncthreads();

    for( int i = tid; i < N; i += blockDim.x )
    {
        int idx =  head_id * ( N * N ) + row * N + i;
        S[idx] = __expf( S[idx] - s_bufMax[0] ) * s_bufSum[0];
    }
}

__global__ void compute_sv( const float* __restrict__ S,
                            const float* __restrict__ V,
                            float* output,
                            int N, int d_model, int h,
                            int dk )
{
    __shared__ float s_bufS[g_tile][g_tile];
    __shared__ float s_bufV[g_tile][g_tile];

    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int head_id = blockIdx.z;

    float acc = 0.0f;
    for( int i = 0; i < divc( N, g_tile ); ++i )
    {
        int tidx = threadIdx.x;
        int tidy = threadIdx.y;

        int S_offset = i * g_tile + tidx;
        int S_idx = ( head_id * N + row ) * N + S_offset;

        s_bufS[tidy][tidx] = ( row < N && S_offset < N )? S[S_idx] : 0.0f;

        int V_offset = i * g_tile + tidy;
        int V_idx = head_id * dk + V_offset * d_model + col;

        s_bufV[tidy][tidx] = ( V_offset < N && col < dk )? V[V_idx] : 0.0f;

        __syncthreads();

        for( int k = 0; k < g_tile; ++k )
        {
            acc += s_bufS[tidy][k] * s_bufV[k][tidx];
        }

        __syncthreads();
    }

    if( row < N && col < dk )
    {
        int output_idx = row * d_model + head_id * dk + col;
        output[output_idx] = acc;
    }
}

extern "C" void solve( const float* Q, const float* K, const float* V, float* output, int N, int d_model, int h )
{
    int dk = d_model / h;
    float dk_i = 1.0f / sqrtf( static_cast<float>( dk ) );

    dim3 nthreads( g_tile, g_tile, 1 );
    dim3 nblocks(
        divc( N, g_tile ),
        divc( N, g_tile ),
        h
    );

    float *S;
    cudaMalloc( (void**)&S, h * N * N * sizeof( float ) );

    compute_qkt<<< nblocks, nthreads >>>( Q, K, S, N, d_model, h, dk, dk_i );

    online_softmax<<< h * N, 512 >>>( S, N, h );

    nblocks.x = divc( dk, g_tile );
    nblocks.y = divc( N, g_tile );

    compute_sv<<< nblocks, nthreads >>>( S, V,output, N, d_model, h, dk );

    cudaFree( S );
}
