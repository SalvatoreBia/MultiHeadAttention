#include <cuda_runtime.h>

#define divc( a, b ) ((a+b-1)/b)


constexpr int g_br = 16;
constexpr int g_maxdk = 128;
constexpr int g_maxcols = g_maxdk / 16;

__global__ void flash_attention( const float* __restrict__ Q,
                                                                        const float* __restrict__ K,
                                                                        const float* __restrict__ V,
                                                                        float* output,
                                                                        int N, int d_model, int h,
                                                                        int dk, float dk_i )
{
    extern __shared__ float s_buf[];
    float* s_Q = s_buf;
    float* s_K = s_buf + ( g_br * dk );
    float* s_V = s_buf + ( g_br * dk * 2 );

    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int head_id = blockIdx.y;

    int Q_row = blockIdx.x * g_br + tidy;
    for( int i = tidx; i < dk; i += blockDim.x )
    {
        int Q_idx = Q_row * d_model + ( head_id * dk ) + i;
        int s_Q_idx = tidy * dk + i;
        s_Q[s_Q_idx] = ( Q_row < N )? Q[Q_idx] : 0.0f;
    }
    __syncthreads();

    float curr_max = -1e20f;
    float curr_sum = 0.0f;
    float scores[g_br];
    float O_acc[g_maxcols] = { 0.0f };
    for( int row_block_id = 0;row_block_id < divc( N, g_br ); ++row_block_id )
    {
        int KV_row = row_block_id * g_br + tidy;
        for( int i = tidx; i < dk; i += blockDim.x )
        {
            int KV_idx = KV_row * d_model + ( head_id * dk ) + i;
            int s_KV_idx = tidy * dk + i;
            bool is_valid = ( KV_row < N );
            s_K[s_KV_idx] = is_valid? K[KV_idx] : 0.0f;
            s_V[s_KV_idx] = is_valid? V[KV_idx] : 0.0f;
        }
        __syncthreads();

        float block_max = -1e20f;
        for( int i = 0; i < g_br; ++i )
        {
            float dot = 0.0f;
            for( int j = 0; j < dk; ++j )
            {
                dot += s_Q[tidy * dk + j] * s_K[i * dk + j];
            }
            bool is_valid = ( row_block_id * g_br + i < N );
            float score = is_valid? ( dot * dk_i ) : -1e20f;
            scores[i] = score;
            block_max = fmaxf( block_max, score );
        }

        float new_max = fmaxf( curr_max, block_max );
        float alpha = __expf( curr_max - new_max );

        float block_sum = 0.0f;
        float P[g_br];
        for( int i = 0; i < g_br; ++i )
        {
            bool is_valid = ( row_block_id * g_br + i < N );
            P[i] = is_valid? __expf( scores[i] - new_max ) : 0.0f;
            block_sum += P[i];
        }
        curr_sum = curr_sum * alpha + block_sum;
        curr_max = new_max;

        int col_idx = 0;
        for( int col = tidx; col < dk; col += blockDim.x )
        {
            float pv = 0.0f;
            for( int k = 0; k < g_br; ++k )
            {
                pv += P[k] * s_V[k * dk + col];
            }
            O_acc[col_idx] = O_acc[col_idx] * alpha + pv;
            col_idx++;
        }

        __syncthreads();
    }

    if( Q_row < N )
    {
        float inv_sum = 1.0f / curr_sum;
        int col_idx = 0;
        for( int col = tidx; col < dk; col += blockDim.x )
        {
            int out_idx = Q_row * d_model + ( head_id * dk ) + col;
            output[out_idx] = O_acc[col_idx] * inv_sum;
            col_idx++;
        }
    }
}


extern "C" void solve( const float* Q,
                                                const float* K,
                                                const float* V,
                                                float* output,
                                                int N, int d_model, int h )
{
    int dk = d_model / h;
    float dk_i = 1.0f / sqrtf( static_cast<float>( dk ) );

    dim3 nthreads( 16, 16 );
    dim3 nblocks( divc( N, g_br ), h );

    size_t smem_bytes = 3 * g_br * dk  * sizeof(float);

    flash_attention<<< nblocks, nthreads, smem_bytes >>>( Q, K, V, output, N, d_model, h, dk, dk_i );
}
