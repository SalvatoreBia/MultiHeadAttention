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

    // accumulatore per il prodotto corrente
    float acc = 0.0f;
    for( int i = 0; i < divc( dk, g_tile ); ++i )
    {
        int tidx = threadIdx.x;
        int tidy = threadIdx.y;

        // questa è una delle parti più complicate da capire:
        //                         il calcolo degli indici
        // 
        // partiamo da Q. dal momento che stiamo facendo una
        // matmul, sappiamo per certo che il thread che si trova a riga X
        // dovrà scorrere gli elementi che si trovano nella medesima riga
        // della matrice a sinistra, che in questo caso è Q.
        // la colonna invece è data dall'indice colonna del thread nella tile
        // più uno stride che ci permette di saltare le tile precedentemente
        // processate ( i.e.    i*g_tile ). L'ultima cosa da fare è saltare all'elemento
        // della testa corrente, perciò aggiungiamo lo stride pari alla
        // larghezza di una singola testa (i.e.     head_id*dk)
        int Q_offset = i * g_tile + tidx;
        int Q_idx = row * d_model + Q_offset + head_id * dk;

        s_bufQ[tidy][tidx] = ( row < N && Q_offset < dk )? Q[Q_idx] : 0.0f;

        // facciamo la stessa cosa per K trasposto.
        // per prima cosa l'offset cambia, visto che adesso le tile scorrono
        // sulle righe e non sulle colonne.
        // 
        // in una matmul classica, a questo punto useremmo col per accedere alla
        // colonna di K, ma visto che dobbiamo trasporla, utilizziamo col
        // per accedere alla riga
        int Kt_offset = i * g_tile + tidy;
        int Kt_idx = col * d_model + Kt_offset + head_id * dk;

        s_bufKt[tidy][tidx] = ( col < N && Kt_offset < dk )? K[Kt_idx] : 0.0f;

        __syncthreads();

        // accumuliamo i prodotti scalari delle tile
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

__global__ void softmax( float* S, int N, int h )
{
    // per calcolare la softmax per cascuna riga e testa di S
    // abbiamo allocato un numero di blocco pari al totale delle righe
    // e cioè h * N.
    // questo array in shared servirà per le tree reduction con le warp
    // intrinsics
    __shared__ float s_buf[g_maxwarps];

    int tid = threadIdx.x;

    // ricaviamo indice della testa corrente
    //  e riga nella testa
    int head_id = blockIdx.x / N;
    int row = blockIdx.x % N;

    // ma prima . . .
    // per rendere la softmax safe e evitare overflow, dobbiamo prima
    // ricavare il massimo input per ciascun blocco, e la formula diventa
    // e^(x - max_x) / sum_i [ e^(x_i - max_x)]
    __shared__ float s_bufMax[g_maxwarps];

    float max_val = -1e20f;
    for( int i = tid; i < N; i += blockDim.x )
    {
        int idx =  head_id * ( N * N ) + row * N + i;
        int x = S[idx];
        if( x > max_val )
        {
            max_val = x;
        }
    }

    float tmp = -1e20f;
    for( int i = 16; i > 0; i >>= 1 )
    {
        max_val = fmaxf( max_val, __shfl_down_sync( 0xffffffff, max_val, i ) );
    }

    if( tid % 32 == 0)
    {
        s_bufMax[tid / 32] = max_val;
    }

    __syncthreads();

    int nwarps = blockDim.x / 32;
    if( tid < 32 )
    {
        max_val = ( tid < nwarps )? s_bufMax[tid] : -1e20f;
        for( int i = 16; i > 0; i >>= 1 )
        {
            max_val = fmaxf( max_val, __shfl_down_sync( 0xffffffff, max_val, i ) );
        }

        if( tid == 0 )
        {
            s_bufMax[0] = max_val;
        }
    }

    __syncthreads();

    // primo loop block-stride, alla fine ogni thread
    // conterrà una somma parziale del denominatore
    // della softmax
    float acc = 0.0f;
    for( int i = tid; i < N; i += blockDim.x )
    {
        int idx =  head_id * ( N * N ) + row * N + i;
        acc += __expf( S[idx] - s_bufMax[0] );
    }

    // prima tree reduction, adesso le somme parziali sono
    // ulteriormente accumulate nel primo thread di ogni warp
    for( int i = 16; i > 0; i >>= 1 )
    {
        acc += __shfl_down_sync( 0xffffffff, acc, i );
    }

    // dopodiché, ogni "capo" del warp carica in smem
    if( tid % 32 == 0 )
    {
        s_buf[tid / 32] = acc;
    }

    __syncthreads();

    // ultima tree reduction. Alla fine di questo blocco if
    // dentro il primo slot della smem ci sarà il denominatore
    if( tid < 32 )
    {
        int nwarps = blockDim.x / 32;
        acc = ( tid < nwarps )? s_buf[tid] : 0.0f;
        for( int i = 16; i > 0; i >>= 1 )
        {
            acc += __shfl_down_sync( 0xffffffff, acc, i );
        }

        if( tid == 0 )
        {
            s_buf[0] = acc;
        }
    }

    __syncthreads();

    // ultimo block-stride loop per aggiornare i logits
    float exp_i = 1.0f / s_buf[0];
    for( int i = tid; i < N; i += blockDim.x )
    {
        int idx =  head_id * ( N * N ) + row * N + i;
        S[idx] = __expf( S[idx] - s_bufMax[0] ) * exp_i;
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
    // si assume che  d_model % h == 0
    int dk = d_model / h;
    float dk_i = 1.0f / sqrtf( static_cast<float>( dk ) );

    // per prima cosa calcoloremo q*k^T scalato di dk_i
    // per fare ciò, optiamo per un approccio tiled matmul
    // quindi dobbiamo dividere la matrice di output in h matrici
    // di dimensione N x N e processare ognuno seguendo l'approccio tiled.
    // SPIEGAZIONE:
    //      se Q è una matrice N x d_model e K^T è d_model x N, allora la risultante
    //      sarà una N x N, ma questa matrice è il risultato di una sola testa, quindi
    //      ce ne servono h di matrici. Per questo motivo, abbiamo messo h alla
    //      dimensione z di nblocks
    dim3 nthreads( g_tile, g_tile, 1 );
    dim3 nblocks(
        divc( N, g_tile ),
        divc( N, g_tile ),
        h
    );

    // allochiamo un buffer che conterrà i risultati della matmul
    // per ogni testa
    float *S;
    cudaMalloc( (void**)&S, h * N * N * sizeof( float ) );

    compute_qkt<<< nblocks, nthreads >>>( Q, K, S, N, d_model, h, dk, dk_i );

    // il prossimo step è passare S sotto una safe softmax
    softmax<<< h * N, 512 >>>( S, N, h );

    // infine calcoliamo S * V, allo stesso modo di come abbiamo fatto con Q e K
    nblocks.x = divc( dk, g_tile );
    nblocks.y = divc( N, g_tile );

    compute_sv<<< nblocks, nthreads >>>( S, V,output, N, d_model, h, dk );

    // liberiamo la memoria
    cudaFree( S );
}
