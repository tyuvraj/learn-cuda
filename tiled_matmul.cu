#include <iostream>
#define TILE_SIZE 32


void matmul_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int i = 0; i < K; ++i) {
                sum += A[row * K + i] * B[i * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}

__global__ void mat_mul_tiled_kernel(
    float* matA, float* matB, float* matC, u_int32_t heightA, u_int32_t widthB, u_int32_t widthA
){
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float A_s[TILE_SIZE][TILE_SIZE];
    __shared__ float B_s[TILE_SIZE][TILE_SIZE];
    
    const unsigned int NUM_TILES = (widthA + TILE_SIZE - 1)/TILE_SIZE;
    float sum = 0;
    for (unsigned int tile_section = 0; tile_section < NUM_TILES; ++tile_section){
        const int tile_offset = TILE_SIZE*tile_section;
        const int innerCol = tile_offset + threadIdx.x;
        const int innerRow = tile_offset + threadIdx.y;

        if(row < heightA && innerCol < widthA){ // can get rid of this if widthA is not a multiple of TILE_SIZE
            A_s[threadIdx.y][threadIdx.x] =  matA[widthA*row + innerCol];
        } else {
            A_s[threadIdx.y][threadIdx.x] = 0;
        }

        if(innerRow < widthA && col < widthB){ // can get rid of this if widthA is not a multiple of TILE_SIZE
            B_s[threadIdx.y][threadIdx.x] =  matB[widthB*innerRow + col];
        } else {
            B_s[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        for (unsigned int  inner_index=0; inner_index < TILE_SIZE; ++inner_index){
            float a = A_s[threadIdx.y][inner_index];
            float b = B_s[inner_index][threadIdx.x];
            sum += a * b;
        }

        __syncthreads();
    }

    u_int32_t output_index = widthB * row + col;
    if (row < heightA and col < widthB){ // can get rid of this if heightA and widthB is not a multiple of blocksize.xy
        matC[output_index] = sum;
    }

}

int main() {
    u_int32_t  matA_size, matB_size, matC_size, matA_bytes, matB_bytes, matC_bytes; 
    
    u_int32_t M = 333, N = 511, K = 1025;
    matA_size = M * K; // MxK input matrix
    matB_size = K * N; // KxN input matrix
    matC_size = M * N; // MxN output matrix

    matA_bytes = matA_size * sizeof(float);
    matB_bytes = matB_size * sizeof(float);
    matC_bytes = matC_size * sizeof(float);


    float *matA = new float[matA_size];
    float *matB = new float[matB_size];
    float *matC = new float[matC_size];
    float *matC_check = new float[matC_size];
    
    for (int i=0; i< matA_size; i++){
        matA[i] =  float(rand())/RAND_MAX;
    }
    
    for (int i=0; i< matB_size; i++){
        matB[i] =  float(rand())/RAND_MAX;
    }
    
    float *matA_d, *matB_d, *matC_d;

    cudaMalloc((void**)&matA_d, matA_bytes);
    cudaMalloc((void**)&matB_d, matB_bytes);
    cudaMalloc((void**)&matC_d, matC_bytes);

    cudaMemcpy(matA_d, matA, matA_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(matB_d, matB, matB_bytes, cudaMemcpyHostToDevice);

    const dim3 blockSize(TILE_SIZE, TILE_SIZE);
    const dim3 gridSize(
        (N + blockSize.x - 1) / blockSize.x,
        (M + blockSize.y - 1) / blockSize.y,
        1
    );


    mat_mul_tiled_kernel<<<gridSize, blockSize>>>(
        matA_d, matB_d, matC_d, M, N, K
    );

    cudaMemcpy(matC, matC_d, matC_bytes, cudaMemcpyDeviceToHost);

    matmul_cpu(matA, matB, matC_check, M, N, K);
    float sum = 0;
    for (int i = 0; i < matC_size; i++) {
        float a = matC_check[i] ;
        float b = matC[i];
        float diff = abs(a - b);
        if(diff > 1e-4){
             std::cout <<  a << " " << b << " " << diff << std::endl;
        }
        sum += diff;
    }
    std::cout << "sum: " << sum << std::endl;

    cudaFree(matA_d);
    cudaFree(matB_d);
    cudaFree(matC_d);
    

    delete[] matA;
    delete[] matB;
    delete[] matC;

    return 0;
}
