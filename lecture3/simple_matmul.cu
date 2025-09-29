#include <iostream>
/**
 * Simple rectangular matrix multiplication
 */
__global__ void mat_mul_kernel(
    float* matA, float* matB, float* matC, u_int32_t M, u_int32_t N, u_int32_t K
){
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    float sum = 0;
    for (int i = 0; i < K; i++) {
        float a = matA[K*row + i];
        float b = matB[N*i + col];
        sum += a * b;
    }
    u_int32_t output_index = N * row + col;
    if (row < M and col < N){
        matC[output_index] = sum;
    }

}

int main() {
    u_int32_t  matA_size, matB_size, matC_size, matA_bytes, matB_bytes, matC_bytes; 
    
    u_int32_t M = 256, N = 512, K = 1024;
    matA_size = M * K; // MxK input matrix
    matB_size = K * N; // KxN input matrix
    matC_size = M * N; // MxN output matrix

    matA_bytes = matA_size * sizeof(float);
    matB_bytes = matB_size * sizeof(float);
    matC_bytes = matC_size * sizeof(float);


    float *matA = new float[matA_size];
    float *matB = new float[matB_size];
    float *matC = new float[matC_size];
    
    for (int i=0; i< matA_size; i++){
        matA[i] =  i; //float(rand())/RAND_MAX;
    }
    
    for (int i=0; i< matB_size; i++){
        matB[i] =  i; //float(rand())/RAND_MAX;
    }
    
    float *matA_d, *matB_d, *matC_d;

    cudaMalloc((void**)&matA_d, matA_bytes);
    cudaMalloc((void**)&matB_d, matB_bytes);
    cudaMalloc((void**)&matC_d, matC_bytes);

    cudaMemcpy(matA_d, matA, matA_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(matB_d, matB, matB_bytes, cudaMemcpyHostToDevice);

    const dim3 blockSize(32, 32);
    const dim3 gridSize(
        (N + blockSize.x - 1) / blockSize.x,
        (M + blockSize.y - 1) / blockSize.y,
        1
    );

    mat_mul_kernel<<<gridSize, blockSize>>>(
        matA_d, matB_d, matC_d, M, N, K
    );

    cudaMemcpy(matC, matC_d, matC_bytes, cudaMemcpyDeviceToHost);

    cudaFree(matA_d);
    cudaFree(matB_d);
    cudaFree(matC_d);
    

    delete[] matA;
    delete[] matB;
    delete[] matC;

    return 0;
}
