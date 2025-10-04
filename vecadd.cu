__global__ void vecAdd(const float* A, const float* B, float* C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char** argv) {
    int N = 1<<20;
    float *x = new float[N];
    float *y = new float[N];
    float *z = new float[N];

    for(int i=0; i<N; i++) {
        x[i] = rand();
        y[i] = rand();
    }

    float *x_d, *y_d, *z_d;
    cudaMalloc((void**)&x_d, N*sizeof(float));
    cudaMalloc((void**)&y_d, N*sizeof(float));
    cudaMalloc((void**)&z_d, N*sizeof(float));

    cudaMemcpy(x_d, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, N*sizeof(float), cudaMemcpyHostToDevice);

    const unsigned int blockSize = 256;
    const unsigned int numBlocks = (N + blockSize - 1) / blockSize;
    vecAdd<<<numBlocks, blockSize>>>(x_d, y_d, z_d, N);
    cudaMemcpy(z, z_d, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(z_d);

    delete[] x;
    delete[] y;
    delete[] z;

    return 0;
}