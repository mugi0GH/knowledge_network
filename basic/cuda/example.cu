#include <iostream>
#include <cuda_runtime.h>

// CUDA 核函数：计算向量加法
__global__ void vectorAdd(int *a, int *b, int *c, int N) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < N) {
        c[index] = a[index] + b[index];
    }
}

int main() {
    int N = 1 << 20;  //1左移20位。 1M（百万）元素，值为1x2的20次方 = 1048576
    size_t size = N * sizeof(int);

    // 主机内存分配
    int *h_a = (int*)malloc(size);
    int *h_b = (int*)malloc(size);
    int *h_c = (int*)malloc(size);

    // 初始化输入数据
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // 设备内存分配
    int *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    // 将数据从主机复制到设备
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // 设置线程和块的维度
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // 启动核函数
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    // 将结果从设备复制到主机
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // 检查结果
    for (int i = 0; i < N; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            std::cerr << "错误：结果不匹配!" << std::endl;
            break;
        }
    }

    std::cout << "程序成功运行!" << std::endl;

    // 清理内存
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
