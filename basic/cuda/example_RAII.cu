#include <iostream>
#include <vector>
#include <memory>
#include <cuda_runtime.h>

// ========= 工具：错误检查宏 =========
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__)          \
                      << " (" << __FILE__ << ":" << __LINE__ << ")\n";        \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// ========= RAII：显存智能指针 =========
template <typename T>
struct CudaDeleter {
    void operator()(T* p) const noexcept {
        if (p) {
            cudaFree(p);   // 用 cudaFree 释放显存
        }
    }
};

template <typename T>
using device_ptr = std::unique_ptr<T, CudaDeleter<T>>;

// 简单的 helper：申请 N 个 T 的显存并返回 unique_ptr
template <typename T>
device_ptr<T> make_device_buffer(std::size_t N) {
    T* raw = nullptr;
    CUDA_CHECK(cudaMalloc(&raw, N * sizeof(T)));
    return device_ptr<T>(raw);
}

// ========= CUDA 核函数：向量加法 =========
__global__ void vectorAdd(const int* a, const int* b, int* c, int N) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < N) {
        c[index] = a[index] + b[index];
    }
}

int main() {
    // ---------------- 参数 ----------------
    int N = 1 << 20;  // 1M 元素
    std::size_t size_bytes = static_cast<std::size_t>(N) * sizeof(int);

    // ---------------- 主机内存：用 std::vector，自动释放 ----------------
    std::vector<int> h_a(N), h_b(N), h_c(N);

    // 初始化输入数据
    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = 2 * i;
    }

    // ---------------- 设备内存：用 unique_ptr + 自定义 deleter ----------------
    auto d_a = make_device_buffer<int>(N);
    auto d_b = make_device_buffer<int>(N);
    auto d_c = make_device_buffer<int>(N);

    // ---------------- 主机 -> 设备 拷贝 ----------------
    CUDA_CHECK(cudaMemcpy(d_a.get(), h_a.data(), size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b.get(), h_b.data(), size_bytes, cudaMemcpyHostToDevice));

    // ---------------- 线程块配置 ----------------
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // ---------------- 启动核函数 ----------------
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(
        d_a.get(), d_b.get(), d_c.get(), N
    );

    // 检查 kernel 是否启动成功
    CUDA_CHECK(cudaGetLastError());
    // 等待 GPU 计算完成
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---------------- 设备 -> 主机 拷贝结果 ----------------
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c.get(), size_bytes, cudaMemcpyDeviceToHost));

    // ---------------- 验证结果 ----------------
    bool success = true;
    for (int i = 0; i < 10; ++i) {  // 只检查前 10 个
        int expected = h_a[i] + h_b[i];
        if (h_c[i] != expected) {
            std::cerr << "结果错误：index " << i
                      << ", got " << h_c[i]
                      << ", expected " << expected << std::endl;
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "前 10 个结果正确，示例运行成功！" << std::endl;
    }

    // 注意：这里不需要手动 free / cudaFree
    // - h_a/h_b/h_c 是 std::vector，出作用域自动释放
    // - d_a/d_b/d_c 是 unique_ptr，出作用域自动调用 CudaDeleter => cudaFree

    return 0;
}
