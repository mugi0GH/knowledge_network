# 🔥 **C++17 完整知识架构列表（工程师版）**

---

# **0. 基础概念与语言环境**

* C++ 编译器（GCC、Clang、MSVC）
* C++17 标准与兼容性
* CMake 构建系统（现代 CMake）
* 项目结构、头文件、编译链接、库的概念（静态/动态）

---

# **1. C++ 语法基础（Core language）**

### ✔ 基础语法

* 变量、常量、数据类型
* 引用 reference（左值引用、右值引用）
* 指针与指针运算
* auto 类型推导
* 类型转换（static_cast / reinterpret_cast / const_cast）

### ✔ 控制结构

* if / switch
* while / for / range-for

---

# **2. 内存模型与资源管理（非常关键）**

### ✔ 内存区域

* stack / heap / global / static
* 对象生命周期与作用域

### ✔ 手动内存管理

* new / delete
* new[] / delete[]
* RAII (Resource Acquisition Is Initialization)

### ✔ 智能指针（C++17 工程会大量使用）

* std::unique_ptr
* std::shared_ptr
* std::weak_ptr
* make_unique / make_shared（减少内存碎片）
* enable_shared_from_this

### ✔ Move 语义与右值引用

* std::move
* 移动构造/移动赋值
* 完美转发 std::forward

这是 CUDA / TensorRT / Zero-copy buffer / Frame pooling 里非常重要的知识点。

---

# **3. 面向对象 OOP（C++ 项目骨架核心）**

* 类、结构体 struct
* 构造函数、析构函数、拷贝构造、移动构造
* 访问控制（public / private / protected）
* 继承 inheritance
* 多态 polymorphism（virtual / override / final）
* 接口类（纯虚函数）
* 抽象类
* 虚函数表 vtable
* 动态绑定与 RTTI（dynamic_cast）

---

# **4. 模板与泛型编程（C++ 的强大核心）**

### ✔ 函数模板 / 类模板

### ✔ 模板特化

* 全特化
* 偏特化

### ✔ 模板元编程基础（TMP）

* constexpr if
* type traits（std::is_same / is_base_of / enable_if）

### ✔ C++17 新增

* **fold expression（模板参数包展开）**
* **inline variables**
* **constexpr lambda**

模板是 TensorRT plugin、CUDA kernels、Eigen、OpenCV 的核心。

---

# **5. 标准库 STL（工程最常用）**

### ✔ 容器 Containers

* vector（最常用）
* array
* deque
* list
* forward_list
* map / multimap
* unordered_map / unordered_set
* set / multiset
* string / string_view

### ✔ 迭代器 Iterators

* input / output / forward / bidirectional / random-access
* iterator invalidation

### ✔ 算法 Algorithms

* sort / find / transform / accumulate / copy
* lower_bound / upper_bound
* remove_if / unique
* all_of / any_of / none_of

### ✔ Utility

* pair / tuple
* optional（C++17 新增）
* variant
* any
* chrono（时间相关）
* filesystem（C++17 正式加入）

---

# **6. 函数式编程**

* lambda 表达式
* 捕获方式（值捕获/引用捕获/隐式捕获）
* std::function
* bind
* functor（重载 operator()）

---

# **7. 并发与多线程（部署与实时视频系统重点）**

### ✔ 线程模型

* std::thread
* detach / join
* thread_local

### ✔ 同步原语

* mutex
* lock_guard
* unique_lock
* shared_mutex（C++17）
* condition_variable
* atomic 原子操作（非常关键）

### ✔ 并发容器

* 无锁队列（使用 atomic 自己实现）
* 环形缓冲区 ring buffer

### ✔ C++17 并行算法

* execution::par
* execution::par_unseq

你的多线程 pipeline（capture, detect, track, pose, analysis）都将用到这些知识。

---

# **8. 错误处理与异常机制**

* try / catch / throw
* noexcept
* std::exception
* 错误码 vs 异常
* 如何写出不使用异常的高性能代码（嵌入式常见）

---

# **9. 工程化与模块化**

### ✔ 模块组织

* 多文件结构
* include guard / #pragma once
* 头文件 vs 源文件

### ✔ 编译流程

* 预处理 → 编译 → 汇编 → 链接

### ✔ 静态库 / 动态库

* .a / .so / .dll 的概念
* 如何用 CMake 链接第三方库

### ✔ API 设计

* PIMPL（隐藏实现细节）
* 接口与抽象层设计

### ✔ C++17 的 inline variable 减少链接冲突

---

# **10. 性能优化（你的 AI pipeline 最重要部分）**

### ✔ 基础性能

* 对齐 alignment
* cache line 友好（struct of arrays vs array of struct）
* SIMD（SSE/AVX）
* move 语义减少拷贝

### ✔ 内存优化

* 内存池（pool allocator）
* 对象复用
* lock-free 结构

### ✔ 编译器优化

* O2 / O3 / march native
* LTO（Link Time Optimization）
* 判断何时 inline 有效

---

# **11. 系统编程**

* 文件 IO
* socket 网络编程（POSIX）
* 多进程（fork、shared memory、pipe）
* epoll 事件循环

---

# **12. C++ 与 CUDA / TensorRT / ONNXRuntime 互操作**

### ✔ CUDA 与 C++交互

* kernel<<<>>> 调度
* host/device 内存模型
* RAII 包装 cudaMalloc
* 流 stream
* 事件 event

### ✔ TensorRT C++17 API

* builder / engine / context
* device buffer 管理
* plugin 动态库编写

### ✔ ONNX Runtime C++ API

### ✔ RKNN C++ API

### ✔ Ascend ACL C++ 接口

---

# **13. 测试与调试工具**

* gdb / lldb
* valgrind（内存泄漏检查）
* sanitizers（ASan, TSan, UBSan）
* unit test（GoogleTest）

---

# **14. 现代 CMake 构建系统**

* target_link_libraries
* PUBLIC / PRIVATE / INTERFACE
* target_include_directories
* find_package
* 配置头文件 configure_file

---

# 📌 总结：这是 C++17 工程师完整能力树

如果把它画成大纲结构：

```
C++17 全栈工程能力树
├── 语法基础
├── 内存模型与 RAII
├── 面向对象
├── 模板与泛型
├── STL
├── 并发与同步
├── 错误处理
├── 工程化（CMake + 项目结构）
├── 性能优化
├── 系统编程
├── CUDA / TensorRT / Edge AI 互操作
├── 测试与调试
└── 现代 CMake
```
