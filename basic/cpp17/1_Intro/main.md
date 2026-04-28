# **0. 基础概念与语言环境**

---

# **0.1 C++ 编译器（GCC / Clang / MSVC）**

C++ 代码需要经过 **编译器（compiler）** 转成机器能运行的可执行文件。常见三大主流编译器：

---

## **① GCC（GNU Compiler Collection）**

* Linux / WSL / Mac（部分）环境最常见
* g++ 用于编译 C++
* 对 C++17 支持非常成熟
* 开源生态最广

编译示例：

```bash
g++ -std=c++17 main.cpp -o app
```

---

## **② Clang（LLVM 编译器）**

* 错误提示信息更友好
* 在 macOS 默认使用
* 也可以在 Linux/Windows 使用

编译示例：

```bash
clang++ -std=c++17 main.cpp -o app
```

---

## **③ MSVC（Microsoft Visual C++）**

* Windows Visual Studio / VSCode 上常用
* 对 C++17 支持良好，但某些细节与 GCC/Clang 不完全一致

编译示例：

```bash
cl /std:c++17 main.cpp
```

---

### **重点总结：**

| 平台          | 编译器             |
| ----------- | --------------- |
| Linux / WSL | GCC 或 Clang     |
| macOS       | Clang           |
| Windows     | MSVC 或 Clang-cl |

所有主流编译器都支持 **C++17**。

---

---

# **0.2 C++17 标准与兼容性**

C++ 按年代发布版本，C++11 → C++14 → **C++17** → C++20 → C++23 …

C++17 是非常成熟且广泛使用的标准。

---

## **C++17 的常用特性（你未来常会遇到）**

### 1）`std::optional<T>`

表示“可能有值可能没值”，避免返回 `nullptr` 或特殊值。

---

### 2）`std::variant`

类型安全的 union，可以让变量在多个类型间切换。

---

### 3）`std::string_view`

* **非常轻量的字符串只读视图**
* 不复制数据，不分配内存
* 适用于性能敏感的代码

```cpp
void foo(std::string_view s);
```

---

### 4）结构化绑定（structured binding）

```cpp
auto [x, y] = getPoint();
```

---

### 5）`constexpr if`

编译期分支，用来减少模板代码复杂度。

---

### 6）并行算法 `std::execution`

允许你通过多线程并行执行大量操作（需要正确配置线程环境）。

---

## **为什么选 C++17？**

* 支持最广、最稳定
* 工具链对它最友好
* 多数开源项目默认 C++17

---

---

# **0.3 CMake 构建系统（现代 CMake）**

CMake 是跨平台 C/C++ 构建系统。

它的任务是：

* 告诉编译器要编译哪些文件
* 设置编译选项（如 C++17）
* 配置包含路径、链接哪些库
* 管理大型工程结构

---

## **现代 CMake（target-based）核心思想**

以 **目标（target）** 为中心，例如：

* 可执行程序（executable target）
* 静态库（static library target）
* 动态库（shared library target）

### 示例目录：

```
project/
  CMakeLists.txt
  src/
     main.cpp
```

### 最小 CMake 示例：

**CMakeLists.txt**

```cmake
cmake_minimum_required(VERSION 3.14)
project(MyApp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

add_executable(myapp src/main.cpp)
```

生成：

```bash
cmake -B build
cmake --build build
```

---

## **现代 CMake 常用命令**

### **add_executable()**

创建一个可执行文件。

### **add_library()**

创建静态库或动态库。

### **target_include_directories()**

给某个目标设置头文件路径。

### **target_link_libraries()**

指定链接哪些库。

---

## 现代 CMake 的优势

* 模块化清晰（每个目标独立）
* 可维护性高
* 不靠全局变量，适合大型项目

---

---

# **0.4 项目结构、头文件、编译链接、库的概念（静态 vs 动态）**

---

# **1）项目结构**

典型 C++ 项目结构：

```
project/
  CMakeLists.txt
  include/         ← 头文件（.h / .hpp）
  src/             ← 源文件（.cpp）
  lib/             ← 第三方静态/动态库
  build/           ← 生成文件（不提交 git）
```

---

# **2）头文件 (.h / .hpp)**

负责：

* 函数声明
* 类的声明
* 模板类的实现（模板必须写头文件）

不负责：

* 函数实现（非模板）

---

# **3）源文件 (.cpp)**

负责：

* 实现头文件中声明的函数/类

---

# **4）编译与链接**

## 编译（compile）

把 `.cpp` 编译成 `.o`（目标文件）。

## 链接（link）

把多个 `.o` + 库文件组合成可执行文件。

---

# **5）静态库 vs 动态库**

---

## **静态库（.a / .lib）**

### 特点：

* 在编译时打包进最终程序
* 运行时不需要额外文件
* 可执行文件变大
* 不支持热更新

生成静态库：

```cmake
add_library(utils STATIC utils.cpp)
```

---

## **动态库（.so / .dll / .dylib）**

### 特点：

* 程序运行时加载
* 文件小
* 可以更换库而不重新编译整个程序
* 部署时必须携带 `.so`

生成动态库：

```cmake
add_library(utils SHARED utils.cpp)
```

链接动态库：

```cmake
target_link_libraries(myapp PRIVATE utils)
```

---

# **总结（本章重点）**

| 主题       | 最重要的理解方式                     |
| -------- | ---------------------------- |
| C++ 编译器  | gcc / clang / msvc 都支持 C++17 |
| C++17 标准 | 更现代、稳定、广泛用于工业                |
| CMake    | 现代 target-based 写法最重要        |
| 头文件      | 声明                           |
| 源文件      | 实现                           |
| 静态库      | 编译时打包                        |
| 动态库      | 运行时加载                        |

---

如果你需要，我可以帮你继续扩展下一节：

### **1. C++ 语言基础（变量、作用域、编译流程、头文件规则）**

或

### **1. 现代 C++17 语法特性精讲（string_view, optional, RAII, 智能指针）**

你想继续哪一部分？
