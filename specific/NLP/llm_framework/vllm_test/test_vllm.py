"""
vLLM 推理测试脚本
使用 Qwen2.5-VL-7B-Instruct 模型进行文本生成测试

注意：VL 模型加载时会同时加载视觉编码器，首次加载需要一些时间。

运行方式：
  pixi run python vllm_test/test_vllm.py
"""

import os
import sys

# 禁用 vLLM 的 CUDA forward compatibility 路径覆盖，
# 避免它重置 LD_LIBRARY_PATH 导致 libstdc++ 找不到
os.environ["VLLM_ENABLE_CUDA_COMPATIBILITY"] = "0"

# 禁用 flashinfer 版本检查，避免 flashinfer-cubin 版本不匹配错误
os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"

# 修复 libstdc++ 版本冲突：优先使用 pixi 环境自带的库
# 注意：vLLM 使用 spawn 方式启动子进程，子进程会重新导入此模块，
# 因此必须在模块级别设置 LD_LIBRARY_PATH，确保子进程也能继承
pixi_lib = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        ".pixi", "envs", "default", "lib")
cuda_lib = "/usr/local/cuda/targets/x86_64-linux/lib"
os.environ.setdefault("LD_LIBRARY_PATH", "")
for lib in [pixi_lib, cuda_lib]:
    if lib not in os.environ["LD_LIBRARY_PATH"]:
        os.environ["LD_LIBRARY_PATH"] = f"{lib}:{os.environ['LD_LIBRARY_PATH']}"

# 设置 CUDA 头文件路径，供 Triton JIT 编译使用
cuda_include = "/usr/local/cuda/targets/x86_64-linux/include"
os.environ.setdefault("C_INCLUDE_PATH", "")
if cuda_include not in os.environ["C_INCLUDE_PATH"]:
    os.environ["C_INCLUDE_PATH"] = f"{cuda_include}:{os.environ['C_INCLUDE_PATH']}"
os.environ.setdefault("CPLUS_INCLUDE_PATH", "")
if cuda_include not in os.environ["CPLUS_INCLUDE_PATH"]:
    os.environ["CPLUS_INCLUDE_PATH"] = f"{cuda_include}:{os.environ['CPLUS_INCLUDE_PATH']}"

# 在导入 vLLM 之前，先手动加载 pixi 环境的 libstdc++.so.6，
# 避免动态链接器使用系统旧版本（缺少 GLIBCXX_3.4.31）
import ctypes
pixi_libstdcpp = os.path.join(pixi_lib, "libstdc++.so.6")
if os.path.exists(pixi_libstdcpp):
    try:
        ctypes.CDLL(pixi_libstdcpp, mode=ctypes.RTLD_GLOBAL)
    except Exception:
        pass  # 如果加载失败，继续尝试

# 在导入 vLLM 之前先导入 torch，确保 LD_LIBRARY_PATH 在 torch 加载时已生效
import torch

from vllm import LLM, SamplingParams


def main():
    # ModelScope 下载时会把 / 替换为 ___
    model_path = "weights/Qwen/Qwen2___5-VL-7B-Instruct"

    print("Loading model...")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        trust_remote_code=True,
        enforce_eager=True,
    )

    # 使用聊天格式的消息列表，让 vLLM 自动应用 chat template
    messages = [
        [{"role": "user", "content": "Hello, what is your name?"}],
        [{"role": "user", "content": "What is the capital of France?"}],
    ]

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        max_tokens=256,
    )

    print("Generating...")
    outputs = llm.chat(messages, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated_text}")
        print("---")


if __name__ == "__main__":
    main()
