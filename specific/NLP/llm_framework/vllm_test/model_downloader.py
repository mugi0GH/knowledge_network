from modelscope import snapshot_download

# Qwen/Qwen2.5-VL-7B-Instruct 是视觉语言模型，支持图像理解和文本生成
# 适合 Agentic LLM 入门，自带视觉能力无需外挂 API
model_dir = snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct', cache_dir='./weights')
print(f"模型已下载到: {model_dir}")
