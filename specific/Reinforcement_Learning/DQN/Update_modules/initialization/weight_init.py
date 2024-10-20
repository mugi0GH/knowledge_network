from torch import nn

    # 定义 He 初始化的权重初始化函数
def he_init_weights(clas):
    # 遍历模型的每一层并初始化
    for layer in clas.modules():
        if isinstance(layer, nn.Linear):
            # 对每个线性层使用 He 初始化
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)  # 将偏置初始化为0