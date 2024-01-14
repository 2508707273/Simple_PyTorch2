import torch

from model1 import loader


@torch.no_grad()
def test():
    # 从磁盘加载模型
    model = torch.load('model1.model')

    # 模型进入测试模式,关闭dropout等功能
    model.eval()

    # 获取一批数据
    x, y = next(iter(loader))

    # 模型计算结果
    out = model(x).argmax(dim=1)

    print(out, y)
    print(out == y)
