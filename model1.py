def get_rectangle():
    import random

    width = random.random()
    height = random.random()

    return width, height, int(width > height)

import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, num_samples):
        super().__init__()
        self.num_samples = num_samples

    def __getitem__(self, index):
        width, height, label = get_rectangle()

        x = torch.tensor([width, height])
        y = torch.tensor(label)
        return x, y

    def __len__(self):
        return self.num_samples

dataset = Dataset(1000)

# print(dataset[0],len(dataset))

loader = torch.utils.data.DataLoader(dataset = dataset, batch_size=8, shuffle=True, drop_last=True)

print(len(loader),next(iter(loader)))

# 全连接网络模型
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, x):
        y = self.linear(x)
        return y

model1 = Model()
print(model1(torch.randn(8,2)).shape)


class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(2, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 2),
            torch.nn.Softmax(dim=1),
        )

    def forward(self, x):
        y = self.fc(x)
        return y

model2 = Model2()
print(model2(torch.randn(8,2)))
print(1e-4)

def train():
    # 优化器
    optimizer = torch.optim.Adam(model2.parameters(), lr=1e-4)

    # 损失函数
    loss_fn = torch.nn.CrossEntropyLoss()

    # 训练
    model2.train()

    for epoch in range(100):
        for i, (x, y) in enumerate(loader):
            out = model2(x)
            loss = loss_fn(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 20 == 0:
            acc = (out.argmax(dim=1) == y).float().mean()
            acc2 = (out.argmax(dim=1) == y).sum().item() / len(y)
            print(f"epoch:{epoch}, loss:{loss.item():.4f}, acc:{acc:.4f}, acc2:{acc2:.4f}")

        torch.save(model2, "model1.model")


train()




