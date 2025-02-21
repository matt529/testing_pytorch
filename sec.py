import torch
import torch.nn as nn
import torch.optim as optim

# 生成訓練數據
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# 定義模型
model = nn.Linear(1, 1)

# 定義損失函數和優化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 優化器

# 訓練模型
for epoch in range(100):
    
    optimizer.zero_grad()   # 清空優化器
    output = model(x)       
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# 測試
test_input = torch.tensor([[5.0]])
predicted = model(test_input)
print(predicted.item())  # 預測 5 對應的輸出值
