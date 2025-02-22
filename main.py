import torch
import torch.nn as nn
import pandas as pd
# import numpy as np

# 這段省略，因為沒有檔案可用
row_df = pd.read_csv('mnist_mock_data.csv')     #假設 n x 785

# 標籤
label = row_df['label'].values                  # label：n * 1

# 特徵

row_df.drop('label', axis=1, inplace=True)      
feature = row_df.values                         # row_df：n * ( 785-1 )

# 定義數據，分為 80% data(數據) 和 20% validation(驗證、訓練)
train_features = feature[:int(len(feature)* 0.8) ]  # 80% 作為訓練
train_labels = label[:int(len(label)* 0.8) ]
val_features = feature[int(len(feature)* 0.8) : ]   # 20% 作為驗證
val_labels = label[int(len(label)* 0.8) : ]

# 需轉成 tensor結構以便訓練 loss function
train_features = torch.tensor(train_features, dtype=torch.float32).cuda()
train_labels = torch.tensor(train_labels, dtype=torch.long).cuda()
val_features = torch.tensor(val_features, dtype=torch.float32).cuda()
val_labels = torch.tensor(val_labels, dtype=torch.long).cuda()  
# 換為整數類別，因為 nn.CrossEntropyLoss() 要求入的 label 為整數類別   

# 定義模型結構
data = torch.rand(1,784).cuda() # 易錯，data 本身也要 cuda，這樣 gpu才能辨識
model = nn.Sequential(
    nn.Linear(784, 444),
    nn.ReLU(),
    nn.Linear(444, 512),
    nn.ReLU(),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, 10),     # output layer with 10 neurons, one for each digit
    nn.Softmax(dim=1)       # activation function for output layer, between 0 to 1, dim=1 表示對每一行進行Softmax操作
)                           # 用 gpu加速

model = model.cuda()        # 寫入 gpu，新手期易錯
predict = model(data)

# 訓練模型
# 梯度下降(gradient descent)(找到一組合適的 w和b 讓損失值越小越好), 想像瞎子下山
lossfunction = nn.CrossEntropyLoss()    #交叉熵損失函數

# 優化器, 不知道用哪個, 無腦用 Adam( ,lr==step_size)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

step_size = 100

# 訓練的 forloop數、迭代數，越多次損失越小，正確率越高
for i in range( step_size ):

    # 清空優化器的梯度(偏微分)
    optimizer.zero_grad()  

    #  預測的過程，算出結果
    predict = model(train_features)  
    result = torch.argmax(predict, axis=1)

    train_acc = (result == train_labels).float().mean()

    # 算出損失值(預測結果, 標準答案)
    loss = lossfunction(predict, train_labels)

    #反向傳播
    loss.backward()

    # #梯度下降
    optimizer.step()    

    print(f"train loss:{loss.item()}, train_acc:{train_acc.item()}")

    # print(f"train loss:{loss.item()}, train_acc:{train_acc.item()}")
    
    # if i % 10 == 0:
    #     print(f"Epoch {i+1}, Loss: {loss.item()}")


# predict = model(data)
# print(predict)


# nn.CrossEntropyLoss()










