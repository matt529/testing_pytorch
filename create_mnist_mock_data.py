import pandas as pd
import numpy as np

# 這段省略，因為沒有檔案可用
# row_df = pd.read_csv('train.csv')


# 設定參數
num_samples = 42000  # 總行數
num_pixels = 784  # 每張圖片的像素數

# 生成隨機標籤（0~9）
labels = np.random.randint(0, 10, size=(num_samples, 1))

# 生成隨機像素數據（大部分為0，少數為1~9）
pixels = np.random.choice([0] * 700 + list(range(1, 10)), size=(num_samples, num_pixels))

# 合併標籤與像素數據
data = np.hstack((labels, pixels))

# 生成 DataFrame, example: label, pixel0, pixel1,..., pixel784
df = pd.DataFrame(data, columns=["label"] + [f"pixel{i}" for i in range(num_pixels)])

# 儲存 CSV, 命名為 mnist_mock_data,  MNIST（手寫數字資料集）, 模擬（mock）的數據集,
df.to_csv("mnist_mock_data.csv", index=False)

# print("CSV 生成完成：mnist_mock_data.csv")