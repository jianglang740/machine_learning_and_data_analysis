#导入相关库
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib .pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

#读取数据
df = pd.read_csv("/Users/clinking/开发/线性回归与分类数据集/house_mini.csv")
print(df)
print(df.info())
#提取数据
a = df["bedrooms"].values  #卧室数量
b = df["bathrooms"].values  #浴室数量
c = df["sqft_living"].values  #居住面积
d = df["sqft_lot"].values  #地块面积
e = df["floors"].values    #楼层数
f = df["zipcode"].values    #邮政编码——标示住宅区
y = df["price"].values
#先绘制散点图查看数据分布
plt.figure(figsize=(18, 10))

# 第1行
plt.subplot(2, 3, 1)  # 2行3列，第1个位置
plt.scatter(a, y)
plt.title("bedrooms vs price")

plt.subplot(2, 3, 2)  # 第2个位置
plt.scatter(b, y)
plt.title("bathrooms vs price")

plt.subplot(2, 3, 3)  # 第3个位置
plt.scatter(c, y)
plt.title("sqft_living vs price")

# 第2行
plt.subplot(2, 3, 4)  # 第4个位置
plt.scatter(d, y)
plt.title("sqft_lot vs price")

plt.subplot(2, 3, 5)  # 第5个位置
plt.scatter(e, y)
plt.title("floors vs price")

plt.subplot(2, 3, 6)  # 第6个位置
plt.scatter(f, y)
plt.title("zipcode vs price")

plt.tight_layout()  # 自动调整子图间距
plt.show()
#划分训练集与测试集
X = df["sqft_living"].values.reshape(-1,1)
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)
#调用模型
model = LinearRegression()
model.fit(X_train,y_train)
alpha = model.intercept_
beta = model.coef_[0]
y_pred_train = model.predict(X_train)
#可视化
figure = plt.figure(figsize = (10,6))
plt.scatter(X_train,y_train,color = "green",alpha = 0.5)
plt.scatter(X_test,y_test,color = "blue",alpha = 0.5)
plt.plot(X_train,y_pred_train,color = "red",alpha = 0.8)
plt.title("可居住面积与房价线性拟合（train）")
plt.show()