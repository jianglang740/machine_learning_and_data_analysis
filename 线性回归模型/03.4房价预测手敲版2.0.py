# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#库导入
import pandas as pd
import matplotlib.pyplot as plt  
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

#数据预览
df = pd.read_csv("/Users/clinking/开发/线性回归与分类数据集/house_mini.csv")
print(df.head())
print(df.info())
print(df.describe())
#预处理
X = df[["sqft_living","sqft_lot","floors"]]
y = df["price"]
# 2. 删除特征变量X中的缺失值行（如果有的话）
X_clean = X.dropna()  # dropna()：删除包含空值的行

# 3. 确保目标变量y和X_clean行数一致（避免数据不匹配）
# 用X_clean的索引筛选y，保证一一对应
y_clean = y[X_clean.index]
print("x的形状如下：",X_clean.shape)
print("y的形状如下：",y_clean.shape)

#划分训练集与测试集
x_train,x_test,y_train,y_test = train_test_split(X_clean,y_clean,test_size = 0.2,random_state = 42)
print("x_train,y_train的形状如下：",x_train.shape,y_train.shape)
#调用模型和提取参数
model = LinearRegression()
model.fit(x_train,y_train)
alpha = model.intercept_
brta = model.coef_[2]
y_pred_train = model.predict(x_train)
y_pred_test = model.predict(x_test)
# 可视化（改进版：2行2列子图 + 真实值vs预测值散点图）
plt.figure(figsize=(18, 10))

# 子图1: sqft_living 与真实价格
plt.subplot(2, 2, 1)
plt.scatter(x_train.iloc[:, 0], y_train, color="green", alpha=0.5)
plt.title("sqft_living vs price")
plt.xlabel("sqft_living")
plt.ylabel("price")

# 子图2: sqft_lot 与 price
plt.subplot(2, 2, 2)
plt.scatter(x_train.iloc[:, 1], y_train, color="green", alpha=0.5)
plt.title("sqft_lot vs price")
plt.xlabel("sqft_lot")
plt.ylabel("price")

# 子图3: floors 与 price
plt.subplot(2, 2, 3)
plt.scatter(x_train.iloc[:, 2], y_train, color="green", alpha=0.5)
plt.title("floors vs price")
plt.xlabel("floors")
plt.ylabel("price")

# 子图4: 真实值 vs 预测值散点图（核心改进）
plt.subplot(2, 2, 4)
plt.scatter(y_test, y_pred_test, color="steelblue", alpha=0.6, s=20)
# 绘制 y=x 参考线（完美预测线）
min_val = min(y_test.min(), y_pred_test.min())
max_val = max(y_test.max(), y_pred_test.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="完美预测线")
plt.title("真实价格 vs 预测价格")
plt.xlabel("真实价格")
plt.ylabel("预测价格")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()  # 自动调整子图间距，避免标签重叠
plt.show()