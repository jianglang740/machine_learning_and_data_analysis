#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 19:02:29 2026

@author: clinking
"""
#库导入
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# 设置绘图风格
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 解决中文显示（Mac用'Arial Unicode MS'）
plt.rcParams['axes.unicode_minus'] = False

#数据提取
df = pd.read_csv("/Users/clinking/开发/线性回归与分类数据集/co2_mini.csv")

#数据预处理
print("数据概览：")
print(df.head())
print(df.describe())
print(df.info())

#随机打乱抽取数据，划分测试集与训练集
print("训练数据如下：")
sample_df = df.sample(frac=1/2, random_state=42)
print(sample_df)
x = sample_df["consumption"].values.reshape(-1,1)
y = sample_df["co2"].values
print("x与y的形状如下：")
print(x.shape)
print(y.shape)
#调用模型
model = LinearRegression()
model.fit(x,y)
y_pred = model.predict(x)
#提取参数
brta = model.coef_[0]
alpha = model.intercept_
r_squared = model.score(x,y)
#绘图
plt.figure(figsize = (10,6))
plt.scatter(x,y,color = "red",alpha=0.5, label="训练集散点数据")
plt.plot(x,y_pred,color = "green",alpha = 0.8,label = f"拟合线: y = {alpha:.3f} + {brta:.3f}x  R² = {r_squared:.3f}")
#图表美化
plt.xlabel("燃油消耗量")
plt.ylabel("二氧化碳排放量")
plt.title("燃油消耗量与二氧化碳排放量一元线性拟合")
plt.legend(fontsize=12) #触发图例渲染
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show
#参数显示
print("拟合数据如下：")
print("alpha的值是：",alpha)
print("brta的值是：",brta)
print("决定系数的值是：",r_squared)