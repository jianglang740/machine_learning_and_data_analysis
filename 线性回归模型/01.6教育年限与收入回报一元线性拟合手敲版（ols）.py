#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 15:16:23 2026

@author: clinking
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd

# 初始化数据
df = pd.read_stata("/Users/clinking/Downloads/Data-Finished-本科计量/grilic.dta")

# 提取数据
print("初始数据: ")
print(df)
new_df = df[["s","lnw"]]
print("提取后的数据:")
print(new_df)
x = new_df["s"]
y = new_df["lnw"]

# 随机抽取 2/3 的数据（核心步骤）
# random_state=42 固定随机种子，保证每次运行结果一致，可复现
# sample是pandas的方法，用来随机打乱数据
sample_df = new_df.sample(frac=2/3, random_state=42)
print("抽样后的数据:")
print(sample_df)

# 4. 拆分x(s)和y(lnw)
new_x = sample_df["s"].values.reshape(-1,1)  # 特征，必须是二维数组（双括号）
new_y = sample_df["lnw"].values  # 目标，一维数组

# 5. 验证抽样结果
print(f"\n原始数据总行数: {len(new_df)}")
print(f"抽样后数据总行数: {len(sample_df)}")
print(f"抽样比例: {len(sample_df)/len(new_df):.2%}")
print("new_x的形状", new_x.shape)
print("new_y的形状", new_y.shape)

# 绘制初始散点图
plt.figure(figsize = (10,6))
# 初始数据：红色散点（覆盖在红色上，更突出）
plt.scatter(x, y, color="red", alpha=0.5, label="初始全部数据")
# 抽样数据：绿色散点
plt.scatter(new_x, new_y, color="green", alpha=0.8, label="2/3抽样数据")

plt.xlabel("教育年限 s", fontsize=12)
plt.ylabel("工资对数 lnw", fontsize=12)
plt.title("s vs lnw：初始数据 vs 2/3抽样数据", fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()

# 调用模型
model = LinearRegression()
model.fit(new_x, new_y)
y_pred = model.predict(new_x)

# 提取参数
coef = model.coef_[0]
alpha = model.intercept_
r_squared = model.score(new_x, new_y)

# 提取参数
plt.figure(figsize = (10,6))
plt.scatter(new_x, new_y, color="green", alpha=0.5, label="抽样散点数据")
# 修正label的格式化错误，添加空格和正确的R²写法
plt.plot(new_x, y_pred, color="#A23B72", linewidth=2, label=f"拟合线: y = {alpha:.3f} + {coef:.3f}x  R² = {r_squared:.3f}")

# 图表美化
plt.xlabel("教育年限 (s)", fontsize=12)
plt.ylabel("工资对数 (lnw)", fontsize=12)
plt.title("教育年限与工资对数的抽样数据散点图及一元线性拟合", fontsize=14, pad=15)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()

# 显示图表
plt.show()