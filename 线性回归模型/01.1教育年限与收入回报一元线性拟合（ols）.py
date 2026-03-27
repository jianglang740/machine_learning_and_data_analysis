import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# 设置绘图风格
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 解决中文显示（Mac用'Arial Unicode MS'）
plt.rcParams['axes.unicode_minus'] = False


# 1. 读取Stata数据
df = pd.read_stata("/Users/clinking/Downloads/Data-Finished-本科计量/grilic.dta")

# 2. 提取变量：s（教育年限）为横轴，lnw（工资对数）为纵轴
#.values 是 pandas 的属性，核心作用是提取 pandas 数据底层的 numpy 数组，丢掉索引 / 列名等元信息；
x = df["s"].values.reshape(-1, 1)  # 转为sklearn要求的二维数组
y = df["lnw"].values
print("X的形状:", x.shape) 
print("Y的形状:", y.shape)

# 3. 一元线性回归拟合
model = LinearRegression()  #调用ols回归模型
model.fit(x, y)            #使用x和y进行拟合得到alpha和brta
y_pred = model.predict(x)  # 拟合后的预测值，即样本回归线

# 4. 提取拟合参数
brta= model.coef_[0]       # 斜率（教育回报率）(取一元回归的斜率)，0是一元回归的唯一斜率，若是多元回归则coef_会有多个值
alpha = model.intercept_ # 截距（调用范式）
r_squared = model.score(x, y)# 决定系数R²（调用范式）

# 5. 绘制散点图+拟合线
plt.figure(figsize=(10, 6))
# 绘制原始数据散点
plt.scatter(x, y, alpha=0.6, color="#2E86AB", label="原始数据点")
# 绘制拟合直线
plt.plot(x, y_pred, color="#A23B72", linewidth=2, 
         label=f"拟合线: y = {alpha:.3f} + {brta:.3f}x\nR² = {r_squared:.3f}")

# 图表美化
plt.xlabel("教育年限 (s)", fontsize=12)
plt.ylabel("工资对数 (lnw)", fontsize=12)
plt.title("教育年限与工资对数的散点图及一元线性拟合", fontsize=14, pad=15)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()

# 显示图表
plt.show()

# 打印回归结果
print("="*50)
print("一元线性回归结果（教育回报率分析）")
print("="*50)
print(f"截距 (intercept): {alpha:.4f}")
print(f"斜率 (slope/教育回报率): {brta:.4f}")
print(f"决定系数 R²: {r_squared:.4f}")
print("="*50)
