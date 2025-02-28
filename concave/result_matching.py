import numpy as np
import matplotlib.pyplot as plt

# 生成示例数据（已知曲线）
x = np.array([250,500,750,1000,1250,1500,1750,2000,2250,2500,2750,3000,3250,3500,3750,4000,4250,4500,4750,5000])
y = np.array([-0.08627059325584645, -0.17682299363791487, -0.27127958509857875, -0.3703652297128294, 
                -0.4752862778966035, -0.5846882260253674, -0.7001139321156399, -0.8229982908818053, 
                -0.9516380264084452, -1.0848171703730034, -1.2299269514831959, -1.3805988911448706, 
                -1.5416823527684764, -1.710542442106814, -1.8863246646022045, -2.074086863591579, 
                -2.2764487426070126, -2.490840379594035, -2.713169231007469])

# 进行多项式拟合
degree = 4  # 选择多项式阶数
coefficients = np.polyfit(x, y, degree)

# 生成多项式函数
poly_func = np.poly1d(coefficients)

# 显示多项式
print("拟合的多项式为：")
print(poly_func)

# 绘制原始数据和拟合曲线
x_fit = np.linspace(min(x), max(x), 100)
y_fit = poly_func(x_fit)

plt.scatter(x, y, color='red', label='Original Data')
plt.plot(x_fit, y_fit, label=f'Fitted Polynomial (degree={degree})')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Fit')
plt.show()
