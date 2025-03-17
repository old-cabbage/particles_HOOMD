import numpy as np
import matplotlib.pyplot as plt

# 生成示例数据（已知曲线)
x = np.array(range(0,5100,200))
y = np.array([0,-0.12576086863975022, -0.26056087305572834, -0.40517459906000447, -0.5608154237733429, -0.7284413644288799, 
-0.90953134372233, -1.1053832159979768, -1.317348602076046, -1.5478212534375415, -1.7984408802263503, 
-2.0710108302100982, -2.368487543861027, -2.6931080063001747, -3.047317110435925, -3.4351003600629926, 
-3.8593676422026575, -4.323387700678301, -4.83177722826859, -5.387281375956857, -5.9945467923492615, 
-6.656193381324303, -7.37793492371769, -8.15669494063865, -9.011100244743533, -9.928318299666635]
)

# 进行多项式拟合
degree = 8  # 选择多项式阶数
coefficients = np.polyfit(x, y, degree)

#coefficients[-1]=0

# 生成多项式函数
poly_func = np.poly1d(coefficients)

# 显示多项式
print("拟合的多项式为：")
print(poly_func)

sump=0
for i in range(5000):
    sump+=poly_func(i+1)
print(poly_func(5001)-sump/5000)

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
