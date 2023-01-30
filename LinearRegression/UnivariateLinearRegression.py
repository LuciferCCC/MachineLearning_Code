# 损失值
# 开始的损失： 14.663548222384891
# 最后的损失： 0.22012803686318996

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from linear_regression import LinearRegression

# 单特征线性回归实验
# 根据表格中各个国家的 gdp 指标预测各个国家的幸福指数

data = pd.read_csv('../data/world-happiness-report-2017.csv')

# 得到训练和测试数据
train_data = data.sample(frac=0.8)  # 从 data 中随机抽取 80% 的数据并返回一个列表
test_data = data.drop(train_data.index)  # 删除重复数据，即返回的数据剩余 20% 的一个列表

input_param_name = 'Economy..GDP.per.Capita.'
output_param_name = 'Happiness.Score'

x_train = train_data[[input_param_name]].values
y_train = train_data[[output_param_name]].values

x_test = test_data[input_param_name].values
y_test = test_data[output_param_name].values

plt.scatter(x_train, y_train, label="Train data")
plt.scatter(x_test, y_test, label="Test data")
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.title('Happy')
plt.legend()  # 添加图例
plt.show()

# 训练模型

num_iterations = 500
learning_rate = 0.01
linear_regression = LinearRegression(x_train, y_train)
(theta, cost_history) = linear_regression.train(learning_rate, num_iterations)

print('开始的损失：', cost_history[0])
print('最后的损失：', cost_history[-1])

plt.plot(range(num_iterations), cost_history)
plt.xlabel('Iter')
plt.ylabel('cost')
plt.title('GD')
plt.show()

# 测试数据
predictions_num = 100  # 测试数据个数：只测试 100 个
x_predictions = np.linspace(x_train.min(), x_train.max(), predictions_num).reshape(predictions_num, 1)  # reshape定义成矩阵
y_predictions = linear_regression.predict(x_predictions)
plt.scatter(x_train, y_train, label="Train data")
plt.scatter(x_test, y_test, label="Test data")
plt.plot(x_predictions, y_predictions, 'r', label='Prediction')  # 绘制预测曲线
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.title('Happy')
plt.legend()  # 添加图例
plt.show()
