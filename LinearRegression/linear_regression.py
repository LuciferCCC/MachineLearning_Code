import numpy as np
from utils.features.prepare_for_training import prepare_for_training


class LinearRegression:

    def __init__(self, data, labels, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
        """
        1. 对数据进行预处理操作
        2. 先得到所有的特征个数
        3. 初始化参数矩阵
        """

        # 预处理操作，返回处理后的数据，平均值，标准差
        (data_processed, features_mean, features_deviation) \
            = prepare_for_training(data, polynomial_degree, sinusoid_degree, normalize_data=True)
        self.data = data_processed
        self.labels = labels
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data

        num_features = self.data.shape[1]  # 返回数据列数
        self.theta = np.zeros((num_features, 1))  # 生成参数 theta

    def train(self, alpha, num_iterations=500):
        """
        训练模块， 执行梯度下降
        :param alpha: 学习率
        :param num_iterations: 训练次数
        :return: theta，每次训练的损失值
        """
        cost_history = self.gradient_descent(alpha, num_iterations)
        return self.theta, cost_history

    def gradient_descent(self, alpha, num_iterations):  # 梯度下降函数
        """
        梯度下降模块
        :param alpha: 学习率
        :param num_iterations: 训练次数
        :return: 本次迭代的损失值
        """
        cost_history = []  # 记录每次迭代之后真实值与预测值之间的差异
        for _ in range(num_iterations):
            self.gradient_step(alpha)
            cost_history.append(self.cost_function(self.data, self.labels))
        return cost_history

    def gradient_step(self, alpha):
        """
        梯度下降参数更新方法，注意是矩阵运算
        :param alpha: 学习率
        """
        num_examples = self.data.shape[0]  # 样本个数
        prediction = LinearRegression.hypothesis(self.data, self.theta)  # 计算出预测值
        delta = prediction - self.labels  # 得到误差
        theta = self.theta
        theta = theta - alpha * (1 / num_examples) * (np.dot(delta.T, self.data)).T  # 注意两个转置细节
        self.theta = theta  # 更新 theta

    def cost_function(self, data, labels):
        """
        损失计算
        :param data: 数据
        :param labels: 真实值
        :return: 损失值
        """
        num_examples = data.shape[0]
        delta = LinearRegression.hypothesis(self.data, self.theta) - labels
        cost = (1 / 2) * np.dot(delta.T, delta) / num_examples  # 平方取一半
        return cost[0][0]

    @staticmethod  # 弄成静态方法，不用实例化就可以调用
    def hypothesis(data, theta):  # 计算预测值
        predictions = np.dot(data, theta)
        return predictions

    def get_cost(self, data, labels):
        """
        返回损失值
        :param data: 数据
        :param labels: 真实值
        :return: 损失值
        """
        data_processed = \
            prepare_for_training(data, self.polynomial_degree, self.sinusoid_degree, self.normalize_data)[0]
        return self.cost_function(data_processed, labels)

    def predict(self, data):
        """
        返回预测值
        :param data: 数据
        :return: 预测值
        """
        data_processed = \
            prepare_for_training(data, self.polynomial_degree, self.sinusoid_degree, self.normalize_data)[0]
        predictions = LinearRegression.hypothesis(data_processed, self.theta)
        return predictions
