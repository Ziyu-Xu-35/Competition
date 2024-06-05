
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime

# 检查GPU是否可用
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# 读取数据
data = pd.read_csv(r"C:\Files and data\Edge_download\trade-price-ir-vegas.csv")

# 检查数据类型
print(data.dtypes)

# 假设第一列是日期列，取出日期列
date_column = pd.to_datetime(data['Value Date'])  # 将日期列转换为datetime类型
data = data.iloc[:, 3:]  # 保留数值列

# 选择数值列
numeric_columns = ['Zero Rate Shock', 'TV', 'Vega']
data = data[numeric_columns]

# 检查是否成功只选择了数值列
print(data.dtypes)

# 合并相同日期的数据，取平均值
data['Value Date'] = date_column
data = data.groupby('Value Date').mean().reset_index()

# 重新设置日期列
date_column = data['Value Date']
data = data[numeric_columns]

# 检查数据范围
print("Zero Rate Shock range before scaling:", data['Zero Rate Shock'].min(), data['Zero Rate Shock'].max())

# 仅使用部分数据以减少训练时间
data = data.sample(frac=0.2, random_state=42)  # 使用20%的数据样本

# 数据预处理
scalers = {col: MinMaxScaler(feature_range=(0, 1)) for col in numeric_columns}
data_scaled = np.array([scalers[col].fit_transform(data[col].values.reshape(-1, 1)).flatten() for col in numeric_columns]).T

# 检查数据范围
for col in numeric_columns:
    print(f"{col} range after scaling: {data_scaled[:, numeric_columns.index(col)].min()} to {data_scaled[:, numeric_columns.index(col)].max()}")

# 创建训练和测试数据
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data)-time_step-1):
        X.append(data[i:(i+time_step)])
        Y.append(data[i + time_step])
    return np.array(X), np.array(Y)

time_step = 10
X, Y = create_dataset(data_scaled, time_step)

# 划分训练集和测试集
train_size = int(len(X) * 0.8)
test_size = len(X) - train_size
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

# 建立LSTM模型
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(time_step, len(numeric_columns))))  # 增加神经元数量
model.add(Dropout(0.2))  # 添加Dropout层
model.add(LSTM(100, return_sequences=False))
model.add(Dropout(0.2))  # 添加Dropout层
model.add(Dense(50))
model.add(Dense(len(numeric_columns)))

model.compile(optimizer='adam', loss='mean_squared_error')

# 训练模型
model.fit(X_train, Y_train, batch_size=32, epochs=100)  # 增加训练轮次和批次

# 预测
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# 反归一化
train_predict_unscaled = np.zeros_like(train_predict)
test_predict_unscaled = np.zeros_like(test_predict)
Y_train_unscaled = np.zeros_like(Y_train)
Y_test_unscaled = np.zeros_like(Y_test)

for i, col in enumerate(numeric_columns):
    train_predict_unscaled[:, i] = scalers[col].inverse_transform(train_predict[:, i].reshape(-1, 1)).flatten()
    test_predict_unscaled[:, i] = scalers[col].inverse_transform(test_predict[:, i].reshape(-1, 1)).flatten()
    Y_train_unscaled[:, i] = scalers[col].inverse_transform(Y_train[:, i].reshape(-1, 1)).flatten()
    Y_test_unscaled[:, i] = scalers[col].inverse_transform(Y_test[:, i].reshape(-1, 1)).flatten()

# 计算准确性
train_rmse = mean_squared_error(Y_train_unscaled, train_predict_unscaled, squared=False)
test_rmse = mean_squared_error(Y_test_unscaled, test_predict_unscaled, squared=False)
print(f'Train RMSE: {train_rmse}')
print(f'Test RMSE: {test_rmse}')

# 设置横坐标
x_axis_train = date_column.values[:len(Y_train_unscaled)]
x_axis_test = date_column.values[len(Y_train_unscaled):len(Y_train_unscaled)+len(Y_test_unscaled)]

# 将numpy.datetime64转换为Python的datetime
x_axis_train = [datetime.utcfromtimestamp(x.astype('O') / 1e9) for x in x_axis_train]
x_axis_test = [datetime.utcfromtimestamp(x.astype('O') / 1e9) for x in x_axis_test]

# 格式化日期去掉年份
x_axis_train_formatted = [date.strftime('%m-%d') for date in x_axis_train]
x_axis_test_formatted = [date.strftime('%m-%d') for date in x_axis_test]

# 可视化结果
def plot_predictions(true_train, pred_train, true_test, pred_test, feature_name):
    plt.figure(figsize=(14, 7))
    plt.plot(x_axis_train_formatted, true_train, label=f'True train data {feature_name}', color='blue')
    plt.plot(x_axis_train_formatted, pred_train, label=f'True train-prediction data {feature_name}', color='orange')
    plt.plot(x_axis_test_formatted, true_test, label=f'True test data {feature_name}', color='green')
    plt.plot(x_axis_test_formatted, pred_test, label=f'True test-prediction data {feature_name}', color='red')
    plt.legend()
    plt.xlabel('Data')
    plt.ylabel(feature_name)
    plt.xticks(rotation=45)
    plt.title(f'{feature_name} prediction')
    plt.show()

for i, col in enumerate(numeric_columns):
    plot_predictions(Y_train_unscaled[:, i], train_predict_unscaled[:, i], Y_test_unscaled[:, i], test_predict_unscaled[:, i], col)