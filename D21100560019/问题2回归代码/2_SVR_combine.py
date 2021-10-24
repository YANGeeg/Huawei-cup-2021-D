import scipy.io as scio
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler



dataFile = r'D:\pycharm\shuxuejianmo\4\data19.mat'
data = scio.loadmat(dataFile)
datanew = data['data19']
dataTest = r'D:\pycharm\shuxuejianmo\4\data_test19.mat'
data_test = scio.loadmat(dataTest)
data_test = data_test['data_test19']
valueFile = r'D:\pycharm\shuxuejianmo\value.mat'
value = scio.loadmat(valueFile)
valuenew = value['value']


x_train = datanew
x_test = data_test
y_train = valuenew
stdsc1 = StandardScaler()
x_train = stdsc1.fit_transform(x_train)
x_test = stdsc1.transform(x_test)

model1 = KNeighborsRegressor(n_neighbors=8, weights='distance', p=1)
model2 = DecisionTreeRegressor()
model3 = GradientBoostingRegressor(n_estimators=80, max_depth=13, min_samples_split=100)
model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)
y_pred1=model1.predict(x_train)
y_pred2=model2.predict(x_train)
y_pred3=model3.predict(x_train)
y_pred2.resize((len(y_pred2), 1))
y_pred3.resize((len(y_pred2), 1))
y_input1 = np.hstack((y_pred1,y_pred2))
y_input = np.hstack((y_input1,y_pred3))

y_pred1=model1.predict(x_test)
y_pred2=model2.predict(x_test)
y_pred3=model3.predict(x_test)
y_pred2.resize((len(y_pred2), 1))
y_pred3.resize((len(y_pred2), 1))
y_input1 = np.hstack((y_pred1, y_pred2))
x_input = np.hstack((y_input1, y_pred3))

stdsc2 = StandardScaler()
y_input = stdsc2.fit_transform(y_input)
x_input = stdsc2.transform(x_input)

model = SVR(C = 10)
model.fit(y_input, y_train)

y_predtr=model.predict(y_input)
y_pred = model.predict(x_input)
y_pred.resize((len(y_pred),1))
# print(y_pred)
print("训练数据集上的均方根误差:",mean_squared_error(y_train,y_predtr))