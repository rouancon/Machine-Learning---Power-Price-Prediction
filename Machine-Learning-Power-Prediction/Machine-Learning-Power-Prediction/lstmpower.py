#imports
from keras.models import Model
from keras.layers import Dense, LSTM, Activation
from keras.models import Sequential
from keras import losses
import xlrd
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

#get datasets from excel file
price2015=[]
price2016=[]
price2017=[]
workbook = xlrd.open_workbook('All-Power-Pricing-Data.xlsx')

#----------IMPORT/CLEAN DATA---------
#2015
worksheet = workbook.sheet_by_name('2015')
done=False
cell=1
while (not done):
    price2015.append(worksheet.cell(cell, 1).value)
    cell=cell+1
    if worksheet.cell(cell, 1).value == 0:
        done=True

price2015x=[[0 for x in range(23)] for y in range(365)]
for i in range(0,365):
    for j in range(0,23):
        price2015x[i][j]=price2015[i*24+j]

price2015x=np.array(price2015x)

price2015y=[[0] for y in range(365)]
for i in range(0,365):
    for j in range(23,24):
        price2015y[i][j-23]=price2015[i*24+j]

price2015y=np.array(price2015y)

#2016
worksheet = workbook.sheet_by_name('2016')
done=False
cell=1
while (not done):
    price2016.append(worksheet.cell(cell, 1).value)
    cell=cell+1
    if worksheet.cell(cell, 1).value == 0:
        done=True
        price2016=np.array(price2016)

price2016x=[[0 for x in range(23)] for y in range(365)]
for i in range(0,365):
    for j in range(0,23):
        price2016x[i][j]=price2016[i*24+j]

price2016x=np.array(price2016x)

price2016y=[[0] for y in range(365)]
for i in range(0,365):
    for j in range(23,24):
        price2016y[i][j-23]=price2016[i*24+j]

price2016y=np.array(price2016y)

#2017
worksheet = workbook.sheet_by_name('2017')
done=False
cell=1
while (not done):
    price2017.append(worksheet.cell(cell, 1).value)
    cell=cell+1
    if worksheet.cell(cell, 1).value == 0:
        done=True
        price2017=np.array(price2017)

price2017x=[[0 for x in range(23)] for y in range(365)]
for i in range(0,365):
    for j in range(0,23):
        price2017x[i][j]=price2017[i*24+j]

price2017x=np.array(price2017x)

price2017y=[[0] for y in range(365)]
for i in range(0,365):
    for j in range(23,24):
        price2017y[i][j-23]=price2017[i*24+j]

price2017y=np.array(price2017y)


#----------DEFINE MODEL------------
model = Sequential()
model.add(LSTM(115, input_shape=(23,1)))
model.add(Dense(2, activation='relu'))
model.add(Dense(1))
model.compile(loss='mse', optimizer='rmsprop', metrics = ['mean_squared_error'])

#train and validate model
price2015x = np.reshape(price2015x, (365,23,1))
price2016x = np.reshape(price2017x, (365,23,1))
price2017x = np.reshape(price2017x, (365,23,1))
price2015y = np.reshape(price2015y, (365,1))
price2016y = np.reshape(price2017y, (365,1))
price2017y = np.reshape(price2017y, (365,1))
model.fit(x=price2015x, y=price2015y, epochs=200, batch_size=32, verbose=2, validation_split=0, validation_data=(price2016x,price2016y), shuffle=False)

#get model results
result = model.predict(x=price2017x)

#reshape array(s) back
result = np.reshape(result, (365,1))
print(result)
print(price2017y)


#----------GRAPH RESULTS----------
plt.plot(price2017y, 'r', result, 'b')
plt.show()

#---------------------------------
del model