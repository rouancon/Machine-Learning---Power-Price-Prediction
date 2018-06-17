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
load2015=[]
price2016=[]
load2016=[]
price2017=[]
load2017=[]
day=[[0 for x in range(24)] for y in range(365)]
workbook = xlrd.open_workbook('All-Power-Pricing-Data.xlsx')

#----------IMPORT DATA---------
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
print(price2015x)

price2015y=[[0] for y in range(365)]
for i in range(0,365):
    for j in range(23,24):
        price2015y[i][j]=price2015[i*24+j+1]

price2015y=np.array(price2015y)
print(price2015y)

done=False
cell=1
while (not done):
    load2015.append(worksheet.cell(cell, 2).value)
    cell=cell+1
    if worksheet.cell(cell, 1).value == 0:
        done=True
        load2015=np.array(load2015)

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

done=False
cell=1
while (not done):
    load2016.append(worksheet.cell(cell, 2).value)
    cell=cell+1
    if worksheet.cell(cell, 1).value == 0:
        done=True
        load2016=np.array(load2016)

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

done=False
cell=1
while (not done):
    load2017.append(worksheet.cell(cell, 2).value)
    cell=cell+1
    if worksheet.cell(cell, 1).value == 0:
        done=True
        load2017=np.array(load2017)

for i in range(0,365):
    for j in range(0,24):
        day[i][j]=j+1

day=np.array(day)
print(day)


#----------DEFINE MODEL------------
model = Sequential()
model.add(LSTM(32, input_shape=(24,1)))
model.add(Dense(24))
model.add(Dense(24))
model.compile(loss='mse', optimizer='rmsprop', metrics = ['accuracy'])

#train and validate model
#load2015 = np.reshape(load2015, (365,24,1))
#load2016 = np.reshape(load2016, (365,24,1))
#load2017 = np.reshape(load2017, (365,24,1))
day = np.reshape(day, (365,24,1))
price2015 = np.reshape(price2015, (365,24))
price2016 = np.reshape(price2016, (365,24))
model.fit(x=day, y=price2015, epochs=30, batch_size=32, verbose=2, validation_split=0, validation_data=(day,price2016), shuffle=False)

#model results
result = model.predict(x=day)

#reshape array(s) back
result = np.reshape(result, (8760,1))
print(result)


#----------GRAPH RESULTS----------
plt.plot(price2017, 'r', result, 'b')
plt.show()

#---------------------------------
del model