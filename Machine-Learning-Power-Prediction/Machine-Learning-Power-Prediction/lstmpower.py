#imports
from keras.models import Model
from keras.layers import Input, Dense
import xlrd
import numpy as np

#get datasets from excel file
price2015=[]
load2015=[]
price2016=[]
load2016=[]
price2017=[]
load2017=[]
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
        price2015=np.array(price2015)

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

#define model
a = Input(shape=(32,))
b = Dense(32)(a)
model = Model(inputs=a, outputs=b)


model.compile(loss='mean_squared_error', optimizer='RMSprop')
model.fit(x=load2015, y=price2015, epochs=5, batch_size=32, verbose=2, validation_split=0, validation_data=(load2016,price2016), shuffle=True)
model.predict(x=load2017)
