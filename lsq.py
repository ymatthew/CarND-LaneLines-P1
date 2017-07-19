import numpy as np
import scipy.optimize as sci
import matplotlib.image as mpimg
import cv2

image = mpimg.imread('test.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)

Xi=np.array([189, 437, 138, 372, 351, 450, 376, 447, 158, 205, 138, 332, 172, 214])
Yi=np.array([517, 337, 538, 380, 394, 322, 381, 329, 538, 504, 539, 408, 529, 498])

def func(p,x):
    k,b=p
    return k*x+b

def error(p,x,y,s):
    print (s)
    return func(p,x)-y 
#x、y都是列表，故返回值也是个列表

p0=[100,2]

s="Test the number of iteration" #试验最小二乘法函数leastsq得调用几次error函数才能找到使得均方误差之和最小的k、b
Para=sci.leastsq(error,p0,args=(Xi,Yi,s)) #把error函数中除了p以外的参数打包到args中
k,b=Para[0]
print("k=",k,'\n',"b=",b)

import matplotlib.pyplot as plt

plt.figure(figsize=(image.shape[0],image.shape[1]))
plt.scatter(Xi,Yi,color="red",label="Sample Point",linewidth=3) #画样本点
x=np.linspace(0,10,1000)
y=k*x+b
plt.plot(x,y,color="orange",label="Fitting Line",linewidth=14) #画拟合直线
#plt.legend()
plt.show()