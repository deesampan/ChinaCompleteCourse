import torch as t
import numpy as np
import pandas as pd
import torchvision as tv
import matplotlib.pyplot as plt

# arr4 = np.eye(5)
# print(arr4)
#
# arr5 = np.random.rand(5)
# print(arr5)
#
arr6 = np.random.randint(10,21,size=(3,4))
print(arr6)
#
# arr7 = np.linspace(0,1,10)
# print(arr7)
#
# py_list = [1,3,5,7,9]
# arr8 = np.array(py_list)
# print(arr8)
#
# test = np.linspace(10,20,5)
# print(test)

test2 = np.arange(1,26).reshape(5,5)
diag = np.diag(test2)
sum = np.sum(diag)


print(test2)
print(diag)
print(sum)


arr1 = np.arange(1,11)
arr12 = arr1.reshape(2,5)

arr13_flatten = arr12.flatten()
arr13_ravel = arr12.ravel()

print(arr13_flatten)
print(arr13_ravel)





arr14_orig = np.arange(12).reshape(4,3)
arr14_transposed = arr14_orig.T

print(arr14_orig)
print(arr14_transposed)
print(arr14_transposed.shape)


print("++++++++++++++ ex3 +++++++++++++++++\n\n")

chess = np.zeros(64).reshape(8,8)

chess[1::2, ::2] = 1
chess[::2, 1::2] = 1
print(chess)



print("++++++++++++++ ex4 +++++++++++++++++\n\n")

ex4 = np.random.randint(1,51,size=(4,5))
third_col = ex4.T[2,:]
print(ex4)
print(third_col)




print("++++++++++++++ ex5 +++++++++++++++++\n\n")

ex5 = np.random.randint(1,101,size=(10,10))

print(ex5)

ex5[ex5 < 50] = 0
ex5[ex5 >= 50] = -1



print(ex5)

print("++++++++++++++ ex6 +++++++++++++++++\n\n")



ex6 = np.arange(25).reshape(-1,5)

ex6[0,:] = -1
ex6[-1,:] = -1
ex6[:,0] = -1
ex6[:,-1] = -1

ex6[2,:] = -1
ex6[:,[0,-1]] = -1

print(ex6)


print("++++++++++++++ ex7 +++++++++++++++++\n\n")


ex7 = np.random.randint(1,101,size=(10,10))
m = np.max(ex7)
r,c = np.where(ex7 == m)
print(ex7)
print(ex7[r[0]])



print("++++++++++++++ ex8 +++++++++++++++++\n\n")



aaa = np.arange(6).reshape(2,3)
bbb = np.ones((2,3),dtype=int) * 5
ele_sum = aaa + bbb
print(ele_sum)


ccc = np.arange(6).reshape(3,2)

matrix_product = aaa @ ccc

print(matrix_product)


print("++++++++++++++ ex9 +++++++++++++++++\n\n")




col_vec = np.array([[1],[2],[3]])
row_vec = np.array([[10,20,30,40]])

broadcasted_sum = col_vec + row_vec
print(broadcasted_sum)
