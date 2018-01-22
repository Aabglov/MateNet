# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
import os
import numpy as np
import time
from scipy.linalg import toeplitz

# x = np.array([[1,2,3]])
# a = np.ones((3,3)) * 2.0
# b = np.ones((3,3)) * 1.5
#
# print(np.dot(x,a))
# print(np.dot(x,b))
#
# x_eye = np.array([[1,2,3,0,0,0],[0,0,0,1,2,3]])
# ab = np.concatenate((a,b))
# print(np.dot(x_eye,ab))

NUM_SAMPLES = 1001
HIDDEN_DIM = 1000
OUTPUT_DIM = 99
BATCH_SIZE = 1

w = []
x = np.random.random((BATCH_SIZE,HIDDEN_DIM))
for _ in range(NUM_SAMPLES):
    w.append(np.random.random((HIDDEN_DIM,OUTPUT_DIM)))

start = time.time()
results = []
for i in range(NUM_SAMPLES):
    results.append(np.dot(x,w[i]).shape)
end = time.time()
print("time:", "{}".format(end-start))

x_eye = np.random.random((NUM_SAMPLES*BATCH_SIZE,NUM_SAMPLES*HIDDEN_DIM))
start = time.time()
w_cat  = np.concatenate(w)
#print(w_cat.shape)
#print(x_eye.shape)
print(np.dot(x_eye,w_cat).shape)
end = time.time()
print("time:", "{}".format(end-start))
