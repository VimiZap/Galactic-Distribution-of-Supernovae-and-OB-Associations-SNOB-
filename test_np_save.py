import numpy as np

test = np.ones((4,4,4)) * np.array([1,2,3,4]).T
test[1]*=2
test[2]*=3
test[3]*=4

np.save('test.npy', test)



test2 = np.load('test.npy')
test2[2] = 0
np.save('test.npy', test2)

test3 = np.load('test.npy')
a = np.random.randint(19, size=(1,4,4))
test3 = np.concatenate((test3, a), axis=0)
print(test3)
np.save('test.npy', test3)
test4 = np.load('test.npy')
print(test4)