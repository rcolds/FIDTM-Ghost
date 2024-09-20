import os
import numpy as np
from sklearn.model_selection import train_test_split

if not os.path.exists('./npydata'):
    os.makedirs('./npydata')

# '''please set your dataset path'''
shanghai_root = 'D:/deeplearning/dataset/wheat/train_data-1200'


try:

    shanghaiAtrain_path = shanghai_root + '/images/'
    # shanghaiAtest_path = shanghai_root + '/part_A_final/test_data/images/'

    train_list = []
    x_train = []
    x_test = []
    for filename in os.listdir(shanghaiAtrain_path):
        if filename.split('.')[1] == 'jpg':
            train_list.append(shanghaiAtrain_path + filename)

    x_train, x_test = train_test_split(train_list, train_size=0.8)
    x_train.sort()
    x_test.sort()
    np.save('./npydata/ShanghaiA_train.npy', x_train)
    np.save('./npydata/ShanghaiA_test.npy', x_test)
#     train_list.sort()
#     np.save('./npydata/ShanghaiA_train.npy', train_list)
#     print(train_list)
#     test_list = []
#     for filename in os.listdir(shanghaiAtest_path):
#         if filename.split('.')[1] == 'jpg':
#             test_list.append(shanghaiAtest_path + filename)
#     test_list.sort()
#     np.save('./npydata/ShanghaiA_test.npy', test_list)
#
#     print("generate ShanghaiA image list successfully")
except:
    print("The ShanghaiA dataset path is wrong. Please check you path.")

# print(np.load("npydata/ShanghaiA_train.npy"))
# print(len(np.load("npydata/ShanghaiA_train.npy")))
# print(np.load("npydata/ShanghaiA_test.npy"))
# print(len(np.load("npydata/ShanghaiA_test.npy")))