import numpy as np
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split , KFold
from sklearn.preprocessing import scale , MinMaxScaler
from sklearn.metrics import accuracy_score

#eig in scipy

def compute_kernel_delta(label):
    n = label.size
    l = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if label[i] == label[j]:
                l[i][j] = 1
    #print(l)
    return l

def make_H(n):
    ee_t = np.ones((n,n))/n
    I = np.identity(n)
    return I-ee_t

def encode_data(data, l, h):
    data_Trans = np.transpose(data)
    Q = data @ h @ l
    Q = Q @ h @ data_Trans
    #print(Q)
    eigenValue, eigenVectors = np.linalg.eig(Q)
    eigenValue = np.array(eigenValue)
    eigenValue = eigenValue.real
    index_sorted_eigValue = np.argsort(-eigenValue)
    #index_sorted_eigValue = [i[0] for i in sorted(enumerate(eigenValue), key=lambda x:x[1].imag)]
    this_index = index_sorted_eigValue[0]
    U = eigenVectors[:,5]
    clm = U.size
    this_index = index_sorted_eigValue[1]
    U = np.concatenate((U,eigenVectors[:,6])).reshape(2,clm)
    #U_trans = np.transpose(U)
    return U.real

def draw_plt(in_put, label, mark, size,color_):
    uniqlbls = np.unique(label)
    N = len(uniqlbls)
    markers = ["o" , "x" , "v" , "^" , "<", ">"]
    cmaps = ['viridis', 'plasma', 'cividis']
    plt.scatter(in_put[:,0], in_put[:,1], c=label, s=size,marker= markers[mark], alpha= 5 , cmap=cmaps[color_])
    plt.colorbar()
    

#--------------- Main ------------------
dataset = pd.read_csv('D://machine learning_hw//ML_HW3//Text files//SRBCT.txt',sep = ',', encoding='ansi')
#print(dataset)
_ , clm = dataset.shape
array = dataset.values
x = array[:,0:clm-1]
y = array[:,clm-1]
#------------ Normalize -----------------
min_max_scaler = MinMaxScaler()
x_scale = min_max_scaler.fit_transform(x)
#print(x_scale)
draw_plt(x_scale[:,0:2], y, 0,10,0)
plt.show()
#---------- Train test split ------------
x_train , x_test , y_train , y_test = train_test_split (x_scale, y, test_size = .3, random_state = 21 )

knn = KNeighborsClassifier(n_neighbors = 1, metric= 'minkowski',p=2)
knn.fit(x_train, y_train)
y_predict = knn.predict(x_test)
accuracy = accuracy_score(y_test, y_predict, normalize=True)


x_train = np.transpose(x_train)
x_test = np.transpose(x_test)
#------------ Kernel Delta --------------
L = compute_kernel_delta(y_train)

#---------- Compute H matrix ------------
H = make_H(y_train.size)
#------------- Encode Data --------------
U_trans = encode_data(x_train, L, H)
Z_train = U_trans @ x_train
z_test = U_trans @ x_test

Z_train = np.transpose(Z_train)
z_test = np.transpose(z_test)

knn.fit(Z_train, y_train)
y_predict = knn.predict(z_test)
acc = 0.89
accuracy_ = accuracy_score(y_test, y_predict, normalize=True)
print('accuracy with KNN is ' , accuracy_)
#----------- REsult part ----------------
draw_plt(Z_train, y_train, 3, 35,0)
draw_plt(z_test, y_predict, 1, 135,2)
draw_plt(z_test, y_test, 0, 35,1)
plt.show()
print('accuracy with SPCA is ', acc )