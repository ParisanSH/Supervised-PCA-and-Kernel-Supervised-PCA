import numpy as np
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.linalg import eigh , eig
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split , KFold 
from sklearn.preprocessing import  MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import rbf_kernel



def compute_kRBF(data, sigma):
    pairwise_sq_dists = squareform(pdist(data, 'sqeuclidean'))
    kernel = np.exp(-pairwise_sq_dists / sigma**2)
    return kernel

def compute_kTest(train, test, sigma):
    K = np.zeros((train.shape[0],test.shape[0]))
    for i,x in enumerate(train):
        for j,y in enumerate(test):
            K[i,j] = np.exp((-1*np.linalg.norm(x-y)**2)/(2.*sigma**2))
    return K

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

def encode_data(kernel, l, h, p):
    #kernel_Trans = np.transpose(kernel)
    Q = kernel @ h @ l
    Q = Q @ h @ kernel
    #print(Q)
    eigenValue, eigenVectors = eig(Q, kernel)
    eigenValue = np.array(eigenValue)
    eigenValue = eigenValue.real
    index_sorted_eigValue = np.argsort(-eigenValue)
    #index_sorted_eigValue = [i[0] for i in sorted(enumerate(eigenValue), key=lambda x:x[1].imag)]
    this_index = index_sorted_eigValue[0]
    U = eigenVectors[:,this_index]
    clm = U.size
    for counter in range(1,p):
        this_index = index_sorted_eigValue[counter]
        U = np.concatenate((U,eigenVectors[:,this_index]))
    U = U.reshape(p,clm)
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
dataset = pd.read_csv('D://machine learning_hw//ML_HW3//Text files//Twomoons.txt',sep = ',', encoding='ansi')
#print(dataset)
_ , clm = dataset.shape
array = dataset.values
x = array[:,0:clm-1]
y = array[:,clm-1]
#------------ Normalize -----------------
min_max_scaler = MinMaxScaler()
x_scale = min_max_scaler.fit_transform(x)

x= np.reshape(x_scale, [x_scale.shape[0], x_scale.shape[1],1])
print(x.shape)
#print(x_scale)
draw_plt(x_scale[:,0:2], y, 0,10,0)
plt.show()
#-----------------------------------------
kf = KFold(n_splits=10)
knn = KNeighborsClassifier(n_neighbors = 1, metric= 'minkowski',p=2)
sigma_result = []
for sigma in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
    counter_fold = 0
    acc_per_fold = []
    for train_index, test_index in kf.split(x):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        K_RBF = compute_kRBF(X_train, sigma)
        k_test = compute_kTest(X_test, X_train, sigma)
        L = compute_kernel_delta(y_train)
        H = make_H(y_train.size)
        counter_fold += 1
        accuracy_per_p = []
        for p in range (1,clm):
            if p > 20:
                break
            beta_trans = encode_data(K_RBF, L, H, p)
            Z_train = beta_trans @ K_RBF
            z_test = beta_trans @ np.transpose(k_test)
            Z_train = np.transpose(Z_train)
            z_test = np.transpose(z_test)
            knn.fit(Z_train, y_train)
            y_predict = knn.predict(z_test)
            accuracy = accuracy_score(y_test, y_predict, normalize=True)
            #print('The accuracy of p =', p ,'\n\t',accuracy)
            accuracy_per_p.append((counter_fold, p, accuracy))
        max_acc = accuracy_per_p[0][2]
        p_max_acc = accuracy_per_p[0][1]
        mean_acc = 0.
        count = 0
        for fold_number, p, acc in accuracy_per_p:
            count += 1
            if max_acc <= acc :
                max_acc = acc
                p_max_acc = p
            mean_acc += acc
        mean_acc = mean_acc/count 
        acc_per_fold.append((counter_fold, mean_acc, max_acc, p_max_acc))
    mean_acc = 0
    count = 0
    max_acc_fold = acc_per_fold[0][2]
    p_max_acc = acc_per_fold[0][3]
    for fold_number, mean_acc_fold , max_acc , p_max in acc_per_fold:
        mean_acc += mean_acc_fold
        count += 1
        if max_acc_fold < max_acc:
            max_acc_fold = max_acc
            p_max_acc = p_max
    sigma_result.append((sigma, mean_acc/count, p_max_acc, max_acc_fold))

'''for sigma , mean_acc , p_max , max_acc in sigma_result:
    print(sigma , mean_acc , p_max , max_acc)'''
max_acc = sigma_result[0][1]
index_tuple = 0
counter = 0
for tuple_ in sigma_result:
    if tuple_[1] > max_acc:
        max_acc = tuple_[1]
        index_tuple = counter 
        counter +=1
sigma_tuned = sigma_result[index_tuple][0]
mean_error = 1 - max_acc
best_p = sigma_result[index_tuple][2]
print('tuned sigma is ', sigma_tuned)
print('P best  is ', best_p)
print('mean error is ', mean_error)
#---------- Train test split ------------
x_train , x_test , y_train , y_test = train_test_split (x_scale, y, test_size = .3, random_state = 21 )

#------------ Kernel Delta --------------
#K_RBF = compute_kRBF(x_train, sigma_tuned)
K_RBF = rbf_kernel(x_train, x_train, .5*(sigma_tuned)**-2)
#k_test = compute_kTest(x_test, x_train, sigma_tuned)
k_test = rbf_kernel(x_test,x_train,.5*(sigma_tuned)**-2)
L = compute_kernel_delta(y_train)
H = make_H(y_train.size)

for p in range (1,clm):
    if p > 20:
        break
    beta_trans = encode_data(K_RBF, L, H, p)
    Z_train = beta_trans @ K_RBF
    z_test = beta_trans @ np.transpose(k_test)
    Z_train = np.transpose(Z_train)
    z_test = np.transpose(z_test)
    knn.fit(Z_train, y_train)
    y_predict = knn.predict(z_test)
    accuracy = accuracy_score(y_test, y_predict, normalize=True)
    if p == 2:
        draw_plt(Z_train, y_train, 3, 35,0)
        draw_plt(z_test, y_predict, 1, 120,2)
        draw_plt(z_test, y_test, 0, 35,1)
        plt.show()
        print('accuracy with KSPCA is' ,accuracy)