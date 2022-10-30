import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x)*sigmoid(1-x)

def cost_function(a,y):
    m = y.shape[1]
    matrix = (y-a)*(y-a)
    return (1/(2*m))*(np.sum(matrix))

def get_data(path):
    df = pd.read_csv (path, header=None)
    pixel_values = df.iloc[0:df.shape[0],0:df.shape[1]-1]
    pixel_values = np.array(pixel_values)

    #Data normalization
    max_pixel_value = np.max(pixel_values)
    pixel_values = pixel_values/max_pixel_value

    temp = df.iloc[:,df.shape[1]-1]
    temp = np.array(temp)

    output = np.zeros((df.shape[0],10))
    for i in range(10):
        output[np.where(temp==i),i]=1
    return pixel_values.T, output.T

def init_params(layer):
    W = []
    b = []
    for i in range(len(layer)-1):
        W.append(np.random.randn(layer[i+1],layer[i])*0.01)
        b.append(np.zeros((layer[i+1],1)))
    return W,b

def minibatch_create(X, y, r):
    X = X.T
    y = y.T
    minibatches = []
    joint = np.hstack((X, y))
    #to randomise the combined dataset
    np.random.shuffle(joint)
    for i in range(joint.shape[0]//r):
        minibatch = joint[i * r:(i + 1)*r, :]
        new_X = minibatch[:, :-10].T
        new_y = minibatch[:, -10:].reshape((-1, 10)).T
        minibatches.append((new_X, new_y))
    return minibatches

def forward_prop(params,X):
    W,b = params
    a = [X]
    z = []
    for i in range(len(b)):
        z.append(W[i]@a[i] + b[i])
        a.append(sigmoid(z[i]))
    return z,a

def back_prop(params,z,a,y):
    W,b = params
    delta = []
    for i in range(len(b)):
      delta.append(None)
    grads = {'W':[], 'b':[]}
    m = y.shape[1]
    for i in range(len(b)-1,-1,-1):
        dA = (a[i+1]-y) if i==len(b)-1 else np.dot(W[i+1].T, delta[i+1])
        delta[i] = dA*(sigmoid_der(z[i]))
        
        grads['W'].append(np.dot(delta[i], a[i].T)/m)
        grads['b'].append(np.sum(delta[i], axis=1, keepdims=True)/m)
        
    grads['W'].reverse()
    grads['b'].reverse()
    
    return grads

def update(params,grads,alpha):
    W,b = params
    for i in range(len(b)):
        W[i] = W[i] - alpha*np.array(grads['W'][i])
        b[i] = b[i] - alpha*np.array(grads['b'][i])
    return W,b

def accuracy(y,y_hat,m):
    matrix = y - y_hat
    positive = 0
    for i in range(m):
        if(np.max(matrix[:,i])==0):
            positive = positive + 1
    return (positive/m)

def model(max_iters, minibatches, params, alpha):
    cost_history = []
    for i in range(max_iters):
        for X,y in minibatches:
            z,a = forward_prop(params,X)
            grads = back_prop(params,z,a,y)
            
            params = update(params,grads,alpha)
            if(cost_function(a[-1],y)<= 0.01):
                return params, cost_history
        
            cost_history.append(cost_function(a[-1], y))
        
    return params, cost_history

train_path = r"C:\Users\prish\Desktop\COL 774\Assignmentt 3\fmnist_train.csv"
test_path = r"C:\Users\prish\Desktop\COL 774\Assignmentt 3\fmnist_test.csv"
#train_path = 'fmnist_train.csv'
#test_path = 'fmnist_test.csv'
paths = [train_path,test_path]

def for_given_parameters(X,y,batch_size,layer_dims,max_iters,alpha,hidden_layers,paths):
    
    minibatches = minibatch_create(X, y, batch_size)
    params = init_params(layer_dims)
    params,cost_history = model(max_iters, minibatches, params, alpha)
    
    z,a = forward_prop(params,X)
    a3 = a[-1]
    y_pred = a3
    m = y.shape[1]
    for i in range(m):
        maximum = np.max(y_pred[:,i])
        for j in range(10):
            if(y_pred[j,i]==maximum):
                y_pred[j,i]=1
            else:
                y_pred[j,i]=0
    print("The final predicted classes are")
    print(y_pred)
    print("Error is " + str(cost_function(a3,y)))
    plt.plot(cost_history)
    acc_train = accuracy(y,y_pred,m)*100
    print ("Accuracy on train set in percentage is "+str(acc_train))
    
    x_test, y_test= get_data(paths[1])
    z,a = forward_prop(params,x_test)
    a3_test = a[-1]

    m_test = x_test.shape[1]
    y_pred_test = a3_test
    for i in range(m_test):
        maximum = np.max(y_pred_test[:,i])
        for j in range(10):
            if(y_pred_test[j,i]==maximum):
                y_pred_test[j,i]=1
            else:
                y_pred_test[j,i]=0
    print("The final predicted classes are")
    print(y_pred)
    print("Error is " + str(cost_function(a3,y)))
    plt.plot(cost_history)
    acc_test = accuracy(y_test,y_pred_test,m_test)*100
    print ("Accuracy on test set in percentage is "+str(acc_test))
    return acc_train,acc_test

batch_size = 100
alpha = 0.1
max_iters = 300
hidden_layers = [100,50]
layer_dims = [784] + hidden_layers + [10]
X, y = get_data(paths[0])
acc_train1, acc_test1 = for_given_parameters(X,y,batch_size,layer_dims,max_iters,alpha,hidden_layers,paths)