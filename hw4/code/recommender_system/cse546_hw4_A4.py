#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import numpy as np
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
import torch
import os
from tqdm.notebook import tqdm


# In[2]:


data = []
pwd = os.getcwd()
path = os.path.join(pwd + '/data/movielens-data/ml-100k/u.data')

#pwd = '/Users/jakegonzales/Documents/Fall 2023/cse 546/movielens-data/ml-100k/u.data'
with open(path) as csvfile:
    spamreader = csv.reader(csvfile, delimiter='\t')
    for row in spamreader:
        data.append([int(row[0])-1, int(row[1])-1, int(row[2])])
data = np.array(data)

num_observations = len(data)  # num_observations = 100,000
num_users = max(data[:,0])+1  # num_users = 943, indexed 0,...,942
num_items = max(data[:,1])+1  # num_items = 1682 indexed 0,...,1681

np.random.seed(1)
num_train = int(0.8*num_observations)
perm = np.random.permutation(data.shape[0])
train = data[perm[0:num_train],:]
test = data[perm[num_train::],:]

print(f"Successfully loaded 100K MovieLens dataset with",
      f"{len(train)} training samples and {len(test)} test samples")


# In[3]:


"""""
Part A - first estimator using averages of ratings 

"""""
# Compute estimate

def compute_estimate(train, test, num_items): 
    
    average_scores = np.zeros(num_items) 

    train_ratings = train[:,1]
    u = np.unique(train_ratings)
    train_data = train_ratings[u]
    #print(train_data)
    
    # loop over data and get averages 
    for data in range(len(train_data)): 
        average_scores[train_data[data]] = np.mean(train[train[:,1] == train_data[data], 2])
       
    # \widehat{R}
    estimate = average_scores[test[:, 1]]
    print(estimate)
    
    test_data = test[:,2]
    
    # average squared error on test dataset 
    error = np.mean(np.square(estimate - test_data))
    
    return error

estimator_error = compute_estimate(train, test, num_items)
print(estimator_error)


# In[4]:


"""""
Part B - rank-d approximation

"""""


# Create the matrix R twiddle (\widetilde{R}).
def r_twiddle(train, R_tilde): 
    
    for i in range(len(train)): 
        R_tilde[train[i,1], train[i,0]] = train[i, 2]
        
    return R_tilde
        
        
def construct_estimator(R_hat, d): 
    
    train_error = np.zeros(len(d))
    test_error = np.zeros(len(d))
    
    for i in tqdm(range(len(d))): 
        
        u, s, v_h = svds(R_tilde, k = d[i]) # k number of singular values
        R_hat = np.matmul(u*s, v_h)
        
        # get error
        train_estimator = R_hat[train[:,1], train[:,0]]
        train_error[i] = np.mean(np.square(train[:,2] - train_estimator))

        test_estimator = R_hat[test[:,1], test[:,0]]
        test_error[i] = np.mean(np.square(test[:,2] - test_estimator))
    
    return R_hat, train_error, test_error


d = [1, 2, 5, 10, 20 ,50] # for best rank-d approximation \widehat{R}^d 
R_tilde = np.zeros((num_items, num_users))

R_tilde = r_twiddle(train, R_tilde)
R_hat, train_error, test_error = construct_estimator(R_tilde, d)

print(train_error)
print(test_error)



# plot errors
plt.plot(d, train_error) 
plt.plot(d, test_error)
plt.xlabel("Dimension d")
plt.ylabel("Mean Squared Error (MSE)")
plt.legend(('Train Error', 'Test Error'))
plt.show()




# In[5]:


"""""
Part C - alternating minimization

"""""


def closed_form_u(V, U):
    
    """""
    minimize the loss function with respect to {u_i} by treating {v_j} as fixed
    
    V: matrix with rows v_i 
    U: matrix with rows u_i
    
    """""
    for i in range(num_items):
        index = r_i[i]
        V_i = V[train[index, 0], :]
        
        a = lambda_ * np.eye(d) + np.dot(V_i.transpose(), V_i) 
        b = np.dot(V_i.transpose(), train[index, 2])
        
        U[i, :] = np.linalg.solve(a,b) 
        
    return U


def closed_form_v(V, U):
    """""
    minimize the loss function with respect to {v_j} by treating {u_i} as fixed
    
    V: matrix with rows v_i 
    U: matrix with rows u_i
    
    """""
    
    for j in range(num_users): 
        index = r_j[j]
        U_j = U[train[index, 1], :] 
        
        a = lambda_ * np.eye(d) + np.dot(U_j.transpose(), U_j) 
        b = np.dot(U_j.transpose(), train[index, 2])
        
        V[j, :] = np.linalg.solve(a,b) 
        
    return V

def get_error(R_hat, data):
    
    # data input is either train or test dataset
    
    N = data.shape[0]
    
    error = np.mean([np.square(data[i, 2] - R_hat(data[i, 0], data[i, 1])) for i in range(N)])
    
    return error 


def construct_alternating_estimator(
    d, r_twiddle, l=10.0, delta=1e-1, sigma=0.1, U=None, V=None
):
    pass
    
    


# In[8]:


# "it is important that the squared error part of the loss is only defined w.r.t. R_ij that actually exist in 
# the training set"

# here, r(j) = {j : (j, i, R_ij) \in train} set of users who have reviewed movie i in training set

r_i = {i : np.where(train[:, 1] == i)[0] for i in range(num_items)}
r_j = {j : np.where(train[:, 0] == j)[0] for j in range(num_users)}

# hyperparameters 
sigma = 0.1
lambda_ = 10 
delta = 1e-1

# Evaluate train and test error for: d = 1, 2, 5, 10, 20, 50.
d_list = [1, 2, 5, 10, 20 ,50]

train_error = []
test_error = []


for d in tqdm(d_list): 
    
    # randomly initialize u and v
    U = sigma * np.random.rand(num_items, d) 
    V = sigma * np.random.rand(num_users, d)
    
    old_U = np.copy(U)
    old_V = np.copy(V)
    
    converge = True
    
    while converge:
    
        U = closed_form_u(V, U)

        V = closed_form_v(V, U)

        if np.max(np.abs(U - old_U)) and np.max(np.abs(V - old_V)) < delta:
            converge = False
        else: 
            old_U = np.copy(U) 
            old_V = np.copy(V) 
            
    
    R_hat = lambda i, j: np.inner(U[j, :], V[i, :])

    # get errors     
    train_error.append(get_error(R_hat, train))
    test_error.append(get_error(R_hat, test))
    
print("Train Error: \n", train_error)
    
print("\n Test Error: \n", test_error)
    
    
# Plot both train and test error as a function of d on the same plot.
# plot errors
plt.plot(d_list, train_error) 
plt.plot(d_list, test_error)
plt.xlabel("Dimension d")
plt.ylabel("Mean Squared Error (MSE)")
plt.legend(('Train Error', 'Test Error'))
plt.title('Alernating Minimization')
plt.show()


# In[ ]:




