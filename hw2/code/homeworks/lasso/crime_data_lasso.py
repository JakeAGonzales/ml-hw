if __name__ == "__main__":
    from ISTA import train  # type: ignore
else:
    from .ISTA import train

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem

def mse(x, y, weight, bias): 

    predict = x.dot(weight) + bias

    return np.mean((predict - y)**2)


@problem.tag("hw2-A", start_line=3)
def main():
    # df_train and df_test are pandas dataframes.
    # Make sure you split them into observations and targets
    eta = 0.001

    df_train, df_test = load_dataset("crime")

    x_df = df_train.drop('ViolentCrimesPerPop', axis=1)
    y_df = df_train['ViolentCrimesPerPop']

    x = np.asarray(df_train.drop('ViolentCrimesPerPop', axis=1))
    y = np.asarray(df_train['ViolentCrimesPerPop'])

    n, d = x.shape

    x_test = np.asarray(df_test.drop('ViolentCrimesPerPop', axis=1))
    y_test = np.asarray(df_test['ViolentCrimesPerPop'])

    lambda_max = np.max(2 * np.sum(x.T * (y - np.mean(y)), axis=0))
    weight = np.zeros((d, ))

    iter = 20
    lambda_iter = [lambda_max/(2**i) for i in range(iter)]
    convergence_delta = 1e-4

    num_zeros = []
    mse_train, mse_test = [], []

    features = ['agePct12t29', 'pctWSocSec', 'pctUrban', 'agePct65up', 'householdsize']
    features_indices = [x_df.columns.get_loc(j) for j in features]

    agePct12t29 = []
    pctWSocSec = []
    pctUrban = []
    agePct65up = []
    householdsize = []

    for lambda_reg in lambda_iter: 
        weight = train(x, y, lambda_reg, eta, convergence_delta, None)[0]
        bias = train(x, y, lambda_reg, eta, convergence_delta, None)[1]

        not_zeros = np.count_nonzero(weight)
        num_zeros.append(not_zeros)
        
        #y_predict = x.dot(weight) + bias
        #y_test_predict = x_test.dot(weight) + bias
        # compute Mean Squared Error (MSE) 
        mse_train_iter = mse(x, y, weight, bias)
        mse_train.append(mse_train_iter)
        mse_test_iter = mse(x_test, y_test, weight, bias)
        mse_test.append(mse_test_iter)

        agePct12t29.append(weight[0])
        pctWSocSec.append(weight[1]) 
        pctUrban.append(weight[2])
        agePct65up.append(weight[3])
        householdsize.append(weight[4])

    # First plot 
    plt.figure(figsize=(10,5))
    plt.plot(lambda_iter, num_zeros)
    plt.xscale('log')
    plt.title('Non-zeros v. Lambda')
    plt.xlabel('Lambda')
    plt.ylabel('Non-zero Weight Elements')
    plt.show()

    # second plot 
    plt.figure(figsize=(10,5))
    plt.plot(lambda_iter, agePct12t29, label= 'agePct12t29')
    plt.plot(lambda_iter, pctWSocSec, label= 'pctWSocSec')
    plt.plot(lambda_iter, pctUrban, label= 'pctUrban')
    plt.plot(lambda_iter, agePct65up, label= 'agePct65up' )
    plt.plot(lambda_iter, householdsize, label= 'householdsize')
    plt.xscale('log')
    plt.title('Lambda')
    plt.xlabel('Lambda')
    plt.ylabel('Regulization Path')
    plt.legend()
    plt.show()

    # third plot
    plt.figure(figsize=(10,5))
    plt.plot(lambda_iter, mse_train, label="Train MSE")
    plt.plot(lambda_iter, mse_test, label='Test MSE')
    plt.xscale('log')
    plt.title('Lambda')
    plt.xlabel('Lambda')
    plt.ylabel('Mean Squared Error: Train v. Test')
    plt.legend()
    plt.show()

    # Question F 
    convergence_delta = 1e-4
    new_lambda = 30
    weight = train(x, y, new_lambda, eta, convergence_delta, None)[0]
    
    print('Largest value with lambda = 30 is ', x_df.columns[np.argmax(weight)], 'with value', np.max(weight) )
    print('Most neagative value with lambda = 30 is ', x_df.columns[np.argmin(weight)], 'with value', np.min(weight) )
    

if __name__ == "__main__":
    main()
