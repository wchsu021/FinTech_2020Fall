import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## (b) 用pseudo inverse求出參數
def pinv_LR(x,y):
    x_inv = np.linalg.pinv(x)
    wp = np.dot(x_inv,y)
    return wp

## (c) Regularization (without bias)
def regression(x,y):
    ld = 1
    wr = np.full(x.shape[1], 0.1).reshape(-1, 1)
    
    first = np.dot(np.transpose(x),x)+0.5*ld*np.eye(x.shape[1])
    first = np.linalg.inv(first)
    second = np.dot(np.transpose(x),y)
    wr = np.dot(first,second)
    return wr

## (d) Regularization with bias
def regression_b(x,y):
    ld = 1
    wr = np.full(x[0].shape[0], 0.1).reshape(-1, 1)
    
    first = np.dot(np.transpose(x),x)+0.5*ld*np.eye(x.shape[1])
    first = np.linalg.inv(first)
    second = np.dot(np.transpose(x),y)
    wr = np.dot(first,second)
    bias = wr[0]
    wr = np.delete(wr, 0, axis=0)
    return wr, bias

## (e) Bayesian Linear Regression (with bias)
def BayesianLR(x,y):
    ld = 1
    wr = np.full(x[0].shape[0], 0.1).reshape(-1, 1)
    bias = 0.1
    
    first = np.dot(np.transpose(x),x)+ld*np.eye(x.shape[1])
    first = np.linalg.inv(first)
    second = np.dot(np.transpose(x),y)
    wr = np.dot(first,second)
    bias = wr[0]
    wr = np.delete(wr, 0, axis=0)
    return wr, bias







if __name__ == '__main__':
    
    df = pd.read_csv('train.csv')

    df = df.drop(columns=['ID','address', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'schoolsup', 'famsup', 'paid', 'nursery', 'G1', 'G2', 'cat'])

    df2 = pd.get_dummies(df)

    cols = list(df2)
    n = len(cols) - 1
    cols.insert(n,cols.pop(cols.index('G3')))
    df2 = df2.loc[:,cols]

    ## (a) 分成80%train, 20%test, 各項標準化
    df_train = df2.sample(frac=1, random_state=7)
    df_train = df_train.reset_index(drop = True)
    train_num = int(0.8*len(df_train))
    train = df_train.iloc[0:train_num,:]
    test = df_train.iloc[train_num:,:]

    cols = list(train)
    cols.pop(-1)

    mu_list = []
    sigma_list = []
    for idx, item in enumerate(cols):
        mu = np.mean(train[item])
        sigma = np.std(train[item])
        mu_list.append(mu)
        sigma_list.append(sigma)
        
        for i in range(len(train[item])):
            temp = train.at[i,item]
            train.loc[i,item] = (temp-mu)/sigma
        

    test = test.reset_index(drop = True)

    for idx, item in enumerate(cols):
        for i in range(len(test[item])):
            temp = test.at[i,item]
            test.loc[i,item] = (temp-mu_list[idx])/sigma_list[idx]

    data_all = df2
    for idx, item in enumerate(cols):
        for i in range(len(data_all[item])):
            temp = data_all.at[i,item]
            data_all.loc[i,item] = (temp-mu_list[idx])/sigma_list[idx]
    
    train_A = train.drop(columns=['G3'])
    train_A = train_A.to_numpy()
    train_G3 = train['G3'].to_numpy()
    train_G3 = train_G3.reshape(-1,1)
    
    ## (b) 用pseudo inverse求出參數
    wp = pinv_LR(train_A, train_G3)

    test_A = test.drop(columns=['G3'])
    test_A = test_A.to_numpy()
    # print(test_A.shape)
    pre_G3_p = np.dot(test_A, wp)
    gt_G3 = test['G3'].to_numpy()
    RMSE_p = np.sqrt(((pre_G3_p - gt_G3) ** 2).mean())


    ## (c) Regularization (without bias)
    wr = regression(train_A, train_G3)

    pre_G3_r = np.dot(test_A, wr)
    RMSE_r = np.sqrt(((pre_G3_r - gt_G3) ** 2).mean())

    ## (d) Regularization with bias
    bias = np.ones(train_A.shape[0])
    train_B = np.insert(train_A, 0, values=bias, axis=1)

    w_rb,bias_rb = regression_b(train_B, train_G3)

    pre_G3_rb = np.dot(test_A, w_rb)+bias_rb
    RMSE_rb = np.sqrt(((pre_G3_rb - gt_G3) ** 2).mean())

    ## (e) Bayesian Linear Regression (with bias)
    w_blr,bias_blr = BayesianLR(train_B, train_G3)

    pre_G3_blr = np.dot(test_A, w_blr)+bias_blr
    RMSE_blr = np.sqrt(((pre_G3_blr - gt_G3) ** 2).mean())

    ## (f) plot

    # gt_G3_all = data_all['G3'].to_numpy()
    # data_all_A = data_all.drop(columns=['G3'])
    # data_all_A = data_all_A.to_numpy()
    # pre_G3_p_all = np.dot(data_all_A, wp)
    # pre_G3_r_all = np.dot(data_all_A, wr)
    # pre_G3_rb_all = np.dot(data_all_A, w_rb)+bias_rb
    # pre_G3_blr_all = np.dot(data_all_A, w_blr)+bias_blr

    RMSE_p = float('{:.4f}'.format(RMSE_p))
    RMSE_r = float('{:.4f}'.format(RMSE_r))
    RMSE_rb = float('{:.4f}'.format(RMSE_rb))
    RMSE_blr = float('{:.4f}'.format(RMSE_blr))

    plt.cla()
    plt.clf()
    plt.plot(gt_G3)
    plt.plot(pre_G3_p)
    plt.plot(pre_G3_r)
    plt.plot(pre_G3_rb)
    plt.plot(pre_G3_blr)
    plt.legend(["Ground Truth", "Linear Regression "+str(RMSE_p), "Linear Regression (reg) "+str(RMSE_r), "Linear Regression (r/b) "+str(RMSE_rb), "Bayesian Linear Regression "+str(RMSE_blr)], loc='lower right')
    # plt.show()
    plt.savefig("MSE.jpg")
    

    ## (g) test no G3.csv + e. with tuned alpha

    test_df = pd.read_csv('test_no_G3.csv')
    test_df = test_df.drop(columns=['ID','address', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'schoolsup', 'famsup', 'paid', 'nursery', 'G1', 'G2', 'cat'])
    test_df2 = pd.get_dummies(test_df)

    ## normalization
    mu_list = []
    sigma_list = []
    for idx, item in enumerate(cols):
        mu = np.mean(test_df2[item])
        sigma = np.std(test_df2[item])
        mu_list.append(mu)
        sigma_list.append(sigma)
        
        for i in range(len(test_df2[item])):
            temp = test_df2.at[i,item]
            test_df2.loc[i,item] = (temp-mu)/sigma

    test_set = test_df2.to_numpy()

    pre_G3_test = np.dot(test_set, w_blr)+bias_blr
    pre_G3_test_T = np.transpose(pre_G3_test)

    f = open('r08942025_1.txt', 'w')
    for i in range(pre_G3_test_T.shape[1]):
        f.write(str(i+1001)+"\t")
        f.write(str(pre_G3_test_T[0][i])+"\n")
    f.close()
    