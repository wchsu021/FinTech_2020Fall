import numpy as np
import pandas as pd
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

def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 1e-6, 1-1e-6)

def adjust(z):
    pred = sigmoid(z)
    pred = np.around(pred)
    pred = np.int_(pred)
    return pred

def predict(w,b,x):
    z = np.dot(w, x.T) + b
    pred = sigmoid(z)
    pred = np.around(pred)
    pred = np.int_(pred)
    return pred

def BayesianLR_a(x,y,alpha):
    
    wr = np.full(x[0].shape[0], 0.1).reshape(-1, 1)
    bias = 0.1
    
    first = np.dot(np.transpose(x),x)+ alpha*np.eye(x.shape[1])
    first = np.linalg.inv(first)
    second = np.dot(np.transpose(x),y)
    wr = np.dot(first,second)
    bias = wr[0]
    wr = np.delete(wr, 0, axis=0)
    return wr, bias

def main_prob1():

	###  problem1  ###

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


def main_prob2():

	###  problem2  ###


    # COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation", "relationship", "race", "gender", "capital_gain", "capital_loss", "hours_per_week", "native_country", "income_bracket"]

    # train_df = pd.read_csv(
    #     "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    #     names=COLUMNS,
    #     sep=r'\s*,\s*',
    #     engine='python',
    #     na_values="?")
    # test_df = pd.read_csv(
    #     "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
    #     names=COLUMNS,
    #     sep=r'\s*,\s*',
    #     skiprows=[0],
    #     engine='python',
    #     na_values="?")

    # train_df = train_df.dropna(how="any", axis=0)
    # test_df = test_df.dropna(how="any", axis=0)

    # train_df = train_df.reset_index(drop = True)
    # test_df = test_df.reset_index(drop = True)

    # train_df2 = pd.get_dummies(train_df)
    # train_df2 = train_df2.drop(columns=['income_bracket_<=50K'])

    """# (a) 80% train, 20% valid, Normalization"""

    # train_num = int(0.8*len(train_df2))
    # train = train_df2.iloc[0:train_num,:]
    # valid = train_df2.iloc[train_num:,:]

    # train = train.reset_index(drop = True)
    # valid = valid.reset_index(drop = True)

    # cols = list(train)
    # cols.pop(-1)

    # mu_list = []
    # sigma_list = []
    # # print(cols)
    # for idx, item in enumerate(cols):
    #     mu = np.mean(train[item])
    #     sigma = np.std(train[item])
    #     mu_list.append(mu)
    #     sigma_list.append(sigma)
    #     # print(item, mu, sigma)
    #     for i in range(len(train[item])):
    #         temp = train.at[i,item]
    #         train.loc[i,item] = (temp-mu)/sigma
    #     # print('----------')

    # for idx, item in enumerate(cols):
    #     for i in range(len(valid[item])):
    #         temp = valid.at[i,item]
    #         valid.loc[i,item] = (temp-mu_list[idx])/sigma_list[idx]
    #     #print('----------')

    ### save normalized file
    # file = 'train_n.csv'
    # train.to_csv(file, index=False)
    # file = 'valid_n.csv'
    # valid.to_csv(file, index=False)
    # print("Finish saving!")
 	
 	### directly load files
    train = pd.read_csv('train_n.csv')
    valid = pd.read_csv('valid_n.csv')
    
    """# (b) 用pseudo inverse求出參數"""
    train_A = train.drop(columns=['income_bracket_>50K'])
    train_A = train_A.to_numpy()
    train_y = train['income_bracket_>50K'].to_numpy()
    train_y = train_y.reshape(-1,1)
    wp = pinv_LR(train_A, train_y)

    valid_A = valid.drop(columns=['income_bracket_>50K'])
    valid_A = valid_A.to_numpy()
    pred_y_p = np.dot(valid_A, wp)
    valid_y = valid['income_bracket_>50K'].to_numpy()
    RMSE_p = np.sqrt(((pred_y_p - valid_y) ** 2).mean())
    # print(np.mean(pred_y_p), np.std(pred_y_p))

    # pred_y_p = adjust(pred_y_p)

    # acc_p = 0
    # for i in range(pred_y_p.shape[0]):
    #     if pred_y_p[i][0] == valid_y[i]:
    #         acc_p += 1
    # acc_p = acc_p/pred_y_p.shape[0]

    acc_p = 0
    for i in range(pred_y_p.shape[0]):
        if pred_y_p[i][0] > 0.5:
            pred_y_p[i][0] = 1
        else:
            pred_y_p[i][0] = 0
        if pred_y_p[i][0] == valid_y[i]:
            acc_p += 1
    acc_p = acc_p/pred_y_p.shape[0]

    """# (c) Regularization (without bias)"""
    wr = regression(train_A, train_y)

    pred_y_r = np.dot(valid_A, wr)

    RMSE_r = np.sqrt(((pred_y_r - valid_y) ** 2).mean())

    # print(np.mean(pred_y_r), np.std(pred_y_r))
    # pred_y_r = adjust(pred_y_r)

    # acc_r = 0
    # for i in range(pred_y_r.shape[0]):
    #     if pred_y_r[i][0] == valid_y[i]:
    #         acc_r += 1
    # acc_r = acc_r/pred_y_r.shape[0]

    acc_r = 0
    for i in range(pred_y_r.shape[0]):
        if pred_y_r[i][0] > 0.5:
            pred_y_r[i][0] = 1
        else:
            pred_y_r[i][0] = 0
        if pred_y_r[i][0] == valid_y[i]:
            acc_r += 1
    acc_r = acc_r/pred_y_r.shape[0]


    """# (d) Regularization with bias"""
    bias = np.ones(train_A.shape[0])
    train_B = np.insert(train_A, 0, values=bias, axis=1)

    w_rb,bias_rb = regression_b(train_B, train_y)

    pred_y_rb = np.dot(valid_A, w_rb)+bias_rb

    RMSE_rb = np.sqrt(((pred_y_rb - valid_y) ** 2).mean())

    # print(np.mean(pred_y_rb), np.std(pred_y_rb))
    # pred_y_rb = adjust(pred_y_rb)

    # acc_rb = 0
    # for i in range(pred_y_rb.shape[0]):
    #     if pred_y_rb[i][0] == valid_y[i]:
    #         acc_rb += 1
    # acc_rb = acc_rb/pred_y_rb.shape[0]

    acc_rb = 0
    for i in range(pred_y_rb.shape[0]):
        if pred_y_rb[i][0] > 0.5:
            pred_y_rb[i][0] = 1
        else:
            pred_y_rb[i][0] = 0
        if pred_y_rb[i][0] == valid_y[i]:
            acc_rb += 1
    acc_rb = acc_rb/pred_y_rb.shape[0]

    """# (e) Bayesian Linear Regression (with bias)"""
    w_blr,bias_blr = BayesianLR(train_B, train_y)

    pred_y_blr = np.dot(valid_A, w_blr)+bias_blr

    RMSE_blr = np.sqrt(((pred_y_blr - valid_y) ** 2).mean())

    # print(np.mean(pred_y_blr), np.std(pred_y_blr))
    # pred_y_blr = adjust(pred_y_blr)

    # acc_blr = 0
    # for i in range(pred_y_blr.shape[0]):
    #     if pred_y_blr[i][0] == valid_y[i]:
    #         acc_blr += 1
    # acc_blr = acc_blr/pred_y_blr.shape[0]

    acc_blr = 0
    for i in range(pred_y_blr.shape[0]):
        if pred_y_blr[i][0] > 0.5:
            pred_y_blr[i][0] = 1
        else:
            pred_y_blr[i][0] = 0
        if pred_y_blr[i][0] == valid_y[i]:
            acc_blr += 1
    acc_blr = acc_blr/pred_y_blr.shape[0]


    print(acc_p)
    print(acc_r)
    print(acc_rb)
    print(acc_blr)

    print("-------------")
    
	## (f) plot

    print(RMSE_p)
    print(RMSE_r)
    print(RMSE_rb)
    print(RMSE_blr)

    # RMSE_p = float('{:.4f}'.format(RMSE_p))
    # RMSE_r = float('{:.4f}'.format(RMSE_r))
    # RMSE_rb = float('{:.4f}'.format(RMSE_rb))
    # RMSE_blr = float('{:.4f}'.format(RMSE_blr))

    # plt.cla()
    # plt.clf()
    # plt.plot(valid_y)
    # plt.plot(pred_y_p)
    # plt.plot(pred_y_r)
    # plt.plot(pred_y_rb)
    # plt.plot(pred_y_blr)
    # plt.legend(["Ground Truth", "Linear Regression "+str(RMSE_p), "Linear Regression (reg) "+str(RMSE_r), "Linear Regression (r/b) "+str(RMSE_rb), "Bayesian Linear Regression "+str(RMSE_blr)], loc='lower right')
    # # plt.show()
    # plt.savefig("MSE_2.jpg")



    """# 1. Tune alpha"""
    a_list = [0.9, 1, 5, 10, 100, 1000, 10000, 100000,1000000]
    R_list = []
    acc_list = []
    for i in a_list:
        w_a, b_a = BayesianLR_a(train_B, train_y, i)
        pre_a = np.dot(valid_A, w_a)+b_a
        RMSE_a = np.sqrt(((pre_a - valid_y) ** 2).mean())
        # pre_a = adjust(pre_a)
        acc_a = 0
        for i in range(pre_a.shape[0]):
            if pre_a[i][0] > 0.5:
                pre_a[i][0] = 1
            else:
                pre_a[i][0] = 0
            if pre_a[i] == valid_y[i]:
                acc_a += 1
        acc_a = acc_a/pre_a.shape[0]
        
        acc_list.append(acc_a)
        R_list.append(RMSE_a)

    print(R_list)
    print(acc_list)

if __name__ == '__main__':
	
	#main_prob1()

	main_prob2()



    
