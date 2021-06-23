import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""# 0.資料前處理"""

"""# Data Type:
* age: continuous.
* workclass(8): Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
* fnlwgt: continuous.
* education(16): Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
* education-num: continuous.
* marital-status(7): Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
* occupation(14): Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
* relationship(6): Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
* race(5): White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
* sex(2): Female, Male.
* capital-gain: continuous.
* capital-loss: continuous.
* hours-per-week: continuous.
* native-country(41): United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
"""

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



def pinv_LR(x,y):
    x_inv = np.linalg.pinv(x)
    wp = np.dot(x_inv,y)
    return wp




def regression(x,y):
    ld = 1
    
    first = np.dot(np.transpose(x),x)+0.5*ld*np.eye(x.shape[1])
    first = np.linalg.inv(first)
    second = np.dot(np.transpose(x),y)
    wr = np.dot(first,second)
    return wr



def regression_b(x,y):
    ld = 1
    
    first = np.dot(np.transpose(x),x)+0.5*ld*np.eye(x.shape[1])
    first = np.linalg.inv(first)
    second = np.dot(np.transpose(x),y)
    wr = np.dot(first,second)
    bias = wr[0]
    wr = np.delete(wr, 0, axis=0)
    return wr, bias


def BayesianLR(x,y):
    ld = 1
    
    first = np.dot(np.transpose(x),x)+ld*np.eye(x.shape[1])
    first = np.linalg.inv(first)
    second = np.dot(np.transpose(x),y)
    wr = np.dot(first,second)
    bias = wr[0]
    wr = np.delete(wr, 0, axis=0)
    return wr, bias



def train_gen(x_train, y_train):
    dim = 104
    
    cnt1 = 0
    cnt2 = 0
    
    mu1 = np.zeros((dim,))
    mu2 = np.zeros((dim,))
    
    for i in range(x_train.shape[0]):
        if y_train[i] == 1:
            cnt1 += 1
            mu1 += x_train[i]
        else:
            cnt2 += 1
            mu2 += x_train[i]
    mu1 /= cnt1
    mu2 /= cnt2

    sigma1 = np.zeros((dim,dim))
    sigma2 = np.zeros((dim,dim))
    for i in range(x_train.shape[0]):
        if y_train[i] == 1:
            sigma1 += np.dot(np.transpose([x_train[i] - mu1]), [(x_train[i] - mu1)])
        else:
            sigma2 += np.dot(np.transpose([x_train[i] - mu2]), [(x_train[i] - mu2)])
    sigma1 /= cnt1
    sigma2 /= cnt2

    
    share_sigma = (cnt1 / x_train.shape[0]) * sigma1 + (cnt2 / x_train.shape[0]) * sigma2

    sigma_inverse = np.linalg.inv(share_sigma)

    N1 = cnt1
    N2 = cnt2

    w = np.dot( (mu1-mu2), sigma_inverse)
    b = (-0.5) * np.dot(np.dot(mu1.T, sigma_inverse), mu1) + (0.5) * np.dot(np.dot(mu2.T, sigma_inverse), mu2) + np.log(float(N1)/N2)
    # return mu1, mu2, share_sigma, cnt1, cnt2
    return w, b




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


if __name__ == '__main__':
    
    '''
    COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation", "relationship", "race", "gender", "capital_gain", "capital_loss", "hours_per_week", "native_country", "income_bracket"]

    train_df = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        names=COLUMNS,
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
    test_df = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
        names=COLUMNS,
        sep=r'\s*,\s*',
        skiprows=[0],
        engine='python',
        na_values="?")

    train_df = train_df.dropna(how="any", axis=0)
    test_df = test_df.dropna(how="any", axis=0)

    train_df = train_df.reset_index(drop = True)
    test_df = test_df.reset_index(drop = True)

    train_df2 = pd.get_dummies(train_df)
    train_df2 = train_df2.drop(columns=['income_bracket_<=50K'])

    """# (a) 80% train, 20% valid, Normalization"""

    train_num = int(0.8*len(train_df2))
    train = train_df2.iloc[0:train_num,:]
    valid = train_df2.iloc[train_num:,:]

    train = train.reset_index(drop = True)
    valid = valid.reset_index(drop = True)

    cols = list(train)
    cols.pop(-1)

    mu_list = []
    sigma_list = []
    # print(cols)
    for idx, item in enumerate(cols):
        mu = np.mean(train[item])
        sigma = np.std(train[item])
        mu_list.append(mu)
        sigma_list.append(sigma)
        # print(item, mu, sigma)
        for i in range(len(train[item])):
            temp = train.at[i,item]
            train.loc[i,item] = (temp-mu)/sigma
        # print('----------')

    for idx, item in enumerate(cols):
        for i in range(len(valid[item])):
            temp = valid.at[i,item]
            valid.loc[i,item] = (temp-mu_list[idx])/sigma_list[idx]
        #print('----------')

    file = 'train_n.csv'
    train.to_csv(file, index=False)
    file = 'valid_n.csv'
    valid.to_csv(file, index=False)

    print("Finish saving!")
    '''

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

    pred_y_p = adjust(pred_y_p)

    acc_p = 0
    for i in range(pred_y_p.shape[0]):
        if pred_y_p[i][0] == valid_y[i]:
            acc_p += 1
    acc_p = acc_p/pred_y_p.shape[0]

    """# (c) Regularization (without bias)"""
    wr = regression(train_A, train_y)

    pred_y_r = np.dot(valid_A, wr)

    pred_y_r = adjust(pred_y_r)

    acc_r = 0
    for i in range(pred_y_r.shape[0]):
        if pred_y_r[i][0] == valid_y[i]:
            acc_r += 1
    acc_r = acc_r/pred_y_r.shape[0]


    """# (d) Regularization with bias"""
    bias = np.ones(train_A.shape[0])
    train_B = np.insert(train_A, 0, values=bias, axis=1)

    w_rb,bias_rb = regression_b(train_B, train_y)

    pred_y_rb = np.dot(valid_A, w_rb)+bias_rb

    pred_y_rb = adjust(pred_y_rb)

    acc_rb = 0
    for i in range(pred_y_rb.shape[0]):
        if pred_y_rb[i][0] == valid_y[i]:
            acc_rb += 1
    acc_rb = acc_rb/pred_y_rb.shape[0]

    """# (e) Bayesian Linear Regression (with bias)"""
    w_blr,bias_blr = BayesianLR(train_B, train_y)

    pred_y_blr = np.dot(valid_A, w_blr)+bias_blr

    pred_y_blr = adjust(pred_y_blr)

    acc_blr = 0
    for i in range(pred_y_blr.shape[0]):
        if pred_y_blr[i][0] == valid_y[i]:
            acc_blr += 1
    acc_blr = acc_blr/pred_y_blr.shape[0]

    """# (e2) Bayesian Linear Regression (with bias)"""
    print(type(train_A))
    print(type(train_y))
    print("---------------")

    wt, bt = train_gen(train_A, train_y)

    pred_y = predict(wt,bt,valid_A)

    acc = 0
    for i in range(pred_y.shape[0]):
        if pred_y[i] == valid_y[i]:
            acc += 1
    acc = acc/pred_y.shape[0]


    print(acc_p)
    print(acc_r)
    print(acc_rb)
    print(acc_blr)
    print(acc)
    """# 1. Tune alpha"""
    a_list = [0.1, 1, 10, 100, 1000, 10000]
    R_list = []
    for i in a_list:
        w_a, b_a = BayesianLR_a(train_B, train_y, i)
        pre_a = np.dot(valid_A, w_a)+b_a
        pre_a = adjust(pre_a)
        acc_a = 0
        for i in range(pre_a.shape[0]):
            if pre_a[i] == valid_y[i]:
                acc_a += 1
        acc_a = acc_a/pre_a.shape[0]
        R_list.append(acc_a)

    print(R_list)