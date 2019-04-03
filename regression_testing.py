from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge, RidgeCV, lars_path
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np


def linear_regression_avg(df, y_column, regular=None, lambda_val=None,
                          coef=False, pr=False):
    """
    This function will take in a DataFrame and a specific column and will
    run a linear regression on 5 train/test splits and will average out
    the scores. It also will find the lambda value for you using helper
    function find_lambda.
    Args:
        df (pandas.DataFrame): DataFrame to run linear regression modeling.
        y_column (string): Name of the column that holds the target value.
        regular (string | Default=None): Value can be set either to Lasso or
            Ridge, controls the type of regularization. Default is None, which
            results in no regularization.
        lambda_val (int | Default=None): Only called when using regularization.
            Represents the lambda value, if None is specified, will find the 
            best lambda_val using find_lambda.
        coef (bool | Default=False): Boolean that controls whether to print
            coefficients or not. If set to True coefficients will print.
        pr (bool | Default=False): Boolean that controls printing train and
            test scores. If set to True, will print scores.
    Return:
        None
    """
    # Set up X and Y values for linear regression
    X = df.drop(columns=y_column).astype(float)
    y = df.loc[:, [y_column]].astype(float)
    y = y.iloc[:, 0]
    X_train_val, x_test, y_train_val, y_test = train_test_split(X, y,
                                                                test_size=.2)
    # Set up list to store test and train scores
    train_val_scores = []
    train_test_scores = []
    final_score = 0
    # Needed to run linear regression
    X_train_val = np.array(X_train_val)
    y_train_val = np.array(y_train_val)
    # Set up KFold Splits
    kf = KFold(n_splits=5, shuffle=True)
    # Run Regularization if specified
    if str(regular).upper() == 'LASSO':
        # Need to Scale for Linear Regression
        scaler = StandardScaler()
        X_train_val = scaler.fit_transform(X_train_val)
        x_test = scaler.transform(x_test)
        # Find Lambda Val if not specified
        if lambda_val is None:
            lambda_val = find_lambda(X_train_val, y_train_val, 5, kf)
            if coef:
                print("Lambda: " + str(lambda_val))
        lm = Lasso(alpha=lambda_val)
    elif str(regular).upper() == 'RIDGE':
        # Need to Scale for Linear Regression
        scaler = StandardScaler()
        X_train_val = scaler.fit_transform(X_train_val)
        x_test = scaler.transform(x_test)
        # Find Lambda Val if not specified
        if lambda_val is None:
            lambda_val = find_lambda(X_train_val, y_train_val, 5, kf)
            if coef:
                print("Lambda: " + str(lambda_val))
        lm = Ridge(alpha=lambda_val)
    else:
        lm = LinearRegression()
        scaler = None
    # Cross Validation
    for train_ind, test_val_ind in kf.split(X_train_val, y_train_val):
        x_tr_val, y_tr_val = X_train_val[train_ind], y_train_val[train_ind]
        x_tr_test = X_train_val[test_val_ind]
        y_tr_test = y_train_val[test_val_ind]
        lm.fit(x_tr_val, y_tr_val)
        train_val_scores.append(lm.score(x_tr_val, y_tr_val))
        train_test_scores.append(lm.score(x_tr_test, y_tr_test))
    final_score = lm.score(x_test, y_test)

    if pr:
        print("Train Val Scores: " + str(pd.Series(train_val_scores).mean()))
        print("Train Test Scores: " + str(pd.Series(train_test_scores).mean()))
        print("Test Scores: " + str(final_score))
    if coef:
        for i in range(len(lm.coef_)):
            print(str(lm.coef_[i])+':' + str(X.columns[i]))
    return(lm, pd.Series(train_val_scores).mean(),
           pd.Series(train_test_scores).mean(), final_score, scaler,
           lambda_val)


def find_lambda(X_train_val, y_train_val, loop_total, kf=None):
    """
    This function will run iteratively over different lambdas in order
    to find the most optimal lambda value for linear regression regularization.
    Args:
        X_train_val (numpy.array): Numpy Array that holds the X Values to be
        plugged into Linear Regression.
        y_train_val (numpy.array): Numpy Array that holds the Y Values to be
        plugged into Linear Regression.
        loop_total (int): How many iterations to find the best lambda.
        kf (sklearn.model_selection._split.KFold | Default = None): The KFold
        split to use if it is entered.
    Returns:
        best_num(int): Best lambda number
    """
    minim = 1
    maxim = 4
    best_R = 0
    old_best_R = 0
    best_num = 0
    old_best_num = 0
    new_r = 0
    # Set up Kfold Split if None
    if kf is None:
        kf = KFold(n_splits=5, shuffle=True)
    # Loop over different powers of 10, set lambda to that number
    for lambda_power in np.arange(minim, maxim, 1):
        lm_lass = Lasso(alpha=10.0**lambda_power)
        # Store Best R Squared Numbers and the older best R Squared numbers
        new_R = np.mean(cross_val_score(lm_lass, X_train_val, y_train_val,
                        cv=kf, scoring='r2'))
        if new_R > best_R:
            old_best_num = best_num
            old_best_R = best_R
            best_num = (10.0**lambda_power)
            best_R = new_R
    maxim = best_num
    best_R = 0
    # Loop over the best lambda, now that we have a range of numbers to loop over.
    for i in range(1, loop_total+1):
        for x in np.linspace(float(old_best_num), float(best_num), 5+1):
            lm_lass = Lasso(alpha=x)
            # Store Best R Squared Numbers and the older best R Squared numbers
            new_R = np.mean(cross_val_score(lm_lass, X_train_val, y_train_val,
                            cv=kf, scoring='r2'))
            if new_R > best_R:
                old_best_num = best_num
                old_best_R = best_R
                best_num = x
                best_R = new_R
        if(maxim == best_num):
            best_num = best_num*2
            maxim = best_num
    return(best_num)


def lambda_tester(df, y_column, l_min, l_max, coef=False, num_of_iterations=5):
    """
    This function runs over a specified lambda min and lambda max
    and will print out the best value and the overall scores of each lambda
    for a Lasso regularization linear regression.
    Args:
        df (pandas.DataFrame): Dataframe to run regression over
        y_column (string): Column name that holds target values.
        l_min (int): Starting point of iteration through lambda values.
        l_max (int): Ending point of iteration through lambda values.
        coef (bool | Default=False): Will print coefficients of best lambda
            values
        num_of_iterations (int): Number of iterations to run regularization
        on, to get an average score
    Return:
        None
    """
    X = df.drop(columns=y_column).astype(float)
    y = df.loc[:, [y_column]].astype(float)
    y = y.iloc[:, 0]
    best_r = -np.inf
    temp_r = 0
    r_train_test = 0
    r_test_test = 0
    best_l = l_min
    best_model = 0
    for lam in range(l_min, l_max):
        train_test = []
        test_test = []
        for i in range(0, num_of_iterations+1):
            fit, train_val, train_test_val, test_test_val = Lasso_Reg(X,
                                                            y, lam)
            train_test.append(train_test_val)
            test_test.append(test_test_val)
        temp_r = np.array(train_test).mean()+np.array(test_test).mean()
        if best_r < temp_r:
            best_l = lam
            best_r = temp_r
            r_train_test = np.array(train_test).mean()
            r_test_test = np.array(test_test).mean()
            best_model = fit
            print(best_l)
            print(r_train_test)
            print(r_test_test)
            print('\n')
    if coef:
        for c in range(len(best_model.coef_)):
            print(str(best_model.coef_[c])+':' + str(X.columns[c]))
    return None


def Lasso_Reg(X, y, lambda_val=None, coef=False, pr=False):
    """
    This function runs a Lasso Regularization, over 5 KFold splits and
    will return the fits and the average scores over the Cross Val splits.
    Args:
        X (numpy.array): X values
        y (numpy.array): y values
        lambda_val (int | Default=None): Lambda value to run lasso on, if
            None, will find the best lambda with find_lambda function
        coef (bool | Default=False): Boolean that controls whether to print
            coefficients or not. If set to True coefficients will print.
        pr (bool | Default=False): Boolean that controls printing train and
            test scores. If set to True, will print scores.
    Return:
        lm (sklearn.linear_model.LinearRegression): Linear Regression model.
        train_val_scores (float): Cross Val Validation data set R squared score
        train_test_scores (float): Cross Val Test data set R squared score
        final_score (float): Final Score on holdout set

    """
    # Set up train/test values
    X_train_val, x_test, y_train_val, y_test = train_test_split(X, y,
                                                        test_size=.2)
    # Lasso Scaled
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_val)
    X_test_scaled = scaler.transform(x_test)
    # Set up Lasso
    if lambda_val is None:
        lambda_val = find_lambda(X_train_val, y_train_val, 5)
        print("Lambda: " + str(lambda_val))
    lm = Lasso(alpha=lambda_val)
    # Get CV Train_Val Scores and CV Train_Test Scores
    train_val_scores = []
    train_test_scores = []
    final_score = 0
    # Set KFolds
    X_train_val = np.array(X_train_val)
    y_train_val = np.array(y_train_val)
    kf = KFold(n_splits=5, shuffle=True)
    for train_ind, test_val_ind in kf.split(X_train_val, y_train_val):
        x_tr_val, y_tr_val = X_train_val[train_ind], y_train_val[train_ind]
        x_tr_test, y_tr_test = X_train_val[test_val_ind], y_train_val[test_val_ind]
        lm.fit(x_tr_val, y_tr_val)
        train_val_scores.append(lm.score(x_tr_val, y_tr_val))
        train_test_scores.append(lm.score(x_tr_test, y_tr_test))
    final_score = lm.score(x_test, y_test)
    if pr:
        print("Train Val Scores: " + str(pd.Series(train_val_scores).mean()))
        print("Train Test Scores: " + str(pd.Series(train_test_scores).mean()))
        print("Test Scores: " + str(final_score))
    if coef:
        for i in range(len(lm.coef_)):
            print(str(lm.coef_[i])+':' + str(X.columns[i]))
    return lm, pd.Series(train_val_scores).mean(), pd.Series(train_test_scores).mean(), final_score


def regression_avg(df, y_column, coef=False, num_of_iterations=5):
    """
    Function used for final model checks, run a bunch of iterations over
    our final model and print out the total scores.
    Args:
        df (pandas.DataFrame): Dataframe to run regression over
        y_column (string): Column name that holds target values.
        coef (bool | Default=False): Will print coefficients of best lambda
            values
        num_of_iterations (int): Number of iterations to run regularization
        on, to get an average score
    Return:
        None
    """
    X = df.drop(columns=y_column).astype(float)
    best_r = -np.inf
    temp_r = 0
    train = []
    train_test = []
    test_test = []
    for i in range(0, num_of_iterations+1):
        fit, train_val, train_test_val, test_test_val, scaler, lambda_num = linear_regression_avg(df, y_column)
        train.append(train_val)
        train_test.append(train_test_val)
        test_test.append(test_test_val)
    print("Train Val Scores: " + str(pd.Series(train).mean()))
    print("Train Test Scores: " + str(pd.Series(train_test).mean()))
    print("Test Scores: " + str(pd.Series(test_test).mean()))
    if coef:
        for i in range(len(fit.coef_)):
            print(str(X.columns[i])+": "+str(fit.coef_[i]))
    return None
