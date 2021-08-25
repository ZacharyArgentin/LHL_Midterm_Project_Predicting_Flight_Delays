import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

from sklearn.ensemble import RandomForestRegressor


def to_hours(s):
    """
    
    Convert military format to hours past midnight. Rounded to 2 decimal places.

    e.g. 930 --> 9.5, 1245 --> 12.75

    Caution: integers that are not valid times will not throw an error, but results will
             be nonsensical. check your data beforehand.
    
    e.g. 590 --> 5.15

    Parameters
    ----------
    s : Pandas Series in military time format

    Returns
    -------
    Pandas Series with dtype float
    """
    # change times like '560' to '600' etc. This is actually not necessary unless you round
    # the military time values beforehand so I think I'll comment it out
    ### s.loc[s.astype(str).str.endswith("60")] = s.loc[s.astype(str).str.endswith("60")].values + 40 

    # converting military time to 'continuous time' (e.g. 830 = 8.5)
    times = (s / 100).values.astype(str)
    minutes = np.array([round(int(time.split(".")[1])/60, 2) for time in list(times)])
    hour = (s // 100).values
    hours = hour + minutes
    s = pd.Series(hours)
    return s


def scale_data(X, y):
    """
    
    Standardize the data.
    
    Parameters
    ----------
    X : Pandas DataFrame or numpy array of feature variables
    
    y : Pandas Series or numpy array of target variable
    
    Returns
    -------
    X_scaled, y_scaled
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.fit_transform(y.values.reshape(-1,1))
    return X_scaled, y_scaled


def run_linear_regression(X, y):
    """

    Run a linear regression.
    
    Parameters
    ----------
    X : Pandas DataFrame or numpy array of feature variables
    
    y : Pandas Series or numpy array of target variable
    
    Returns
    -------
    numpy array of predictions for the target variable , R**2 training score, R**2 testing score 
    """
    X_test, X_train, y_test, y_train = train_test_split(X, y, train_size=0.8)
    model = LinearRegression()
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2_train = model.score(X_train, y_train)
    r2_test = model.score(X_test, y_test)
    return y_pred, r2_train, r2_test


def run_polynomial_regression(X, y, degree):
    """

    Run a polynomial regression. Prints out the degree of the Polynomial, the training score,
    and the testing score.
    
    Parameters
    ----------
    X : Pandas DataFrame or numpy array of feature variables
    
    y : Pandas Series or numpy array of target variable

    degree : The degree of the polynomial
    
    Returns
    -------
    numpy array of predictions for the target variable , R**2 training score, R**2 testing score
    """
    poly = PolynomialFeatures(degree)
    X = poly.fit_transform(X)
    X_test, X_train, y_test, y_train = train_test_split(X, y, train_size=0.8)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2_train = model.score(X_train, y_train)
    r2_test = model.score(X_test, y_test)
    print("---Results---")
    print(f"degree = {degree}")
    print(f"Train score = {r2_train}")
    print(f"Test score = {r2_test}")
    return y_pred, r2_train, r2_test


def run_random_forest(X, y, n_estimators=100, max_depth=10):
    """
    
    Run a random forest regression. Prints out the training score and the testing score

    Parameters
    ----------
    X : Pandas DataFrame or numpy array of feature variables
    
    y : Pandas Series or numpy array of target variable

    n_estimators : number of trees in the forest

    max_depth : max depth of each tree
    
    Returns
    -------
    numpy array of predictions for the target variable , R**2 training score, R**2 testing score
    """
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
    X_test, X_train, y_test, y_train = train_test_split(X, y, train_size=0.8)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2_train = model.score(X_train, y_train)
    r2_test = model.score(X_test, y_test)
    print(f"train score = {r2_train}")
    print(f"test score = {r2_test}")
    return y_pred, r2_train, r2_test