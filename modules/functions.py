from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

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
    R**2 score, numpy array of predictions for the target variable 
    """
    X_test, X_train, y_test, y_train = train_test_split(X, y, train_size=0.8)
    model = LinearRegression()
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = model.score(X_test, y_test)
    return r2, y_pred


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
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print("---Results---")
    print(f"degree = {degree}")
    print(f"Train score = {train_score}")
    print(f"Test score = {test_score}")
    return y_pred, train_score, test_score

