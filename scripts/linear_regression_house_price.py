import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (StandardScaler,
                                   PolynomialFeatures)
from helper_function_files import normality_test, boxcox_test
from scipy.special import inv_boxcox
from plot_helper_files import facet_grid


def load_clean_dataframe():
    USA_house_price_df = pd.read_csv("input/USA_Housing.csv")
    USA_house_price_df = USA_house_price_df.drop(labels = 'Address', axis =1)
    USA_house_price_df = USA_house_price_df.rename(columns={'Avg. Area Income': 'Income',
                                                            'Avg. Area House Age': 'House_Age',
                                                            'Avg. Area Number of Rooms': 'No_Rooms',
                                                            'Avg. Area Number of Bedrooms': 'No_Bedrooms'
                                                            })
    return USA_house_price_df


def lr_regression(col,df):
    lr = LinearRegression()
    y_col = col
    X = df.drop(y_col, axis=1)
    y = df[y_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=4321)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    MAE_linear = metrics.mean_absolute_error(y_test, y_pred).round(2)
    MSE_linear = metrics.mean_squared_error(y_test, y_pred).round(2)
    RMSE_linear = np.sqrt(MSE_linear).round(2)
    r2 = r2_score(y_test, y_pred)
    lr_metrics = pd.DataFrame([MAE_linear, MSE_linear, RMSE_linear,r2],
                              index=['MAE_linear', 'MSE_linear', 'RMSE_linear', 'R^2'], columns=['Linear_Regression'])
    return lr_metrics


def polynomial_regression(col,df):
    X = df.drop(col, axis=1)
    y = col
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4321)
    bc_price = boxcox_test(y_train)
    y_train_bc = bc_price[0]
    lamda = bc_price[1]
    pred_collector = []
    # for i in range(1, 20):
    poly = PolynomialFeatures(degree=2)
    poly.fit(X_train,y_train_bc)
    y_test_pred = poly.predict(X_test)
    inv_boxcox(y_train_bc, lamda)
    y_train_pred = inv_boxcox(y_pred_bc, lam2)
    pred_collector.append(y_train_pred,y_test_pred)
    return r2_score(y_pred_tran,y_pred_bc)


if __name__ == "__main__":
    USA_house_price_df = load_clean_dataframe()
    linear_reg_metrics = lr_regression("Price",USA_house_price_df)
    boxcox_tst = normality_test(USA_house_price_df['Price'])[2]  # Boxcox demonstrated higher p-value.
    bc_price= boxcox_test(USA_house_price_df['Price'])

