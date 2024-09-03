from sklearn import linear_model
from sklearn.linear_model import LinearRegression, Ridge
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# np.random.seed(0)
# X = 2 * np.random.randn(100, 1)
# y = 4 + 3 * X + np.random.randn(100, 1)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model = LinearRegression()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# plt.scatter(X_test, y_test, color='blue', label='Actual')
# plt.scatter(X_test, y_pred, color='red', label='Predicted')
# plt.plot(X_test, y_pred, color='green', linewidth=2)
# plt.title('Linear Regression Predictions')
# plt.xlabel('Feature')
# plt.ylabel('Target')
# plt.legend()
# plt.show()
# //////////////////////////////////////////////////////////////////////////////////////////////////////
np.random.seed(0)
X = 2 * np.random.randn(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def calculate_mse(model, X, y):
    y_pred = model.predict(X)
    return mean_squared_error(y, y_pred)


linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
mse_linear = calculate_mse(linear_model, X_test, y_test)
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
mse_ridge = calculate_mse(ridge_model, X_test, y_test)
plt.scatter(X_test, y_test, color='blue', label='Actual', alpha=0.5)
plt.scatter(X_test, linear_model.predict(X_test), color='red', label='Linear Regression', alpha=0.5)
plt.scatter(X_test, ridge_model.predict(X_test), color='green', label='Ridge Regression', alpha=0.5)
plt.title('Model Predictions Comparison')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.legend()
plt.show()
