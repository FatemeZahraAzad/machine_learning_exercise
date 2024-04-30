from matplotlib import pyplot as plt
import sys
from packaging import version
import sklearn
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from pathlib import Path
import tarfile
import urllib.request
from pathlib import Path
import seaborn as sns
import urllib.request

plt.rc('font', size=12)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=12)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

np.random.seed(42)

data_root = "https://github.com/ageron/data/raw/main/"
lifesat = pd.read_csv(data_root + 'lifesat/lifesat.csv')
x = lifesat[["GDP per capita (USD)"]].values
y = lifesat[["Life satisfaction"]].values

lifesat.plot(kind="scatter", grid=True, x="GDP per capita (USD)", y="Life satisfaction")
plt.axis([23_500, 62_500, 4, 9])
plt.show()

model = KNeighborsRegressor(n_neighbors=3)
model.fit(x, y)

X_new = [[37_655.2]]
print(model.predict(X_new))

