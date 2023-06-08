# Example code from the regressor library docs I used to test the library
# as well as help me debug what was wrong in my code when I tried to use it
import numpy as np
from sklearn import datasets
from regressors import stats
from sklearn import linear_model
boston = datasets.load_boston()
which_betas = np.ones(13, dtype=bool)
which_betas[3] = False  # Eliminate dummy variable
X = boston.data[:, which_betas]
y = boston.target
ols = linear_model.LinearRegression()
ols.fit(X, y)
xlabels = boston.feature_names[which_betas]
print(ols.coef_.shape)
print(ols.intercept_.shape)
print(stats.summary(ols, X, y, xlabels))
print("Done")