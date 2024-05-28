from desReg.des.DESRegression import DESRegression
from desReg.dataset import load_Student_Mark

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


data = load_Student_Mark()
X = data.iloc[:,:-1].to_numpy()
y = np.ravel(data.iloc[:, -1:]) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# DES declaration
homogeneous_DES = DESRegression()
# DES fitting
homogeneous_DES.fit(X_train, y_train)
# DES prediction
y_pred = homogeneous_DES.predict(X_test)
print('MSE error:', mean_squared_error(y_test, y_pred))