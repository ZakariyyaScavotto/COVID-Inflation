from customLR import myLR
import numpy as np

dummyData = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
dummyTarget = np.array([1, 2, 3, 4, 5])

model = myLR()
model.fit(dummyData, dummyTarget)
model.save('myLR.pkl')

loaded_model = myLR.load('myLR.pkl')
print(loaded_model.predict(dummyData))
# Expected output: [1. 2. 3. 4. 5.]

# Now, test with dataset with new features
dummyData = np.array([[16, 17, 18], [19, 20, 21], [22, 23, 24], [25, 26, 27], [28, 29, 30]])
loaded_model.fit(dummyData, dummyTarget)
loaded_model.save('myLR.pkl')

loaded_model = myLR.load('myLR.pkl')
print(loaded_model.predict(dummyData))

# Test on pandas DataFrame
import pandas as pd

dummyData = pd.DataFrame({'A': [1, 3, 5, 7, 9], 'B': [2, 4, 6, 8, 10]})
dummyTarget = np.array([1, 2, 3, 4, 5])

model = myLR()
model.fit(dummyData, dummyTarget)
model.save('myLR.pkl')

loaded_model = myLR.load('myLR.pkl')
print(loaded_model.predict(dummyData))

# Now, test with dataset with new features
dummyData = pd.DataFrame({'A': [16, 19, 22, 25, 28], 'B': [17, 20, 23, 26, 29], 'C': [18, 21, 24, 27, 30]})
loaded_model.fit(dummyData, dummyTarget)
loaded_model.save('myLR.pkl')

loaded_model = myLR.load('myLR.pkl')
print(loaded_model.predict(dummyData))
