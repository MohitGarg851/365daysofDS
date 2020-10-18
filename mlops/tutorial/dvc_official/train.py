from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
import pickle
import os

X, y = make_regression(10000,n_features = 11)

# Train a model
reg = LinearRegression().fit(X, y.ravel())
# Print out training r2
print(reg.score(X,y.ravel() ))

# Write the model to a file
if not os.path.isdir("model/"):
    os.mkdir("model")

filename = 'model/weights.pkl'
pickle.dump(reg, open(filename, 'wb'))
