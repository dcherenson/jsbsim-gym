from sklearn.preprocessing import PolynomialFeatures
import numpy as np

features = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7']
poly = PolynomialFeatures(degree=2, include_bias=True)
poly.fit(np.zeros((1, 8)))

for i, fn in enumerate(poly.get_feature_names_out(features)):
    print(f"{i}: {fn}")
