import joblib
import numpy as np
from jsbsim_gym.calibration import NominalModel
from jsbsim_gym.uncertainty import RuntimeUncertaintySampler

print('Loading uncertainty model...')
model = joblib.load('f16_uncertainty_model.pkl')
nom_model = model['nominal_model']

weights = []
intercepts = []

for t_idx, t in enumerate(nom_model.targets):
    ridge_m = nom_model.ridge_models[t]
    weights.append(ridge_m.coef_)
    intercepts.append(ridge_m.intercept_)

W = np.stack(weights, axis=-1)  # shape: (45, 6)
B = np.stack(intercepts, axis=0) # shape: (6,)

print(f'Extracted W shape: {W.shape}')
print(f'Extracted B shape: {B.shape}')

np.savez('jsbsim_gym/mppi_nominal_weights.npz', W=W, B=B)
print('Saved to jsbsim_gym/mppi_nominal_weights.npz')
