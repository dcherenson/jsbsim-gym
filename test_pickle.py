import joblib
from jsbsim_gym.uncertainty import RuntimeUncertaintySampler
from jsbsim_gym.calibration import NominalModel
model = joblib.load('f16_uncertainty_model.pkl')
if isinstance(model, dict):
    print("Keys:", model.keys())
else:
    print("Type:", type(model))
    print("Attributes:", dir(model))
