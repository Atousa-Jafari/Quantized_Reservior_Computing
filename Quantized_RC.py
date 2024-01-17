from reservoirpy.datasets import mackey_glass

X = mackey_glass(n_timesteps=2000)

import numpy as np

def quantize_weight(weights, n_bits=8):
    """ Quantize weights to n_bits """

    R_min = np.min(weights)
    R_max = np.max(weights)

    S = (R_max - R_min) / (2**n_bits - 1)
    print(S)
    Z = np.round(-R_min/S)
    print(Z)
    q_weights = np.round(Z + weights/S)
    #q_weights=q_weights.astype(np.int8)
    print(q_weights.dtype)
    return q_weights

#small example for me...

quantize_weight([1.2,3.2,7.5],8)

def quantize_activation(activations, n_bits=16):
    """Quantize the activations with hard tanh function to n_bits"""
    R_min = -1   # Lower bound of hard tanh
    R_max = 1    # Upper bound of hard tanh

    S = (R_max - R_min) / (2**n_bits - 1)
    Z = np.round(-R_min/S)
    q_activations = np.round(Z + activations/S)

    return q_activations

def quantize_input_state(uin, n_bits=8):
    """ Quantize weights to n_bits """

    R_min = np.min(uin)
    R_max = np.max(uin)

    S = (R_max - R_min) / (2**n_bits - 1)
    Z = np.round(-R_min/S)

    q_input_state = np.round(Z + uin/S)
    return q_input_state

from reservoirpy.nodes import Reservoir, Ridge

reservoir = Reservoir(units=100, lr=1, sr=0.2,input_connectivity=0.2, rc_connectivity=0.1)
readout = Ridge(output_dim=1, ridge=1e-5)

reservoir.params
#reservoir.state()
esn = reservoir >> readout

esn.fit(X[:1000], X[1:1001], warmup=200)
reservoir.params
reservoir.state()
reservoir.internal_state
print(type(reservoir.W))
predictions = esn.run(X[1001:-1])
len(X[1001:-1])
from reservoirpy.observables import rmse, rsquare

print("RMSE:", rmse(X[1002:], predictions), "R^2 score:", rsquare(X[1002:], predictions))
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 9))
plt.title(" its future.")
plt.xlabel("$t$")
plt.plot(predictions, label="Predicted", color="blue")
plt.plot(X[1001:-1], label="Real ", color="red")
plt.legend()
plt.show()
#print(reservoir.W.data)
print(type(reservoir.W))
reservoir.W.data=quantize_weight(reservoir.W.data, n_bits=8)
print(type(reservoir.W))
print(reservoir.W.shape)
X=quantize_input_state(X,n_bits=8)
print(reservoir.Win.data)
print(type(reservoir.Win))
reservoir.Win.data=quantize_weight(reservoir.Win.data,n_bits=8)
print(type(reservoir.Win))
print(reservoir.Win.data)
a2D = np.array([[1], [3]])
a2D.shape
predictions2=np.eye(998,1)
for i in range (998):
  reservoir.state()[0]=quantize_activation(reservoir.state(),16)
  predictions2[i]=esn.run(X[i+1001])

  #reservoir.rc_connectivity=1 not
#reservoir.sr=10
#readout.ridge=1e-6
esn.fit(X[:1000], X[1:1001], warmup=100)
#esn.input_connectivity=1
predictions2 = esn.run(X[1001:-1],reset=True)
#print(predictions2.dtype)
# Convert back to float
#predictions2 = predictions2.astype(np.float64)
print(predictions.max())
print(predictions2.max())
#print(predictions2.dtype)
  # Reverse the scaling
scale_factor = (predictions.max() - predictions.min()) / (predictions2.max() - predictions2.min())
predictions2 = (predictions2 - predictions2.min()) * scale_factor + predictions.min()
#predictions2 = (predictions2 - predictions.min()) / (predictions.max() - predictions.min())
print(predictions2.max())
#print(predictions)
#print(predictions2)
#min_range = -128
#max_range = 127
#predictions = (predictions - predictions.min()) * (max_range - min_range) / (predictions.max() - predictions.min()) + min_range
#predictions= predictions.astype(np.int8)

from reservoirpy.observables import rmse, rsquare
print(predictions2)
print("RMSE:", rmse(X[1001:-1], predictions2), "R^2 score:", rsquare(X[1001:-1], predictions2))

print("RMSE:", rmse(predictions, predictions2), "R^2 score:", rsquare(predictions, predictions2))
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 9))
plt.title(" its future.")
plt.xlabel("$t$")
plt.plot(predictions2, label="Predicted", color="blue")
plt.plot(predictions, label="Real ", color="red")
plt.legend()
plt.show()