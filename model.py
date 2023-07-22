import keras as k
import numpy as np

input_data = np.array([0.3, 0.7, 0.9])
output_data = np.array([0.5, 0.9, 1.0])

# y = ax + b

model = k.models.Sequential()
model.add(k.layers.Dense(units=1, activation='linear', input_shape=[1]))
model.compile(loss='mse', optimizer='sgd')

fit_results = model.fit(x=input_data, y=output_data, epochs=100)

predicted = model.predict([0.5])
print(predicted)