```python
from lite.core import LiteCNN
from lite.presets import Preset

import os
from tensorflow.keras.datasets import cifar10 # -> In this project I used the cifar10 as dataset

#----Data preparation process----

class_names = ['car', 'plane', 'cat', 'dog', 'bird', 'deer', 'horse', 'frog', 'ship', 'truck']

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

my_file = os.path.join(os.path.dirname(__file__), 'car.jpg') # -> here path to your image

x_train = x_train[:2000]
y_train = y_train[:2000]
x_test = x_test[:400]
y_test = y_test[:400]
x_train = x_train / 255
x_test = x_test / 255

# ----Model build part----
preset = Preset()
model = LiteCNN()
model.add(preset.get_preset('resnet50', 10))
model.add_flatten()
model.add_dense(512, activation='relu')
model.add_dense(10, activation='softmax')
model.compile()
model.train(x_train, y_train, x_test, y_test, epochs=10)
prediction = model.predict(my_file)

print(f"My prediction: {prediction}")
```
