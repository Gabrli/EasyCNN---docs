```python
from easycnn.core import EasyCNN
from easycnn.visualizer import TrainingVisualizer # Import visualizer function from EasyCnn
import os
from tensorflow.keras.datasets import cifar10

# ----Data preparation process----

class_names = ['car', 'plane', 'cat', 'dog', 'bird', 'deer', 'horse', 'frog', 'ship', 'truck']

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

my_file = os.path.join(os.path.dirname(__file__), 'car.jpg')

x_train = x_train[:2000]
y_train = y_train[:2000]
x_test = x_test[:400]
y_test = y_test[:400]
x_train = x_train / 255
x_test = x_test / 255

#----Model build part----

model = EasyCNN()
model.add_conv(32, 3)
model.add_max_pool(2)
model.add_conv(64, 3)
model.add_max_pool(2)
model.add_conv(128, 3)
model.add_max_pool(2)
model.add_flatten()
model.add_dense(10, activation='softmax')
model.compile()

history = model.train(x_train, y_train, x_test, y_test, epochs=5) # -> We need history data of model training process to draw the charts.

visualizer = TrainingVisualizer() # -> initialization of the visualizer function
visualizer.plot_training(history) # -> put history variable as parameter to draw function

```