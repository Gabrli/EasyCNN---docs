
# EasyCNN - Documentation

Documentation for EasyCNN (python liblary).

[EasyCNN](https://github.com/Gabrli/easyCNN)




## Installation

Install liblary with pip

```bash
  pip install easycnn
```
    
## Imports

```python
from easycnn.core import EasyCNN
from easycnn.visualizer import TrainingVisualizer
from easycnn.exporter import EasyExporter
from easycnn.presets import (one of prepare model at the list)
```
## Model initiation

```bash
model = EasyCNN()
```
## Model Build Methods


| Name | Parameters     | Description                |
| :-------- | :------- | :------------------------- |
| `add_conv` | `filters:int*, kernel_size:int*, activation:str(optional)` | **Required**. Basic layer|
| `add_max_pool` | `pool_size:int*` |Special layer|
| `add_flatten` | `none` |**Required**. Basic layer|
| `add_dense` | `filters:int*, activation:str(optional)` |**Required**. Basic layer|
| `add_dropout` | `value:int or float (optional)` |Special layer|
| `compile` | `optimizer:str, loss:str, metrics:list[str] -> all optional` |**Required**. Method|
| `train` | `x:np.array*, y:int*, test_x:np.array, test_y:np.array, epochs:int(optional),` |**Required**. Method|
| `save` | `filename* (my-model.h5)` |Method|
| `load` | `filename* (my-model.h5)` |Method|
| `predict` | `x:np:array (image)*` |Method|


## Simple Usage/Examples

```python
from easycnn.core import EasyCNN
from easycnn.visualizer import TrainingVisualizer
import os
from tensorflow.keras.datasets import cifar10

class_names = ['car', 'plane', 'cat', 'dog', 'bird', 'deer', 'horse', 'frog', 'ship', 'truck']

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

my_file = os.path.join(os.path.dirname(__file__), 'car.jpg')

x_train = x_train[:2000]
y_train = y_train[:2000]
x_test = x_test[:400]
y_test = y_test[:400]
x_train = x_train / 255
x_test = x_test / 255


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
history = model.train(x_train, y_train, x_test, y_test, epochs=5)
prediction = model.predict(my_file)

visualizer = TrainingVisualizer()
visualizer.plot_training(history)

```

## Visualizing Training result


| Name | Parameters     | Description                       |
| :-------- | :------- | :-------------------------------- |
| `plot_training`      | `history: Any` | **Required**. Method |

![image](https://github.com/user-attachments/assets/6cf34bbf-ded7-44b0-99bb-8446371b27d6)




## 15 Presets Of The most popular models from keras 
`ResNet50`, `ResNet152`, `ResNet50V2`, `ResNet101`, `ResNet101V2`, `ResNet152V2`, `VGG16`, `VGG19`, `InceptionV3`, `MobileNet`, `MobileNetV2`, `MobileNetV3Small`, `InceptionResNetV2`, `MobileNetV3Large`, `Xception`

## Simple Usage/Examples of preset

```python
from easycnn.presets import resnet50Preset
```
```python
model = EasyCNN()
model.add(resnet50Preset(10))
model.add_flatten()
model.add_dense(512, activation='relu')
model.add_dense(10, activation='softmax')
model.compile()
```

## Conversion from Keras to ONNX

| Name | Parameters     | Description                       |
| :-------- | :------- | :-------------------------------- |
| `export_to_onnx`      | `name ( new model name ): str, path: str` | **Required**. Method |

```python
exporter = EasyExporter()
exporter.export_to_onnx("test_model", "model.h5")
```




