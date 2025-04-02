```python
from easycnn.core import EasyCNN
from easycnn.presets import resnet50Preset

model = EasyCNN()
model.add(resnet50Preset(10)) # used preset
model.add_flatten()
model.add_dense(512, activation='relu')
model.add_dense(10, activation='softmax')
model.compile()
```