```python
from litecnn.core import LiteCNN
from litecnn.presets import resnet50Preset

model = LiteCNN()
model.add(resnet50Preset(10)) # used preset
model.add_flatten()
model.add_dense(512, activation='relu')
model.add_dense(10, activation='softmax')
model.compile()
```
