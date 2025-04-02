```python
from easycnn.core import EasyCNN

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

```