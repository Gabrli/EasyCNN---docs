```python
from litecnn.visualizer import TrainingVisualizer

#...code before

history = model.train(x_train, y_train, x_test, y_test, epochs=5)

visualizer = TrainingVisualizer()
visualizer.plot_training(history)
```

![image](https://github.com/user-attachments/assets/6cf34bbf-ded7-44b0-99bb-8446371b27d6)