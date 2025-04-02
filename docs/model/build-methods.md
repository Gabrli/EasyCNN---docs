
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