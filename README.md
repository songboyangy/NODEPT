利用Neural ODE研究信息整体流行度的代码

调试好的参数
```shell
# 下面参数msle:2.0316
python main.py --dataset twitter  --prefix test  --epoch 150 --lr 1e-4 --patience 10 --memory_size 16 --bs 64 --predict_single_timestamp 10

```
较好的模型
model_path='saved_models/test_twitter_CTCP_2024-03-23_05-40-51'

test

```shell
python main.py --dataset twitter  --prefix test  --epoch 150 --lr 1e-4 --patience 10 --memory_size 16 --bs 64 --predict_single_timestamp 10
``` 