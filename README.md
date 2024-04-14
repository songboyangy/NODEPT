利用Neural ODE研究信息整体流行度的代码

调试好的参数
```shell
# 下面参数msle:2.0316
python main.py --dataset twitter  --prefix test  --epoch 150 --lr 1e-4 --patience 10 --memory_size 16 --bs 64 --predict_single_timestamp 20

```
较好的模型
model_path='saved_models/test_twitter_CTCP_2024-03-23_05-40-51'

test

```shell

python main.py --dataset twitter  --prefix w_o_DL  --epoch 150 --lr 1e-4 --patience 10 --memory_size 16 --bs 64 --predict_timestamp [5,10,15] --gpu 3  --use_temporal

python main.py --dataset weibo  --prefix w_o_DL  --epoch 150 --lr 1e-4 --patience 10 --memory_size 16 --bs 128 --predict_timestamp [5,10,15] --use_temporal 

python test.py --predict_timestamps 1,2,3



python main.py --dataset aps  --prefix obs5_res20  --epoch 150 --lr 1e-4 --patience 10 --memory_size 16 --bs 64 --predict_timestamp [8,10,15,20] --gpu 3
```
