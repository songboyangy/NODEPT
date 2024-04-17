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

python main.py --dataset twitter  --prefix casode  --epoch 150 --lr 1e-4 --patience 10 --memory_size 16 --bs 64 --predict_timestamp [5,10,15] --gpu 2   --use_dynamic --use_temporal 

python main.py --dataset weibo  --prefix w_o_TL  --epoch 150 --lr 1e-4 --patience 10 --memory_size 16 --bs 128 --predict_timestamp [5,10,15]  --gpu 0 --use_dynamic --use_temporal --self_evolution

python test.py --predict_timestamps 1,2,3



python main.py --dataset aps  --prefix w_o_IL  --epoch 150 --lr 1e-4 --patience 10 --memory_size 16 --bs 64 --predict_timestamp [8,10,15,20] --gpu 1   --use_dynamic --use_temporal --self_evolution --test --test_model_path w_o_IL_aps_CTCP_2024-04-16_07-48-52
```
