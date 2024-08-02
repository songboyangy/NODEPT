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

python main.py --dataset twitter  --prefix increase50memory_influence  --epoch 150 --lr 1e-4 --patience 10 --memory_size 16 --bs 64 --predict_timestamp [5] --gpu 2   --use_dynamic --use_temporal 


python main.py --dataset weibo  --prefix increase50memory_influence  --epoch 150 --lr 1e-4 --patience 10 --memory_size 16 --bs 128 --predict_timestamp [5]  --gpu 0 --use_dynamic --use_temporal 

python test.py --predict_timestamps 1,2,3



python main.py --dataset aps  --prefix increase50memory_influence  --epoch 150 --lr 1e-4 --patience 10 --memory_size 16 --bs 64 --predict_timestamp [5,10,15] --gpu 1   --use_dynamic --use_temporal 

python main.py --dataset twitter  --prefix increase_test  --epoch 150 --lr 1e-4 --patience 10 --memory_size 16 --bs 64 --predict_timestamp [5,10,15] --gpu 2   --use_dynamic --use_temporal --test --test_model_path increase_twitter_CTCP_2024-07-19_05-35-37
```

# 训练完成的model
## twitter
saved_models/test_twitter_CTCP_2024-04-13_14-02-47.pth


## weibo

saved_models/test_weibo_obs2_res15.pth


## aps
saved_models/obs5_res20_aps_CTCP_2024-04-12_08-18-34.pth