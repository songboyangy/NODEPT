import argparse
import ast

# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser()

# 添加参数
parser.add_argument('--predict_timestamps', type=str, default='', help='time_point_timestamp_to_predict')

# 解析参数
args = parser.parse_args()
print(args.predict_timestamps.split(','))
# 如果参数被提供，则将字符串解析为列表
if args.predict_timestamps:
    predict_timestamps =  [int(num) for num in args.predict_timestamps.split(',')]
    args.predict_timestamps = predict_timestamps
    print("Predict timestamps:", args.predict_timestamps)
else:
    print("No predict timestamps provided.")
