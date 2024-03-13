import pandas as pd
import numpy as np

# 创建一个包含NumPy数组的DataFrame
data = {'array_col': [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]}
df = pd.DataFrame(data)

print(df)
