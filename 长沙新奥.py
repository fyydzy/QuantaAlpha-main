# 读取数据
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# df2 = pd.read_pickle(r'C:\Users\lenovo\Downloads\JA-Netfilter-v2.2.3\data_train_new_2022.pkl')
# df_pre = df2
# df_pre = df_pre[df_pre['客户'] == '长沙新奥']
# df_pre = df_pre[['时间', '销量']]
# df_pre['时间'] = pd.to_datetime(df_pre['时间'])
# df = df_pre.set_index('时间').resample('D').sum().reset_index().rename({'时间': 'ds', '销量': 'y'}, axis=1)
# df.to_csv('test.csv', index=None)

warnings.filterwarnings("ignore")


def build_fourier_features(date_index, yearly_order=3, weekly_order=1):
    """构造周周期 + 年周期 Fourier 特征，供 ARIMAX 显式学习季节性。"""
    t = np.arange(len(date_index), dtype=float)
    features = {}

    for k in range(1, weekly_order + 1):
        angle = 2 * np.pi * k * t / 7.0
        features[f"weekly_sin_{k}"] = np.sin(angle)
        features[f"weekly_cos_{k}"] = np.cos(angle)

    for k in range(1, yearly_order + 1):
        angle = 2 * np.pi * k * t / 365.25
        features[f"yearly_sin_{k}"] = np.sin(angle)
        features[f"yearly_cos_{k}"] = np.cos(angle)

    return pd.DataFrame(features, index=date_index)

df = pd.read_csv("test.csv", parse_dates=["ds"])
print(df)
from prophet import Prophet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings


warnings.filterwarnings('ignore')
# 数据范围
min_date = df['ds'].min()
max_date = df['ds'].max()
print(min_date, max_date)
m = Prophet()
m.fit(df)
horizon = 100
future = m.make_future_dataframe( periods= horizon)
future.head(2)
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper',
         'trend','trend_lower','trend_upper',
         'weekly','yearly']].head(5)