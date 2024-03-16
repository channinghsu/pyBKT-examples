import time

import pandas as pd
from pyBKT.models import Model

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度
pd.set_option('max_colwidth', 100)
# 设置1000列时才换行
pd.set_option('display.width', 1000)

if __name__ == '__main__':
    start = time.time()
    model = Model(num_fits=55)  # 创建模型对象
    # data = '../data/as_delete_nan.csv'
    data = '../data/as_predict_set.csv'
    model.fit(data_path=data, forgets=True, multigs=True)

    # 使用默认的RMSE进行评估，然后指定AUC和ACC进行评估
    print(model.evaluate(data_path=data))
    print(model.evaluate(data_path=data, metric='auc'))
    print(model.evaluate(data_path=data, metric='accuracy'))

    # 进行交叉验证
    print(model.crossvalidate(data_path=data, forgets=True, multigs=True))
    print(model.crossvalidate(data_path=data, forgets=True, metric='auc', multigs=True))
    print(model.crossvalidate(data_path=data, forgets=True, metric='accuracy', multigs=True))
    print(model.crossvalidate(data_path=data, forgets=True, metric='doa', multigs=True))

