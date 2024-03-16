from pyBKT.models import Model
import pandas as pd
#显示所有列
pd.set_option('display.max_columns',None)
#显示所有行
pd.set_option('display.max_rows',None)
#设置value的显示长度
pd.set_option('max_colwidth',100)
#设置1000列时才换行
pd.set_option('display.width',1000)
if __name__ == '__main__':
    model = Model(num_fits=55)  # 创建模型对象
    model.fit(data_path='./data/predict_set.csv', multigs = True)
    # model.fit(data_path='data/ct.csv', forgets=True)

    # 使用默认的RMSE进行评估，然后指定AUC和ACC进行评估
    print("Training RMSE: %f" % model.evaluate(data_path="./data/predict_set.csv"))
    print("Training AUC: %f" % model.evaluate(data_path="./data/predict_set.csv", metric='auc'))
    print("Training ACC: %f" % model.evaluate(data_path="./data/predict_set.csv", metric='accuracy'))

    # 进行交叉验证进行交叉验证

    print(model.crossvalidate(data_path='./data/predict_set.csv'))
    print(model.crossvalidate(data_path='./data/predict_set.csv', metric='auc'))
    print(model.crossvalidate(data_path='./data/predict_set.csv', metric='accuracy'))

