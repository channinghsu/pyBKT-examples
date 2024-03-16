import pandas as pd
from sklearn.model_selection import train_test_split

# 读取数据集文件 ct.csv
df = pd.read_csv('../data/as_delete_nan.csv')  # 根据实际情况选择正确的编码
# 按知识点分组，并获取每个知识点的总题数
knowledge_point_counts = df['skill_name'].value_counts()

# 定义一个函数，用于从每个知识点的题目中抽取一定比例的题目作为训练集
def split_by_knowledge_point(df, knowledge_point, train_size=0.7, random_state=23):
    knowledge_point_df = df[df['skill_name'] == knowledge_point]
    train_set, predict_set = train_test_split(
        knowledge_point_df,
        test_size=1 - train_size,
        random_state=random_state
    )
    return train_set, predict_set

# 存储训练集和预测集的DataFrame
train_sets = []
predict_sets = []

# 对每个知识点执行拆分
for knowledge_point in knowledge_point_counts.index:
    train_set, predict_set = split_by_knowledge_point(df, knowledge_point)
    train_sets.append(train_set)
    predict_sets.append(predict_set)

# 将训练集和预测集合并为最终的DataFrame
train_df = pd.concat(train_sets)
predict_df = pd.concat(predict_sets)

# 输出训练集和预测集的信息
print("训练集总题数:", len(train_df))
print("预测集总题数:", len(predict_df))

# 将训练集和预测集保存为TSV文件
# train_df.to_csv('train_set.csv', index=False)
predict_df.to_csv('as_predict_set.csv', index=False)
