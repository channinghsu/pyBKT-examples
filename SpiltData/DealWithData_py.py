import pandas as pd
from sklearn.model_selection import train_test_split

# 读取数据集文件 ct.csv
# df = pd.read_csv('../data/as_delete_nan.csv')  # 根据实际情况选择正确的编码
# df = pd.read_csv('../data/student-problem-middle.csv')
df = pd.read_csv('../data/ct_relation.csv')
# 按知识点分组，并获取每个知识点的总题数
knowledge_point_counts = df['skill'].value_counts()


# 定义一个函数，用于从每个知识点的题目中抽取一定比例的题目作为训练集
def split_by_knowledge_point(df, knowledge_point, train_size=0.7, random_state=23):
    knowledge_point_df = df[df['skill'] == knowledge_point]
    if len(knowledge_point_df) <= 1:
        print(f"Skipping knowledge point {knowledge_point} due to insufficient data.")
        return None, None
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

# 为 "num" 列创建新的连续整数序列
train_df['num'] = range(1, len(train_df) + 1)
predict_df['num'] = range(1, len(predict_df) + 1)

# 输出训练集和预测集的信息
print("训练集总题数:", len(train_df))
print("预测集总题数:", len(predict_df))

# 将 train_df 保存为 TSV 文件
# train_df.to_csv('../data/mooc_radar_train_set.tsv', sep='\t', index=False)
train_df.to_csv('../data/ct_train.tsv', sep='\t', index=False)

# 将 predict_df 保存为 CSV 文件
# predict_df.to_csv('../data/mooc_radar_predict_set.csv', index=False)
predict_df.to_csv('../data/ct_predict.csv', index=False)
