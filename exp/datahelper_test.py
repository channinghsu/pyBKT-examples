import numpy as np

# 示例数据
data = [
    {"user_id": "student1", "skill_id": "skill1", "problem_id": 1, "multigs": ["guess1"]},
    {"user_id": "student2", "skill_id": "skill2", "problem_id": 2, "multigs": ["guess2"]},
    {"user_id": "student3", "skill_id": "skill3", "problem_id": 3, "multigs": ["guess3"]},
    {"user_id": "student4", "skill_id": "skill4", "problem_id": 4, "multigs": ["guess1"]}
]

# 默认设置
defaults = {"multigs": "multigs"}

# 检查是否存在 multigs
if "multigs" not in defaults:
    raise KeyError("multigs default column not specified")

# 检查数据中是否包含指定的 multigs 列
elif defaults["multigs"] not in data[0]:
    raise KeyError("specified multigs default column not in data")

# 获取所有不同的 multigs 值，并按照要求进行排序
all_multigs = np.sort(np.unique([m for record in data for m in record["multigs"]]))

# 创建一个映射字典，如果没有提供 gs_ref 映射，则创建一个新的映射
gs_ref = None
if gs_ref is None:
    gs_ref = dict(zip(all_multigs, range(len(all_multigs))))
else:
    # 检查是否所有的 multigs 值都已经被映射
    for multigs_item in all_multigs:
        if multigs_item not in gs_ref:
            raise ValueError("Guess rate", multigs_item, "not previously fitted")
print("gs_ref:", gs_ref)
# 根据映射将每个 multigs 值转换为一个唯一的行索引
data_ref = np.array([gs_ref[multigs_item] for record in data for multigs_item in record["multigs"]])
print("data_ref:", data_ref)
# 构建 n 维数据
data_temp = np.zeros((len(all_multigs), len(data)), dtype=int)
for i in range(len(data_temp[0])):
    data_temp[data_ref[i]][i] = 1  # 标记为 1，表示该记录存在于对应的 multigs 中
print("data_temp:", data_temp)

# 存储结果
Data = {"data": data_temp}

# 输出结果
print("Resulting n-dimensional data:")
print(Data["data"])
