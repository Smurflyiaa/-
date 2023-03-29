# -*- encoding:utf-8 -*-

import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer, KNNImputer


def get_attribute_type(attribute):
    """
    判断属性的类型

    参数：
    attribute: pandas.Series，表示一个属性的数据列

    返回：
    字符串，表示属性的类型，可能的取值包括"数值属性"、"标称属性"、"文本型属性"、"其他"
    """
    if np.issubdtype(attribute.dtype, np.number):  # 如果数据类型是数值类型
        return "数值属性"
    elif attribute.dtype == 'object':  # 如果数据类型是字符串
        text = attribute.dropna().tolist()
        unique_tokens = set(text)  # 去重
        if len(unique_tokens) > 2000:  # 如果唯一词汇量超过2000个，则认为是文本型属性
            return "文本型属性"
        else:
            return "标称属性"
    else:  # 其他情况
        return "其他"


# 读取数据集
df = pd.read_csv('movies_dataset.csv')

# 遍历每一列数据
num_attr = []
nom_attr = []
for col_name in df.columns:
    attr_type = get_attribute_type(df[col_name])
    print(col_name, "的类型是：", attr_type)
    if attr_type == "数值属性":
        num_attr.append(col_name)
        # 统计五数概括和缺失值个数
        attribute = df[col_name]
        five_num = attribute.describe()[['min', '25%', '50%', '75%', 'max']]
        missing_values = attribute.isnull().sum()
        print("  五数概括：\n", five_num)
        print("  缺失值个数：", missing_values)

        # 绘制盒图
        plt.boxplot(attribute.dropna().values)
        plt.title(col_name + "的盒图")
        plt.show()
    elif attr_type == "标称属性":
        nom_attr.append(col_name)
        # 统计每个可能取值的频数
        attribute = df[col_name]
        freq_count = attribute.value_counts()
        print("  频数统计：\n", freq_count)

        # 绘制直方图
        plt.hist(attribute.dropna().values, bins=len(freq_count))
        plt.xticks(rotation=90)
        plt.title(col_name + "的直方图")
        plt.show()

print("数值属性名称列表：", num_attr)
print("标称属性名称列表：", nom_attr)

# 策略1：将缺失部分剔除
df1 = df.dropna()

# 策略2：用最高频率值来填补缺失值
df2 = df.fillna(df.mode().iloc[0])

# 策略3：通过属性的相关关系来填补缺失值
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean.fit(df)
df3 = pd.DataFrame(imp_mean.transform(df), columns=df.columns)

# 策略4：通过数据对象之间的相似性来填补缺失值
df4 = df.copy()
imp = KNNImputer(n_neighbors=5, weights='uniform')
df4.iloc[:, :] = imp.fit_transform(df4)

# 输出处理后的数据集
print("策略1：将缺失部分剔除")
print(df1)

print("\n策略2：用最高频率值来填补缺失值")
print(df2)

print("\n策略3：通过属性的相关关系来填补缺失值")
print(df3)

print("\n策略4：通过数据对象之间的相似性来填补缺失值")
print(df4)

# 导出新数据集
df1.to_csv('data_1.csv', index=False)
df2.to_csv('data_2.csv', index=False)
df3.to_csv('data_3.csv', index=False)
df4.to_csv('data_4.csv', index=False)
