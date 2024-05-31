import pandas as pd

result_csv = pd.read_csv('data/Data2022-05-26-20-35-38/violinplot-sum.csv')  # 硬脂酸数据
index_columns = result_csv.columns
index_rows = result_csv.index
df_T = pd.DataFrame(result_csv.values.T, columns=index_rows, index=index_columns)
df_T.columns = ['GA4CBST', 'GA', 'DE', 'PSO', 'SA']
# yingzhisuan#显示数据
# 绘制非分组小提琴图
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")

sns.violinplot(data=df_T, palette="Set3", scale='count', cut=1, linewidth=1)

sns.despine(left=True, bottom=True)
plt.show()
