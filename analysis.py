import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
df12 = pd.read_csv("C:/Users/Lenovo/Desktop/01_district_wise_crimes_committed_ipc_2001_2012.csv")
df13 = pd.read_csv("C:/Users/Lenovo/Desktop/01_district_wise_crimes_committed_ipc_2013.csv")
df14 = pd.read_csv("C:/Users/Lenovo/Desktop/01_district_wise_crimes_committed_ipc_2014.csv")
df = pd.concat([df12, df13, df14])
df['state_ut'] = df.state_ut.str.replace('\s+&\s+', '&')
df['district_ut'] = df.district.apply(lambda x: x.replace('\s+&\s+', '&'))
no_total = df.loc[df.district != 'Total', :]
group_by = ["state_ut"]
columns_of_interest = ["total_ipc_crimes"]
state_grp = no_total[columns_of_interest + group_by].groupby(["state_ut"])
state_agg = state_grp.sum().reset_index().sort_values(by='total_ipc_crimes')
state_agg['total_ipc_crimes'] = state_agg.total_ipc_crimes
fig = plt.figure()
fig.set_size_inches(20, 5)
ax = fig.add_subplot(111)
ax.set_xticklabels(state_agg.state_ut, rotation=90)
b = sn.barplot(x='state_ut', y='total_ipc_crimes', data=state_agg, ax=ax)