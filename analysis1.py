

import numpy as np
import pandas as pd
import matplotlib.pyplot as mplt
import seaborn as sn
get_ipython().magic(u'matplotlib inline')
sn.set_style("darkgrid")



df12 = pd.read_csv("C:/Users/Lenovo/Desktop/01_district_wise_crimes_committed_ipc_2001_2012 (2).csv")
df13 = pd.read_csv("C:/Users/Lenovo/Desktop/01_district_wise_crimes_committed_ipc_2013 (2).csv")
df14 = pd.read_csv("C:/Users/Lenovo/Desktop/01_district_wise_crimes_committed_ipc_2014 (2).csv")




df = pd.concat([df12, df13, df14])




df12.shape, df13.shape, df14.shape, df.shape



df['state_ut'] = df.state_ut.str.replace('\s+&\s+', '&')
df['district_ut'] = df.district.apply(lambda x: x.replace('\s+&\s+', '&'))


list(df14.columns)


no_total = df.loc[df.district != 'Total', :]




yr_totals = no_total.groupby('year')
yr_agg = yr_totals.sum().reset_index()



for cl in yr_agg.columns:
    if cl == 'year':
        continue
    if yr_agg[cl].isnull().sum() > 10:
        yr_agg = yr_agg.drop(cl, axis=1)
    else:
        mean_cl = np.mean(yr_agg[cl])
        yr_agg[cl] = yr_agg[cl].fillna(mean_cl)
        yr_agg[cl] = yr_agg[cl].apply(lambda x: x/10000)




yr_agg.describe()




cols = list(yr_agg.columns)
cols.remove('year')
cols.remove('total_ipc_crimes')
cols.remove('other_ipc_crimes')
fig = mplt.figure()
fig.set_size_inches(15, 10)
ax = mplt.subplot(111)
# ax.set_xlim([2001, 2014])
ax.set_title("Crime committed in India (in ten thousands)")
ax.set_xlabel("Year")
ax.set_ylabel("Number of cases filed")
for col in cols:
    ax.plot(yr_agg.year, yr_agg[col], label=col.replace('_', ' '))
ax.legend(loc=5, bbox_to_anchor=(1.5, .5))




corr = yr_agg[cols].corr()
mask = np.array(corr)
mask[np.tril_indices_from(mask)] = False
fig, ax = mplt.subplots()
fig.set_size_inches(15, 15)
sn.heatmap(corr, mask=mask, ax=ax, square=True)




group_by = ["state_ut"]
columns_of_interest = ["total_ipc_crimes"]
state_grp = no_total[columns_of_interest + group_by].groupby(["state_ut"])
state_agg = state_grp.sum().reset_index().sort_values(by='total_ipc_crimes')
state_agg['total_ipc_crimes'] = state_agg.total_ipc_crimes




fig = mplt.figure()
fig.set_size_inches(20, 5)
ax = fig.add_subplot(111)
ax.set_xticklabels(state_agg.state_ut, rotation=90)
b = sn.barplot(x='state_ut', y='total_ipc_crimes', data=state_agg, ax=ax)




columns_of_interest = [
    'state_ut',
    'district',
    'year',
    'murder',
    'rape',
    'kidnapping_abduction_total',
    'dacoity',
    'dacoity_with_murder',
    'robbery',
    'theft',
    'unlawful_assembly',
    'riots',
    'riots_communal',
    'riots_industrial',
    'riots_political',
    'riots_caste_conflict',
    'riots_students',
    'counterfeiting',
    'dowry_deaths',
    'sexual_harassment',
    'importation_of_girls_from_foriegn_country',
    'humantrafficking',
    'unnatural_offence'
]






