#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.formula.api as smf


# In[2]:


#path = r'/Users/katiacordoba/Documents/GitHub/policy_lab_ndi/data/indices_and_controls.csv'
path = r'/Users/edtro/OneDrive/Documents/GitHub/policy_lab_ndi/data/indices_and_controls.csv'

ndi_df = pd.read_csv(path)


# In[3]:


#Replace NaN with mean of column 
column_means = ndi_df.mean()
ndi_df = ndi_df.fillna(column_means)


# In[4]:


pairplot = sns.pairplot(data = ndi_df)


# In[5]:


#pairplot.savefig("Indices, transparency and controls pairplot.png")


# In[6]:


reg_accountability = smf.ols('accountability_index ~ transparency_index + gdp + gini + C(country_standard)', ndi_df).fit(cov_type='cluster', cov_kwds={'groups': ndi_df['country_standard']})


# In[7]:


reg_corruption = smf.ols('corruption_index ~ transparency_index + gdp + gini + C(country_standard)', ndi_df).fit(cov_type='cluster', cov_kwds={'groups': ndi_df['country_standard']})


# In[8]:


reg_trust = smf.ols('trust_index ~ transparency_index + gdp + gini + C(country_standard)', ndi_df).fit(cov_type='cluster', cov_kwds={'groups': ndi_df['country_standard']})


# In[9]:


reg_effectiveness = smf.ols('effectiveness_index ~ transparency_index + gdp + gini + C(country_standard)', ndi_df).fit(cov_type='cluster', cov_kwds={'groups': ndi_df['country_standard']})


# In[10]:


reg_bugetparticipation = smf.ols('budget_participation_index ~ transparency_index + gdp + gini + C(country_standard)', ndi_df).fit(cov_type='cluster', cov_kwds={'groups': ndi_df['country_standard']})


# In[11]:


reg_buget_transparency = smf.ols('budget_transparency_index ~ transparency_index + gdp + gini + C(country_standard)', ndi_df).fit(cov_type='cluster', cov_kwds={'groups': ndi_df['country_standard']})


# In[14]:


tables = []
values = [reg_accountability, reg_corruption, reg_trust, reg_effectiveness, reg_bugetparticipation, reg_buget_transparency]
for value in values:
    LRresult = value.summary2().tables[1]
    some_values = ['Intercept', 'transparency_index', 'gdp', 'gini']
    LRresult = LRresult.loc[LRresult.index.isin(some_values)]
    tables.append(LRresult)


# In[23]:


#Call on tables to show all of them or by index: tables[i] 
tables


# In[24]:


#Select only year 2020 for next two models 
df_2020 = ndi_df.loc[ndi_df['year'] == 2020]


# In[25]:


#COVID outcomes model
reg_covid = smf.ols('covid_index ~ transparency_index + gdp + gini + percap_domestic_health_expenditure + median_age + aged_65_older', df_2020).fit()


# In[26]:


LRresult = reg_covid.summary2().tables[1]
LRresult


# In[27]:


reg_pandemic_violations = smf.ols('pandemic_dem_violation_index ~ transparency_index + gdp + gini + percap_domestic_health_expenditure + median_age + aged_65_older', df_2020).fit()


# In[28]:


LRresult = reg_pandemic_violations.summary2().tables[1]
LRresult


# In[ ]:




