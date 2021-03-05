#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
get_ipython().system('pip install linearmodels')
from linearmodels import PanelOLS
import statsmodels.formula.api as smf
from statsmodels.base.covtype import get_robustcov_results


# In[2]:


path = r'/Users/katiacordoba/Documents/GitHub/policy_lab_ndi/data/indices_and_controls.csv'
ndi_df = pd.read_csv(path)


# In[3]:


ndi_df


# In[4]:


#Replace NaN with mean of column 
column_means = ndi_df.mean()
ndi_df = ndi_df.fillna(column_means)


# In[5]:


ndi_df


# In[6]:


#Drop last 15 rows with NA values in country_standard
ndi_df = ndi_df.dropna(subset=['country_standard'])


# In[7]:


ndi_df


# In[8]:


#Covid model data (only has 2020)
data2020 = pd.read_csv(r'/Users/katiacordoba/Documents/GitHub/policy_lab_ndi/data/merged_data_yr_2020.csv')
covid_index_df = pd.read_csv(r'/Users/katiacordoba/Documents/GitHub/policy_lab_ndi/data/covid-19_index.csv')


# In[9]:


covid_df = pd.merge(data2020, covid_index_df, on="country_standard")


# In[10]:


#pairplot = sns.pairplot(data = ndi_df)


# In[11]:


#pairplot.savefig("Indices, transparency and controls pairplot.png")


# In[27]:


reg_corruption = smf.ols('corruption_index ~ transparency_index + gdp + gdp_per_capita + gini_2020 + C(country_standard)', ndi_df).fit(cov_type='cluster', cov_kwds={'groups': ndi_df['country_standard']})


# In[28]:


reg_corruption.summary()


# In[56]:


reg_trust = smf.ols('trust_index ~ transparency_index + gdp + gdp_per_capita + gini_2020 + C(country_standard)', ndi_df).fit(cov_type='cluster', cov_kwds={'groups': ndi_df['country_standard']})


# In[57]:


reg_trust.summary()


# In[52]:


reg_effectiveness = smf.ols('effectiveness_index ~ transparency_index + gdp + gdp_per_capita + gini_2020 + C(country_standard)', ndi_df).fit(cov_type='cluster', cov_kwds={'groups': ndi_df['country_standard']})


# In[53]:


reg_effectiveness.summary()


# In[31]:


reg_bugetparticipation = smf.ols('budget_participation_index ~ transparency_index + gdp + gdp_per_capita + gini_2020 + C(country_standard)', ndi_df).fit(cov_type='cluster', cov_kwds={'groups': ndi_df['country_standard']})


# In[32]:


reg_bugetparticipation.summary()


# In[20]:


df_2020 = ndi_df.loc[ndi_df['year'] == 2020]


# In[21]:


df_2020


# In[43]:


#COVID outcomes model
reg_covid = smf.ols('covid_index ~ transparency_index + gdp + gdp_per_capita + gini_2020 + percap_domestic_health_expenditure + median_age + aged_65_older', df_2020).fit()


# In[45]:


reg_covid.summary()


# In[24]:


#Fixed effects or not, only 2020? 
#reg_pandemic_violations = smf.ols('pandemic_dem_violation_index ~ transparency_index + gdp + gdp_per_capita + gini_2020 + percap_domestic_health_expenditure + median_age + aged_65_older + C(country_standard)', ndi_df).fit(cov_type='cluster', cov_kwds={'groups': ndi_df['country_standard']})


# In[48]:


reg_pandemic_violations = smf.ols('pandemic_dem_violation_index ~ transparency_index + gdp + gdp_per_capita + gini_2020 + percap_domestic_health_expenditure + median_age + aged_65_older', df_2020).fit()


# In[49]:


reg_pandemic_violations.summary()


# In[ ]:




