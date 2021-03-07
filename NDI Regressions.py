#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
get_ipython().system('pip install linearmodels')
from linearmodels import PanelOLS
import statsmodels.formula.api as smf
from statsmodels.base.covtype import get_robustcov_results


# In[4]:


path = r'/Users/katiacordoba/Documents/GitHub/policy_lab_ndi/data/indices_and_controls.csv'
ndi_df = pd.read_csv(path)


# In[5]:


ndi_df


# In[6]:


#Replace NaN with mean of column 
column_means = ndi_df.mean()
ndi_df = ndi_df.fillna(column_means)


# In[7]:


ndi_df


# In[8]:


pairplot = sns.pairplot(data = ndi_df)


# In[9]:


#pairplot.savefig("Indices, transparency and controls pairplot.png")


# In[12]:


reg_accountability = smf.ols('accountability_index ~ transparency_index + gdp + gini + C(country_standard)', ndi_df).fit(cov_type='cluster', cov_kwds={'groups': ndi_df['country_standard']})


# In[13]:


reg_accountability.summary()


# In[10]:


reg_corruption = smf.ols('corruption_index ~ transparency_index + gdp + gini + C(country_standard)', ndi_df).fit(cov_type='cluster', cov_kwds={'groups': ndi_df['country_standard']})


# In[11]:


reg_corruption.summary()


# In[14]:


reg_trust = smf.ols('trust_index ~ transparency_index + gdp + gini + C(country_standard)', ndi_df).fit(cov_type='cluster', cov_kwds={'groups': ndi_df['country_standard']})


# In[15]:


reg_trust.summary()


# In[16]:


reg_effectiveness = smf.ols('effectiveness_index ~ transparency_index + gdp + gini + C(country_standard)', ndi_df).fit(cov_type='cluster', cov_kwds={'groups': ndi_df['country_standard']})


# In[17]:


reg_effectiveness.summary()


# In[18]:


reg_bugetparticipation = smf.ols('budget_participation_index ~ transparency_index + gdp + gini + C(country_standard)', ndi_df).fit(cov_type='cluster', cov_kwds={'groups': ndi_df['country_standard']})


# In[19]:


reg_bugetparticipation.summary()


# In[20]:


reg_buget_transparency = smf.ols('budget_transparency_index ~ transparency_index + gdp + gini + C(country_standard)', ndi_df).fit(cov_type='cluster', cov_kwds={'groups': ndi_df['country_standard']})


# In[21]:


reg_buget_transparency.summary()


# In[24]:


df_2020 = ndi_df.loc[ndi_df['year'] == 2020]


# In[25]:


df_2020


# In[26]:


#COVID outcomes model
reg_covid = smf.ols('covid_index ~ transparency_index + gdp + gini + percap_domestic_health_expenditure + median_age + aged_65_older', df_2020).fit()


# In[27]:


reg_covid.summary()


# In[28]:


reg_pandemic_violations = smf.ols('pandemic_dem_violation_index ~ transparency_index + gdp + gini + percap_domestic_health_expenditure + median_age + aged_65_older', df_2020).fit()


# In[30]:


reg_pandemic_violations.summary()
