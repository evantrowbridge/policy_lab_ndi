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


path = r'/Users/katiacordoba/Documents/GitHub/policy_lab_ndi/data/indices_and_controls.csv'
ndi_df = pd.read_csv(path)


# In[3]:


ndi_df


# In[24]:


#Transparency without covid models
#pairplot = sns.pairplot(data = ndi_df, vars=['transparency_index', 'budget_transparency_index', 'accountability_index', 'trust_index', 'corruption_index', 'effectiveness_index', 'budget_participation_index', 'gdp_percap'])


# In[5]:


#pairplot.savefig("Transparency without covid models.png")


# In[6]:


#Accountability
available_data = ndi_df.loc[:,['accountability_index', 'transparency_index', 'gdp_percap_ppp', 'country_standard']].dropna(how='any')
reg_accountability = smf.ols('accountability_index ~ transparency_index + gdp_percap_ppp + C(country_standard)', available_data).fit(cov_type='cluster', cov_kwds={'groups': available_data['country_standard']})


# In[7]:


#Corruption
corr = ndi_df.loc[:,['corruption_index', 'transparency_index', 'gdp_percap_ppp', 'country_standard']].dropna(how='any')
reg_corruption = smf.ols('corruption_index ~ transparency_index + gdp_percap_ppp + C(country_standard)', corr).fit(cov_type='cluster', cov_kwds={'groups': corr['country_standard']})


# In[8]:


#Trust
trust = ndi_df.loc[:,['trust_index', 'transparency_index', 'gdp_percap_ppp', 'country_standard']].dropna(how='any')
reg_trust = smf.ols('trust_index ~ transparency_index + gdp_percap_ppp + C(country_standard)', trust).fit(cov_type='cluster', cov_kwds={'groups': trust['country_standard']})


# In[9]:


#Effectiveness
effect = ndi_df.loc[:,['effectiveness_index', 'transparency_index', 'gdp_percap_ppp', 'country_standard']].dropna(how='any')
reg_effectiveness = smf.ols('effectiveness_index ~ transparency_index + gdp_percap_ppp + C(country_standard)', effect, missing='drop').fit(cov_type='cluster', cov_kwds={'groups': effect['country_standard']})


# In[10]:


#Budget Participation
budget_particip = ndi_df.loc[:,['budget_participation_index', 'transparency_index', 'gdp_percap_ppp', 'country_standard']].dropna(how='any')
reg_bugetparticipation = smf.ols('budget_participation_index ~ transparency_index + gdp_percap_ppp + C(country_standard)', budget_particip).fit(cov_type='cluster', cov_kwds={'groups': budget_particip['country_standard']})


# In[11]:


#Budget Transparency
budget_transp = ndi_df.loc[:,['budget_transparency_index', 'transparency_index', 'gdp_percap_ppp', 'country_standard']].dropna(how='any')
reg_buget_transparency = smf.ols('budget_transparency_index ~ transparency_index + gdp_percap_ppp + C(country_standard)', budget_transp).fit(cov_type='cluster', cov_kwds={'groups': budget_transp['country_standard']})


# In[12]:


tables = []
values = [reg_accountability, reg_corruption, reg_trust, reg_effectiveness, reg_bugetparticipation, reg_buget_transparency]
for value in values:
    LRresult = value.summary2().tables[1]
    some_values = ['Intercept', 'transparency_index', 'gdp_percap_ppp']
    LRresult = LRresult.loc[LRresult.index.isin(some_values)]#.style.apply(highlight_1, axis=1)
    tables.append(LRresult)


# In[13]:


#Call on tables to show all of them or by index: tables[i] 
tables[5]


# In[14]:


#Select years greater than 2018 for next two models 
df_2020 = ndi_df.loc[ndi_df['year'] > 2018]


# In[15]:


df_2020


# In[16]:


#Fill NA values with previous year
df_2020['transparency_index'].fillna(method='ffill', inplace=True)


# In[17]:


#Keep only 2020
df_2020 = df_2020.loc[df_2020['year'] == 2020]


# In[18]:


df_2020


# In[25]:


pp = sns.pairplot(data = df_2020, vars=['pandemic_dem_violation_index', 'covid_index', 'gdp_percap', 'gdp_percap_ppp_covid', 'percap_domestic_health_expenditure_ppp']) 


# In[20]:


#COVID outcomes model Not significant (significant without age controls) 
reg_covid = smf.ols('covid_index ~ transparency_index +  gdp_percap_ppp_covid + percap_domestic_health_expenditure_ppp + median_age + aged_65_older', df_2020).fit()


# In[21]:


LRresult = reg_covid.summary2().tables[1]
LRresult


# In[22]:


#Significant and positive (negative without gdp and health) more to explore here 
reg_pandemic_violations = smf.ols('pandemic_dem_violation_index ~ transparency_index + gdp_percap_ppp_covid + percap_domestic_health_expenditure_ppp + aged_65_older + median_age', df_2020).fit()


# In[23]:


LRresult = reg_pandemic_violations.summary2().tables[1]
LRresult


# In[ ]:




