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
path2 = r'/Users/katiacordoba/Documents/GitHub/policy_lab_ndi/data/indices_and_controls_cross_section.csv'
ndi_df = pd.read_csv(path)
nonpanel_df = pd.read_csv(path2)


# In[3]:


ndi_df


# In[4]:


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
reg_budgetparticipation = smf.ols('budget_participation_index ~ transparency_index_2019 + gdp_percap_ppp_covid', nonpanel_df).fit()


# In[11]:


print(reg_budgetparticipation.summary())


# In[12]:


#Budget Transparency
reg_budget_transparency = smf.ols('budget_transparency_index ~ transparency_index_2019 + gdp_percap_ppp_covid', nonpanel_df).fit()


# In[13]:


print(reg_budget_transparency.summary())


# In[14]:


tables = []
values = [reg_accountability, reg_corruption, reg_trust, reg_effectiveness]
for value in values:
    LRresult = value.summary2().tables[1]
    some_values = ['Intercept', 'transparency_index', 'gdp_percap_ppp', 'gdp_percap_ppp_covid', 'transparency_index_2019']
    LRresult = LRresult.loc[LRresult.index.isin(some_values)]#.style.apply(highlight_1, axis=1)
    tables.append(LRresult)


# In[25]:


#Call on tables to show all of them or by index: tables[i] 
tables[3]


# In[26]:


#COVID Models
nonpanel_df


# In[27]:


#Pairplot COVID variables
#pairplot2 = sns.pairplot(data = covid_df, vars=['transparency_index_2019', 'pandemic_dem_violation_index', 'covid_index', 'gdp_percap', 'gdp_percap_ppp_covid', 'percap_domestic_health_expenditure_ppp']) 


# In[28]:


#COVID
reg_covid = smf.ols('covid_index ~ transparency_index_2019 + gdp_percap_ppp_covid + percap_domestic_health_expenditure_ppp + median_age + aged_65_older', nonpanel_df).fit()


# In[29]:


print(reg_covid.summary())


# In[31]:


#Pandemic Democracy Violations
reg_pandemic_violations = smf.ols('pandemic_dem_violation_index ~ transparency_index_2019 + gdp_percap_ppp_covid + percap_domestic_health_expenditure_ppp + aged_65_older + median_age', nonpanel_df).fit()


# In[32]:


print(reg_pandemic_violations.summary())

