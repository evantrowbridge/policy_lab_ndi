```python
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.formula.api as smf
```


```python
path = r'/Users/katiacordoba/Documents/GitHub/policy_lab_ndi/data/indices_and_controls.csv'
ndi_df = pd.read_csv(path)
```


```python
ndi_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country_standard</th>
      <th>year</th>
      <th>transparency_index</th>
      <th>budget_transparency_index</th>
      <th>accountability_index</th>
      <th>trust_index</th>
      <th>corruption_index</th>
      <th>effectiveness_index</th>
      <th>budget_participation_index</th>
      <th>pandemic_dem_violation_index</th>
      <th>covid_index</th>
      <th>gdp</th>
      <th>gini</th>
      <th>gdp_percap</th>
      <th>percap_domestic_health_expenditure</th>
      <th>median_age</th>
      <th>aged_65_older</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>2006</td>
      <td>0.398481</td>
      <td>0.0</td>
      <td>0.621583</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.109936</td>
      <td>0.0</td>
      <td>0.214286</td>
      <td>0.012655</td>
      <td>1.929110e+10</td>
      <td>NaN</td>
      <td>2156.419482</td>
      <td>2.578007</td>
      <td>18.6</td>
      <td>2.581</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>2007</td>
      <td>0.404977</td>
      <td>0.0</td>
      <td>0.623198</td>
      <td>NaN</td>
      <td>0.910703</td>
      <td>0.133040</td>
      <td>0.0</td>
      <td>0.214286</td>
      <td>0.012655</td>
      <td>1.929110e+10</td>
      <td>NaN</td>
      <td>2156.419482</td>
      <td>2.578007</td>
      <td>18.6</td>
      <td>2.581</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Afghanistan</td>
      <td>2008</td>
      <td>0.391248</td>
      <td>0.0</td>
      <td>0.629477</td>
      <td>NaN</td>
      <td>0.933634</td>
      <td>0.143647</td>
      <td>0.0</td>
      <td>0.214286</td>
      <td>0.012655</td>
      <td>1.929110e+10</td>
      <td>NaN</td>
      <td>2156.419482</td>
      <td>2.578007</td>
      <td>18.6</td>
      <td>2.581</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Afghanistan</td>
      <td>2009</td>
      <td>0.115751</td>
      <td>0.0</td>
      <td>0.630297</td>
      <td>NaN</td>
      <td>0.933071</td>
      <td>0.144919</td>
      <td>0.0</td>
      <td>0.214286</td>
      <td>0.012655</td>
      <td>1.929110e+10</td>
      <td>NaN</td>
      <td>2156.419482</td>
      <td>2.578007</td>
      <td>18.6</td>
      <td>2.581</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Afghanistan</td>
      <td>2010</td>
      <td>0.112175</td>
      <td>0.0</td>
      <td>0.627897</td>
      <td>NaN</td>
      <td>0.939024</td>
      <td>0.147968</td>
      <td>0.0</td>
      <td>0.214286</td>
      <td>0.012655</td>
      <td>1.929110e+10</td>
      <td>NaN</td>
      <td>2156.419482</td>
      <td>2.578007</td>
      <td>18.6</td>
      <td>2.581</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3973</th>
      <td>Zimbabwe</td>
      <td>2017</td>
      <td>0.388089</td>
      <td>0.0</td>
      <td>0.543230</td>
      <td>NaN</td>
      <td>0.851687</td>
      <td>0.237870</td>
      <td>0.0</td>
      <td>0.500000</td>
      <td>0.271321</td>
      <td>2.144076e+10</td>
      <td>44.3</td>
      <td>2961.446428</td>
      <td>39.249222</td>
      <td>19.6</td>
      <td>2.822</td>
    </tr>
    <tr>
      <th>3974</th>
      <td>Zimbabwe</td>
      <td>2018</td>
      <td>0.146885</td>
      <td>0.0</td>
      <td>0.533216</td>
      <td>NaN</td>
      <td>0.847306</td>
      <td>0.236812</td>
      <td>0.0</td>
      <td>0.500000</td>
      <td>0.271321</td>
      <td>2.144076e+10</td>
      <td>44.3</td>
      <td>2961.446428</td>
      <td>39.249222</td>
      <td>19.6</td>
      <td>2.822</td>
    </tr>
    <tr>
      <th>3975</th>
      <td>Zimbabwe</td>
      <td>2019</td>
      <td>0.394583</td>
      <td>0.0</td>
      <td>0.504282</td>
      <td>NaN</td>
      <td>0.836403</td>
      <td>0.261814</td>
      <td>0.0</td>
      <td>0.500000</td>
      <td>0.271321</td>
      <td>2.144076e+10</td>
      <td>44.3</td>
      <td>2961.446428</td>
      <td>39.249222</td>
      <td>19.6</td>
      <td>2.822</td>
    </tr>
    <tr>
      <th>3976</th>
      <td>Zimbabwe</td>
      <td>2020</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.402851</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.500000</td>
      <td>0.271321</td>
      <td>2.144076e+10</td>
      <td>44.3</td>
      <td>2961.446428</td>
      <td>39.249222</td>
      <td>19.6</td>
      <td>2.822</td>
    </tr>
    <tr>
      <th>3977</th>
      <td>Zimbabwe</td>
      <td>2021</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.500000</td>
      <td>0.271321</td>
      <td>2.144076e+10</td>
      <td>44.3</td>
      <td>2961.446428</td>
      <td>39.249222</td>
      <td>19.6</td>
      <td>2.822</td>
    </tr>
  </tbody>
</table>
<p>3978 rows × 17 columns</p>
</div>




```python
#Transparency without covid models
pairplot = sns.pairplot(data = ndi_df, vars=['transparency_index', 'budget_transparency_index', 'accountability_index', 'trust_index', 'corruption_index', 'effectiveness_index', 'budget_participation_index', 'gdp_percap'])
```


![png](output_3_0.png)



```python
#Transparency and covid models
pairplot_covid = sns.pairplot(data = ndi_df, vars=['transparency_index', 'pandemic_dem_violation_index', 'covid_index', 'gdp_percap', 'percap_domestic_health_expenditure', 'median_age', 'aged_65_older'])
```


![png](output_4_0.png)



```python
#pairplot.savefig("Transparency without covid models.png")
```


```python
#pairplot_covid.savefig("Transparency on covid models.png")
```


```python
#Significant and positive
available_data = ndi_df.loc[:,['accountability_index', 'transparency_index', 'gdp_percap', 'country_standard']].dropna(how='any')
reg_accountability = smf.ols('accountability_index ~ transparency_index + gdp_percap + C(country_standard)', available_data).fit(cov_type='cluster', cov_kwds={'groups': available_data['country_standard']})
```


```python
#Significant and negative
corr = ndi_df.loc[:,['corruption_index', 'transparency_index', 'gdp_percap', 'country_standard']].dropna(how='any')
reg_corruption = smf.ols('corruption_index ~ transparency_index + gdp_percap + C(country_standard)', corr).fit(cov_type='cluster', cov_kwds={'groups': corr['country_standard']})
```


```python
#Not significant positive, very few observations
trust = ndi_df.loc[:,['trust_index', 'transparency_index', 'gdp_percap', 'country_standard']].dropna(how='any')
reg_trust = smf.ols('trust_index ~ transparency_index + gdp_percap + C(country_standard)', trust).fit(cov_type='cluster', cov_kwds={'groups': trust['country_standard']})
```


```python
#Significant and positive
effect = ndi_df.loc[:,['effectiveness_index', 'transparency_index', 'gdp_percap', 'country_standard']].dropna(how='any')
reg_effectiveness = smf.ols('effectiveness_index ~ transparency_index + gdp_percap + C(country_standard)', effect, missing='drop').fit(cov_type='cluster', cov_kwds={'groups': effect['country_standard']})
```


```python
#Not significant negative
budget_particip = ndi_df.loc[:,['budget_participation_index', 'transparency_index', 'gdp_percap', 'country_standard']].dropna(how='any')
reg_bugetparticipation = smf.ols('budget_participation_index ~ transparency_index + gdp_percap + C(country_standard)', budget_particip).fit(cov_type='cluster', cov_kwds={'groups': budget_particip['country_standard']})
```


```python
#Not significant positive
budget_transp = ndi_df.loc[:,['budget_transparency_index', 'transparency_index', 'gdp_percap', 'country_standard']].dropna(how='any')
reg_buget_transparency = smf.ols('budget_transparency_index ~ transparency_index + gdp_percap + C(country_standard)', budget_transp).fit(cov_type='cluster', cov_kwds={'groups': budget_transp['country_standard']})
```


```python
tables = []
values = [reg_accountability, reg_corruption, reg_trust, reg_effectiveness, reg_bugetparticipation, reg_buget_transparency]
for value in values:
    LRresult = value.summary2().tables[1]
    some_values = ['Intercept', 'transparency_index', 'gdp_percap']
    LRresult = LRresult.loc[LRresult.index.isin(some_values)]#.style.apply(highlight_1, axis=1)
    tables.append(LRresult)
```

    /Users/katiacordoba/opt/anaconda3/lib/python3.7/site-packages/statsmodels/base/model.py:1832: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 171, but rank is 1
      'rank is %d' % (J, J_), ValueWarning)
    /Users/katiacordoba/opt/anaconda3/lib/python3.7/site-packages/statsmodels/base/model.py:1832: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 179, but rank is 1
      'rank is %d' % (J, J_), ValueWarning)
    /Users/katiacordoba/opt/anaconda3/lib/python3.7/site-packages/statsmodels/base/model.py:1832: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 78, but rank is 1
      'rank is %d' % (J, J_), ValueWarning)
    /Users/katiacordoba/opt/anaconda3/lib/python3.7/site-packages/statsmodels/base/model.py:1832: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 160, but rank is 1
      'rank is %d' % (J, J_), ValueWarning)
    /Users/katiacordoba/opt/anaconda3/lib/python3.7/site-packages/statsmodels/base/model.py:1832: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 188, but rank is 1
      'rank is %d' % (J, J_), ValueWarning)
    /Users/katiacordoba/opt/anaconda3/lib/python3.7/site-packages/statsmodels/base/model.py:1832: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 188, but rank is 1
      'rank is %d' % (J, J_), ValueWarning)



```python
#Call on tables to show all of them or by index: tables[i] 
tables[0]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Coef.</th>
      <th>Std.Err.</th>
      <th>z</th>
      <th>P&gt;|z|</th>
      <th>[0.025</th>
      <th>0.975]</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Intercept</th>
      <td>5.749225e-01</td>
      <td>1.147231e-02</td>
      <td>50.113937</td>
      <td>0.000000</td>
      <td>0.552437</td>
      <td>0.597408</td>
    </tr>
    <tr>
      <th>transparency_index</th>
      <td>2.587581e-01</td>
      <td>6.702221e-02</td>
      <td>3.860782</td>
      <td>0.000113</td>
      <td>0.127397</td>
      <td>0.390119</td>
    </tr>
    <tr>
      <th>gdp_percap</th>
      <td>-2.773302e-07</td>
      <td>7.738812e-07</td>
      <td>-0.358363</td>
      <td>0.720072</td>
      <td>-0.000002</td>
      <td>0.000001</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Select years greater than 2018 for next two models 
df_2020 = ndi_df.loc[ndi_df['year'] > 2018]
```


```python
#Fill NA values with previous year
df_2020['transparency_index'].fillna(method='ffill', inplace=True)
```

    /Users/katiacordoba/opt/anaconda3/lib/python3.7/site-packages/pandas/core/generic.py:6287: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      self._update_inplace(new_data)



```python
#Keep only 2020
df_2020 = df_2020.loc[df_2020['year'] == 2020]
```


```python
#COVID outcomes model Not significant (significant without age controls) 
reg_covid = smf.ols('covid_index ~ transparency_index + gdp_percap + percap_domestic_health_expenditure + median_age + aged_65_older', df_2020).fit()
```


```python
LRresult = reg_covid.summary2().tables[1]
LRresult
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Coef.</th>
      <th>Std.Err.</th>
      <th>t</th>
      <th>P&gt;|t|</th>
      <th>[0.025</th>
      <th>0.975]</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Intercept</th>
      <td>-0.051443</td>
      <td>0.124502</td>
      <td>-0.413192</td>
      <td>0.679981</td>
      <td>-0.297191</td>
      <td>0.194305</td>
    </tr>
    <tr>
      <th>transparency_index</th>
      <td>0.064414</td>
      <td>0.078500</td>
      <td>0.820556</td>
      <td>0.413035</td>
      <td>-0.090534</td>
      <td>0.219362</td>
    </tr>
    <tr>
      <th>gdp_percap</th>
      <td>0.000007</td>
      <td>0.000002</td>
      <td>3.873579</td>
      <td>0.000152</td>
      <td>0.000004</td>
      <td>0.000011</td>
    </tr>
    <tr>
      <th>percap_domestic_health_expenditure</th>
      <td>-0.000028</td>
      <td>0.000029</td>
      <td>-0.988977</td>
      <td>0.324063</td>
      <td>-0.000085</td>
      <td>0.000028</td>
    </tr>
    <tr>
      <th>median_age</th>
      <td>0.006733</td>
      <td>0.006658</td>
      <td>1.011316</td>
      <td>0.313286</td>
      <td>-0.006408</td>
      <td>0.019874</td>
    </tr>
    <tr>
      <th>aged_65_older</th>
      <td>0.005374</td>
      <td>0.009466</td>
      <td>0.567756</td>
      <td>0.570941</td>
      <td>-0.013310</td>
      <td>0.024059</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Significant without either median age or aged_65_older (negative both cases)
reg_pandemic_violations = smf.ols('pandemic_dem_violation_index ~ transparency_index + gdp_percap + percap_domestic_health_expenditure + median_age + aged_65_older', df_2020).fit()
```


```python
LRresult = reg_pandemic_violations.summary2().tables[1]
LRresult
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Coef.</th>
      <th>Std.Err.</th>
      <th>t</th>
      <th>P&gt;|t|</th>
      <th>[0.025</th>
      <th>0.975]</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Intercept</th>
      <td>3.204529e-01</td>
      <td>0.106300</td>
      <td>3.014622</td>
      <td>0.003063</td>
      <td>0.110266</td>
      <td>0.530639</td>
    </tr>
    <tr>
      <th>transparency_index</th>
      <td>-1.264424e-01</td>
      <td>0.070727</td>
      <td>-1.787757</td>
      <td>0.076010</td>
      <td>-0.266291</td>
      <td>0.013406</td>
    </tr>
    <tr>
      <th>gdp_percap</th>
      <td>-8.706109e-07</td>
      <td>0.000002</td>
      <td>-0.519559</td>
      <td>0.604203</td>
      <td>-0.000004</td>
      <td>0.000002</td>
    </tr>
    <tr>
      <th>percap_domestic_health_expenditure</th>
      <td>-4.469910e-05</td>
      <td>0.000025</td>
      <td>-1.790454</td>
      <td>0.075573</td>
      <td>-0.000094</td>
      <td>0.000005</td>
    </tr>
    <tr>
      <th>median_age</th>
      <td>8.245122e-03</td>
      <td>0.005779</td>
      <td>1.426704</td>
      <td>0.155924</td>
      <td>-0.003182</td>
      <td>0.019672</td>
    </tr>
    <tr>
      <th>aged_65_older</th>
      <td>-1.144379e-02</td>
      <td>0.008383</td>
      <td>-1.365086</td>
      <td>0.174447</td>
      <td>-0.028020</td>
      <td>0.005132</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
