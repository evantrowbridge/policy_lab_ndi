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
      <th>accountability_index</th>
      <th>trust_index</th>
      <th>corruption_index</th>
      <th>effectiveness_index</th>
      <th>budget_participation_index</th>
      <th>budget_transparency_index</th>
      <th>pandemic_dem_violation_index</th>
      <th>covid_index</th>
      <th>gdp</th>
      <th>gini</th>
      <th>gdp_percap</th>
      <th>gdp_percap_ppp_covid</th>
      <th>percap_domestic_health_expenditure</th>
      <th>percap_domestic_health_expenditure_ppp</th>
      <th>median_age</th>
      <th>aged_65_older</th>
      <th>gdp_percap_ppp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>2006</td>
      <td>0.398481</td>
      <td>0.621583</td>
      <td>NaN</td>
      <td>0.899492</td>
      <td>0.109936</td>
      <td>1.0</td>
      <td>0.333333</td>
      <td>0.214286</td>
      <td>0.013742</td>
      <td>1.929110e+10</td>
      <td>NaN</td>
      <td>507.103432</td>
      <td>2156.419482</td>
      <td>2.578007</td>
      <td>9.641537</td>
      <td>18.6</td>
      <td>2.581</td>
      <td>1077.761907</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>2007</td>
      <td>0.404977</td>
      <td>0.623198</td>
      <td>NaN</td>
      <td>0.910703</td>
      <td>0.133040</td>
      <td>1.0</td>
      <td>0.333333</td>
      <td>0.214286</td>
      <td>0.013742</td>
      <td>1.929110e+10</td>
      <td>NaN</td>
      <td>507.103432</td>
      <td>2156.419482</td>
      <td>2.578007</td>
      <td>9.641537</td>
      <td>18.6</td>
      <td>2.581</td>
      <td>1228.704135</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Afghanistan</td>
      <td>2008</td>
      <td>0.391248</td>
      <td>0.629477</td>
      <td>NaN</td>
      <td>0.933634</td>
      <td>0.143647</td>
      <td>1.0</td>
      <td>0.333333</td>
      <td>0.214286</td>
      <td>0.013742</td>
      <td>1.929110e+10</td>
      <td>NaN</td>
      <td>507.103432</td>
      <td>2156.419482</td>
      <td>2.578007</td>
      <td>9.641537</td>
      <td>18.6</td>
      <td>2.581</td>
      <td>1272.573204</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Afghanistan</td>
      <td>2009</td>
      <td>0.115751</td>
      <td>0.630297</td>
      <td>NaN</td>
      <td>0.933071</td>
      <td>0.144919</td>
      <td>1.0</td>
      <td>0.333333</td>
      <td>0.214286</td>
      <td>0.013742</td>
      <td>1.929110e+10</td>
      <td>NaN</td>
      <td>507.103432</td>
      <td>2156.419482</td>
      <td>2.578007</td>
      <td>9.641537</td>
      <td>18.6</td>
      <td>2.581</td>
      <td>1519.692548</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Afghanistan</td>
      <td>2010</td>
      <td>0.112175</td>
      <td>0.627897</td>
      <td>NaN</td>
      <td>0.939024</td>
      <td>0.147968</td>
      <td>1.0</td>
      <td>0.333333</td>
      <td>0.214286</td>
      <td>0.013742</td>
      <td>1.929110e+10</td>
      <td>NaN</td>
      <td>507.103432</td>
      <td>2156.419482</td>
      <td>2.578007</td>
      <td>9.641537</td>
      <td>18.6</td>
      <td>2.581</td>
      <td>1710.575645</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3530</th>
      <td>Zimbabwe</td>
      <td>2016</td>
      <td>0.390190</td>
      <td>0.501289</td>
      <td>NaN</td>
      <td>0.850148</td>
      <td>0.241040</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>0.279450</td>
      <td>2.144076e+10</td>
      <td>44.3</td>
      <td>1463.985910</td>
      <td>2961.446428</td>
      <td>39.249222</td>
      <td>55.387615</td>
      <td>19.6</td>
      <td>2.822</td>
      <td>2806.458631</td>
    </tr>
    <tr>
      <th>3531</th>
      <td>Zimbabwe</td>
      <td>2017</td>
      <td>0.388089</td>
      <td>0.543230</td>
      <td>NaN</td>
      <td>0.851687</td>
      <td>0.237870</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>0.279450</td>
      <td>2.144076e+10</td>
      <td>44.3</td>
      <td>1463.985910</td>
      <td>2961.446428</td>
      <td>39.249222</td>
      <td>55.387615</td>
      <td>19.6</td>
      <td>2.822</td>
      <td>3028.245976</td>
    </tr>
    <tr>
      <th>3532</th>
      <td>Zimbabwe</td>
      <td>2018</td>
      <td>0.146885</td>
      <td>0.533216</td>
      <td>NaN</td>
      <td>0.847306</td>
      <td>0.236812</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>0.279450</td>
      <td>2.144076e+10</td>
      <td>44.3</td>
      <td>1463.985910</td>
      <td>2961.446428</td>
      <td>39.249222</td>
      <td>55.387615</td>
      <td>19.6</td>
      <td>2.822</td>
      <td>3206.277079</td>
    </tr>
    <tr>
      <th>3533</th>
      <td>Zimbabwe</td>
      <td>2019</td>
      <td>0.394583</td>
      <td>0.504282</td>
      <td>NaN</td>
      <td>0.836403</td>
      <td>0.261814</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>0.279450</td>
      <td>2.144076e+10</td>
      <td>44.3</td>
      <td>1463.985910</td>
      <td>2961.446428</td>
      <td>39.249222</td>
      <td>55.387615</td>
      <td>19.6</td>
      <td>2.822</td>
      <td>2961.446428</td>
    </tr>
    <tr>
      <th>3534</th>
      <td>Zimbabwe</td>
      <td>2020</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.402851</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>0.279450</td>
      <td>2.144076e+10</td>
      <td>44.3</td>
      <td>1463.985910</td>
      <td>2961.446428</td>
      <td>39.249222</td>
      <td>55.387615</td>
      <td>19.6</td>
      <td>2.822</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>3535 rows × 20 columns</p>
</div>




```python
#Transparency without covid models
#pairplot = sns.pairplot(data = ndi_df, vars=['transparency_index', 'budget_transparency_index', 'accountability_index', 'trust_index', 'corruption_index', 'effectiveness_index', 'budget_participation_index', 'gdp_percap'])
```


![png](output_3_0.png)



```python
#pairplot.savefig("Transparency without covid models.png")
```


```python
#Accountability
available_data = ndi_df.loc[:,['accountability_index', 'transparency_index', 'gdp_percap_ppp', 'country_standard']].dropna(how='any')
reg_accountability = smf.ols('accountability_index ~ transparency_index + gdp_percap_ppp + C(country_standard)', available_data).fit(cov_type='cluster', cov_kwds={'groups': available_data['country_standard']})
```


```python
#Corruption
corr = ndi_df.loc[:,['corruption_index', 'transparency_index', 'gdp_percap_ppp', 'country_standard']].dropna(how='any')
reg_corruption = smf.ols('corruption_index ~ transparency_index + gdp_percap_ppp + C(country_standard)', corr).fit(cov_type='cluster', cov_kwds={'groups': corr['country_standard']})
```


```python
#Trust
trust = ndi_df.loc[:,['trust_index', 'transparency_index', 'gdp_percap_ppp', 'country_standard']].dropna(how='any')
reg_trust = smf.ols('trust_index ~ transparency_index + gdp_percap_ppp + C(country_standard)', trust).fit(cov_type='cluster', cov_kwds={'groups': trust['country_standard']})
```


```python
#Effectiveness
effect = ndi_df.loc[:,['effectiveness_index', 'transparency_index', 'gdp_percap_ppp', 'country_standard']].dropna(how='any')
reg_effectiveness = smf.ols('effectiveness_index ~ transparency_index + gdp_percap_ppp + C(country_standard)', effect, missing='drop').fit(cov_type='cluster', cov_kwds={'groups': effect['country_standard']})
```


```python
#Budget Participation
budget_particip = ndi_df.loc[:,['budget_participation_index', 'transparency_index', 'gdp_percap_ppp', 'country_standard']].dropna(how='any')
reg_bugetparticipation = smf.ols('budget_participation_index ~ transparency_index + gdp_percap_ppp + C(country_standard)', budget_particip).fit(cov_type='cluster', cov_kwds={'groups': budget_particip['country_standard']})
```


```python
#Budget Transparency
budget_transp = ndi_df.loc[:,['budget_transparency_index', 'transparency_index', 'gdp_percap_ppp', 'country_standard']].dropna(how='any')
reg_buget_transparency = smf.ols('budget_transparency_index ~ transparency_index + gdp_percap_ppp + C(country_standard)', budget_transp).fit(cov_type='cluster', cov_kwds={'groups': budget_transp['country_standard']})
```


```python
tables = []
values = [reg_accountability, reg_corruption, reg_trust, reg_effectiveness, reg_bugetparticipation, reg_buget_transparency]
for value in values:
    LRresult = value.summary2().tables[1]
    some_values = ['Intercept', 'transparency_index', 'gdp_percap_ppp']
    LRresult = LRresult.loc[LRresult.index.isin(some_values)]#.style.apply(highlight_1, axis=1)
    tables.append(LRresult)
```

    /Users/katiacordoba/opt/anaconda3/lib/python3.7/site-packages/statsmodels/base/model.py:1832: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 172, but rank is 2
      'rank is %d' % (J, J_), ValueWarning)
    /Users/katiacordoba/opt/anaconda3/lib/python3.7/site-packages/statsmodels/base/model.py:1832: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 190, but rank is 2
      'rank is %d' % (J, J_), ValueWarning)
    /Users/katiacordoba/opt/anaconda3/lib/python3.7/site-packages/statsmodels/base/model.py:1832: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 77, but rank is 2
      'rank is %d' % (J, J_), ValueWarning)
    /Users/katiacordoba/opt/anaconda3/lib/python3.7/site-packages/statsmodels/base/model.py:1832: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 189, but rank is 2
      'rank is %d' % (J, J_), ValueWarning)
    /Users/katiacordoba/opt/anaconda3/lib/python3.7/site-packages/statsmodels/base/model.py:1832: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 191, but rank is 190
      'rank is %d' % (J, J_), ValueWarning)
    /Users/katiacordoba/opt/anaconda3/lib/python3.7/site-packages/statsmodels/base/model.py:1832: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 191, but rank is 190
      'rank is %d' % (J, J_), ValueWarning)



```python
#Call on tables to show all of them or by index: tables[i] 
tables[5]
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
      <td>3.333333e-01</td>
      <td>8.098985e-12</td>
      <td>4.115742e+10</td>
      <td>0.000000</td>
      <td>3.333333e-01</td>
      <td>3.333333e-01</td>
    </tr>
    <tr>
      <th>transparency_index</th>
      <td>9.189871e-13</td>
      <td>2.194662e-13</td>
      <td>4.187374e+00</td>
      <td>0.000028</td>
      <td>4.888413e-13</td>
      <td>1.349133e-12</td>
    </tr>
    <tr>
      <th>gdp_percap_ppp</th>
      <td>4.586154e-17</td>
      <td>9.519366e-18</td>
      <td>4.817710e+00</td>
      <td>0.000001</td>
      <td>2.720393e-17</td>
      <td>6.451915e-17</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Select years greater than 2018 for next two models 
df_2020 = ndi_df.loc[ndi_df['year'] > 2018]
```


```python
df_2020
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
      <th>accountability_index</th>
      <th>trust_index</th>
      <th>corruption_index</th>
      <th>effectiveness_index</th>
      <th>budget_participation_index</th>
      <th>budget_transparency_index</th>
      <th>pandemic_dem_violation_index</th>
      <th>covid_index</th>
      <th>gdp</th>
      <th>gini</th>
      <th>gdp_percap</th>
      <th>gdp_percap_ppp_covid</th>
      <th>percap_domestic_health_expenditure</th>
      <th>percap_domestic_health_expenditure_ppp</th>
      <th>median_age</th>
      <th>aged_65_older</th>
      <th>gdp_percap_ppp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>13</th>
      <td>Afghanistan</td>
      <td>2019</td>
      <td>0.163570</td>
      <td>0.590704</td>
      <td>NaN</td>
      <td>0.900596</td>
      <td>0.139305</td>
      <td>1.0</td>
      <td>0.333333</td>
      <td>0.214286</td>
      <td>0.013742</td>
      <td>1.929110e+10</td>
      <td>NaN</td>
      <td>507.103432</td>
      <td>2156.419482</td>
      <td>2.578007</td>
      <td>9.641537</td>
      <td>18.6</td>
      <td>2.581</td>
      <td>2156.419482</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Albania</td>
      <td>2019</td>
      <td>0.554243</td>
      <td>0.725007</td>
      <td>NaN</td>
      <td>0.691994</td>
      <td>0.528098</td>
      <td>0.5</td>
      <td>0.666667</td>
      <td>0.357143</td>
      <td>0.540872</td>
      <td>1.527918e+10</td>
      <td>33.2</td>
      <td>5353.244856</td>
      <td>14648.267402</td>
      <td>148.436569</td>
      <td>376.501373</td>
      <td>38.0</td>
      <td>13.188</td>
      <td>14648.267402</td>
    </tr>
    <tr>
      <th>41</th>
      <td>Algeria</td>
      <td>2019</td>
      <td>0.157443</td>
      <td>0.416367</td>
      <td>NaN</td>
      <td>0.702728</td>
      <td>0.352232</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.428571</td>
      <td>0.011908</td>
      <td>1.710913e+11</td>
      <td>27.6</td>
      <td>3973.964072</td>
      <td>12019.928356</td>
      <td>168.449661</td>
      <td>633.798828</td>
      <td>29.1</td>
      <td>6.211</td>
      <td>12019.928356</td>
    </tr>
    <tr>
      <th>55</th>
      <td>American Samoa</td>
      <td>2019</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.143812</td>
      <td>0.631710</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.360000e+08</td>
      <td>NaN</td>
      <td>11466.690706</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>69</th>
      <td>Andorra</td>
      <td>2019</td>
      <td>0.926096</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.284371</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.919778</td>
      <td>3.154058e+09</td>
      <td>NaN</td>
      <td>40886.391165</td>
      <td>NaN</td>
      <td>1916.984497</td>
      <td>2450.407959</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3491</th>
      <td>Yemen</td>
      <td>2019</td>
      <td>0.067556</td>
      <td>0.274884</td>
      <td>NaN</td>
      <td>0.938612</td>
      <td>0.021317</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.031210</td>
      <td>2.258108e+10</td>
      <td>36.7</td>
      <td>774.334490</td>
      <td>3688.519849</td>
      <td>7.451180</td>
      <td>14.315764</td>
      <td>20.3</td>
      <td>2.922</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3505</th>
      <td>Zambia</td>
      <td>2019</td>
      <td>0.500186</td>
      <td>0.633221</td>
      <td>NaN</td>
      <td>0.710546</td>
      <td>0.339228</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.428571</td>
      <td>0.406037</td>
      <td>2.330977e+10</td>
      <td>57.1</td>
      <td>1305.063254</td>
      <td>3624.024939</td>
      <td>29.700403</td>
      <td>81.467789</td>
      <td>17.7</td>
      <td>2.480</td>
      <td>3624.024939</td>
    </tr>
    <tr>
      <th>3519</th>
      <td>Zanzibar</td>
      <td>2019</td>
      <td>NaN</td>
      <td>0.481492</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3533</th>
      <td>Zimbabwe</td>
      <td>2019</td>
      <td>0.394583</td>
      <td>0.504282</td>
      <td>NaN</td>
      <td>0.836403</td>
      <td>0.261814</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>0.279450</td>
      <td>2.144076e+10</td>
      <td>44.3</td>
      <td>1463.985910</td>
      <td>2961.446428</td>
      <td>39.249222</td>
      <td>55.387615</td>
      <td>19.6</td>
      <td>2.822</td>
      <td>2961.446428</td>
    </tr>
    <tr>
      <th>3534</th>
      <td>Zimbabwe</td>
      <td>2020</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.402851</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>0.279450</td>
      <td>2.144076e+10</td>
      <td>44.3</td>
      <td>1463.985910</td>
      <td>2961.446428</td>
      <td>39.249222</td>
      <td>55.387615</td>
      <td>19.6</td>
      <td>2.822</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>264 rows × 20 columns</p>
</div>




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
df_2020
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
      <th>accountability_index</th>
      <th>trust_index</th>
      <th>corruption_index</th>
      <th>effectiveness_index</th>
      <th>budget_participation_index</th>
      <th>budget_transparency_index</th>
      <th>pandemic_dem_violation_index</th>
      <th>covid_index</th>
      <th>gdp</th>
      <th>gini</th>
      <th>gdp_percap</th>
      <th>gdp_percap_ppp_covid</th>
      <th>percap_domestic_health_expenditure</th>
      <th>percap_domestic_health_expenditure_ppp</th>
      <th>median_age</th>
      <th>aged_65_older</th>
      <th>gdp_percap_ppp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>960</th>
      <td>Ethiopia</td>
      <td>2020</td>
      <td>0.156400</td>
      <td>NaN</td>
      <td>0.541624</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.285714</td>
      <td>0.235592</td>
      <td>9.591259e+10</td>
      <td>35.0</td>
      <td>855.760862</td>
      <td>2319.707378</td>
      <td>5.661227</td>
      <td>15.572806</td>
      <td>19.8</td>
      <td>3.526</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1199</th>
      <td>Guatemala</td>
      <td>2020</td>
      <td>0.496823</td>
      <td>NaN</td>
      <td>0.077618</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>0.666667</td>
      <td>0.357143</td>
      <td>0.377813</td>
      <td>7.671036e+10</td>
      <td>48.3</td>
      <td>4619.985258</td>
      <td>9019.693804</td>
      <td>93.486130</td>
      <td>173.908356</td>
      <td>22.9</td>
      <td>4.694</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1453</th>
      <td>Iran</td>
      <td>2020</td>
      <td>0.116422</td>
      <td>NaN</td>
      <td>0.583046</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.357143</td>
      <td>0.514688</td>
      <td>4.539965e+11</td>
      <td>40.8</td>
      <td>5550.060957</td>
      <td>12937.475980</td>
      <td>222.420959</td>
      <td>776.789001</td>
      <td>32.4</td>
      <td>5.440</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1664</th>
      <td>Kyrgyzstan</td>
      <td>2020</td>
      <td>0.479062</td>
      <td>NaN</td>
      <td>0.341828</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.571429</td>
      <td>0.006079</td>
      <td>8.454620e+09</td>
      <td>27.7</td>
      <td>1309.392992</td>
      <td>5485.560329</td>
      <td>36.723625</td>
      <td>111.334984</td>
      <td>26.3</td>
      <td>4.489</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1805</th>
      <td>Macao SAR China</td>
      <td>2020</td>
      <td>0.973647</td>
      <td>NaN</td>
      <td>0.576210</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.385912e+10</td>
      <td>NaN</td>
      <td>84096.396311</td>
      <td>129451.063933</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2072</th>
      <td>Myanmar (Burma)</td>
      <td>2020</td>
      <td>0.432114</td>
      <td>NaN</td>
      <td>0.659180</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.357143</td>
      <td>0.362541</td>
      <td>7.608585e+10</td>
      <td>30.7</td>
      <td>1407.813143</td>
      <td>5369.707495</td>
      <td>8.779543</td>
      <td>43.261250</td>
      <td>29.1</td>
      <td>5.732</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2166</th>
      <td>New Zealand</td>
      <td>2020</td>
      <td>0.979150</td>
      <td>NaN</td>
      <td>0.555966</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.5</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.691548</td>
      <td>2.069288e+11</td>
      <td>NaN</td>
      <td>42084.353375</td>
      <td>45382.123508</td>
      <td>3021.277100</td>
      <td>3011.491943</td>
      <td>37.9</td>
      <td>15.322</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2181</th>
      <td>Nicaragua</td>
      <td>2020</td>
      <td>0.152579</td>
      <td>NaN</td>
      <td>0.199680</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.428571</td>
      <td>0.017535</td>
      <td>1.252092e+10</td>
      <td>46.2</td>
      <td>1912.903745</td>
      <td>5646.399468</td>
      <td>103.936005</td>
      <td>283.472290</td>
      <td>27.3</td>
      <td>5.445</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2606</th>
      <td>Portugal</td>
      <td>2020</td>
      <td>0.938392</td>
      <td>NaN</td>
      <td>0.339723</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.5</td>
      <td>0.333333</td>
      <td>0.000000</td>
      <td>0.783658</td>
      <td>2.387851e+11</td>
      <td>33.8</td>
      <td>23252.058518</td>
      <td>37918.446865</td>
      <td>1361.255127</td>
      <td>1992.470581</td>
      <td>46.2</td>
      <td>21.502</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3167</th>
      <td>Tajikistan</td>
      <td>2020</td>
      <td>0.059708</td>
      <td>NaN</td>
      <td>0.865473</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.428571</td>
      <td>0.020865</td>
      <td>8.116627e+09</td>
      <td>34.0</td>
      <td>870.787589</td>
      <td>3529.311126</td>
      <td>16.184902</td>
      <td>67.545227</td>
      <td>23.3</td>
      <td>3.466</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3364</th>
      <td>Ukraine</td>
      <td>2020</td>
      <td>0.542325</td>
      <td>NaN</td>
      <td>0.188459</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.5</td>
      <td>0.666667</td>
      <td>0.214286</td>
      <td>0.544317</td>
      <td>1.537811e+11</td>
      <td>26.1</td>
      <td>3659.031312</td>
      <td>13341.210519</td>
      <td>109.496864</td>
      <td>327.214996</td>
      <td>41.4</td>
      <td>16.462</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3477</th>
      <td>Vietnam</td>
      <td>2020</td>
      <td>0.115130</td>
      <td>NaN</td>
      <td>0.811616</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.214286</td>
      <td>0.293488</td>
      <td>2.619212e+11</td>
      <td>35.7</td>
      <td>2715.276036</td>
      <td>8397.021042</td>
      <td>69.108612</td>
      <td>200.541077</td>
      <td>32.6</td>
      <td>7.150</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3534</th>
      <td>Zimbabwe</td>
      <td>2020</td>
      <td>0.394583</td>
      <td>NaN</td>
      <td>0.402851</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>0.279450</td>
      <td>2.144076e+10</td>
      <td>44.3</td>
      <td>1463.985910</td>
      <td>2961.446428</td>
      <td>39.249222</td>
      <td>55.387615</td>
      <td>19.6</td>
      <td>2.822</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
pp = sns.pairplot(data = df_2020, vars=['pandemic_dem_violation_index', 'covid_index', 'gdp_percap', 'gdp_percap_ppp_covid', 'percap_domestic_health_expenditure_ppp']) 
```


![png](output_18_0.png)



```python
#COVID outcomes model Not significant (significant without age controls) 
reg_covid = smf.ols('covid_index ~ transparency_index +  gdp_percap_ppp_covid + percap_domestic_health_expenditure_ppp + median_age + aged_65_older', df_2020).fit()
```


```python
LRresult = reg_covid.summary2().tables[1]
LRresult
```

    /Users/katiacordoba/opt/anaconda3/lib/python3.7/site-packages/scipy/stats/stats.py:1535: UserWarning: kurtosistest only valid for n>=20 ... continuing anyway, n=12
      "anyway, n=%i" % int(n))





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
      <td>-0.014515</td>
      <td>0.458062</td>
      <td>-0.031687</td>
      <td>0.975749</td>
      <td>-1.135352</td>
      <td>1.106323</td>
    </tr>
    <tr>
      <th>transparency_index</th>
      <td>-0.045456</td>
      <td>0.430344</td>
      <td>-0.105628</td>
      <td>0.919320</td>
      <td>-1.098470</td>
      <td>1.007557</td>
    </tr>
    <tr>
      <th>gdp_percap_ppp_covid</th>
      <td>0.000044</td>
      <td>0.000050</td>
      <td>0.876181</td>
      <td>0.414629</td>
      <td>-0.000078</td>
      <td>0.000166</td>
    </tr>
    <tr>
      <th>percap_domestic_health_expenditure_ppp</th>
      <td>-0.000425</td>
      <td>0.000572</td>
      <td>-0.743798</td>
      <td>0.485096</td>
      <td>-0.001825</td>
      <td>0.000974</td>
    </tr>
    <tr>
      <th>median_age</th>
      <td>0.003920</td>
      <td>0.021946</td>
      <td>0.178624</td>
      <td>0.864112</td>
      <td>-0.049780</td>
      <td>0.057621</td>
    </tr>
    <tr>
      <th>aged_65_older</th>
      <td>-0.005663</td>
      <td>0.039832</td>
      <td>-0.142184</td>
      <td>0.891589</td>
      <td>-0.103128</td>
      <td>0.091801</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Significant and positive (negative without gdp and health) more to explore here 
reg_pandemic_violations = smf.ols('pandemic_dem_violation_index ~ transparency_index + gdp_percap_ppp_covid + percap_domestic_health_expenditure_ppp + aged_65_older + median_age', df_2020).fit()
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
      <td>0.198705</td>
      <td>0.213220</td>
      <td>0.931926</td>
      <td>0.387338</td>
      <td>-0.323025</td>
      <td>0.720436</td>
    </tr>
    <tr>
      <th>transparency_index</th>
      <td>0.432958</td>
      <td>0.200318</td>
      <td>2.161356</td>
      <td>0.073938</td>
      <td>-0.057202</td>
      <td>0.923117</td>
    </tr>
    <tr>
      <th>gdp_percap_ppp_covid</th>
      <td>-0.000036</td>
      <td>0.000023</td>
      <td>-1.546996</td>
      <td>0.172827</td>
      <td>-0.000093</td>
      <td>0.000021</td>
    </tr>
    <tr>
      <th>percap_domestic_health_expenditure_ppp</th>
      <td>0.000299</td>
      <td>0.000266</td>
      <td>1.123849</td>
      <td>0.304021</td>
      <td>-0.000352</td>
      <td>0.000951</td>
    </tr>
    <tr>
      <th>aged_65_older</th>
      <td>-0.023920</td>
      <td>0.018541</td>
      <td>-1.290112</td>
      <td>0.244499</td>
      <td>-0.069288</td>
      <td>0.021448</td>
    </tr>
    <tr>
      <th>median_age</th>
      <td>0.013618</td>
      <td>0.010216</td>
      <td>1.333057</td>
      <td>0.230895</td>
      <td>-0.011379</td>
      <td>0.038615</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
