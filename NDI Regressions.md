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
path2 = r'/Users/katiacordoba/Documents/GitHub/policy_lab_ndi/data/indices_and_controls_cross_section.csv'
ndi_df = pd.read_csv(path)
covid_df = pd.read_csv(path2)
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
      <td>0.576651</td>
      <td>1.345239e-02</td>
      <td>42.866080</td>
      <td>0.000000</td>
      <td>0.550285</td>
      <td>6.030175e-01</td>
    </tr>
    <tr>
      <th>transparency_index</th>
      <td>0.257076</td>
      <td>6.841921e-02</td>
      <td>3.757367</td>
      <td>0.000172</td>
      <td>0.122977</td>
      <td>3.911752e-01</td>
    </tr>
    <tr>
      <th>gdp_percap_ppp</th>
      <td>-0.000001</td>
      <td>4.914317e-07</td>
      <td>-2.287029</td>
      <td>0.022194</td>
      <td>-0.000002</td>
      <td>-1.607299e-07</td>
    </tr>
  </tbody>
</table>
</div>




```python
#COVID Models
covid_df
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
      <th>transparency_index_2019</th>
      <th>transparency_index_mean</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>0.163570</td>
      <td>0.196071</td>
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
    </tr>
    <tr>
      <th>1</th>
      <td>Albania</td>
      <td>0.554243</td>
      <td>0.551349</td>
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
    </tr>
    <tr>
      <th>2</th>
      <td>Algeria</td>
      <td>0.157443</td>
      <td>0.169187</td>
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
    </tr>
    <tr>
      <th>3</th>
      <td>Andorra</td>
      <td>0.926096</td>
      <td>0.941675</td>
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
    </tr>
    <tr>
      <th>4</th>
      <td>Angola</td>
      <td>0.189579</td>
      <td>0.149982</td>
      <td>0.357143</td>
      <td>0.021043</td>
      <td>8.881570e+10</td>
      <td>51.3</td>
      <td>2790.726615</td>
      <td>6965.511374</td>
      <td>36.737221</td>
      <td>69.060318</td>
      <td>16.8</td>
      <td>2.405</td>
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
    </tr>
    <tr>
      <th>181</th>
      <td>Venezuela</td>
      <td>0.111563</td>
      <td>0.358687</td>
      <td>0.928571</td>
      <td>0.011356</td>
      <td>4.823593e+11</td>
      <td>46.9</td>
      <td>16054.490513</td>
      <td>17527.447795</td>
      <td>122.942413</td>
      <td>183.498871</td>
      <td>29.0</td>
      <td>6.614</td>
    </tr>
    <tr>
      <th>182</th>
      <td>Vietnam</td>
      <td>0.115130</td>
      <td>0.108154</td>
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
    </tr>
    <tr>
      <th>183</th>
      <td>Yemen</td>
      <td>0.067556</td>
      <td>0.181291</td>
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
    </tr>
    <tr>
      <th>184</th>
      <td>Zambia</td>
      <td>0.500186</td>
      <td>0.509566</td>
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
    </tr>
    <tr>
      <th>185</th>
      <td>Zimbabwe</td>
      <td>0.394583</td>
      <td>0.171093</td>
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
    </tr>
  </tbody>
</table>
<p>186 rows × 13 columns</p>
</div>




```python
#pp = sns.pairplot(data = covid_df, vars=['transparency_index_2019', 'pandemic_dem_violation_index', 'covid_index', 'gdp_percap', 'gdp_percap_ppp_covid', 'percap_domestic_health_expenditure_ppp']) 
```


![png](output_14_0.png)



```python
#COVID
reg_covid = smf.ols('covid_index ~ transparency_index_2019 + gdp_percap_ppp_covid + percap_domestic_health_expenditure_ppp + median_age + aged_65_older', covid_df).fit()
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
      <td>-0.079224</td>
      <td>0.120948</td>
      <td>-0.655027</td>
      <td>0.513357</td>
      <td>-0.318018</td>
      <td>0.159570</td>
    </tr>
    <tr>
      <th>transparency_index_2019</th>
      <td>0.068764</td>
      <td>0.079589</td>
      <td>0.863986</td>
      <td>0.388842</td>
      <td>-0.088373</td>
      <td>0.225900</td>
    </tr>
    <tr>
      <th>gdp_percap_ppp_covid</th>
      <td>0.000007</td>
      <td>0.000002</td>
      <td>3.246821</td>
      <td>0.001412</td>
      <td>0.000003</td>
      <td>0.000010</td>
    </tr>
    <tr>
      <th>percap_domestic_health_expenditure_ppp</th>
      <td>-0.000022</td>
      <td>0.000033</td>
      <td>-0.668516</td>
      <td>0.504733</td>
      <td>-0.000086</td>
      <td>0.000043</td>
    </tr>
    <tr>
      <th>median_age</th>
      <td>0.008591</td>
      <td>0.006241</td>
      <td>1.376600</td>
      <td>0.170490</td>
      <td>-0.003730</td>
      <td>0.020912</td>
    </tr>
    <tr>
      <th>aged_65_older</th>
      <td>0.003581</td>
      <td>0.009110</td>
      <td>0.393071</td>
      <td>0.694772</td>
      <td>-0.014405</td>
      <td>0.021567</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Pandemic Democracy Violations
reg_pandemic_violations = smf.ols('pandemic_dem_violation_index ~ transparency_index_2019 + gdp_percap_ppp_covid + percap_domestic_health_expenditure_ppp + aged_65_older + median_age', covid_df).fit()
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
      <td>3.248865e-01</td>
      <td>0.100008</td>
      <td>3.248609</td>
      <td>0.001464</td>
      <td>0.127102</td>
      <td>0.522671</td>
    </tr>
    <tr>
      <th>transparency_index_2019</th>
      <td>-1.383298e-01</td>
      <td>0.069534</td>
      <td>-1.989387</td>
      <td>0.048679</td>
      <td>-0.275846</td>
      <td>-0.000813</td>
    </tr>
    <tr>
      <th>gdp_percap_ppp_covid</th>
      <td>-8.427588e-07</td>
      <td>0.000002</td>
      <td>-0.492633</td>
      <td>0.623072</td>
      <td>-0.000004</td>
      <td>0.000003</td>
    </tr>
    <tr>
      <th>percap_domestic_health_expenditure_ppp</th>
      <td>-3.840644e-05</td>
      <td>0.000026</td>
      <td>-1.470562</td>
      <td>0.143737</td>
      <td>-0.000090</td>
      <td>0.000013</td>
    </tr>
    <tr>
      <th>aged_65_older</th>
      <td>-1.174270e-02</td>
      <td>0.007691</td>
      <td>-1.526886</td>
      <td>0.129129</td>
      <td>-0.026952</td>
      <td>0.003467</td>
    </tr>
    <tr>
      <th>median_age</th>
      <td>8.325270e-03</td>
      <td>0.005217</td>
      <td>1.595861</td>
      <td>0.112858</td>
      <td>-0.001992</td>
      <td>0.018642</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
