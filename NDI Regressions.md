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
nonpanel_df = pd.read_csv(path2)
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
      <th>budget_participation_index_noNA</th>
      <th>budget_transparency_index</th>
      <th>...</th>
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
      <td>1.0</td>
      <td>0.333333</td>
      <td>...</td>
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
      <td>1.0</td>
      <td>0.333333</td>
      <td>...</td>
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
      <td>1.0</td>
      <td>0.333333</td>
      <td>...</td>
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
      <td>1.0</td>
      <td>0.333333</td>
      <td>...</td>
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
      <td>1.0</td>
      <td>0.333333</td>
      <td>...</td>
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
      <td>NaN</td>
      <td>0.000000</td>
      <td>...</td>
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
      <td>NaN</td>
      <td>0.000000</td>
      <td>...</td>
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
      <td>NaN</td>
      <td>0.000000</td>
      <td>...</td>
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
      <td>NaN</td>
      <td>0.000000</td>
      <td>...</td>
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
      <td>NaN</td>
      <td>0.000000</td>
      <td>...</td>
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
<p>3535 rows × 22 columns</p>
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
reg_budgetparticipation = smf.ols('budget_participation_index ~ transparency_index_2019 + gdp_percap_ppp_covid', nonpanel_df).fit()
```


```python
print(reg_budgetparticipation.summary())
```

                                    OLS Regression Results                                
    ======================================================================================
    Dep. Variable:     budget_participation_index   R-squared:                       0.060
    Model:                                    OLS   Adj. R-squared:                  0.050
    Method:                         Least Squares   F-statistic:                     6.035
    Date:                        Mon, 15 Mar 2021   Prob (F-statistic):            0.00288
    Time:                                21:40:23   Log-Likelihood:                -65.897
    No. Observations:                         193   AIC:                             137.8
    Df Residuals:                             190   BIC:                             147.6
    Df Model:                                   2                                         
    Covariance Type:                    nonrobust                                         
    ===========================================================================================
                                  coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------------------
    Intercept                   0.0572      0.052      1.092      0.276      -0.046       0.161
    transparency_index_2019     0.2825      0.085      3.330      0.001       0.115       0.450
    gdp_percap_ppp_covid    -3.902e-07   1.22e-06     -0.321      0.749   -2.79e-06    2.01e-06
    ==============================================================================
    Omnibus:                       34.953   Durbin-Watson:                   1.907
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):               48.826
    Skew:                           1.222   Prob(JB):                     2.50e-11
    Kurtosis:                       3.310   Cond. No.                     1.18e+05
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.18e+05. This might indicate that there are
    strong multicollinearity or other numerical problems.



```python
#Budget Transparency
reg_budget_transparency = smf.ols('budget_transparency_index ~ transparency_index_2019 + gdp_percap_ppp_covid', nonpanel_df).fit()
```


```python
print(reg_budget_transparency.summary())
```

                                    OLS Regression Results                               
    =====================================================================================
    Dep. Variable:     budget_transparency_index   R-squared:                       0.150
    Model:                                   OLS   Adj. R-squared:                  0.141
    Method:                        Least Squares   F-statistic:                     16.70
    Date:                       Mon, 15 Mar 2021   Prob (F-statistic):           2.08e-07
    Time:                               21:40:23   Log-Likelihood:                -51.756
    No. Observations:                        193   AIC:                             109.5
    Df Residuals:                            190   BIC:                             119.3
    Df Model:                                  2                                         
    Covariance Type:                   nonrobust                                         
    ===========================================================================================
                                  coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------------------
    Intercept                  -0.0065      0.049     -0.134      0.894      -0.103       0.090
    transparency_index_2019     0.3805      0.079      4.826      0.000       0.225       0.536
    gdp_percap_ppp_covid     1.302e-06   1.13e-06      1.152      0.251   -9.27e-07    3.53e-06
    ==============================================================================
    Omnibus:                       24.915   Durbin-Watson:                   1.992
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):               15.321
    Skew:                           0.548   Prob(JB):                     0.000471
    Kurtosis:                       2.162   Cond. No.                     1.18e+05
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.18e+05. This might indicate that there are
    strong multicollinearity or other numerical problems.



```python
tables = []
values = [reg_accountability, reg_corruption, reg_trust, reg_effectiveness]
for value in values:
    LRresult = value.summary2().tables[1]
    some_values = ['Intercept', 'transparency_index', 'gdp_percap_ppp', 'gdp_percap_ppp_covid', 'transparency_index_2019']
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



```python
#Call on tables to show all of them or by index: tables[i] 
tables[3]
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
      <td>1.345998e-01</td>
      <td>5.207374e-03</td>
      <td>25.847929</td>
      <td>2.567082e-147</td>
      <td>1.243936e-01</td>
      <td>1.448061e-01</td>
    </tr>
    <tr>
      <th>transparency_index</th>
      <td>1.064131e-01</td>
      <td>2.629020e-02</td>
      <td>4.047633</td>
      <td>5.173817e-05</td>
      <td>5.488525e-02</td>
      <td>1.579410e-01</td>
    </tr>
    <tr>
      <th>gdp_percap_ppp</th>
      <td>-1.431830e-07</td>
      <td>3.177727e-07</td>
      <td>-0.450583</td>
      <td>6.522902e-01</td>
      <td>-7.660061e-07</td>
      <td>4.796401e-07</td>
    </tr>
  </tbody>
</table>
</div>




```python
#COVID Models
nonpanel_df
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
      <th>budget_participation_index</th>
      <th>budget_participation_index_noNA</th>
      <th>budget_transparency_index</th>
      <th>budget_transparency_index_noNA</th>
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
      <td>1.0</td>
      <td>1.0</td>
      <td>0.333333</td>
      <td>0.0</td>
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
      <td>0.5</td>
      <td>0.0</td>
      <td>0.666667</td>
      <td>0.5</td>
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
      <td>0.0</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>NaN</td>
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
      <td>American Samoa</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>NaN</td>
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
    </tr>
    <tr>
      <th>4</th>
      <td>Andorra</td>
      <td>0.926096</td>
      <td>0.941675</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>NaN</td>
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
      <th>211</th>
      <td>Vietnam</td>
      <td>0.115130</td>
      <td>0.108154</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>NaN</td>
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
      <th>212</th>
      <td>Yemen</td>
      <td>0.067556</td>
      <td>0.181291</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>NaN</td>
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
      <th>213</th>
      <td>Zambia</td>
      <td>0.500186</td>
      <td>0.509566</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>NaN</td>
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
      <th>214</th>
      <td>Zanzibar</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
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
      <th>215</th>
      <td>Zimbabwe</td>
      <td>0.394583</td>
      <td>0.171093</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>NaN</td>
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
<p>216 rows × 17 columns</p>
</div>




```python
#Pairplot COVID variables
#pairplot2 = sns.pairplot(data = covid_df, vars=['transparency_index_2019', 'pandemic_dem_violation_index', 'covid_index', 'gdp_percap', 'gdp_percap_ppp_covid', 'percap_domestic_health_expenditure_ppp']) 
```


```python
#COVID
reg_covid = smf.ols('covid_index ~ transparency_index_2019 + gdp_percap_ppp_covid + percap_domestic_health_expenditure_ppp + median_age + aged_65_older', nonpanel_df).fit()
```


```python
print(reg_covid.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:            covid_index   R-squared:                       0.437
    Model:                            OLS   Adj. R-squared:                  0.420
    Method:                 Least Squares   F-statistic:                     25.77
    Date:                Mon, 15 Mar 2021   Prob (F-statistic):           3.39e-19
    Time:                        21:42:07   Log-Likelihood:                0.61291
    No. Observations:                 172   AIC:                             10.77
    Df Residuals:                     166   BIC:                             29.66
    Df Model:                           5                                         
    Covariance Type:            nonrobust                                         
    ==========================================================================================================
                                                 coef    std err          t      P>|t|      [0.025      0.975]
    ----------------------------------------------------------------------------------------------------------
    Intercept                                 -0.0792      0.121     -0.655      0.513      -0.318       0.160
    transparency_index_2019                    0.0688      0.080      0.864      0.389      -0.088       0.226
    gdp_percap_ppp_covid                    6.501e-06      2e-06      3.247      0.001    2.55e-06    1.05e-05
    percap_domestic_health_expenditure_ppp -2.181e-05   3.26e-05     -0.669      0.505   -8.62e-05    4.26e-05
    median_age                                 0.0086      0.006      1.377      0.170      -0.004       0.021
    aged_65_older                              0.0036      0.009      0.393      0.695      -0.014       0.022
    ==============================================================================
    Omnibus:                        6.227   Durbin-Watson:                   2.053
    Prob(Omnibus):                  0.044   Jarque-Bera (JB):                6.210
    Skew:                          -0.465   Prob(JB):                       0.0448
    Kurtosis:                       3.005   Cond. No.                     2.10e+05
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 2.1e+05. This might indicate that there are
    strong multicollinearity or other numerical problems.



```python
#Pandemic Democracy Violations
reg_pandemic_violations = smf.ols('pandemic_dem_violation_index ~ transparency_index_2019 + gdp_percap_ppp_covid + percap_domestic_health_expenditure_ppp + aged_65_older + median_age', nonpanel_df).fit()
```


```python
print(reg_pandemic_violations.summary())
```

                                     OLS Regression Results                                 
    ========================================================================================
    Dep. Variable:     pandemic_dem_violation_index   R-squared:                       0.258
    Model:                                      OLS   Adj. R-squared:                  0.230
    Method:                           Least Squares   F-statistic:                     9.379
    Date:                          Mon, 15 Mar 2021   Prob (F-statistic):           1.09e-07
    Time:                                  21:42:07   Log-Likelihood:                 42.792
    No. Observations:                           141   AIC:                            -73.58
    Df Residuals:                               135   BIC:                            -55.89
    Df Model:                                     5                                         
    Covariance Type:                      nonrobust                                         
    ==========================================================================================================
                                                 coef    std err          t      P>|t|      [0.025      0.975]
    ----------------------------------------------------------------------------------------------------------
    Intercept                                  0.3249      0.100      3.249      0.001       0.127       0.523
    transparency_index_2019                   -0.1383      0.070     -1.989      0.049      -0.276      -0.001
    gdp_percap_ppp_covid                   -8.428e-07   1.71e-06     -0.493      0.623   -4.23e-06    2.54e-06
    percap_domestic_health_expenditure_ppp -3.841e-05   2.61e-05     -1.471      0.144   -9.01e-05    1.32e-05
    aged_65_older                             -0.0117      0.008     -1.527      0.129      -0.027       0.003
    median_age                                 0.0083      0.005      1.596      0.113      -0.002       0.019
    ==============================================================================
    Omnibus:                       23.492   Durbin-Watson:                   1.896
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):               30.999
    Skew:                           0.943   Prob(JB):                     1.86e-07
    Kurtosis:                       4.312   Cond. No.                     2.08e+05
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 2.08e+05. This might indicate that there are
    strong multicollinearity or other numerical problems.

