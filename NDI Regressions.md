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
#Replace NaN with mean of column 
column_means = ndi_df.mean()
ndi_df = ndi_df.fillna(column_means)
```


```python
pairplot = sns.pairplot(data = ndi_df)
```


```python
#pairplot.savefig("Indices, transparency and controls pairplot.png")
```


```python
reg_accountability = smf.ols('accountability_index ~ transparency_index + gdp + gini + C(country_standard)', ndi_df).fit(cov_type='cluster', cov_kwds={'groups': ndi_df['country_standard']})
```


```python
reg_corruption = smf.ols('corruption_index ~ transparency_index + gdp + gini + C(country_standard)', ndi_df).fit(cov_type='cluster', cov_kwds={'groups': ndi_df['country_standard']})
```


```python
reg_trust = smf.ols('trust_index ~ transparency_index + gdp + gini + C(country_standard)', ndi_df).fit(cov_type='cluster', cov_kwds={'groups': ndi_df['country_standard']})
```


```python
reg_effectiveness = smf.ols('effectiveness_index ~ transparency_index + gdp + gini + C(country_standard)', ndi_df).fit(cov_type='cluster', cov_kwds={'groups': ndi_df['country_standard']})
```


```python
reg_bugetparticipation = smf.ols('budget_participation_index ~ transparency_index + gdp + gini + C(country_standard)', ndi_df).fit(cov_type='cluster', cov_kwds={'groups': ndi_df['country_standard']})
```


```python
reg_buget_transparency = smf.ols('budget_transparency_index ~ transparency_index + gdp + gini + C(country_standard)', ndi_df).fit(cov_type='cluster', cov_kwds={'groups': ndi_df['country_standard']})
```


```python
tables = []
values = [reg_accountability, reg_corruption, reg_trust, reg_effectiveness, reg_bugetparticipation, reg_buget_transparency]
for value in values:
    LRresult = value.summary2().tables[1]
    some_values = ['Intercept', 'transparency_index', 'gdp', 'gini']
    LRresult = LRresult.loc[LRresult.index.isin(some_values)]
    tables.append(LRresult)
```

    /Users/katiacordoba/opt/anaconda3/lib/python3.7/site-packages/statsmodels/base/model.py:1832: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 232, but rank is 229
      'rank is %d' % (J, J_), ValueWarning)
    /Users/katiacordoba/opt/anaconda3/lib/python3.7/site-packages/statsmodels/base/model.py:1832: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 232, but rank is 229
      'rank is %d' % (J, J_), ValueWarning)
    /Users/katiacordoba/opt/anaconda3/lib/python3.7/site-packages/statsmodels/base/model.py:1832: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 232, but rank is 229
      'rank is %d' % (J, J_), ValueWarning)
    /Users/katiacordoba/opt/anaconda3/lib/python3.7/site-packages/statsmodels/base/model.py:1832: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 232, but rank is 228
      'rank is %d' % (J, J_), ValueWarning)
    /Users/katiacordoba/opt/anaconda3/lib/python3.7/site-packages/statsmodels/base/model.py:1832: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 232, but rank is 229
      'rank is %d' % (J, J_), ValueWarning)
    /Users/katiacordoba/opt/anaconda3/lib/python3.7/site-packages/statsmodels/base/model.py:1832: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 232, but rank is 229
      'rank is %d' % (J, J_), ValueWarning)



```python
#Call on tables to show all of them or by index: tables[i] 
tables
```




    [                           Coef.      Std.Err.          z         P>|z|  \
     Intercept           5.286789e-01  8.499324e-03  62.202472  0.000000e+00   
     transparency_index  4.436738e-01  3.670441e-02  12.087753  1.225937e-33   
     gdp                -1.383056e-14  1.968341e-15  -7.026506  2.117689e-12   
     gini               -2.360768e-04  1.704729e-05 -13.848348  1.301722e-43   
     
                               [0.025        0.975]  
     Intercept           5.120206e-01  5.453373e-01  
     transparency_index  3.717345e-01  5.156131e-01  
     gdp                -1.768844e-14 -9.972684e-15  
     gini               -2.694889e-04 -2.026648e-04  ,
                                Coef.      Std.Err.           z         P>|z|  \
     Intercept           6.789926e-01  8.418546e-03   80.654386  0.000000e+00   
     transparency_index -3.245027e-01  3.711495e-02   -8.743180  2.266387e-18   
     gdp                -3.446067e-14  1.999557e-15  -17.234153  1.471947e-66   
     gini                7.072667e-03  1.642901e-05  430.498576  0.000000e+00   
     
                               [0.025        0.975]  
     Intercept           6.624926e-01  6.954927e-01  
     transparency_index -3.972467e-01 -2.517587e-01  
     gdp                -3.837973e-14 -3.054161e-14  
     gini                7.040467e-03  7.104868e-03  ,
                                Coef.      Std.Err.           z     P>|z|  \
     Intercept           3.672616e-01  1.394115e-03  263.436977  0.000000   
     transparency_index  3.206261e-03  6.173190e-03    0.519385  0.603492   
     gdp                 6.564655e-16  3.327905e-16    1.972609  0.048540   
     gini                1.415570e-03  2.748348e-06  515.062140  0.000000   
     
                               [0.025        0.975]  
     Intercept           3.645291e-01  3.699940e-01  
     transparency_index -8.892968e-03  1.530549e-02  
     gdp                 4.208181e-18  1.308723e-15  
     gini                1.410184e-03  1.420957e-03  ,
                                Coef.      Std.Err.           z          P>|z|  \
     Intercept           1.800468e-01  7.593386e-03   23.711003  2.776585e-124   
     transparency_index  3.346380e-01  3.367567e-02    9.937085   2.871067e-23   
     gdp                 4.155137e-14  1.815051e-15   22.892667  5.497383e-116   
     gini               -1.629718e-03  1.503014e-05 -108.429966   0.000000e+00   
     
                               [0.025        0.975]  
     Intercept           1.651640e-01  1.949296e-01  
     transparency_index  2.686349e-01  4.006411e-01  
     gdp                 3.799393e-14  4.510880e-14  
     gini               -1.659176e-03 -1.600259e-03  ,
                                Coef.      Std.Err.          z          P>|z|  \
     Intercept           6.509903e-02  1.582245e-03  41.143458   0.000000e+00   
     transparency_index  1.100233e-02  7.122230e-03   1.544787   1.223979e-01   
     gdp                -6.319537e-15  4.032768e-16 -15.670469   2.407945e-55   
     gini               -1.243817e-04  4.719937e-06 -26.352405  4.816760e-153   
     
                               [0.025        0.975]  
     Intercept           6.199789e-02  6.820017e-02  
     transparency_index -2.956988e-03  2.496164e-02  
     gdp                -7.109945e-15 -5.529129e-15  
     gini               -1.336326e-04 -1.151308e-04  ,
                                Coef.      Std.Err.         z         P>|z|  \
     Intercept           1.729465e-02  1.836162e-03  9.418914  4.557777e-21   
     transparency_index  2.189404e-02  8.014782e-03  2.731707  6.300714e-03   
     gdp                -3.761452e-16  4.395792e-16 -0.855694  3.921671e-01   
     gini               -3.654306e-05  4.477711e-06 -8.161103  3.319791e-16   
     
                               [0.025        0.975]  
     Intercept           1.369584e-02  2.089347e-02  
     transparency_index  6.185352e-03  3.760272e-02  
     gdp                -1.237705e-15  4.854142e-16  
     gini               -4.531921e-05 -2.776691e-05  ]




```python
#Select only year 2020 for next two models 
df_2020 = ndi_df.loc[ndi_df['year'] == 2020]
```


```python
#COVID outcomes model
reg_covid = smf.ols('covid_index ~ transparency_index + gdp + gini + percap_domestic_health_expenditure + median_age + aged_65_older', df_2020).fit()
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
      <td>-1.128887e-01</td>
      <td>1.204474e-01</td>
      <td>-0.937245</td>
      <td>0.349730</td>
      <td>-3.503563e-01</td>
      <td>1.245789e-01</td>
    </tr>
    <tr>
      <th>transparency_index</th>
      <td>-6.459594e-02</td>
      <td>6.892106e-02</td>
      <td>-0.937245</td>
      <td>0.349730</td>
      <td>-2.004770e-01</td>
      <td>7.128513e-02</td>
    </tr>
    <tr>
      <th>gdp</th>
      <td>5.838804e-15</td>
      <td>9.643596e-15</td>
      <td>0.605459</td>
      <td>0.545540</td>
      <td>-1.317399e-14</td>
      <td>2.485160e-14</td>
    </tr>
    <tr>
      <th>gini</th>
      <td>-1.993693e-03</td>
      <td>2.852820e-03</td>
      <td>-0.698850</td>
      <td>0.485434</td>
      <td>-7.618160e-03</td>
      <td>3.630774e-03</td>
    </tr>
    <tr>
      <th>percap_domestic_health_expenditure</th>
      <td>4.532022e-05</td>
      <td>1.837211e-05</td>
      <td>2.466795</td>
      <td>0.014449</td>
      <td>9.098750e-06</td>
      <td>8.154170e-05</td>
    </tr>
    <tr>
      <th>median_age</th>
      <td>1.993618e-02</td>
      <td>4.812323e-03</td>
      <td>4.142735</td>
      <td>0.000050</td>
      <td>1.044846e-02</td>
      <td>2.942390e-02</td>
    </tr>
    <tr>
      <th>aged_65_older</th>
      <td>-5.910733e-03</td>
      <td>7.610942e-03</td>
      <td>-0.776610</td>
      <td>0.438280</td>
      <td>-2.091606e-02</td>
      <td>9.094594e-03</td>
    </tr>
  </tbody>
</table>
</div>




```python
reg_pandemic_violations = smf.ols('pandemic_dem_violation_index ~ transparency_index + gdp + gini + percap_domestic_health_expenditure + median_age + aged_65_older', df_2020).fit()
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
      <td>3.415484e-01</td>
      <td>7.724160e-02</td>
      <td>4.421820</td>
      <td>0.000016</td>
      <td>1.892630e-01</td>
      <td>4.938339e-01</td>
    </tr>
    <tr>
      <th>transparency_index</th>
      <td>1.954371e-01</td>
      <td>4.419834e-02</td>
      <td>4.421820</td>
      <td>0.000016</td>
      <td>1.082980e-01</td>
      <td>2.825761e-01</td>
    </tr>
    <tr>
      <th>gdp</th>
      <td>2.006482e-14</td>
      <td>6.184335e-15</td>
      <td>3.244459</td>
      <td>0.001373</td>
      <td>7.872116e-15</td>
      <td>3.225753e-14</td>
    </tr>
    <tr>
      <th>gini</th>
      <td>-1.958582e-03</td>
      <td>1.829483e-03</td>
      <td>-1.070566</td>
      <td>0.285617</td>
      <td>-5.565493e-03</td>
      <td>1.648328e-03</td>
    </tr>
    <tr>
      <th>percap_domestic_health_expenditure</th>
      <td>-5.795900e-05</td>
      <td>1.178184e-05</td>
      <td>-4.919351</td>
      <td>0.000002</td>
      <td>-8.118744e-05</td>
      <td>-3.473055e-05</td>
    </tr>
    <tr>
      <th>median_age</th>
      <td>2.672412e-03</td>
      <td>3.086091e-03</td>
      <td>0.865954</td>
      <td>0.387523</td>
      <td>-3.411961e-03</td>
      <td>8.756785e-03</td>
    </tr>
    <tr>
      <th>aged_65_older</th>
      <td>-8.849708e-03</td>
      <td>4.880816e-03</td>
      <td>-1.813162</td>
      <td>0.071262</td>
      <td>-1.847246e-02</td>
      <td>7.730480e-04</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
