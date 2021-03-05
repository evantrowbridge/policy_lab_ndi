```python
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
!pip install linearmodels
from linearmodels import PanelOLS
import statsmodels.formula.api as smf
from statsmodels.base.covtype import get_robustcov_results
```

    Requirement already satisfied: linearmodels in ./opt/anaconda3/lib/python3.7/site-packages (4.21)
    Requirement already satisfied: scipy>=1.2 in ./opt/anaconda3/lib/python3.7/site-packages (from linearmodels) (1.4.1)
    Requirement already satisfied: numpy>=1.16 in ./opt/anaconda3/lib/python3.7/site-packages (from linearmodels) (1.18.1)
    Requirement already satisfied: patsy in ./opt/anaconda3/lib/python3.7/site-packages (from linearmodels) (0.5.1)
    Requirement already satisfied: property-cached>=1.6.3 in ./opt/anaconda3/lib/python3.7/site-packages (from linearmodels) (1.6.4)
    Requirement already satisfied: pandas>=0.24 in ./opt/anaconda3/lib/python3.7/site-packages (from linearmodels) (0.25.3)
    Requirement already satisfied: mypy-extensions>=0.4 in ./opt/anaconda3/lib/python3.7/site-packages (from linearmodels) (0.4.3)
    Requirement already satisfied: statsmodels>=0.11 in ./opt/anaconda3/lib/python3.7/site-packages (from linearmodels) (0.11.0)
    Requirement already satisfied: Cython>=0.29.21 in ./opt/anaconda3/lib/python3.7/site-packages (from linearmodels) (0.29.22)
    Requirement already satisfied: pyhdfe>=0.1 in ./opt/anaconda3/lib/python3.7/site-packages (from linearmodels) (0.1.0)
    Requirement already satisfied: six in ./opt/anaconda3/lib/python3.7/site-packages (from patsy->linearmodels) (1.14.0)
    Requirement already satisfied: pytz>=2017.2 in ./opt/anaconda3/lib/python3.7/site-packages (from pandas>=0.24->linearmodels) (2019.3)
    Requirement already satisfied: python-dateutil>=2.6.1 in ./opt/anaconda3/lib/python3.7/site-packages (from pandas>=0.24->linearmodels) (2.8.1)



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
      <th>trust_index</th>
      <th>corruption_index</th>
      <th>effectiveness_index</th>
      <th>budget_participation_index</th>
      <th>pandemic_dem_violation_index</th>
      <th>covid_index</th>
      <th>gdp</th>
      <th>gdp_per_capita</th>
      <th>gini_2020</th>
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
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.090550</td>
      <td>0.0</td>
      <td>0.214286</td>
      <td>0.012655</td>
      <td>1.929110e+10</td>
      <td>1803.987</td>
      <td>0.655</td>
      <td>2.578007</td>
      <td>18.6</td>
      <td>2.581</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>2007</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.114156</td>
      <td>0.0</td>
      <td>0.214286</td>
      <td>0.012655</td>
      <td>1.929110e+10</td>
      <td>1803.987</td>
      <td>0.655</td>
      <td>2.578007</td>
      <td>18.6</td>
      <td>2.581</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Afghanistan</td>
      <td>2008</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.124995</td>
      <td>0.0</td>
      <td>0.214286</td>
      <td>0.012655</td>
      <td>1.929110e+10</td>
      <td>1803.987</td>
      <td>0.655</td>
      <td>2.578007</td>
      <td>18.6</td>
      <td>2.581</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Afghanistan</td>
      <td>2009</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.126294</td>
      <td>0.0</td>
      <td>0.214286</td>
      <td>0.012655</td>
      <td>1.929110e+10</td>
      <td>1803.987</td>
      <td>0.655</td>
      <td>2.578007</td>
      <td>18.6</td>
      <td>2.581</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Afghanistan</td>
      <td>2010</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.129410</td>
      <td>0.0</td>
      <td>0.214286</td>
      <td>0.012655</td>
      <td>1.929110e+10</td>
      <td>1803.987</td>
      <td>0.655</td>
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
    </tr>
    <tr>
      <th>3625</th>
      <td>NaN</td>
      <td>2016</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>661.562211</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3626</th>
      <td>NaN</td>
      <td>2017</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>661.562211</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3627</th>
      <td>NaN</td>
      <td>2018</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>661.562211</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3628</th>
      <td>NaN</td>
      <td>2019</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>661.562211</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3629</th>
      <td>NaN</td>
      <td>2020</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>661.562211</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>3630 rows × 15 columns</p>
</div>




```python
#Replace NaN with mean of column 
column_means = ndi_df.mean()
ndi_df = ndi_df.fillna(column_means)
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
      <th>trust_index</th>
      <th>corruption_index</th>
      <th>effectiveness_index</th>
      <th>budget_participation_index</th>
      <th>pandemic_dem_violation_index</th>
      <th>covid_index</th>
      <th>gdp</th>
      <th>gdp_per_capita</th>
      <th>gini_2020</th>
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
      <td>0.612422</td>
      <td>0.532376</td>
      <td>0.924998</td>
      <td>0.090550</td>
      <td>0.0</td>
      <td>0.214286</td>
      <td>0.012655</td>
      <td>1.929110e+10</td>
      <td>1803.987000</td>
      <td>0.655000</td>
      <td>2.578007</td>
      <td>18.600000</td>
      <td>2.581000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>2007</td>
      <td>0.612422</td>
      <td>0.532376</td>
      <td>0.924998</td>
      <td>0.114156</td>
      <td>0.0</td>
      <td>0.214286</td>
      <td>0.012655</td>
      <td>1.929110e+10</td>
      <td>1803.987000</td>
      <td>0.655000</td>
      <td>2.578007</td>
      <td>18.600000</td>
      <td>2.581000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Afghanistan</td>
      <td>2008</td>
      <td>0.612422</td>
      <td>0.532376</td>
      <td>0.924998</td>
      <td>0.124995</td>
      <td>0.0</td>
      <td>0.214286</td>
      <td>0.012655</td>
      <td>1.929110e+10</td>
      <td>1803.987000</td>
      <td>0.655000</td>
      <td>2.578007</td>
      <td>18.600000</td>
      <td>2.581000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Afghanistan</td>
      <td>2009</td>
      <td>0.612422</td>
      <td>0.532376</td>
      <td>0.924998</td>
      <td>0.126294</td>
      <td>0.0</td>
      <td>0.214286</td>
      <td>0.012655</td>
      <td>1.929110e+10</td>
      <td>1803.987000</td>
      <td>0.655000</td>
      <td>2.578007</td>
      <td>18.600000</td>
      <td>2.581000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Afghanistan</td>
      <td>2010</td>
      <td>0.612422</td>
      <td>0.532376</td>
      <td>0.924998</td>
      <td>0.129410</td>
      <td>0.0</td>
      <td>0.214286</td>
      <td>0.012655</td>
      <td>1.929110e+10</td>
      <td>1803.987000</td>
      <td>0.655000</td>
      <td>2.578007</td>
      <td>18.600000</td>
      <td>2.581000</td>
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
    </tr>
    <tr>
      <th>3625</th>
      <td>NaN</td>
      <td>2016</td>
      <td>0.612422</td>
      <td>0.532376</td>
      <td>0.924998</td>
      <td>0.503408</td>
      <td>0.0</td>
      <td>0.336622</td>
      <td>0.354439</td>
      <td>4.822151e+11</td>
      <td>18862.965835</td>
      <td>0.728947</td>
      <td>661.562211</td>
      <td>30.359825</td>
      <td>8.649119</td>
    </tr>
    <tr>
      <th>3626</th>
      <td>NaN</td>
      <td>2017</td>
      <td>0.612422</td>
      <td>0.532376</td>
      <td>0.924998</td>
      <td>0.503408</td>
      <td>0.0</td>
      <td>0.336622</td>
      <td>0.354439</td>
      <td>4.822151e+11</td>
      <td>18862.965835</td>
      <td>0.728947</td>
      <td>661.562211</td>
      <td>30.359825</td>
      <td>8.649119</td>
    </tr>
    <tr>
      <th>3627</th>
      <td>NaN</td>
      <td>2018</td>
      <td>0.612422</td>
      <td>0.532376</td>
      <td>0.924998</td>
      <td>0.503408</td>
      <td>0.0</td>
      <td>0.336622</td>
      <td>0.354439</td>
      <td>4.822151e+11</td>
      <td>18862.965835</td>
      <td>0.728947</td>
      <td>661.562211</td>
      <td>30.359825</td>
      <td>8.649119</td>
    </tr>
    <tr>
      <th>3628</th>
      <td>NaN</td>
      <td>2019</td>
      <td>0.612422</td>
      <td>0.532376</td>
      <td>0.924998</td>
      <td>0.503408</td>
      <td>0.0</td>
      <td>0.336622</td>
      <td>0.354439</td>
      <td>4.822151e+11</td>
      <td>18862.965835</td>
      <td>0.728947</td>
      <td>661.562211</td>
      <td>30.359825</td>
      <td>8.649119</td>
    </tr>
    <tr>
      <th>3629</th>
      <td>NaN</td>
      <td>2020</td>
      <td>0.612422</td>
      <td>0.532376</td>
      <td>0.924998</td>
      <td>0.503408</td>
      <td>0.0</td>
      <td>0.336622</td>
      <td>0.354439</td>
      <td>4.822151e+11</td>
      <td>18862.965835</td>
      <td>0.728947</td>
      <td>661.562211</td>
      <td>30.359825</td>
      <td>8.649119</td>
    </tr>
  </tbody>
</table>
<p>3630 rows × 15 columns</p>
</div>




```python
#Drop last 15 rows with NA values in country_standard
ndi_df = ndi_df.dropna(subset=['country_standard'])
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
      <th>trust_index</th>
      <th>corruption_index</th>
      <th>effectiveness_index</th>
      <th>budget_participation_index</th>
      <th>pandemic_dem_violation_index</th>
      <th>covid_index</th>
      <th>gdp</th>
      <th>gdp_per_capita</th>
      <th>gini_2020</th>
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
      <td>0.612422</td>
      <td>0.532376</td>
      <td>0.924998</td>
      <td>0.090550</td>
      <td>0.0</td>
      <td>0.214286</td>
      <td>0.012655</td>
      <td>1.929110e+10</td>
      <td>1803.987</td>
      <td>0.655</td>
      <td>2.578007</td>
      <td>18.6</td>
      <td>2.581</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>2007</td>
      <td>0.612422</td>
      <td>0.532376</td>
      <td>0.924998</td>
      <td>0.114156</td>
      <td>0.0</td>
      <td>0.214286</td>
      <td>0.012655</td>
      <td>1.929110e+10</td>
      <td>1803.987</td>
      <td>0.655</td>
      <td>2.578007</td>
      <td>18.6</td>
      <td>2.581</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Afghanistan</td>
      <td>2008</td>
      <td>0.612422</td>
      <td>0.532376</td>
      <td>0.924998</td>
      <td>0.124995</td>
      <td>0.0</td>
      <td>0.214286</td>
      <td>0.012655</td>
      <td>1.929110e+10</td>
      <td>1803.987</td>
      <td>0.655</td>
      <td>2.578007</td>
      <td>18.6</td>
      <td>2.581</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Afghanistan</td>
      <td>2009</td>
      <td>0.612422</td>
      <td>0.532376</td>
      <td>0.924998</td>
      <td>0.126294</td>
      <td>0.0</td>
      <td>0.214286</td>
      <td>0.012655</td>
      <td>1.929110e+10</td>
      <td>1803.987</td>
      <td>0.655</td>
      <td>2.578007</td>
      <td>18.6</td>
      <td>2.581</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Afghanistan</td>
      <td>2010</td>
      <td>0.612422</td>
      <td>0.532376</td>
      <td>0.924998</td>
      <td>0.129410</td>
      <td>0.0</td>
      <td>0.214286</td>
      <td>0.012655</td>
      <td>1.929110e+10</td>
      <td>1803.987</td>
      <td>0.655</td>
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
    </tr>
    <tr>
      <th>3610</th>
      <td>Zimbabwe</td>
      <td>2017</td>
      <td>0.377128</td>
      <td>0.316887</td>
      <td>0.465453</td>
      <td>0.221270</td>
      <td>0.0</td>
      <td>0.500000</td>
      <td>0.271321</td>
      <td>2.144076e+10</td>
      <td>1899.775</td>
      <td>0.719</td>
      <td>39.249222</td>
      <td>19.6</td>
      <td>2.822</td>
    </tr>
    <tr>
      <th>3611</th>
      <td>Zimbabwe</td>
      <td>2018</td>
      <td>0.479432</td>
      <td>0.316887</td>
      <td>0.473051</td>
      <td>0.220189</td>
      <td>0.0</td>
      <td>0.500000</td>
      <td>0.271321</td>
      <td>2.144076e+10</td>
      <td>1899.775</td>
      <td>0.719</td>
      <td>39.249222</td>
      <td>19.6</td>
      <td>2.822</td>
    </tr>
    <tr>
      <th>3612</th>
      <td>Zimbabwe</td>
      <td>2019</td>
      <td>0.461171</td>
      <td>0.316887</td>
      <td>0.492241</td>
      <td>0.245736</td>
      <td>0.0</td>
      <td>0.500000</td>
      <td>0.271321</td>
      <td>2.144076e+10</td>
      <td>1899.775</td>
      <td>0.719</td>
      <td>39.249222</td>
      <td>19.6</td>
      <td>2.822</td>
    </tr>
    <tr>
      <th>3613</th>
      <td>Zimbabwe</td>
      <td>2020</td>
      <td>0.461171</td>
      <td>0.272614</td>
      <td>0.492241</td>
      <td>0.245736</td>
      <td>0.0</td>
      <td>0.500000</td>
      <td>0.271321</td>
      <td>2.144076e+10</td>
      <td>1899.775</td>
      <td>0.719</td>
      <td>39.249222</td>
      <td>19.6</td>
      <td>2.822</td>
    </tr>
    <tr>
      <th>3614</th>
      <td>Zimbabwe</td>
      <td>2021</td>
      <td>0.461171</td>
      <td>0.272614</td>
      <td>0.492241</td>
      <td>0.503408</td>
      <td>0.0</td>
      <td>0.500000</td>
      <td>0.271321</td>
      <td>2.144076e+10</td>
      <td>1899.775</td>
      <td>0.719</td>
      <td>39.249222</td>
      <td>19.6</td>
      <td>2.822</td>
    </tr>
  </tbody>
</table>
<p>3615 rows × 15 columns</p>
</div>




```python
pairplot = sns.pairplot(data = ndi_df)
```


![png](output_7_0.png)



```python
#pairplot.savefig("Indices, transparency and controls pairplot.png")
```


```python
reg_corruption = smf.ols('corruption_index ~ transparency_index + gdp + gdp_per_capita + gini_2020 + C(country_standard)', ndi_df).fit(cov_type='cluster', cov_kwds={'groups': ndi_df['country_standard']})
```


```python
reg_corruption.summary()
```

    /Users/katiacordoba/opt/anaconda3/lib/python3.7/site-packages/statsmodels/base/model.py:1832: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 232, but rank is 228
      'rank is %d' % (J, J_), ValueWarning)





<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>    <td>corruption_index</td> <th>  R-squared:         </th> <td>   0.741</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.741</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>2.269e+08</td>
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 05 Mar 2021</td> <th>  Prob (F-statistic):</th>  <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>13:17:42</td>     <th>  Log-Likelihood:    </th> <td>  1859.7</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>  3615</td>      <th>  AIC:               </th> <td>  -3709.</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  3610</td>      <th>  BIC:               </th> <td>  -3679.</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     4</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>       <td>cluster</td>     <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
                             <td></td>                                <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>                                               <td>   -0.2935</td> <td>    0.026</td> <td>  -11.477</td> <td> 0.000</td> <td>   -0.344</td> <td>   -0.243</td>
</tr>
<tr>
  <th>C(country_standard)[T.Albania]</th>                          <td>    0.0221</td> <td>    0.011</td> <td>    1.937</td> <td> 0.053</td> <td>   -0.000</td> <td>    0.044</td>
</tr>
<tr>
  <th>C(country_standard)[T.Algeria]</th>                          <td>    0.1439</td> <td>    0.013</td> <td>   10.746</td> <td> 0.000</td> <td>    0.118</td> <td>    0.170</td>
</tr>
<tr>
  <th>C(country_standard)[T.American Samoa]</th>                   <td>    0.0517</td> <td>    0.003</td> <td>   16.881</td> <td> 0.000</td> <td>    0.046</td> <td>    0.058</td>
</tr>
<tr>
  <th>C(country_standard)[T.Andorra]</th>                          <td>    0.0560</td> <td>    0.003</td> <td>   20.024</td> <td> 0.000</td> <td>    0.050</td> <td>    0.061</td>
</tr>
<tr>
  <th>C(country_standard)[T.Angola]</th>                           <td>    0.0512</td> <td>    0.009</td> <td>    5.471</td> <td> 0.000</td> <td>    0.033</td> <td>    0.070</td>
</tr>
<tr>
  <th>C(country_standard)[T.Anguilla]</th>                         <td>    0.0496</td> <td>    0.003</td> <td>   19.143</td> <td> 0.000</td> <td>    0.044</td> <td>    0.055</td>
</tr>
<tr>
  <th>C(country_standard)[T.Antigua & Barbuda]</th>                <td>   -0.0159</td> <td>    0.001</td> <td>  -11.789</td> <td> 0.000</td> <td>   -0.019</td> <td>   -0.013</td>
</tr>
<tr>
  <th>C(country_standard)[T.Argentina]</th>                        <td>   -0.1446</td> <td>    0.014</td> <td>  -10.680</td> <td> 0.000</td> <td>   -0.171</td> <td>   -0.118</td>
</tr>
<tr>
  <th>C(country_standard)[T.Armenia]</th>                          <td>    0.1158</td> <td>    0.006</td> <td>   18.502</td> <td> 0.000</td> <td>    0.104</td> <td>    0.128</td>
</tr>
<tr>
  <th>C(country_standard)[T.Aruba]</th>                            <td>    0.0516</td> <td>    0.003</td> <td>   17.426</td> <td> 0.000</td> <td>    0.046</td> <td>    0.057</td>
</tr>
<tr>
  <th>C(country_standard)[T.Australia]</th>                        <td>    0.1367</td> <td>    0.016</td> <td>    8.631</td> <td> 0.000</td> <td>    0.106</td> <td>    0.168</td>
</tr>
<tr>
  <th>C(country_standard)[T.Austria]</th>                          <td>    0.0076</td> <td>    0.011</td> <td>    0.667</td> <td> 0.505</td> <td>   -0.015</td> <td>    0.030</td>
</tr>
<tr>
  <th>C(country_standard)[T.Azerbaijan]</th>                       <td>    0.1805</td> <td>    0.018</td> <td>    9.762</td> <td> 0.000</td> <td>    0.144</td> <td>    0.217</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bahamas]</th>                          <td>    0.1996</td> <td>    0.004</td> <td>   54.279</td> <td> 0.000</td> <td>    0.192</td> <td>    0.207</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bahrain]</th>                          <td>    0.0949</td> <td>    0.030</td> <td>    3.212</td> <td> 0.001</td> <td>    0.037</td> <td>    0.153</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bangladesh]</th>                       <td>    0.1118</td> <td>    0.002</td> <td>   63.980</td> <td> 0.000</td> <td>    0.108</td> <td>    0.115</td>
</tr>
<tr>
  <th>C(country_standard)[T.Barbados]</th>                         <td>    0.1768</td> <td>    0.017</td> <td>   10.127</td> <td> 0.000</td> <td>    0.143</td> <td>    0.211</td>
</tr>
<tr>
  <th>C(country_standard)[T.Belarus]</th>                          <td>    0.3070</td> <td>    0.016</td> <td>   19.573</td> <td> 0.000</td> <td>    0.276</td> <td>    0.338</td>
</tr>
<tr>
  <th>C(country_standard)[T.Belgium]</th>                          <td>    0.0986</td> <td>    0.018</td> <td>    5.379</td> <td> 0.000</td> <td>    0.063</td> <td>    0.135</td>
</tr>
<tr>
  <th>C(country_standard)[T.Belize]</th>                           <td>    0.1148</td> <td>    0.004</td> <td>   27.615</td> <td> 0.000</td> <td>    0.107</td> <td>    0.123</td>
</tr>
<tr>
  <th>C(country_standard)[T.Benin]</th>                            <td>    0.0333</td> <td>    0.018</td> <td>    1.838</td> <td> 0.066</td> <td>   -0.002</td> <td>    0.069</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bermuda]</th>                          <td>    0.0495</td> <td>    0.003</td> <td>   19.001</td> <td> 0.000</td> <td>    0.044</td> <td>    0.055</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bhutan]</th>                           <td>    0.4289</td> <td>    0.009</td> <td>   46.802</td> <td> 0.000</td> <td>    0.411</td> <td>    0.447</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bolivia]</th>                          <td>    0.0255</td> <td>    0.004</td> <td>    6.241</td> <td> 0.000</td> <td>    0.018</td> <td>    0.034</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bosnia & Herzegovina]</th>             <td>    0.0816</td> <td>    0.008</td> <td>   10.157</td> <td> 0.000</td> <td>    0.066</td> <td>    0.097</td>
</tr>
<tr>
  <th>C(country_standard)[T.Botswana]</th>                         <td>    0.1430</td> <td>    0.014</td> <td>   10.588</td> <td> 0.000</td> <td>    0.117</td> <td>    0.169</td>
</tr>
<tr>
  <th>C(country_standard)[T.Brazil]</th>                           <td>   -0.1889</td> <td>    0.012</td> <td>  -15.154</td> <td> 0.000</td> <td>   -0.213</td> <td>   -0.164</td>
</tr>
<tr>
  <th>C(country_standard)[T.British Virgin Islands]</th>           <td>    0.0516</td> <td>    0.003</td> <td>   17.298</td> <td> 0.000</td> <td>    0.046</td> <td>    0.057</td>
</tr>
<tr>
  <th>C(country_standard)[T.Brunei]</th>                           <td>   -0.2835</td> <td>    0.018</td> <td>  -16.083</td> <td> 0.000</td> <td>   -0.318</td> <td>   -0.249</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bulgaria]</th>                         <td>   -0.0618</td> <td>    0.017</td> <td>   -3.543</td> <td> 0.000</td> <td>   -0.096</td> <td>   -0.028</td>
</tr>
<tr>
  <th>C(country_standard)[T.Burkina Faso]</th>                     <td>    0.1673</td> <td>    0.011</td> <td>   15.149</td> <td> 0.000</td> <td>    0.146</td> <td>    0.189</td>
</tr>
<tr>
  <th>C(country_standard)[T.Burundi]</th>                          <td>    0.1538</td> <td>    0.011</td> <td>   13.903</td> <td> 0.000</td> <td>    0.132</td> <td>    0.175</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cambodia]</th>                         <td>    0.1063</td> <td>    0.011</td> <td>    9.911</td> <td> 0.000</td> <td>    0.085</td> <td>    0.127</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cameroon]</th>                         <td>    0.0896</td> <td>    0.008</td> <td>   11.920</td> <td> 0.000</td> <td>    0.075</td> <td>    0.104</td>
</tr>
<tr>
  <th>C(country_standard)[T.Canada]</th>                           <td>    0.0900</td> <td>    0.012</td> <td>    7.823</td> <td> 0.000</td> <td>    0.067</td> <td>    0.113</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cape Verde]</th>                       <td>    0.1900</td> <td>    0.024</td> <td>    7.836</td> <td> 0.000</td> <td>    0.142</td> <td>    0.237</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cayman Islands]</th>                   <td>   -0.2175</td> <td>    0.008</td> <td>  -26.993</td> <td> 0.000</td> <td>   -0.233</td> <td>   -0.202</td>
</tr>
<tr>
  <th>C(country_standard)[T.Central African Republic]</th>         <td>    0.0560</td> <td>    0.005</td> <td>   11.752</td> <td> 0.000</td> <td>    0.047</td> <td>    0.065</td>
</tr>
<tr>
  <th>C(country_standard)[T.Chad]</th>                             <td>    0.1010</td> <td>    0.011</td> <td>    8.897</td> <td> 0.000</td> <td>    0.079</td> <td>    0.123</td>
</tr>
<tr>
  <th>C(country_standard)[T.Channel Islands]</th>                  <td>    0.0548</td> <td>    0.006</td> <td>    9.886</td> <td> 0.000</td> <td>    0.044</td> <td>    0.066</td>
</tr>
<tr>
  <th>C(country_standard)[T.Chile]</th>                            <td>    0.1415</td> <td>    0.018</td> <td>    8.027</td> <td> 0.000</td> <td>    0.107</td> <td>    0.176</td>
</tr>
<tr>
  <th>C(country_standard)[T.China]</th>                            <td>    0.1987</td> <td>    0.030</td> <td>    6.583</td> <td> 0.000</td> <td>    0.140</td> <td>    0.258</td>
</tr>
<tr>
  <th>C(country_standard)[T.Colombia]</th>                         <td>   -0.0385</td> <td>    0.007</td> <td>   -5.247</td> <td> 0.000</td> <td>   -0.053</td> <td>   -0.024</td>
</tr>
<tr>
  <th>C(country_standard)[T.Comoros]</th>                          <td>    0.0222</td> <td>    0.003</td> <td>    7.577</td> <td> 0.000</td> <td>    0.016</td> <td>    0.028</td>
</tr>
<tr>
  <th>C(country_standard)[T.Congo - Brazzaville]</th>              <td>    0.0614</td> <td>    0.013</td> <td>    4.879</td> <td> 0.000</td> <td>    0.037</td> <td>    0.086</td>
</tr>
<tr>
  <th>C(country_standard)[T.Congo - Kinshasa]</th>                 <td>    0.0683</td> <td>    0.009</td> <td>    7.342</td> <td> 0.000</td> <td>    0.050</td> <td>    0.087</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cook Islands]</th>                     <td>    0.0516</td> <td>    0.003</td> <td>   17.405</td> <td> 0.000</td> <td>    0.046</td> <td>    0.057</td>
</tr>
<tr>
  <th>C(country_standard)[T.Costa Rica]</th>                       <td>    0.0273</td> <td>    0.023</td> <td>    1.167</td> <td> 0.243</td> <td>   -0.019</td> <td>    0.073</td>
</tr>
<tr>
  <th>C(country_standard)[T.Croatia]</th>                          <td>   -0.0218</td> <td>    0.018</td> <td>   -1.244</td> <td> 0.213</td> <td>   -0.056</td> <td>    0.013</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cuba]</th>                             <td>    0.3573</td> <td>    0.022</td> <td>   16.330</td> <td> 0.000</td> <td>    0.314</td> <td>    0.400</td>
</tr>
<tr>
  <th>C(country_standard)[T.Curaçao]</th>                          <td>    0.0516</td> <td>    0.003</td> <td>   17.361</td> <td> 0.000</td> <td>    0.046</td> <td>    0.057</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cyprus]</th>                           <td>    0.0087</td> <td>    0.011</td> <td>    0.764</td> <td> 0.445</td> <td>   -0.014</td> <td>    0.031</td>
</tr>
<tr>
  <th>C(country_standard)[T.Czechia]</th>                          <td>   -0.1061</td> <td>    0.014</td> <td>   -7.536</td> <td> 0.000</td> <td>   -0.134</td> <td>   -0.078</td>
</tr>
<tr>
  <th>C(country_standard)[T.Côte d’Ivoire]</th>                    <td>    0.0688</td> <td>    0.007</td> <td>    9.402</td> <td> 0.000</td> <td>    0.054</td> <td>    0.083</td>
</tr>
<tr>
  <th>C(country_standard)[T.Denmark]</th>                          <td>    0.0813</td> <td>    0.012</td> <td>    6.888</td> <td> 0.000</td> <td>    0.058</td> <td>    0.104</td>
</tr>
<tr>
  <th>C(country_standard)[T.Djibouti]</th>                         <td>    0.2510</td> <td>    0.011</td> <td>   22.823</td> <td> 0.000</td> <td>    0.229</td> <td>    0.273</td>
</tr>
<tr>
  <th>C(country_standard)[T.Dominica]</th>                         <td>    0.2360</td> <td>    0.003</td> <td>   82.463</td> <td> 0.000</td> <td>    0.230</td> <td>    0.242</td>
</tr>
<tr>
  <th>C(country_standard)[T.Dominican Republic]</th>               <td>   -0.0966</td> <td>    0.005</td> <td>  -18.034</td> <td> 0.000</td> <td>   -0.107</td> <td>   -0.086</td>
</tr>
<tr>
  <th>C(country_standard)[T.Ecuador]</th>                          <td>    0.0085</td> <td>    0.003</td> <td>    2.457</td> <td> 0.014</td> <td>    0.002</td> <td>    0.015</td>
</tr>
<tr>
  <th>C(country_standard)[T.Egypt]</th>                            <td>    0.1865</td> <td>    0.013</td> <td>   13.832</td> <td> 0.000</td> <td>    0.160</td> <td>    0.213</td>
</tr>
<tr>
  <th>C(country_standard)[T.El Salvador]</th>                      <td>   -0.0239</td> <td>    0.013</td> <td>   -1.806</td> <td> 0.071</td> <td>   -0.050</td> <td>    0.002</td>
</tr>
<tr>
  <th>C(country_standard)[T.Equatorial Guinea]</th>                <td>   -0.0042</td> <td>    0.026</td> <td>   -0.164</td> <td> 0.870</td> <td>   -0.055</td> <td>    0.046</td>
</tr>
<tr>
  <th>C(country_standard)[T.Eritrea]</th>                          <td>    0.3524</td> <td>    0.023</td> <td>   15.416</td> <td> 0.000</td> <td>    0.308</td> <td>    0.397</td>
</tr>
<tr>
  <th>C(country_standard)[T.Estonia]</th>                          <td>    0.1055</td> <td>    0.019</td> <td>    5.456</td> <td> 0.000</td> <td>    0.068</td> <td>    0.143</td>
</tr>
<tr>
  <th>C(country_standard)[T.Eswatini]</th>                         <td>    0.2868</td> <td>    0.012</td> <td>   23.132</td> <td> 0.000</td> <td>    0.262</td> <td>    0.311</td>
</tr>
<tr>
  <th>C(country_standard)[T.Ethiopia]</th>                         <td>    0.3498</td> <td>    0.006</td> <td>   54.071</td> <td> 0.000</td> <td>    0.337</td> <td>    0.363</td>
</tr>
<tr>
  <th>C(country_standard)[T.Faroe Islands]</th>                    <td>    0.0496</td> <td>    0.003</td> <td>   19.127</td> <td> 0.000</td> <td>    0.044</td> <td>    0.055</td>
</tr>
<tr>
  <th>C(country_standard)[T.Fiji]</th>                             <td>    0.2059</td> <td>    0.003</td> <td>   71.793</td> <td> 0.000</td> <td>    0.200</td> <td>    0.211</td>
</tr>
<tr>
  <th>C(country_standard)[T.Finland]</th>                          <td>    0.1912</td> <td>    0.016</td> <td>   12.323</td> <td> 0.000</td> <td>    0.161</td> <td>    0.222</td>
</tr>
<tr>
  <th>C(country_standard)[T.France]</th>                           <td>    0.0149</td> <td>    0.013</td> <td>    1.111</td> <td> 0.267</td> <td>   -0.011</td> <td>    0.041</td>
</tr>
<tr>
  <th>C(country_standard)[T.French Guiana]</th>                    <td>    0.0516</td> <td>    0.003</td> <td>   17.451</td> <td> 0.000</td> <td>    0.046</td> <td>    0.057</td>
</tr>
<tr>
  <th>C(country_standard)[T.French Polynesia]</th>                 <td>    0.0516</td> <td>    0.003</td> <td>   17.384</td> <td> 0.000</td> <td>    0.046</td> <td>    0.057</td>
</tr>
<tr>
  <th>C(country_standard)[T.Gabon]</th>                            <td>    0.0646</td> <td>    0.010</td> <td>    6.660</td> <td> 0.000</td> <td>    0.046</td> <td>    0.084</td>
</tr>
<tr>
  <th>C(country_standard)[T.Gambia]</th>                           <td>    0.1576</td> <td>    0.003</td> <td>   57.789</td> <td> 0.000</td> <td>    0.152</td> <td>    0.163</td>
</tr>
<tr>
  <th>C(country_standard)[T.Georgia]</th>                          <td>    0.2652</td> <td>    0.012</td> <td>   22.308</td> <td> 0.000</td> <td>    0.242</td> <td>    0.289</td>
</tr>
<tr>
  <th>C(country_standard)[T.Germany]</th>                          <td>    0.0387</td> <td>    0.010</td> <td>    4.058</td> <td> 0.000</td> <td>    0.020</td> <td>    0.057</td>
</tr>
<tr>
  <th>C(country_standard)[T.Ghana]</th>                            <td>    0.0496</td> <td>    0.022</td> <td>    2.254</td> <td> 0.024</td> <td>    0.006</td> <td>    0.093</td>
</tr>
<tr>
  <th>C(country_standard)[T.Gibraltar]</th>                        <td>    0.0495</td> <td>    0.003</td> <td>   19.005</td> <td> 0.000</td> <td>    0.044</td> <td>    0.055</td>
</tr>
<tr>
  <th>C(country_standard)[T.Greece]</th>                           <td>   -0.1175</td> <td>    0.020</td> <td>   -5.774</td> <td> 0.000</td> <td>   -0.157</td> <td>   -0.078</td>
</tr>
<tr>
  <th>C(country_standard)[T.Greenland]</th>                        <td>    0.0496</td> <td>    0.003</td> <td>   19.116</td> <td> 0.000</td> <td>    0.044</td> <td>    0.055</td>
</tr>
<tr>
  <th>C(country_standard)[T.Grenada]</th>                          <td>    0.1220</td> <td>    0.001</td> <td>   87.286</td> <td> 0.000</td> <td>    0.119</td> <td>    0.125</td>
</tr>
<tr>
  <th>C(country_standard)[T.Guam]</th>                             <td>    0.0516</td> <td>    0.003</td> <td>   17.431</td> <td> 0.000</td> <td>    0.046</td> <td>    0.057</td>
</tr>
<tr>
  <th>C(country_standard)[T.Guatemala]</th>                        <td>   -0.0404</td> <td>    0.008</td> <td>   -5.233</td> <td> 0.000</td> <td>   -0.056</td> <td>   -0.025</td>
</tr>
<tr>
  <th>C(country_standard)[T.Guernsey]</th>                         <td>    0.0173</td> <td>    0.034</td> <td>    0.510</td> <td> 0.610</td> <td>   -0.049</td> <td>    0.084</td>
</tr>
<tr>
  <th>C(country_standard)[T.Guinea]</th>                           <td>    0.0544</td> <td>    0.001</td> <td>   44.846</td> <td> 0.000</td> <td>    0.052</td> <td>    0.057</td>
</tr>
<tr>
  <th>C(country_standard)[T.Guinea-Bissau]</th>                    <td>   -0.0491</td> <td>    0.003</td> <td>  -16.220</td> <td> 0.000</td> <td>   -0.055</td> <td>   -0.043</td>
</tr>
<tr>
  <th>C(country_standard)[T.Guyana]</th>                           <td>   -0.0311</td> <td>    0.014</td> <td>   -2.157</td> <td> 0.031</td> <td>   -0.059</td> <td>   -0.003</td>
</tr>
<tr>
  <th>C(country_standard)[T.Haiti]</th>                            <td>   -0.0948</td> <td>    0.003</td> <td>  -37.262</td> <td> 0.000</td> <td>   -0.100</td> <td>   -0.090</td>
</tr>
<tr>
  <th>C(country_standard)[T.Honduras]</th>                         <td>    0.0484</td> <td>    0.003</td> <td>   17.351</td> <td> 0.000</td> <td>    0.043</td> <td>    0.054</td>
</tr>
<tr>
  <th>C(country_standard)[T.Hong Kong SAR China]</th>              <td>    0.1599</td> <td>    0.010</td> <td>   15.381</td> <td> 0.000</td> <td>    0.140</td> <td>    0.180</td>
</tr>
<tr>
  <th>C(country_standard)[T.Hungary]</th>                          <td>    0.0096</td> <td>    0.008</td> <td>    1.206</td> <td> 0.228</td> <td>   -0.006</td> <td>    0.025</td>
</tr>
<tr>
  <th>C(country_standard)[T.Iceland]</th>                          <td>    0.0934</td> <td>    0.012</td> <td>    7.550</td> <td> 0.000</td> <td>    0.069</td> <td>    0.118</td>
</tr>
<tr>
  <th>C(country_standard)[T.India]</th>                            <td>   -0.0554</td> <td>    0.008</td> <td>   -7.362</td> <td> 0.000</td> <td>   -0.070</td> <td>   -0.041</td>
</tr>
<tr>
  <th>C(country_standard)[T.Indonesia]</th>                        <td>   -0.0727</td> <td>    0.006</td> <td>  -11.252</td> <td> 0.000</td> <td>   -0.085</td> <td>   -0.060</td>
</tr>
<tr>
  <th>C(country_standard)[T.Iran]</th>                             <td>    0.0351</td> <td>    0.013</td> <td>    2.626</td> <td> 0.009</td> <td>    0.009</td> <td>    0.061</td>
</tr>
<tr>
  <th>C(country_standard)[T.Iraq]</th>                             <td>   -0.0725</td> <td>    0.003</td> <td>  -21.953</td> <td> 0.000</td> <td>   -0.079</td> <td>   -0.066</td>
</tr>
<tr>
  <th>C(country_standard)[T.Ireland]</th>                          <td>   -0.2289</td> <td>    0.003</td> <td>  -80.949</td> <td> 0.000</td> <td>   -0.234</td> <td>   -0.223</td>
</tr>
<tr>
  <th>C(country_standard)[T.Isle of Man]</th>                      <td>    0.0495</td> <td>    0.003</td> <td>   19.022</td> <td> 0.000</td> <td>    0.044</td> <td>    0.055</td>
</tr>
<tr>
  <th>C(country_standard)[T.Israel]</th>                           <td>   -0.0205</td> <td>    0.010</td> <td>   -2.062</td> <td> 0.039</td> <td>   -0.040</td> <td>   -0.001</td>
</tr>
<tr>
  <th>C(country_standard)[T.Italy]</th>                            <td>   -0.2304</td> <td>    0.017</td> <td>  -13.545</td> <td> 0.000</td> <td>   -0.264</td> <td>   -0.197</td>
</tr>
<tr>
  <th>C(country_standard)[T.Jamaica]</th>                          <td>   -0.0615</td> <td>    0.020</td> <td>   -3.096</td> <td> 0.002</td> <td>   -0.100</td> <td>   -0.023</td>
</tr>
<tr>
  <th>C(country_standard)[T.Japan]</th>                            <td>    0.1928</td> <td>    0.014</td> <td>   13.552</td> <td> 0.000</td> <td>    0.165</td> <td>    0.221</td>
</tr>
<tr>
  <th>C(country_standard)[T.Jersey]</th>                           <td>    0.0495</td> <td>    0.003</td> <td>   19.009</td> <td> 0.000</td> <td>    0.044</td> <td>    0.055</td>
</tr>
<tr>
  <th>C(country_standard)[T.Jordan]</th>                           <td>    0.3089</td> <td>    0.001</td> <td>  218.689</td> <td> 0.000</td> <td>    0.306</td> <td>    0.312</td>
</tr>
<tr>
  <th>C(country_standard)[T.Kazakhstan]</th>                       <td>    0.0701</td> <td>    0.020</td> <td>    3.497</td> <td> 0.000</td> <td>    0.031</td> <td>    0.109</td>
</tr>
<tr>
  <th>C(country_standard)[T.Kenya]</th>                            <td>   -0.0367</td> <td>    0.009</td> <td>   -4.161</td> <td> 0.000</td> <td>   -0.054</td> <td>   -0.019</td>
</tr>
<tr>
  <th>C(country_standard)[T.Kiribati]</th>                         <td>    0.0516</td> <td>    0.003</td> <td>   17.378</td> <td> 0.000</td> <td>    0.046</td> <td>    0.057</td>
</tr>
<tr>
  <th>C(country_standard)[T.Kosovo]</th>                           <td>    0.0548</td> <td>    0.006</td> <td>    9.882</td> <td> 0.000</td> <td>    0.044</td> <td>    0.066</td>
</tr>
<tr>
  <th>C(country_standard)[T.Kuwait]</th>                           <td>   -0.3829</td> <td>    0.017</td> <td>  -22.963</td> <td> 0.000</td> <td>   -0.416</td> <td>   -0.350</td>
</tr>
<tr>
  <th>C(country_standard)[T.Kyrgyzstan]</th>                       <td>    0.0094</td> <td>    0.008</td> <td>    1.127</td> <td> 0.260</td> <td>   -0.007</td> <td>    0.026</td>
</tr>
<tr>
  <th>C(country_standard)[T.Laos]</th>                             <td>    0.1708</td> <td>    0.020</td> <td>    8.730</td> <td> 0.000</td> <td>    0.132</td> <td>    0.209</td>
</tr>
<tr>
  <th>C(country_standard)[T.Latvia]</th>                           <td>   -0.0624</td> <td>    0.014</td> <td>   -4.354</td> <td> 0.000</td> <td>   -0.090</td> <td>   -0.034</td>
</tr>
<tr>
  <th>C(country_standard)[T.Lebanon]</th>                          <td>   -0.1084</td> <td>    0.001</td> <td>  -99.310</td> <td> 0.000</td> <td>   -0.111</td> <td>   -0.106</td>
</tr>
<tr>
  <th>C(country_standard)[T.Lesotho]</th>                          <td>    0.1218</td> <td>    0.010</td> <td>   12.255</td> <td> 0.000</td> <td>    0.102</td> <td>    0.141</td>
</tr>
<tr>
  <th>C(country_standard)[T.Liberia]</th>                          <td>    0.0400</td> <td>    0.012</td> <td>    3.433</td> <td> 0.001</td> <td>    0.017</td> <td>    0.063</td>
</tr>
<tr>
  <th>C(country_standard)[T.Libya]</th>                            <td>   -0.1198</td> <td>    0.007</td> <td>  -16.317</td> <td> 0.000</td> <td>   -0.134</td> <td>   -0.105</td>
</tr>
<tr>
  <th>C(country_standard)[T.Liechtenstein]</th>                    <td>    0.0559</td> <td>    0.003</td> <td>   20.030</td> <td> 0.000</td> <td>    0.050</td> <td>    0.061</td>
</tr>
<tr>
  <th>C(country_standard)[T.Lithuania]</th>                        <td>   -0.0109</td> <td>    0.018</td> <td>   -0.600</td> <td> 0.549</td> <td>   -0.047</td> <td>    0.025</td>
</tr>
<tr>
  <th>C(country_standard)[T.Luxembourg]</th>                       <td>   -0.2814</td> <td>    0.002</td> <td> -116.761</td> <td> 0.000</td> <td>   -0.286</td> <td>   -0.277</td>
</tr>
<tr>
  <th>C(country_standard)[T.Macao SAR China]</th>                  <td>    0.0548</td> <td>    0.003</td> <td>   19.337</td> <td> 0.000</td> <td>    0.049</td> <td>    0.060</td>
</tr>
<tr>
  <th>C(country_standard)[T.Madagascar]</th>                       <td>    0.0288</td> <td>    0.006</td> <td>    4.653</td> <td> 0.000</td> <td>    0.017</td> <td>    0.041</td>
</tr>
<tr>
  <th>C(country_standard)[T.Malawi]</th>                           <td>    0.0221</td> <td>    0.011</td> <td>    2.034</td> <td> 0.042</td> <td>    0.001</td> <td>    0.043</td>
</tr>
<tr>
  <th>C(country_standard)[T.Malaysia]</th>                         <td>    0.1072</td> <td>    0.008</td> <td>   13.380</td> <td> 0.000</td> <td>    0.091</td> <td>    0.123</td>
</tr>
<tr>
  <th>C(country_standard)[T.Maldives]</th>                         <td>    0.0443</td> <td>    0.001</td> <td>   42.153</td> <td> 0.000</td> <td>    0.042</td> <td>    0.046</td>
</tr>
<tr>
  <th>C(country_standard)[T.Mali]</th>                             <td>    0.0690</td> <td>    0.008</td> <td>    8.565</td> <td> 0.000</td> <td>    0.053</td> <td>    0.085</td>
</tr>
<tr>
  <th>C(country_standard)[T.Malta]</th>                            <td>   -0.0272</td> <td>    0.012</td> <td>   -2.182</td> <td> 0.029</td> <td>   -0.052</td> <td>   -0.003</td>
</tr>
<tr>
  <th>C(country_standard)[T.Marshall Islands]</th>                 <td>    0.1885</td> <td>    0.008</td> <td>   23.511</td> <td> 0.000</td> <td>    0.173</td> <td>    0.204</td>
</tr>
<tr>
  <th>C(country_standard)[T.Martinique]</th>                       <td>    0.0516</td> <td>    0.003</td> <td>   17.441</td> <td> 0.000</td> <td>    0.046</td> <td>    0.057</td>
</tr>
<tr>
  <th>C(country_standard)[T.Mauritania]</th>                       <td>    0.1477</td> <td>    0.002</td> <td>   64.694</td> <td> 0.000</td> <td>    0.143</td> <td>    0.152</td>
</tr>
<tr>
  <th>C(country_standard)[T.Mauritius]</th>                        <td>    0.0120</td> <td>    0.020</td> <td>    0.591</td> <td> 0.555</td> <td>   -0.028</td> <td>    0.052</td>
</tr>
<tr>
  <th>C(country_standard)[T.Mexico]</th>                           <td>   -0.1537</td> <td>    0.004</td> <td>  -43.233</td> <td> 0.000</td> <td>   -0.161</td> <td>   -0.147</td>
</tr>
<tr>
  <th>C(country_standard)[T.Micronesia (Federated States of)]</th> <td>    0.0495</td> <td>    0.003</td> <td>   18.948</td> <td> 0.000</td> <td>    0.044</td> <td>    0.055</td>
</tr>
<tr>
  <th>C(country_standard)[T.Moldova]</th>                          <td>    0.0510</td> <td>    0.012</td> <td>    4.321</td> <td> 0.000</td> <td>    0.028</td> <td>    0.074</td>
</tr>
<tr>
  <th>C(country_standard)[T.Monaco]</th>                           <td>    0.0559</td> <td>    0.003</td> <td>   20.036</td> <td> 0.000</td> <td>    0.050</td> <td>    0.061</td>
</tr>
<tr>
  <th>C(country_standard)[T.Mongolia]</th>                         <td>   -0.0569</td> <td>    0.018</td> <td>   -3.176</td> <td> 0.001</td> <td>   -0.092</td> <td>   -0.022</td>
</tr>
<tr>
  <th>C(country_standard)[T.Montenegro]</th>                       <td>    0.0970</td> <td>    0.009</td> <td>   11.400</td> <td> 0.000</td> <td>    0.080</td> <td>    0.114</td>
</tr>
<tr>
  <th>C(country_standard)[T.Morocco]</th>                          <td>    0.0988</td> <td>    0.004</td> <td>   25.460</td> <td> 0.000</td> <td>    0.091</td> <td>    0.106</td>
</tr>
<tr>
  <th>C(country_standard)[T.Mozambique]</th>                       <td>    0.0651</td> <td>    0.006</td> <td>   10.457</td> <td> 0.000</td> <td>    0.053</td> <td>    0.077</td>
</tr>
<tr>
  <th>C(country_standard)[T.Myanmar (Burma)]</th>                  <td>    0.1442</td> <td>    0.002</td> <td>   67.716</td> <td> 0.000</td> <td>    0.140</td> <td>    0.148</td>
</tr>
<tr>
  <th>C(country_standard)[T.Namibia]</th>                          <td>    0.0846</td> <td>    0.015</td> <td>    5.537</td> <td> 0.000</td> <td>    0.055</td> <td>    0.115</td>
</tr>
<tr>
  <th>C(country_standard)[T.Nauru]</th>                            <td>    0.0517</td> <td>    0.003</td> <td>   17.125</td> <td> 0.000</td> <td>    0.046</td> <td>    0.058</td>
</tr>
<tr>
  <th>C(country_standard)[T.Nepal]</th>                            <td>    0.0282</td> <td>    0.011</td> <td>    2.531</td> <td> 0.011</td> <td>    0.006</td> <td>    0.050</td>
</tr>
<tr>
  <th>C(country_standard)[T.Netherlands]</th>                      <td>   -0.0307</td> <td>    0.008</td> <td>   -4.052</td> <td> 0.000</td> <td>   -0.046</td> <td>   -0.016</td>
</tr>
<tr>
  <th>C(country_standard)[T.New Caledonia]</th>                    <td>    0.0515</td> <td>    0.003</td> <td>   17.525</td> <td> 0.000</td> <td>    0.046</td> <td>    0.057</td>
</tr>
<tr>
  <th>C(country_standard)[T.New Zealand]</th>                      <td>    0.2861</td> <td>    0.019</td> <td>   14.993</td> <td> 0.000</td> <td>    0.249</td> <td>    0.323</td>
</tr>
<tr>
  <th>C(country_standard)[T.Nicaragua]</th>                        <td>    0.0962</td> <td>    0.009</td> <td>   10.708</td> <td> 0.000</td> <td>    0.079</td> <td>    0.114</td>
</tr>
<tr>
  <th>C(country_standard)[T.Niger]</th>                            <td>    0.0900</td> <td>    0.012</td> <td>    7.529</td> <td> 0.000</td> <td>    0.067</td> <td>    0.113</td>
</tr>
<tr>
  <th>C(country_standard)[T.Nigeria]</th>                          <td>   -0.0978</td> <td>    0.004</td> <td>  -22.265</td> <td> 0.000</td> <td>   -0.106</td> <td>   -0.089</td>
</tr>
<tr>
  <th>C(country_standard)[T.Niue]</th>                             <td>    0.0515</td> <td>    0.003</td> <td>   17.526</td> <td> 0.000</td> <td>    0.046</td> <td>    0.057</td>
</tr>
<tr>
  <th>C(country_standard)[T.North Korea]</th>                      <td>    0.0496</td> <td>    0.003</td> <td>   19.176</td> <td> 0.000</td> <td>    0.045</td> <td>    0.055</td>
</tr>
<tr>
  <th>C(country_standard)[T.North Macedonia]</th>                  <td>    0.0506</td> <td>    0.005</td> <td>   10.408</td> <td> 0.000</td> <td>    0.041</td> <td>    0.060</td>
</tr>
<tr>
  <th>C(country_standard)[T.Northern Mariana Islands]</th>         <td>    0.0517</td> <td>    0.003</td> <td>   17.005</td> <td> 0.000</td> <td>    0.046</td> <td>    0.058</td>
</tr>
<tr>
  <th>C(country_standard)[T.Norway]</th>                           <td>   -0.0897</td> <td>    0.007</td> <td>  -12.737</td> <td> 0.000</td> <td>   -0.104</td> <td>   -0.076</td>
</tr>
<tr>
  <th>C(country_standard)[T.Oman]</th>                             <td>    0.1081</td> <td>    0.023</td> <td>    4.765</td> <td> 0.000</td> <td>    0.064</td> <td>    0.153</td>
</tr>
<tr>
  <th>C(country_standard)[T.Pakistan]</th>                         <td>    0.0897</td> <td>    0.002</td> <td>   42.767</td> <td> 0.000</td> <td>    0.086</td> <td>    0.094</td>
</tr>
<tr>
  <th>C(country_standard)[T.Palau]</th>                            <td>    0.0516</td> <td>    0.003</td> <td>   17.416</td> <td> 0.000</td> <td>    0.046</td> <td>    0.057</td>
</tr>
<tr>
  <th>C(country_standard)[T.Palestinian Territories]</th>          <td>    0.3880</td> <td>    0.012</td> <td>   32.823</td> <td> 0.000</td> <td>    0.365</td> <td>    0.411</td>
</tr>
<tr>
  <th>C(country_standard)[T.Panama]</th>                           <td>   -0.2243</td> <td>    0.012</td> <td>  -18.537</td> <td> 0.000</td> <td>   -0.248</td> <td>   -0.201</td>
</tr>
<tr>
  <th>C(country_standard)[T.Papua New Guinea]</th>                 <td>   -0.0605</td> <td>    0.008</td> <td>   -7.472</td> <td> 0.000</td> <td>   -0.076</td> <td>   -0.045</td>
</tr>
<tr>
  <th>C(country_standard)[T.Paraguay]</th>                         <td>   -0.0899</td> <td>    0.006</td> <td>  -15.052</td> <td> 0.000</td> <td>   -0.102</td> <td>   -0.078</td>
</tr>
<tr>
  <th>C(country_standard)[T.Peru]</th>                             <td>   -0.1507</td> <td>    0.016</td> <td>   -9.442</td> <td> 0.000</td> <td>   -0.182</td> <td>   -0.119</td>
</tr>
<tr>
  <th>C(country_standard)[T.Philippines]</th>                      <td>    0.0953</td> <td>    0.005</td> <td>   21.030</td> <td> 0.000</td> <td>    0.086</td> <td>    0.104</td>
</tr>
<tr>
  <th>C(country_standard)[T.Poland]</th>                           <td>    0.0387</td> <td>    0.016</td> <td>    2.451</td> <td> 0.014</td> <td>    0.008</td> <td>    0.070</td>
</tr>
<tr>
  <th>C(country_standard)[T.Portugal]</th>                         <td>    0.0299</td> <td>    0.020</td> <td>    1.529</td> <td> 0.126</td> <td>   -0.008</td> <td>    0.068</td>
</tr>
<tr>
  <th>C(country_standard)[T.Puerto Rico]</th>                      <td>    0.0515</td> <td>    0.003</td> <td>   17.592</td> <td> 0.000</td> <td>    0.046</td> <td>    0.057</td>
</tr>
<tr>
  <th>C(country_standard)[T.Qatar]</th>                            <td>   -0.2182</td> <td>    0.056</td> <td>   -3.899</td> <td> 0.000</td> <td>   -0.328</td> <td>   -0.109</td>
</tr>
<tr>
  <th>C(country_standard)[T.Romania]</th>                          <td>   -0.0443</td> <td>    0.015</td> <td>   -3.013</td> <td> 0.003</td> <td>   -0.073</td> <td>   -0.015</td>
</tr>
<tr>
  <th>C(country_standard)[T.Russia]</th>                           <td>   -0.0807</td> <td>    0.023</td> <td>   -3.496</td> <td> 0.000</td> <td>   -0.126</td> <td>   -0.035</td>
</tr>
<tr>
  <th>C(country_standard)[T.Rwanda]</th>                           <td>    0.5190</td> <td>    0.009</td> <td>   60.393</td> <td> 0.000</td> <td>    0.502</td> <td>    0.536</td>
</tr>
<tr>
  <th>C(country_standard)[T.Réunion]</th>                          <td>    0.0515</td> <td>    0.003</td> <td>   17.704</td> <td> 0.000</td> <td>    0.046</td> <td>    0.057</td>
</tr>
<tr>
  <th>C(country_standard)[T.Saint Martin (French part)]</th>       <td>    0.0515</td> <td>    0.003</td> <td>   17.579</td> <td> 0.000</td> <td>    0.046</td> <td>    0.057</td>
</tr>
<tr>
  <th>C(country_standard)[T.Samoa]</th>                            <td>    0.1597</td> <td>    0.007</td> <td>   24.018</td> <td> 0.000</td> <td>    0.147</td> <td>    0.173</td>
</tr>
<tr>
  <th>C(country_standard)[T.San Marino]</th>                       <td>   -0.2787</td> <td>    0.010</td> <td>  -26.603</td> <td> 0.000</td> <td>   -0.299</td> <td>   -0.258</td>
</tr>
<tr>
  <th>C(country_standard)[T.Saudi Arabia]</th>                     <td>    0.1012</td> <td>    0.041</td> <td>    2.462</td> <td> 0.014</td> <td>    0.021</td> <td>    0.182</td>
</tr>
<tr>
  <th>C(country_standard)[T.Senegal]</th>                          <td>    0.1031</td> <td>    0.018</td> <td>    5.694</td> <td> 0.000</td> <td>    0.068</td> <td>    0.139</td>
</tr>
<tr>
  <th>C(country_standard)[T.Serbia]</th>                           <td>    0.0466</td> <td>    0.010</td> <td>    4.821</td> <td> 0.000</td> <td>    0.028</td> <td>    0.066</td>
</tr>
<tr>
  <th>C(country_standard)[T.Seychelles]</th>                       <td>    0.1583</td> <td>    0.001</td> <td>  130.286</td> <td> 0.000</td> <td>    0.156</td> <td>    0.161</td>
</tr>
<tr>
  <th>C(country_standard)[T.Sierra Leone]</th>                     <td>    0.0693</td> <td>    0.010</td> <td>    6.736</td> <td> 0.000</td> <td>    0.049</td> <td>    0.090</td>
</tr>
<tr>
  <th>C(country_standard)[T.Singapore]</th>                        <td>    0.0118</td> <td>    0.025</td> <td>    0.479</td> <td> 0.632</td> <td>   -0.036</td> <td>    0.060</td>
</tr>
<tr>
  <th>C(country_standard)[T.Sint Maarten]</th>                     <td>    0.0515</td> <td>    0.003</td> <td>   17.658</td> <td> 0.000</td> <td>    0.046</td> <td>    0.057</td>
</tr>
<tr>
  <th>C(country_standard)[T.Slovakia]</th>                         <td>   -0.0191</td> <td>    0.022</td> <td>   -0.853</td> <td> 0.393</td> <td>   -0.063</td> <td>    0.025</td>
</tr>
<tr>
  <th>C(country_standard)[T.Slovenia]</th>                         <td>    0.0115</td> <td>    0.018</td> <td>    0.648</td> <td> 0.517</td> <td>   -0.023</td> <td>    0.046</td>
</tr>
<tr>
  <th>C(country_standard)[T.Solomon Islands]</th>                  <td>    0.1374</td> <td>    0.017</td> <td>    8.270</td> <td> 0.000</td> <td>    0.105</td> <td>    0.170</td>
</tr>
<tr>
  <th>C(country_standard)[T.Somalia]</th>                          <td>   -0.1660</td> <td>    0.017</td> <td>   -9.759</td> <td> 0.000</td> <td>   -0.199</td> <td>   -0.133</td>
</tr>
<tr>
  <th>C(country_standard)[T.South Africa]</th>                     <td>   -0.0610</td> <td>    0.015</td> <td>   -3.934</td> <td> 0.000</td> <td>   -0.091</td> <td>   -0.031</td>
</tr>
<tr>
  <th>C(country_standard)[T.South Korea]</th>                      <td>   -0.1002</td> <td>    0.008</td> <td>  -13.197</td> <td> 0.000</td> <td>   -0.115</td> <td>   -0.085</td>
</tr>
<tr>
  <th>C(country_standard)[T.South Sudan]</th>                      <td>    0.1177</td> <td>    0.017</td> <td>    7.104</td> <td> 0.000</td> <td>    0.085</td> <td>    0.150</td>
</tr>
<tr>
  <th>C(country_standard)[T.Spain]</th>                            <td>   -0.0726</td> <td>    0.016</td> <td>   -4.594</td> <td> 0.000</td> <td>   -0.104</td> <td>   -0.042</td>
</tr>
<tr>
  <th>C(country_standard)[T.Sri Lanka]</th>                        <td>    0.0800</td> <td>    0.005</td> <td>   17.014</td> <td> 0.000</td> <td>    0.071</td> <td>    0.089</td>
</tr>
<tr>
  <th>C(country_standard)[T.St. Helena]</th>                       <td>    0.0173</td> <td>    0.034</td> <td>    0.511</td> <td> 0.610</td> <td>   -0.049</td> <td>    0.084</td>
</tr>
<tr>
  <th>C(country_standard)[T.St. Kitts & Nevis]</th>                <td>    0.0049</td> <td>    0.001</td> <td>    5.838</td> <td> 0.000</td> <td>    0.003</td> <td>    0.007</td>
</tr>
<tr>
  <th>C(country_standard)[T.St. Lucia]</th>                        <td>    0.2839</td> <td>    0.005</td> <td>   58.689</td> <td> 0.000</td> <td>    0.274</td> <td>    0.293</td>
</tr>
<tr>
  <th>C(country_standard)[T.St. Vincent & Grenadines]</th>         <td>    0.2670</td> <td>    0.003</td> <td>  100.215</td> <td> 0.000</td> <td>    0.262</td> <td>    0.272</td>
</tr>
<tr>
  <th>C(country_standard)[T.Sudan]</th>                            <td>    0.0702</td> <td>    0.012</td> <td>    5.716</td> <td> 0.000</td> <td>    0.046</td> <td>    0.094</td>
</tr>
<tr>
  <th>C(country_standard)[T.Suriname]</th>                         <td>   -0.1332</td> <td>    0.013</td> <td>   -9.938</td> <td> 0.000</td> <td>   -0.160</td> <td>   -0.107</td>
</tr>
<tr>
  <th>C(country_standard)[T.Sweden]</th>                           <td>    0.0478</td> <td>    0.010</td> <td>    4.756</td> <td> 0.000</td> <td>    0.028</td> <td>    0.068</td>
</tr>
<tr>
  <th>C(country_standard)[T.Switzerland]</th>                      <td>    0.0201</td> <td>    0.011</td> <td>    1.799</td> <td> 0.072</td> <td>   -0.002</td> <td>    0.042</td>
</tr>
<tr>
  <th>C(country_standard)[T.Syria]</th>                            <td>    0.0280</td> <td>    0.027</td> <td>    1.033</td> <td> 0.301</td> <td>   -0.025</td> <td>    0.081</td>
</tr>
<tr>
  <th>C(country_standard)[T.São Tomé & Príncipe]</th>              <td>    0.1168</td> <td>    0.022</td> <td>    5.291</td> <td> 0.000</td> <td>    0.074</td> <td>    0.160</td>
</tr>
<tr>
  <th>C(country_standard)[T.Taiwan]</th>                           <td>    0.2031</td> <td>    0.017</td> <td>   11.845</td> <td> 0.000</td> <td>    0.169</td> <td>    0.237</td>
</tr>
<tr>
  <th>C(country_standard)[T.Tajikistan]</th>                       <td>    0.2219</td> <td>    0.014</td> <td>   15.489</td> <td> 0.000</td> <td>    0.194</td> <td>    0.250</td>
</tr>
<tr>
  <th>C(country_standard)[T.Tanzania]</th>                         <td>    0.1009</td> <td>    0.012</td> <td>    8.568</td> <td> 0.000</td> <td>    0.078</td> <td>    0.124</td>
</tr>
<tr>
  <th>C(country_standard)[T.Thailand]</th>                         <td>    0.0949</td> <td>    0.013</td> <td>    7.169</td> <td> 0.000</td> <td>    0.069</td> <td>    0.121</td>
</tr>
<tr>
  <th>C(country_standard)[T.Timor-Leste]</th>                      <td>    0.0654</td> <td>    0.019</td> <td>    3.464</td> <td> 0.001</td> <td>    0.028</td> <td>    0.102</td>
</tr>
<tr>
  <th>C(country_standard)[T.Togo]</th>                             <td>    0.0988</td> <td>    0.002</td> <td>   44.555</td> <td> 0.000</td> <td>    0.094</td> <td>    0.103</td>
</tr>
<tr>
  <th>C(country_standard)[T.Tonga]</th>                            <td>    0.0517</td> <td>    0.003</td> <td>   17.124</td> <td> 0.000</td> <td>    0.046</td> <td>    0.058</td>
</tr>
<tr>
  <th>C(country_standard)[T.Trinidad & Tobago]</th>                <td>   -0.2346</td> <td>    0.013</td> <td>  -17.646</td> <td> 0.000</td> <td>   -0.261</td> <td>   -0.209</td>
</tr>
<tr>
  <th>C(country_standard)[T.Tunisia]</th>                          <td>    0.0583</td> <td>    0.018</td> <td>    3.172</td> <td> 0.002</td> <td>    0.022</td> <td>    0.094</td>
</tr>
<tr>
  <th>C(country_standard)[T.Turkey]</th>                           <td>    0.0950</td> <td>    0.013</td> <td>    7.451</td> <td> 0.000</td> <td>    0.070</td> <td>    0.120</td>
</tr>
<tr>
  <th>C(country_standard)[T.Turkmenistan]</th>                     <td>    0.0871</td> <td>    0.023</td> <td>    3.866</td> <td> 0.000</td> <td>    0.043</td> <td>    0.131</td>
</tr>
<tr>
  <th>C(country_standard)[T.Turks & Caicos Islands]</th>           <td>    0.0495</td> <td>    0.003</td> <td>   19.042</td> <td> 0.000</td> <td>    0.044</td> <td>    0.055</td>
</tr>
<tr>
  <th>C(country_standard)[T.Tuvalu]</th>                           <td>    0.0516</td> <td>    0.003</td> <td>   17.168</td> <td> 0.000</td> <td>    0.046</td> <td>    0.058</td>
</tr>
<tr>
  <th>C(country_standard)[T.U.S. Virgin Islands]</th>              <td>    0.0516</td> <td>    0.003</td> <td>   17.379</td> <td> 0.000</td> <td>    0.046</td> <td>    0.057</td>
</tr>
<tr>
  <th>C(country_standard)[T.Uganda]</th>                           <td>    0.0670</td> <td>    0.001</td> <td>   67.155</td> <td> 0.000</td> <td>    0.065</td> <td>    0.069</td>
</tr>
<tr>
  <th>C(country_standard)[T.Ukraine]</th>                          <td>   -0.0947</td> <td>    0.001</td> <td>  -67.248</td> <td> 0.000</td> <td>   -0.097</td> <td>   -0.092</td>
</tr>
<tr>
  <th>C(country_standard)[T.United Arab Emirates]</th>             <td>    0.1066</td> <td>    0.036</td> <td>    2.967</td> <td> 0.003</td> <td>    0.036</td> <td>    0.177</td>
</tr>
<tr>
  <th>C(country_standard)[T.United Kingdom]</th>                   <td>    0.0768</td> <td>    0.012</td> <td>    6.231</td> <td> 0.000</td> <td>    0.053</td> <td>    0.101</td>
</tr>
<tr>
  <th>C(country_standard)[T.United States]</th>                    <td>   -0.1963</td> <td>    0.005</td> <td>  -40.131</td> <td> 0.000</td> <td>   -0.206</td> <td>   -0.187</td>
</tr>
<tr>
  <th>C(country_standard)[T.Uruguay]</th>                          <td>    0.1953</td> <td>    0.021</td> <td>    9.386</td> <td> 0.000</td> <td>    0.154</td> <td>    0.236</td>
</tr>
<tr>
  <th>C(country_standard)[T.Uzbekistan]</th>                       <td>    0.1462</td> <td>    0.017</td> <td>    8.361</td> <td> 0.000</td> <td>    0.112</td> <td>    0.180</td>
</tr>
<tr>
  <th>C(country_standard)[T.Vanuatu]</th>                          <td>    0.0661</td> <td>    0.024</td> <td>    2.770</td> <td> 0.006</td> <td>    0.019</td> <td>    0.113</td>
</tr>
<tr>
  <th>C(country_standard)[T.Vatican City]</th>                     <td>    0.0819</td> <td>    0.034</td> <td>    2.443</td> <td> 0.015</td> <td>    0.016</td> <td>    0.148</td>
</tr>
<tr>
  <th>C(country_standard)[T.Venezuela]</th>                        <td>   -0.0904</td> <td>    0.014</td> <td>   -6.265</td> <td> 0.000</td> <td>   -0.119</td> <td>   -0.062</td>
</tr>
<tr>
  <th>C(country_standard)[T.Vietnam]</th>                          <td>    0.2170</td> <td>    0.012</td> <td>   18.327</td> <td> 0.000</td> <td>    0.194</td> <td>    0.240</td>
</tr>
<tr>
  <th>C(country_standard)[T.Western Sahara]</th>                   <td>    0.0423</td> <td>    0.008</td> <td>    5.089</td> <td> 0.000</td> <td>    0.026</td> <td>    0.059</td>
</tr>
<tr>
  <th>C(country_standard)[T.Yemen]</th>                            <td>    0.0501</td> <td>    0.016</td> <td>    3.206</td> <td> 0.001</td> <td>    0.019</td> <td>    0.081</td>
</tr>
<tr>
  <th>C(country_standard)[T.Zambia]</th>                           <td>    0.0647</td> <td>    0.005</td> <td>   13.423</td> <td> 0.000</td> <td>    0.055</td> <td>    0.074</td>
</tr>
<tr>
  <th>C(country_standard)[T.Zimbabwe]</th>                         <td>    0.0417</td> <td>    0.003</td> <td>   15.085</td> <td> 0.000</td> <td>    0.036</td> <td>    0.047</td>
</tr>
<tr>
  <th>transparency_index</th>                                      <td>    1.0103</td> <td>    0.096</td> <td>   10.494</td> <td> 0.000</td> <td>    0.822</td> <td>    1.199</td>
</tr>
<tr>
  <th>gdp</th>                                                     <td> 1.236e-14</td> <td> 5.39e-16</td> <td>   22.938</td> <td> 0.000</td> <td> 1.13e-14</td> <td> 1.34e-14</td>
</tr>
<tr>
  <th>gdp_per_capita</th>                                          <td> 8.809e-06</td> <td> 3.48e-07</td> <td>   25.287</td> <td> 0.000</td> <td> 8.13e-06</td> <td> 9.49e-06</td>
</tr>
<tr>
  <th>gini_2020</th>                                               <td>    0.5180</td> <td>    0.033</td> <td>   15.516</td> <td> 0.000</td> <td>    0.453</td> <td>    0.583</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>453.059</td> <th>  Durbin-Watson:     </th> <td>   0.687</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>3228.240</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.368</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 7.571</td>  <th>  Cond. No.          </th> <td>4.96e+26</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors are robust to cluster correlation (cluster)<br/>[2] The smallest eigenvalue is 5.07e-26. This might indicate that there are<br/>strong multicollinearity problems or that the design matrix is singular.




```python
reg_trust = smf.ols('trust_index ~ transparency_index + gdp + gdp_per_capita + gini_2020 + C(country_standard)', ndi_df).fit(cov_type='cluster', cov_kwds={'groups': ndi_df['country_standard']})
```


```python
reg_trust.summary()
```

    /Users/katiacordoba/opt/anaconda3/lib/python3.7/site-packages/statsmodels/base/model.py:1832: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 232, but rank is 228
      'rank is %d' % (J, J_), ValueWarning)





<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>trust_index</td>   <th>  R-squared:         </th> <td>   0.699</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.699</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>1.070e+11</td>
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 05 Mar 2021</td> <th>  Prob (F-statistic):</th>  <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>13:17:42</td>     <th>  Log-Likelihood:    </th> <td>  4846.7</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>  3615</td>      <th>  AIC:               </th> <td>  -9683.</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  3610</td>      <th>  BIC:               </th> <td>  -9652.</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     4</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>       <td>cluster</td>     <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
                             <td></td>                                <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>                                               <td>    0.4586</td> <td>    0.009</td> <td>   53.189</td> <td> 0.000</td> <td>    0.442</td> <td>    0.476</td>
</tr>
<tr>
  <th>C(country_standard)[T.Albania]</th>                          <td>   -0.0065</td> <td>    0.004</td> <td>   -1.712</td> <td> 0.087</td> <td>   -0.014</td> <td>    0.001</td>
</tr>
<tr>
  <th>C(country_standard)[T.Algeria]</th>                          <td>   -0.1325</td> <td>    0.005</td> <td>  -29.226</td> <td> 0.000</td> <td>   -0.141</td> <td>   -0.124</td>
</tr>
<tr>
  <th>C(country_standard)[T.American Samoa]</th>                   <td>   -0.0169</td> <td>    0.001</td> <td>  -17.983</td> <td> 0.000</td> <td>   -0.019</td> <td>   -0.015</td>
</tr>
<tr>
  <th>C(country_standard)[T.Andorra]</th>                          <td>   -0.0146</td> <td>    0.001</td> <td>  -15.500</td> <td> 0.000</td> <td>   -0.016</td> <td>   -0.013</td>
</tr>
<tr>
  <th>C(country_standard)[T.Angola]</th>                           <td>   -0.0043</td> <td>    0.003</td> <td>   -1.371</td> <td> 0.170</td> <td>   -0.011</td> <td>    0.002</td>
</tr>
<tr>
  <th>C(country_standard)[T.Anguilla]</th>                         <td>   -0.0164</td> <td>    0.001</td> <td>  -19.290</td> <td> 0.000</td> <td>   -0.018</td> <td>   -0.015</td>
</tr>
<tr>
  <th>C(country_standard)[T.Antigua & Barbuda]</th>                <td>   -0.0224</td> <td>    0.000</td> <td>  -49.768</td> <td> 0.000</td> <td>   -0.023</td> <td>   -0.022</td>
</tr>
<tr>
  <th>C(country_standard)[T.Argentina]</th>                        <td>   -0.2167</td> <td>    0.005</td> <td>  -47.302</td> <td> 0.000</td> <td>   -0.226</td> <td>   -0.208</td>
</tr>
<tr>
  <th>C(country_standard)[T.Armenia]</th>                          <td>   -0.2114</td> <td>    0.002</td> <td> -100.440</td> <td> 0.000</td> <td>   -0.216</td> <td>   -0.207</td>
</tr>
<tr>
  <th>C(country_standard)[T.Aruba]</th>                            <td>   -0.0169</td> <td>    0.001</td> <td>  -18.331</td> <td> 0.000</td> <td>   -0.019</td> <td>   -0.015</td>
</tr>
<tr>
  <th>C(country_standard)[T.Australia]</th>                        <td>    0.0488</td> <td>    0.005</td> <td>    9.063</td> <td> 0.000</td> <td>    0.038</td> <td>    0.059</td>
</tr>
<tr>
  <th>C(country_standard)[T.Austria]</th>                          <td>   -0.0315</td> <td>    0.004</td> <td>   -8.214</td> <td> 0.000</td> <td>   -0.039</td> <td>   -0.024</td>
</tr>
<tr>
  <th>C(country_standard)[T.Azerbaijan]</th>                       <td>    0.0361</td> <td>    0.006</td> <td>    5.771</td> <td> 0.000</td> <td>    0.024</td> <td>    0.048</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bahamas]</th>                          <td>   -0.0243</td> <td>    0.001</td> <td>  -19.634</td> <td> 0.000</td> <td>   -0.027</td> <td>   -0.022</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bahrain]</th>                          <td>   -0.0105</td> <td>    0.010</td> <td>   -1.052</td> <td> 0.293</td> <td>   -0.030</td> <td>    0.009</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bangladesh]</th>                       <td>    0.0994</td> <td>    0.001</td> <td>  192.045</td> <td> 0.000</td> <td>    0.098</td> <td>    0.100</td>
</tr>
<tr>
  <th>C(country_standard)[T.Barbados]</th>                         <td>   -0.0254</td> <td>    0.006</td> <td>   -4.295</td> <td> 0.000</td> <td>   -0.037</td> <td>   -0.014</td>
</tr>
<tr>
  <th>C(country_standard)[T.Belarus]</th>                          <td>   -0.1002</td> <td>    0.005</td> <td>  -18.878</td> <td> 0.000</td> <td>   -0.111</td> <td>   -0.090</td>
</tr>
<tr>
  <th>C(country_standard)[T.Belgium]</th>                          <td>   -0.0217</td> <td>    0.006</td> <td>   -3.489</td> <td> 0.000</td> <td>   -0.034</td> <td>   -0.009</td>
</tr>
<tr>
  <th>C(country_standard)[T.Belize]</th>                           <td>   -0.0179</td> <td>    0.001</td> <td>  -12.776</td> <td> 0.000</td> <td>   -0.021</td> <td>   -0.015</td>
</tr>
<tr>
  <th>C(country_standard)[T.Benin]</th>                            <td>   -0.0134</td> <td>    0.006</td> <td>   -2.196</td> <td> 0.028</td> <td>   -0.025</td> <td>   -0.001</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bermuda]</th>                          <td>   -0.0164</td> <td>    0.001</td> <td>  -18.875</td> <td> 0.000</td> <td>   -0.018</td> <td>   -0.015</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bhutan]</th>                           <td>   -0.0138</td> <td>    0.003</td> <td>   -4.437</td> <td> 0.000</td> <td>   -0.020</td> <td>   -0.008</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bolivia]</th>                          <td>   -0.1127</td> <td>    0.001</td> <td>  -82.188</td> <td> 0.000</td> <td>   -0.115</td> <td>   -0.110</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bosnia & Herzegovina]</th>             <td>   -0.0064</td> <td>    0.003</td> <td>   -2.346</td> <td> 0.019</td> <td>   -0.012</td> <td>   -0.001</td>
</tr>
<tr>
  <th>C(country_standard)[T.Botswana]</th>                         <td>   -0.0252</td> <td>    0.005</td> <td>   -5.508</td> <td> 0.000</td> <td>   -0.034</td> <td>   -0.016</td>
</tr>
<tr>
  <th>C(country_standard)[T.Brazil]</th>                           <td>   -0.1207</td> <td>    0.004</td> <td>  -28.607</td> <td> 0.000</td> <td>   -0.129</td> <td>   -0.112</td>
</tr>
<tr>
  <th>C(country_standard)[T.British Virgin Islands]</th>           <td>   -0.0168</td> <td>    0.001</td> <td>  -18.843</td> <td> 0.000</td> <td>   -0.019</td> <td>   -0.015</td>
</tr>
<tr>
  <th>C(country_standard)[T.Brunei]</th>                           <td>   -0.0308</td> <td>    0.006</td> <td>   -5.159</td> <td> 0.000</td> <td>   -0.043</td> <td>   -0.019</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bulgaria]</th>                         <td>   -0.0968</td> <td>    0.006</td> <td>  -16.404</td> <td> 0.000</td> <td>   -0.108</td> <td>   -0.085</td>
</tr>
<tr>
  <th>C(country_standard)[T.Burkina Faso]</th>                     <td>   -0.1502</td> <td>    0.004</td> <td>  -40.250</td> <td> 0.000</td> <td>   -0.158</td> <td>   -0.143</td>
</tr>
<tr>
  <th>C(country_standard)[T.Burundi]</th>                          <td>    0.0033</td> <td>    0.004</td> <td>    0.887</td> <td> 0.375</td> <td>   -0.004</td> <td>    0.011</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cambodia]</th>                         <td>   -0.0015</td> <td>    0.004</td> <td>   -0.420</td> <td> 0.675</td> <td>   -0.009</td> <td>    0.006</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cameroon]</th>                         <td>   -0.0053</td> <td>    0.003</td> <td>   -2.098</td> <td> 0.036</td> <td>   -0.010</td> <td>   -0.000</td>
</tr>
<tr>
  <th>C(country_standard)[T.Canada]</th>                           <td>    0.0768</td> <td>    0.004</td> <td>   19.679</td> <td> 0.000</td> <td>    0.069</td> <td>    0.084</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cape Verde]</th>                       <td>   -0.0200</td> <td>    0.008</td> <td>   -2.430</td> <td> 0.015</td> <td>   -0.036</td> <td>   -0.004</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cayman Islands]</th>                   <td>   -0.0215</td> <td>    0.003</td> <td>   -7.864</td> <td> 0.000</td> <td>   -0.027</td> <td>   -0.016</td>
</tr>
<tr>
  <th>C(country_standard)[T.Central African Republic]</th>         <td>   -0.0086</td> <td>    0.002</td> <td>   -5.365</td> <td> 0.000</td> <td>   -0.012</td> <td>   -0.005</td>
</tr>
<tr>
  <th>C(country_standard)[T.Chad]</th>                             <td>   -0.0016</td> <td>    0.004</td> <td>   -0.407</td> <td> 0.684</td> <td>   -0.009</td> <td>    0.006</td>
</tr>
<tr>
  <th>C(country_standard)[T.Channel Islands]</th>                  <td>   -0.0176</td> <td>    0.001</td> <td>  -12.662</td> <td> 0.000</td> <td>   -0.020</td> <td>   -0.015</td>
</tr>
<tr>
  <th>C(country_standard)[T.Chile]</th>                            <td>   -0.0960</td> <td>    0.006</td> <td>  -16.076</td> <td> 0.000</td> <td>   -0.108</td> <td>   -0.084</td>
</tr>
<tr>
  <th>C(country_standard)[T.China]</th>                            <td>    0.1133</td> <td>    0.010</td> <td>   11.028</td> <td> 0.000</td> <td>    0.093</td> <td>    0.133</td>
</tr>
<tr>
  <th>C(country_standard)[T.Colombia]</th>                         <td>   -0.1363</td> <td>    0.002</td> <td>  -55.021</td> <td> 0.000</td> <td>   -0.141</td> <td>   -0.131</td>
</tr>
<tr>
  <th>C(country_standard)[T.Comoros]</th>                          <td>   -0.0130</td> <td>    0.001</td> <td>  -13.420</td> <td> 0.000</td> <td>   -0.015</td> <td>   -0.011</td>
</tr>
<tr>
  <th>C(country_standard)[T.Congo - Brazzaville]</th>              <td>   -0.0058</td> <td>    0.004</td> <td>   -1.359</td> <td> 0.174</td> <td>   -0.014</td> <td>    0.003</td>
</tr>
<tr>
  <th>C(country_standard)[T.Congo - Kinshasa]</th>                 <td>   -0.0046</td> <td>    0.003</td> <td>   -1.477</td> <td> 0.140</td> <td>   -0.011</td> <td>    0.002</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cook Islands]</th>                     <td>   -0.0168</td> <td>    0.001</td> <td>  -18.948</td> <td> 0.000</td> <td>   -0.019</td> <td>   -0.015</td>
</tr>
<tr>
  <th>C(country_standard)[T.Costa Rica]</th>                       <td>   -0.0253</td> <td>    0.008</td> <td>   -3.192</td> <td> 0.001</td> <td>   -0.041</td> <td>   -0.010</td>
</tr>
<tr>
  <th>C(country_standard)[T.Croatia]</th>                          <td>   -0.0155</td> <td>    0.006</td> <td>   -2.611</td> <td> 0.009</td> <td>   -0.027</td> <td>   -0.004</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cuba]</th>                             <td>   -0.0034</td> <td>    0.007</td> <td>   -0.459</td> <td> 0.646</td> <td>   -0.018</td> <td>    0.011</td>
</tr>
<tr>
  <th>C(country_standard)[T.Curaçao]</th>                          <td>   -0.0168</td> <td>    0.001</td> <td>  -18.745</td> <td> 0.000</td> <td>   -0.019</td> <td>   -0.015</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cyprus]</th>                           <td>    0.1471</td> <td>    0.004</td> <td>   38.178</td> <td> 0.000</td> <td>    0.140</td> <td>    0.155</td>
</tr>
<tr>
  <th>C(country_standard)[T.Czechia]</th>                          <td>   -0.0259</td> <td>    0.005</td> <td>   -5.442</td> <td> 0.000</td> <td>   -0.035</td> <td>   -0.017</td>
</tr>
<tr>
  <th>C(country_standard)[T.Côte d’Ivoire]</th>                    <td>   -0.0111</td> <td>    0.002</td> <td>   -4.499</td> <td> 0.000</td> <td>   -0.016</td> <td>   -0.006</td>
</tr>
<tr>
  <th>C(country_standard)[T.Denmark]</th>                          <td>   -0.0412</td> <td>    0.004</td> <td>  -10.295</td> <td> 0.000</td> <td>   -0.049</td> <td>   -0.033</td>
</tr>
<tr>
  <th>C(country_standard)[T.Djibouti]</th>                         <td>   -0.0019</td> <td>    0.004</td> <td>   -0.522</td> <td> 0.602</td> <td>   -0.009</td> <td>    0.005</td>
</tr>
<tr>
  <th>C(country_standard)[T.Dominica]</th>                         <td>   -0.0198</td> <td>    0.001</td> <td>  -20.758</td> <td> 0.000</td> <td>   -0.022</td> <td>   -0.018</td>
</tr>
<tr>
  <th>C(country_standard)[T.Dominican Republic]</th>               <td>   -0.0146</td> <td>    0.002</td> <td>   -8.071</td> <td> 0.000</td> <td>   -0.018</td> <td>   -0.011</td>
</tr>
<tr>
  <th>C(country_standard)[T.Ecuador]</th>                          <td>   -0.0635</td> <td>    0.001</td> <td>  -54.530</td> <td> 0.000</td> <td>   -0.066</td> <td>   -0.061</td>
</tr>
<tr>
  <th>C(country_standard)[T.Egypt]</th>                            <td>   -0.1485</td> <td>    0.005</td> <td>  -32.557</td> <td> 0.000</td> <td>   -0.157</td> <td>   -0.140</td>
</tr>
<tr>
  <th>C(country_standard)[T.El Salvador]</th>                      <td>   -0.0165</td> <td>    0.004</td> <td>   -3.689</td> <td> 0.000</td> <td>   -0.025</td> <td>   -0.008</td>
</tr>
<tr>
  <th>C(country_standard)[T.Equatorial Guinea]</th>                <td>   -0.0086</td> <td>    0.009</td> <td>   -0.983</td> <td> 0.325</td> <td>   -0.026</td> <td>    0.009</td>
</tr>
<tr>
  <th>C(country_standard)[T.Eritrea]</th>                          <td>    0.0142</td> <td>    0.008</td> <td>    1.830</td> <td> 0.067</td> <td>   -0.001</td> <td>    0.029</td>
</tr>
<tr>
  <th>C(country_standard)[T.Estonia]</th>                          <td>    0.0773</td> <td>    0.007</td> <td>   11.786</td> <td> 0.000</td> <td>    0.064</td> <td>    0.090</td>
</tr>
<tr>
  <th>C(country_standard)[T.Eswatini]</th>                         <td>   -0.0032</td> <td>    0.004</td> <td>   -0.766</td> <td> 0.443</td> <td>   -0.011</td> <td>    0.005</td>
</tr>
<tr>
  <th>C(country_standard)[T.Ethiopia]</th>                         <td>   -0.2725</td> <td>    0.002</td> <td> -126.048</td> <td> 0.000</td> <td>   -0.277</td> <td>   -0.268</td>
</tr>
<tr>
  <th>C(country_standard)[T.Faroe Islands]</th>                    <td>   -0.0164</td> <td>    0.001</td> <td>  -19.261</td> <td> 0.000</td> <td>   -0.018</td> <td>   -0.015</td>
</tr>
<tr>
  <th>C(country_standard)[T.Fiji]</th>                             <td>   -0.0082</td> <td>    0.001</td> <td>   -8.710</td> <td> 0.000</td> <td>   -0.010</td> <td>   -0.006</td>
</tr>
<tr>
  <th>C(country_standard)[T.Finland]</th>                          <td>   -0.0314</td> <td>    0.005</td> <td>   -5.957</td> <td> 0.000</td> <td>   -0.042</td> <td>   -0.021</td>
</tr>
<tr>
  <th>C(country_standard)[T.France]</th>                           <td>   -0.0933</td> <td>    0.005</td> <td>  -20.455</td> <td> 0.000</td> <td>   -0.102</td> <td>   -0.084</td>
</tr>
<tr>
  <th>C(country_standard)[T.French Guiana]</th>                    <td>   -0.0169</td> <td>    0.001</td> <td>  -18.281</td> <td> 0.000</td> <td>   -0.019</td> <td>   -0.015</td>
</tr>
<tr>
  <th>C(country_standard)[T.French Polynesia]</th>                 <td>   -0.0169</td> <td>    0.001</td> <td>  -18.535</td> <td> 0.000</td> <td>   -0.019</td> <td>   -0.015</td>
</tr>
<tr>
  <th>C(country_standard)[T.Gabon]</th>                            <td>   -0.0070</td> <td>    0.003</td> <td>   -2.128</td> <td> 0.033</td> <td>   -0.013</td> <td>   -0.001</td>
</tr>
<tr>
  <th>C(country_standard)[T.Gambia]</th>                           <td>   -0.0094</td> <td>    0.001</td> <td>  -10.491</td> <td> 0.000</td> <td>   -0.011</td> <td>   -0.008</td>
</tr>
<tr>
  <th>C(country_standard)[T.Georgia]</th>                          <td>   -0.1159</td> <td>    0.004</td> <td>  -28.792</td> <td> 0.000</td> <td>   -0.124</td> <td>   -0.108</td>
</tr>
<tr>
  <th>C(country_standard)[T.Germany]</th>                          <td>    0.0642</td> <td>    0.003</td> <td>   19.754</td> <td> 0.000</td> <td>    0.058</td> <td>    0.071</td>
</tr>
<tr>
  <th>C(country_standard)[T.Ghana]</th>                            <td>    0.1849</td> <td>    0.007</td> <td>   24.832</td> <td> 0.000</td> <td>    0.170</td> <td>    0.199</td>
</tr>
<tr>
  <th>C(country_standard)[T.Gibraltar]</th>                        <td>   -0.0164</td> <td>    0.001</td> <td>  -18.900</td> <td> 0.000</td> <td>   -0.018</td> <td>   -0.015</td>
</tr>
<tr>
  <th>C(country_standard)[T.Greece]</th>                           <td>   -0.0445</td> <td>    0.007</td> <td>   -6.467</td> <td> 0.000</td> <td>   -0.058</td> <td>   -0.031</td>
</tr>
<tr>
  <th>C(country_standard)[T.Greenland]</th>                        <td>   -0.0164</td> <td>    0.001</td> <td>  -19.178</td> <td> 0.000</td> <td>   -0.018</td> <td>   -0.015</td>
</tr>
<tr>
  <th>C(country_standard)[T.Grenada]</th>                          <td>   -0.0210</td> <td>    0.000</td> <td>  -46.595</td> <td> 0.000</td> <td>   -0.022</td> <td>   -0.020</td>
</tr>
<tr>
  <th>C(country_standard)[T.Guam]</th>                             <td>   -0.0169</td> <td>    0.001</td> <td>  -18.459</td> <td> 0.000</td> <td>   -0.019</td> <td>   -0.015</td>
</tr>
<tr>
  <th>C(country_standard)[T.Guatemala]</th>                        <td>   -0.0720</td> <td>    0.003</td> <td>  -27.593</td> <td> 0.000</td> <td>   -0.077</td> <td>   -0.067</td>
</tr>
<tr>
  <th>C(country_standard)[T.Guernsey]</th>                         <td>   -0.0105</td> <td>    0.006</td> <td>   -1.686</td> <td> 0.092</td> <td>   -0.023</td> <td>    0.002</td>
</tr>
<tr>
  <th>C(country_standard)[T.Guinea]</th>                           <td>   -0.0084</td> <td>    0.000</td> <td>  -25.900</td> <td> 0.000</td> <td>   -0.009</td> <td>   -0.008</td>
</tr>
<tr>
  <th>C(country_standard)[T.Guinea-Bissau]</th>                    <td>   -0.0064</td> <td>    0.001</td> <td>   -6.510</td> <td> 0.000</td> <td>   -0.008</td> <td>   -0.005</td>
</tr>
<tr>
  <th>C(country_standard)[T.Guyana]</th>                           <td>   -0.0162</td> <td>    0.005</td> <td>   -3.326</td> <td> 0.001</td> <td>   -0.026</td> <td>   -0.007</td>
</tr>
<tr>
  <th>C(country_standard)[T.Haiti]</th>                            <td>   -0.1885</td> <td>    0.001</td> <td> -225.057</td> <td> 0.000</td> <td>   -0.190</td> <td>   -0.187</td>
</tr>
<tr>
  <th>C(country_standard)[T.Honduras]</th>                         <td>   -0.0092</td> <td>    0.001</td> <td>  -10.010</td> <td> 0.000</td> <td>   -0.011</td> <td>   -0.007</td>
</tr>
<tr>
  <th>C(country_standard)[T.Hong Kong SAR China]</th>              <td>   -0.0073</td> <td>    0.004</td> <td>   -2.059</td> <td> 0.039</td> <td>   -0.014</td> <td>   -0.000</td>
</tr>
<tr>
  <th>C(country_standard)[T.Hungary]</th>                          <td>   -0.1055</td> <td>    0.003</td> <td>  -39.039</td> <td> 0.000</td> <td>   -0.111</td> <td>   -0.100</td>
</tr>
<tr>
  <th>C(country_standard)[T.Iceland]</th>                          <td>   -0.0266</td> <td>    0.004</td> <td>   -6.357</td> <td> 0.000</td> <td>   -0.035</td> <td>   -0.018</td>
</tr>
<tr>
  <th>C(country_standard)[T.India]</th>                            <td>    0.2472</td> <td>    0.003</td> <td>   96.616</td> <td> 0.000</td> <td>    0.242</td> <td>    0.252</td>
</tr>
<tr>
  <th>C(country_standard)[T.Indonesia]</th>                        <td>    0.0915</td> <td>    0.002</td> <td>   41.800</td> <td> 0.000</td> <td>    0.087</td> <td>    0.096</td>
</tr>
<tr>
  <th>C(country_standard)[T.Iran]</th>                             <td>   -0.1697</td> <td>    0.005</td> <td>  -37.498</td> <td> 0.000</td> <td>   -0.179</td> <td>   -0.161</td>
</tr>
<tr>
  <th>C(country_standard)[T.Iraq]</th>                             <td>   -0.1871</td> <td>    0.001</td> <td> -171.250</td> <td> 0.000</td> <td>   -0.189</td> <td>   -0.185</td>
</tr>
<tr>
  <th>C(country_standard)[T.Ireland]</th>                          <td>   -0.0410</td> <td>    0.001</td> <td>  -45.007</td> <td> 0.000</td> <td>   -0.043</td> <td>   -0.039</td>
</tr>
<tr>
  <th>C(country_standard)[T.Isle of Man]</th>                      <td>   -0.0164</td> <td>    0.001</td> <td>  -18.940</td> <td> 0.000</td> <td>   -0.018</td> <td>   -0.015</td>
</tr>
<tr>
  <th>C(country_standard)[T.Israel]</th>                           <td>   -0.0296</td> <td>    0.003</td> <td>   -8.796</td> <td> 0.000</td> <td>   -0.036</td> <td>   -0.023</td>
</tr>
<tr>
  <th>C(country_standard)[T.Italy]</th>                            <td>   -0.0305</td> <td>    0.006</td> <td>   -5.292</td> <td> 0.000</td> <td>   -0.042</td> <td>   -0.019</td>
</tr>
<tr>
  <th>C(country_standard)[T.Jamaica]</th>                          <td>   -0.0230</td> <td>    0.007</td> <td>   -3.415</td> <td> 0.001</td> <td>   -0.036</td> <td>   -0.010</td>
</tr>
<tr>
  <th>C(country_standard)[T.Japan]</th>                            <td>    0.0408</td> <td>    0.005</td> <td>    8.476</td> <td> 0.000</td> <td>    0.031</td> <td>    0.050</td>
</tr>
<tr>
  <th>C(country_standard)[T.Jersey]</th>                           <td>   -0.0164</td> <td>    0.001</td> <td>  -18.956</td> <td> 0.000</td> <td>   -0.018</td> <td>   -0.015</td>
</tr>
<tr>
  <th>C(country_standard)[T.Jordan]</th>                           <td>    0.0426</td> <td>    0.000</td> <td>  102.754</td> <td> 0.000</td> <td>    0.042</td> <td>    0.043</td>
</tr>
<tr>
  <th>C(country_standard)[T.Kazakhstan]</th>                       <td>   -0.0241</td> <td>    0.007</td> <td>   -3.547</td> <td> 0.000</td> <td>   -0.037</td> <td>   -0.011</td>
</tr>
<tr>
  <th>C(country_standard)[T.Kenya]</th>                            <td>   -0.0132</td> <td>    0.003</td> <td>   -4.440</td> <td> 0.000</td> <td>   -0.019</td> <td>   -0.007</td>
</tr>
<tr>
  <th>C(country_standard)[T.Kiribati]</th>                         <td>   -0.0168</td> <td>    0.001</td> <td>  -18.690</td> <td> 0.000</td> <td>   -0.019</td> <td>   -0.015</td>
</tr>
<tr>
  <th>C(country_standard)[T.Kosovo]</th>                           <td>   -0.0176</td> <td>    0.001</td> <td>  -12.645</td> <td> 0.000</td> <td>   -0.020</td> <td>   -0.015</td>
</tr>
<tr>
  <th>C(country_standard)[T.Kuwait]</th>                           <td>    0.0193</td> <td>    0.006</td> <td>    3.424</td> <td> 0.001</td> <td>    0.008</td> <td>    0.030</td>
</tr>
<tr>
  <th>C(country_standard)[T.Kyrgyzstan]</th>                       <td>   -0.0294</td> <td>    0.003</td> <td>  -10.444</td> <td> 0.000</td> <td>   -0.035</td> <td>   -0.024</td>
</tr>
<tr>
  <th>C(country_standard)[T.Laos]</th>                             <td>   -0.0054</td> <td>    0.007</td> <td>   -0.816</td> <td> 0.414</td> <td>   -0.018</td> <td>    0.008</td>
</tr>
<tr>
  <th>C(country_standard)[T.Latvia]</th>                           <td>   -0.0282</td> <td>    0.005</td> <td>   -5.808</td> <td> 0.000</td> <td>   -0.038</td> <td>   -0.019</td>
</tr>
<tr>
  <th>C(country_standard)[T.Lebanon]</th>                          <td>   -0.1918</td> <td>    0.000</td> <td> -561.858</td> <td> 0.000</td> <td>   -0.192</td> <td>   -0.191</td>
</tr>
<tr>
  <th>C(country_standard)[T.Lesotho]</th>                          <td>   -0.0189</td> <td>    0.003</td> <td>   -5.625</td> <td> 0.000</td> <td>   -0.025</td> <td>   -0.012</td>
</tr>
<tr>
  <th>C(country_standard)[T.Liberia]</th>                          <td>   -0.0117</td> <td>    0.004</td> <td>   -2.966</td> <td> 0.003</td> <td>   -0.019</td> <td>   -0.004</td>
</tr>
<tr>
  <th>C(country_standard)[T.Libya]</th>                            <td>   -0.1613</td> <td>    0.002</td> <td>  -65.129</td> <td> 0.000</td> <td>   -0.166</td> <td>   -0.156</td>
</tr>
<tr>
  <th>C(country_standard)[T.Liechtenstein]</th>                    <td>   -0.0146</td> <td>    0.001</td> <td>  -15.680</td> <td> 0.000</td> <td>   -0.016</td> <td>   -0.013</td>
</tr>
<tr>
  <th>C(country_standard)[T.Lithuania]</th>                        <td>   -0.0201</td> <td>    0.006</td> <td>   -3.258</td> <td> 0.001</td> <td>   -0.032</td> <td>   -0.008</td>
</tr>
<tr>
  <th>C(country_standard)[T.Luxembourg]</th>                       <td>   -0.0362</td> <td>    0.001</td> <td>  -49.580</td> <td> 0.000</td> <td>   -0.038</td> <td>   -0.035</td>
</tr>
<tr>
  <th>C(country_standard)[T.Macao SAR China]</th>                  <td>   -0.0147</td> <td>    0.001</td> <td>  -15.699</td> <td> 0.000</td> <td>   -0.017</td> <td>   -0.013</td>
</tr>
<tr>
  <th>C(country_standard)[T.Madagascar]</th>                       <td>   -0.0089</td> <td>    0.002</td> <td>   -4.284</td> <td> 0.000</td> <td>   -0.013</td> <td>   -0.005</td>
</tr>
<tr>
  <th>C(country_standard)[T.Malawi]</th>                           <td>   -0.0137</td> <td>    0.004</td> <td>   -3.726</td> <td> 0.000</td> <td>   -0.021</td> <td>   -0.006</td>
</tr>
<tr>
  <th>C(country_standard)[T.Malaysia]</th>                         <td>    0.2018</td> <td>    0.003</td> <td>   74.332</td> <td> 0.000</td> <td>    0.197</td> <td>    0.207</td>
</tr>
<tr>
  <th>C(country_standard)[T.Maldives]</th>                         <td>   -0.0119</td> <td>    0.000</td> <td>  -38.190</td> <td> 0.000</td> <td>   -0.012</td> <td>   -0.011</td>
</tr>
<tr>
  <th>C(country_standard)[T.Mali]</th>                             <td>    0.2586</td> <td>    0.003</td> <td>   95.088</td> <td> 0.000</td> <td>    0.253</td> <td>    0.264</td>
</tr>
<tr>
  <th>C(country_standard)[T.Malta]</th>                            <td>   -0.0178</td> <td>    0.004</td> <td>   -4.209</td> <td> 0.000</td> <td>   -0.026</td> <td>   -0.009</td>
</tr>
<tr>
  <th>C(country_standard)[T.Marshall Islands]</th>                 <td>   -0.0113</td> <td>    0.003</td> <td>   -4.178</td> <td> 0.000</td> <td>   -0.017</td> <td>   -0.006</td>
</tr>
<tr>
  <th>C(country_standard)[T.Martinique]</th>                       <td>   -0.0169</td> <td>    0.001</td> <td>  -18.279</td> <td> 0.000</td> <td>   -0.019</td> <td>   -0.015</td>
</tr>
<tr>
  <th>C(country_standard)[T.Mauritania]</th>                       <td>   -0.0020</td> <td>    0.001</td> <td>   -2.845</td> <td> 0.004</td> <td>   -0.003</td> <td>   -0.001</td>
</tr>
<tr>
  <th>C(country_standard)[T.Mauritius]</th>                        <td>   -0.0173</td> <td>    0.007</td> <td>   -2.518</td> <td> 0.012</td> <td>   -0.031</td> <td>   -0.004</td>
</tr>
<tr>
  <th>C(country_standard)[T.Mexico]</th>                           <td>   -0.1842</td> <td>    0.001</td> <td> -153.304</td> <td> 0.000</td> <td>   -0.187</td> <td>   -0.182</td>
</tr>
<tr>
  <th>C(country_standard)[T.Micronesia (Federated States of)]</th> <td>   -0.0164</td> <td>    0.001</td> <td>  -18.863</td> <td> 0.000</td> <td>   -0.018</td> <td>   -0.015</td>
</tr>
<tr>
  <th>C(country_standard)[T.Moldova]</th>                          <td>   -0.1181</td> <td>    0.004</td> <td>  -29.614</td> <td> 0.000</td> <td>   -0.126</td> <td>   -0.110</td>
</tr>
<tr>
  <th>C(country_standard)[T.Monaco]</th>                           <td>   -0.0146</td> <td>    0.001</td> <td>  -15.684</td> <td> 0.000</td> <td>   -0.016</td> <td>   -0.013</td>
</tr>
<tr>
  <th>C(country_standard)[T.Mongolia]</th>                         <td>   -0.0135</td> <td>    0.006</td> <td>   -2.219</td> <td> 0.026</td> <td>   -0.025</td> <td>   -0.002</td>
</tr>
<tr>
  <th>C(country_standard)[T.Montenegro]</th>                       <td>   -0.0089</td> <td>    0.003</td> <td>   -3.095</td> <td> 0.002</td> <td>   -0.015</td> <td>   -0.003</td>
</tr>
<tr>
  <th>C(country_standard)[T.Morocco]</th>                          <td>   -0.0786</td> <td>    0.001</td> <td>  -60.372</td> <td> 0.000</td> <td>   -0.081</td> <td>   -0.076</td>
</tr>
<tr>
  <th>C(country_standard)[T.Mozambique]</th>                       <td>   -0.0083</td> <td>    0.002</td> <td>   -3.970</td> <td> 0.000</td> <td>   -0.012</td> <td>   -0.004</td>
</tr>
<tr>
  <th>C(country_standard)[T.Myanmar (Burma)]</th>                  <td>    0.0236</td> <td>    0.001</td> <td>   36.462</td> <td> 0.000</td> <td>    0.022</td> <td>    0.025</td>
</tr>
<tr>
  <th>C(country_standard)[T.Namibia]</th>                          <td>   -0.0225</td> <td>    0.005</td> <td>   -4.353</td> <td> 0.000</td> <td>   -0.033</td> <td>   -0.012</td>
</tr>
<tr>
  <th>C(country_standard)[T.Nauru]</th>                            <td>   -0.0168</td> <td>    0.001</td> <td>  -18.872</td> <td> 0.000</td> <td>   -0.019</td> <td>   -0.015</td>
</tr>
<tr>
  <th>C(country_standard)[T.Nepal]</th>                            <td>   -0.0106</td> <td>    0.004</td> <td>   -2.800</td> <td> 0.005</td> <td>   -0.018</td> <td>   -0.003</td>
</tr>
<tr>
  <th>C(country_standard)[T.Netherlands]</th>                      <td>    0.0016</td> <td>    0.003</td> <td>    0.637</td> <td> 0.524</td> <td>   -0.003</td> <td>    0.007</td>
</tr>
<tr>
  <th>C(country_standard)[T.New Caledonia]</th>                    <td>   -0.0169</td> <td>    0.001</td> <td>  -18.120</td> <td> 0.000</td> <td>   -0.019</td> <td>   -0.015</td>
</tr>
<tr>
  <th>C(country_standard)[T.New Zealand]</th>                      <td>    0.1226</td> <td>    0.006</td> <td>   18.937</td> <td> 0.000</td> <td>    0.110</td> <td>    0.135</td>
</tr>
<tr>
  <th>C(country_standard)[T.Nicaragua]</th>                        <td>   -0.0733</td> <td>    0.003</td> <td>  -24.171</td> <td> 0.000</td> <td>   -0.079</td> <td>   -0.067</td>
</tr>
<tr>
  <th>C(country_standard)[T.Niger]</th>                            <td>   -0.0078</td> <td>    0.004</td> <td>   -1.919</td> <td> 0.055</td> <td>   -0.016</td> <td>    0.000</td>
</tr>
<tr>
  <th>C(country_standard)[T.Nigeria]</th>                          <td>   -0.0912</td> <td>    0.001</td> <td>  -61.710</td> <td> 0.000</td> <td>   -0.094</td> <td>   -0.088</td>
</tr>
<tr>
  <th>C(country_standard)[T.Niue]</th>                             <td>   -0.0169</td> <td>    0.001</td> <td>  -18.145</td> <td> 0.000</td> <td>   -0.019</td> <td>   -0.015</td>
</tr>
<tr>
  <th>C(country_standard)[T.North Korea]</th>                      <td>   -0.0164</td> <td>    0.001</td> <td>  -19.285</td> <td> 0.000</td> <td>   -0.018</td> <td>   -0.015</td>
</tr>
<tr>
  <th>C(country_standard)[T.North Macedonia]</th>                  <td>   -0.0134</td> <td>    0.002</td> <td>   -8.201</td> <td> 0.000</td> <td>   -0.017</td> <td>   -0.010</td>
</tr>
<tr>
  <th>C(country_standard)[T.Northern Mariana Islands]</th>         <td>   -0.0168</td> <td>    0.001</td> <td>  -19.063</td> <td> 0.000</td> <td>   -0.018</td> <td>   -0.015</td>
</tr>
<tr>
  <th>C(country_standard)[T.Norway]</th>                           <td>    0.2554</td> <td>    0.002</td> <td>  107.395</td> <td> 0.000</td> <td>    0.251</td> <td>    0.260</td>
</tr>
<tr>
  <th>C(country_standard)[T.Oman]</th>                             <td>   -0.0155</td> <td>    0.008</td> <td>   -2.019</td> <td> 0.043</td> <td>   -0.031</td> <td>   -0.000</td>
</tr>
<tr>
  <th>C(country_standard)[T.Pakistan]</th>                         <td>   -0.0573</td> <td>    0.001</td> <td>  -94.675</td> <td> 0.000</td> <td>   -0.058</td> <td>   -0.056</td>
</tr>
<tr>
  <th>C(country_standard)[T.Palau]</th>                            <td>   -0.0170</td> <td>    0.001</td> <td>  -17.283</td> <td> 0.000</td> <td>   -0.019</td> <td>   -0.015</td>
</tr>
<tr>
  <th>C(country_standard)[T.Palestinian Territories]</th>          <td>   -0.0880</td> <td>    0.004</td> <td>  -21.986</td> <td> 0.000</td> <td>   -0.096</td> <td>   -0.080</td>
</tr>
<tr>
  <th>C(country_standard)[T.Panama]</th>                           <td>   -0.0253</td> <td>    0.004</td> <td>   -6.191</td> <td> 0.000</td> <td>   -0.033</td> <td>   -0.017</td>
</tr>
<tr>
  <th>C(country_standard)[T.Papua New Guinea]</th>                 <td>   -0.0149</td> <td>    0.003</td> <td>   -5.436</td> <td> 0.000</td> <td>   -0.020</td> <td>   -0.010</td>
</tr>
<tr>
  <th>C(country_standard)[T.Paraguay]</th>                         <td>   -0.0161</td> <td>    0.002</td> <td>   -7.972</td> <td> 0.000</td> <td>   -0.020</td> <td>   -0.012</td>
</tr>
<tr>
  <th>C(country_standard)[T.Peru]</th>                             <td>   -0.3837</td> <td>    0.005</td> <td>  -71.013</td> <td> 0.000</td> <td>   -0.394</td> <td>   -0.373</td>
</tr>
<tr>
  <th>C(country_standard)[T.Philippines]</th>                      <td>    0.1562</td> <td>    0.002</td> <td>  102.291</td> <td> 0.000</td> <td>    0.153</td> <td>    0.159</td>
</tr>
<tr>
  <th>C(country_standard)[T.Poland]</th>                           <td>   -0.0880</td> <td>    0.005</td> <td>  -16.481</td> <td> 0.000</td> <td>   -0.098</td> <td>   -0.078</td>
</tr>
<tr>
  <th>C(country_standard)[T.Portugal]</th>                         <td>   -0.0211</td> <td>    0.007</td> <td>   -3.180</td> <td> 0.001</td> <td>   -0.034</td> <td>   -0.008</td>
</tr>
<tr>
  <th>C(country_standard)[T.Puerto Rico]</th>                      <td>   -0.0169</td> <td>    0.001</td> <td>  -18.192</td> <td> 0.000</td> <td>   -0.019</td> <td>   -0.015</td>
</tr>
<tr>
  <th>C(country_standard)[T.Qatar]</th>                            <td>    0.2104</td> <td>    0.019</td> <td>   11.098</td> <td> 0.000</td> <td>    0.173</td> <td>    0.248</td>
</tr>
<tr>
  <th>C(country_standard)[T.Romania]</th>                          <td>   -0.1474</td> <td>    0.005</td> <td>  -29.637</td> <td> 0.000</td> <td>   -0.157</td> <td>   -0.138</td>
</tr>
<tr>
  <th>C(country_standard)[T.Russia]</th>                           <td>   -0.2075</td> <td>    0.008</td> <td>  -26.558</td> <td> 0.000</td> <td>   -0.223</td> <td>   -0.192</td>
</tr>
<tr>
  <th>C(country_standard)[T.Rwanda]</th>                           <td>   -0.0477</td> <td>    0.003</td> <td>  -16.450</td> <td> 0.000</td> <td>   -0.053</td> <td>   -0.042</td>
</tr>
<tr>
  <th>C(country_standard)[T.Réunion]</th>                          <td>   -0.0168</td> <td>    0.001</td> <td>  -18.785</td> <td> 0.000</td> <td>   -0.019</td> <td>   -0.015</td>
</tr>
<tr>
  <th>C(country_standard)[T.Saint Martin (French part)]</th>       <td>   -0.0169</td> <td>    0.001</td> <td>  -18.474</td> <td> 0.000</td> <td>   -0.019</td> <td>   -0.015</td>
</tr>
<tr>
  <th>C(country_standard)[T.Samoa]</th>                            <td>   -0.0132</td> <td>    0.002</td> <td>   -5.887</td> <td> 0.000</td> <td>   -0.018</td> <td>   -0.009</td>
</tr>
<tr>
  <th>C(country_standard)[T.San Marino]</th>                       <td>   -0.0230</td> <td>    0.004</td> <td>   -6.471</td> <td> 0.000</td> <td>   -0.030</td> <td>   -0.016</td>
</tr>
<tr>
  <th>C(country_standard)[T.Saudi Arabia]</th>                     <td>   -0.0186</td> <td>    0.014</td> <td>   -1.334</td> <td> 0.182</td> <td>   -0.046</td> <td>    0.009</td>
</tr>
<tr>
  <th>C(country_standard)[T.Senegal]</th>                          <td>   -0.0154</td> <td>    0.006</td> <td>   -2.511</td> <td> 0.012</td> <td>   -0.027</td> <td>   -0.003</td>
</tr>
<tr>
  <th>C(country_standard)[T.Serbia]</th>                           <td>   -0.2030</td> <td>    0.003</td> <td>  -62.094</td> <td> 0.000</td> <td>   -0.209</td> <td>   -0.197</td>
</tr>
<tr>
  <th>C(country_standard)[T.Seychelles]</th>                       <td>   -0.0168</td> <td>    0.000</td> <td>  -42.214</td> <td> 0.000</td> <td>   -0.018</td> <td>   -0.016</td>
</tr>
<tr>
  <th>C(country_standard)[T.Sierra Leone]</th>                     <td>   -0.0082</td> <td>    0.003</td> <td>   -2.371</td> <td> 0.018</td> <td>   -0.015</td> <td>   -0.001</td>
</tr>
<tr>
  <th>C(country_standard)[T.Singapore]</th>                        <td>    0.1062</td> <td>    0.008</td> <td>   12.769</td> <td> 0.000</td> <td>    0.090</td> <td>    0.123</td>
</tr>
<tr>
  <th>C(country_standard)[T.Sint Maarten]</th>                     <td>   -0.0169</td> <td>    0.001</td> <td>  -18.549</td> <td> 0.000</td> <td>   -0.019</td> <td>   -0.015</td>
</tr>
<tr>
  <th>C(country_standard)[T.Slovakia]</th>                         <td>   -0.0072</td> <td>    0.008</td> <td>   -0.946</td> <td> 0.344</td> <td>   -0.022</td> <td>    0.008</td>
</tr>
<tr>
  <th>C(country_standard)[T.Slovenia]</th>                         <td>   -0.1830</td> <td>    0.006</td> <td>  -30.527</td> <td> 0.000</td> <td>   -0.195</td> <td>   -0.171</td>
</tr>
<tr>
  <th>C(country_standard)[T.Solomon Islands]</th>                  <td>   -0.0121</td> <td>    0.006</td> <td>   -2.150</td> <td> 0.032</td> <td>   -0.023</td> <td>   -0.001</td>
</tr>
<tr>
  <th>C(country_standard)[T.Somalia]</th>                          <td>   -0.0073</td> <td>    0.006</td> <td>   -1.266</td> <td> 0.206</td> <td>   -0.019</td> <td>    0.004</td>
</tr>
<tr>
  <th>C(country_standard)[T.South Africa]</th>                     <td>    0.1493</td> <td>    0.005</td> <td>   28.488</td> <td> 0.000</td> <td>    0.139</td> <td>    0.160</td>
</tr>
<tr>
  <th>C(country_standard)[T.South Korea]</th>                      <td>    0.0522</td> <td>    0.003</td> <td>   20.237</td> <td> 0.000</td> <td>    0.047</td> <td>    0.057</td>
</tr>
<tr>
  <th>C(country_standard)[T.South Sudan]</th>                      <td>    0.0011</td> <td>    0.006</td> <td>    0.194</td> <td> 0.846</td> <td>   -0.010</td> <td>    0.012</td>
</tr>
<tr>
  <th>C(country_standard)[T.Spain]</th>                            <td>   -0.0231</td> <td>    0.005</td> <td>   -4.325</td> <td> 0.000</td> <td>   -0.034</td> <td>   -0.013</td>
</tr>
<tr>
  <th>C(country_standard)[T.Sri Lanka]</th>                        <td>   -0.0104</td> <td>    0.002</td> <td>   -6.598</td> <td> 0.000</td> <td>   -0.014</td> <td>   -0.007</td>
</tr>
<tr>
  <th>C(country_standard)[T.St. Helena]</th>                       <td>   -0.0105</td> <td>    0.006</td> <td>   -1.682</td> <td> 0.093</td> <td>   -0.023</td> <td>    0.002</td>
</tr>
<tr>
  <th>C(country_standard)[T.St. Kitts & Nevis]</th>                <td>   -0.0159</td> <td>    0.000</td> <td>  -61.835</td> <td> 0.000</td> <td>   -0.016</td> <td>   -0.015</td>
</tr>
<tr>
  <th>C(country_standard)[T.St. Lucia]</th>                        <td>   -0.0133</td> <td>    0.002</td> <td>   -8.175</td> <td> 0.000</td> <td>   -0.017</td> <td>   -0.010</td>
</tr>
<tr>
  <th>C(country_standard)[T.St. Vincent & Grenadines]</th>         <td>   -0.0197</td> <td>    0.001</td> <td>  -22.194</td> <td> 0.000</td> <td>   -0.021</td> <td>   -0.018</td>
</tr>
<tr>
  <th>C(country_standard)[T.Sudan]</th>                            <td>    0.0018</td> <td>    0.004</td> <td>    0.429</td> <td> 0.668</td> <td>   -0.006</td> <td>    0.010</td>
</tr>
<tr>
  <th>C(country_standard)[T.Suriname]</th>                         <td>   -0.0273</td> <td>    0.005</td> <td>   -6.011</td> <td> 0.000</td> <td>   -0.036</td> <td>   -0.018</td>
</tr>
<tr>
  <th>C(country_standard)[T.Sweden]</th>                           <td>    0.2328</td> <td>    0.003</td> <td>   68.173</td> <td> 0.000</td> <td>    0.226</td> <td>    0.240</td>
</tr>
<tr>
  <th>C(country_standard)[T.Switzerland]</th>                      <td>    0.2359</td> <td>    0.004</td> <td>   62.451</td> <td> 0.000</td> <td>    0.228</td> <td>    0.243</td>
</tr>
<tr>
  <th>C(country_standard)[T.Syria]</th>                            <td>    0.0002</td> <td>    0.009</td> <td>    0.025</td> <td> 0.980</td> <td>   -0.018</td> <td>    0.018</td>
</tr>
<tr>
  <th>C(country_standard)[T.São Tomé & Príncipe]</th>              <td>   -0.0125</td> <td>    0.007</td> <td>   -1.674</td> <td> 0.094</td> <td>   -0.027</td> <td>    0.002</td>
</tr>
<tr>
  <th>C(country_standard)[T.Taiwan]</th>                           <td>   -0.0292</td> <td>    0.006</td> <td>   -5.022</td> <td> 0.000</td> <td>   -0.041</td> <td>   -0.018</td>
</tr>
<tr>
  <th>C(country_standard)[T.Tajikistan]</th>                       <td>    0.0557</td> <td>    0.005</td> <td>   11.505</td> <td> 0.000</td> <td>    0.046</td> <td>    0.065</td>
</tr>
<tr>
  <th>C(country_standard)[T.Tanzania]</th>                         <td>   -0.0065</td> <td>    0.004</td> <td>   -1.638</td> <td> 0.101</td> <td>   -0.014</td> <td>    0.001</td>
</tr>
<tr>
  <th>C(country_standard)[T.Thailand]</th>                         <td>    0.0738</td> <td>    0.004</td> <td>   16.429</td> <td> 0.000</td> <td>    0.065</td> <td>    0.083</td>
</tr>
<tr>
  <th>C(country_standard)[T.Timor-Leste]</th>                      <td>   -0.0023</td> <td>    0.006</td> <td>   -0.352</td> <td> 0.725</td> <td>   -0.015</td> <td>    0.010</td>
</tr>
<tr>
  <th>C(country_standard)[T.Togo]</th>                             <td>   -0.0081</td> <td>    0.001</td> <td>  -11.416</td> <td> 0.000</td> <td>   -0.009</td> <td>   -0.007</td>
</tr>
<tr>
  <th>C(country_standard)[T.Tonga]</th>                            <td>   -0.0168</td> <td>    0.001</td> <td>  -18.592</td> <td> 0.000</td> <td>   -0.019</td> <td>   -0.015</td>
</tr>
<tr>
  <th>C(country_standard)[T.Trinidad & Tobago]</th>                <td>   -0.0440</td> <td>    0.005</td> <td>   -9.768</td> <td> 0.000</td> <td>   -0.053</td> <td>   -0.035</td>
</tr>
<tr>
  <th>C(country_standard)[T.Tunisia]</th>                          <td>   -0.1899</td> <td>    0.006</td> <td>  -30.482</td> <td> 0.000</td> <td>   -0.202</td> <td>   -0.178</td>
</tr>
<tr>
  <th>C(country_standard)[T.Turkey]</th>                           <td>    0.1780</td> <td>    0.004</td> <td>   41.209</td> <td> 0.000</td> <td>    0.170</td> <td>    0.186</td>
</tr>
<tr>
  <th>C(country_standard)[T.Turkmenistan]</th>                     <td>    0.0068</td> <td>    0.008</td> <td>    0.897</td> <td> 0.370</td> <td>   -0.008</td> <td>    0.022</td>
</tr>
<tr>
  <th>C(country_standard)[T.Turks & Caicos Islands]</th>           <td>   -0.0164</td> <td>    0.001</td> <td>  -19.074</td> <td> 0.000</td> <td>   -0.018</td> <td>   -0.015</td>
</tr>
<tr>
  <th>C(country_standard)[T.Tuvalu]</th>                           <td>   -0.0169</td> <td>    0.001</td> <td>  -18.146</td> <td> 0.000</td> <td>   -0.019</td> <td>   -0.015</td>
</tr>
<tr>
  <th>C(country_standard)[T.U.S. Virgin Islands]</th>              <td>   -0.0168</td> <td>    0.001</td> <td>  -18.619</td> <td> 0.000</td> <td>   -0.019</td> <td>   -0.015</td>
</tr>
<tr>
  <th>C(country_standard)[T.Uganda]</th>                           <td>   -0.0068</td> <td>    0.000</td> <td>  -30.669</td> <td> 0.000</td> <td>   -0.007</td> <td>   -0.006</td>
</tr>
<tr>
  <th>C(country_standard)[T.Ukraine]</th>                          <td>   -0.2163</td> <td>    0.000</td> <td> -484.480</td> <td> 0.000</td> <td>   -0.217</td> <td>   -0.215</td>
</tr>
<tr>
  <th>C(country_standard)[T.United Arab Emirates]</th>             <td>   -0.0229</td> <td>    0.012</td> <td>   -1.882</td> <td> 0.060</td> <td>   -0.047</td> <td>    0.001</td>
</tr>
<tr>
  <th>C(country_standard)[T.United Kingdom]</th>                   <td>   -0.0406</td> <td>    0.004</td> <td>   -9.717</td> <td> 0.000</td> <td>   -0.049</td> <td>   -0.032</td>
</tr>
<tr>
  <th>C(country_standard)[T.United States]</th>                    <td>   -0.0837</td> <td>    0.002</td> <td>  -49.276</td> <td> 0.000</td> <td>   -0.087</td> <td>   -0.080</td>
</tr>
<tr>
  <th>C(country_standard)[T.Uruguay]</th>                          <td>    0.1338</td> <td>    0.007</td> <td>   18.971</td> <td> 0.000</td> <td>    0.120</td> <td>    0.148</td>
</tr>
<tr>
  <th>C(country_standard)[T.Uzbekistan]</th>                       <td>    0.2547</td> <td>    0.006</td> <td>   43.025</td> <td> 0.000</td> <td>    0.243</td> <td>    0.266</td>
</tr>
<tr>
  <th>C(country_standard)[T.Vanuatu]</th>                          <td>   -0.0146</td> <td>    0.008</td> <td>   -1.814</td> <td> 0.070</td> <td>   -0.030</td> <td>    0.001</td>
</tr>
<tr>
  <th>C(country_standard)[T.Vatican City]</th>                     <td>   -0.0230</td> <td>    0.007</td> <td>   -3.356</td> <td> 0.001</td> <td>   -0.036</td> <td>   -0.010</td>
</tr>
<tr>
  <th>C(country_standard)[T.Venezuela]</th>                        <td>   -0.0090</td> <td>    0.005</td> <td>   -1.847</td> <td> 0.065</td> <td>   -0.019</td> <td>    0.001</td>
</tr>
<tr>
  <th>C(country_standard)[T.Vietnam]</th>                          <td>    0.4421</td> <td>    0.004</td> <td>  110.383</td> <td> 0.000</td> <td>    0.434</td> <td>    0.450</td>
</tr>
<tr>
  <th>C(country_standard)[T.Western Sahara]</th>                   <td>   -0.0148</td> <td>    0.002</td> <td>   -7.474</td> <td> 0.000</td> <td>   -0.019</td> <td>   -0.011</td>
</tr>
<tr>
  <th>C(country_standard)[T.Yemen]</th>                            <td>   -0.2690</td> <td>    0.005</td> <td>  -50.912</td> <td> 0.000</td> <td>   -0.279</td> <td>   -0.259</td>
</tr>
<tr>
  <th>C(country_standard)[T.Zambia]</th>                           <td>   -0.0711</td> <td>    0.002</td> <td>  -43.883</td> <td> 0.000</td> <td>   -0.074</td> <td>   -0.068</td>
</tr>
<tr>
  <th>C(country_standard)[T.Zimbabwe]</th>                         <td>   -0.1450</td> <td>    0.001</td> <td> -162.450</td> <td> 0.000</td> <td>   -0.147</td> <td>   -0.143</td>
</tr>
<tr>
  <th>transparency_index</th>                                      <td>    0.0456</td> <td>    0.033</td> <td>    1.397</td> <td> 0.162</td> <td>   -0.018</td> <td>    0.110</td>
</tr>
<tr>
  <th>gdp</th>                                                     <td> 3.966e-15</td> <td> 1.81e-16</td> <td>   21.858</td> <td> 0.000</td> <td> 3.61e-15</td> <td> 4.32e-15</td>
</tr>
<tr>
  <th>gdp_per_capita</th>                                          <td> 2.201e-07</td> <td> 1.18e-07</td> <td>    1.866</td> <td> 0.062</td> <td>-1.11e-08</td> <td> 4.51e-07</td>
</tr>
<tr>
  <th>gini_2020</th>                                               <td>    0.0773</td> <td>    0.011</td> <td>    6.843</td> <td> 0.000</td> <td>    0.055</td> <td>    0.099</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>762.005</td> <th>  Durbin-Watson:     </th> <td>   0.542</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>15230.852</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.472</td>  <th>  Prob(JB):          </th> <td>    0.00</td> 
</tr>
<tr>
  <th>Kurtosis:</th>      <td>13.011</td>  <th>  Cond. No.          </th> <td>4.96e+26</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors are robust to cluster correlation (cluster)<br/>[2] The smallest eigenvalue is 5.07e-26. This might indicate that there are<br/>strong multicollinearity problems or that the design matrix is singular.




```python
reg_effectiveness = smf.ols('effectiveness_index ~ transparency_index + gdp + gdp_per_capita + gini_2020 + C(country_standard)', ndi_df).fit(cov_type='cluster', cov_kwds={'groups': ndi_df['country_standard']})
```


```python
reg_effectiveness.summary()
```

    /Users/katiacordoba/opt/anaconda3/lib/python3.7/site-packages/statsmodels/base/model.py:1832: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 232, but rank is 228
      'rank is %d' % (J, J_), ValueWarning)





<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>    <td>effectiveness_index</td> <th>  R-squared:         </th>  <td>   0.916</td> 
</tr>
<tr>
  <th>Model:</th>                    <td>OLS</td>         <th>  Adj. R-squared:    </th>  <td>   0.916</td> 
</tr>
<tr>
  <th>Method:</th>              <td>Least Squares</td>    <th>  F-statistic:       </th>  <td>9.710e+06</td>
</tr>
<tr>
  <th>Date:</th>              <td>Fri, 05 Mar 2021</td>   <th>  Prob (F-statistic):</th>   <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                  <td>13:17:43</td>       <th>  Log-Likelihood:    </th>  <td>  5365.7</td> 
</tr>
<tr>
  <th>No. Observations:</th>       <td>  3615</td>        <th>  AIC:               </th> <td>-1.072e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>           <td>  3610</td>        <th>  BIC:               </th> <td>-1.069e+04</td>
</tr>
<tr>
  <th>Df Model:</th>               <td>     4</td>        <th>                     </th>      <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>        <td>cluster</td>       <th>                     </th>      <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
                             <td></td>                                <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>                                               <td>   -0.3934</td> <td>    0.005</td> <td>  -72.236</td> <td> 0.000</td> <td>   -0.404</td> <td>   -0.383</td>
</tr>
<tr>
  <th>C(country_standard)[T.Albania]</th>                          <td>    0.2240</td> <td>    0.004</td> <td>   50.160</td> <td> 0.000</td> <td>    0.215</td> <td>    0.233</td>
</tr>
<tr>
  <th>C(country_standard)[T.Algeria]</th>                          <td>   -0.0564</td> <td>    0.002</td> <td>  -23.677</td> <td> 0.000</td> <td>   -0.061</td> <td>   -0.052</td>
</tr>
<tr>
  <th>C(country_standard)[T.American Samoa]</th>                   <td>    0.0949</td> <td>    0.002</td> <td>   39.723</td> <td> 0.000</td> <td>    0.090</td> <td>    0.100</td>
</tr>
<tr>
  <th>C(country_standard)[T.Andorra]</th>                          <td>    0.1018</td> <td>    0.001</td> <td>  127.711</td> <td> 0.000</td> <td>    0.100</td> <td>    0.103</td>
</tr>
<tr>
  <th>C(country_standard)[T.Angola]</th>                           <td>    0.0273</td> <td>    0.002</td> <td>   14.773</td> <td> 0.000</td> <td>    0.024</td> <td>    0.031</td>
</tr>
<tr>
  <th>C(country_standard)[T.Anguilla]</th>                         <td>    0.0920</td> <td>    0.001</td> <td>   91.786</td> <td> 0.000</td> <td>    0.090</td> <td>    0.094</td>
</tr>
<tr>
  <th>C(country_standard)[T.Antigua & Barbuda]</th>                <td>   -0.0058</td> <td>    0.000</td> <td>  -13.240</td> <td> 0.000</td> <td>   -0.007</td> <td>   -0.005</td>
</tr>
<tr>
  <th>C(country_standard)[T.Argentina]</th>                        <td>    0.0736</td> <td>    0.002</td> <td>   31.087</td> <td> 0.000</td> <td>    0.069</td> <td>    0.078</td>
</tr>
<tr>
  <th>C(country_standard)[T.Armenia]</th>                          <td>    0.1787</td> <td>    0.001</td> <td>  119.902</td> <td> 0.000</td> <td>    0.176</td> <td>    0.182</td>
</tr>
<tr>
  <th>C(country_standard)[T.Aruba]</th>                            <td>    0.0946</td> <td>    0.002</td> <td>   46.626</td> <td> 0.000</td> <td>    0.091</td> <td>    0.099</td>
</tr>
<tr>
  <th>C(country_standard)[T.Australia]</th>                        <td>    0.2239</td> <td>    0.003</td> <td>   82.350</td> <td> 0.000</td> <td>    0.219</td> <td>    0.229</td>
</tr>
<tr>
  <th>C(country_standard)[T.Austria]</th>                          <td>    0.1095</td> <td>    0.002</td> <td>   55.049</td> <td> 0.000</td> <td>    0.106</td> <td>    0.113</td>
</tr>
<tr>
  <th>C(country_standard)[T.Azerbaijan]</th>                       <td>   -0.0133</td> <td>    0.003</td> <td>   -4.025</td> <td> 0.000</td> <td>   -0.020</td> <td>   -0.007</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bahamas]</th>                          <td>   -0.0778</td> <td>    0.001</td> <td> -107.397</td> <td> 0.000</td> <td>   -0.079</td> <td>   -0.076</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bahrain]</th>                          <td>   -0.2375</td> <td>    0.005</td> <td>  -46.771</td> <td> 0.000</td> <td>   -0.247</td> <td>   -0.228</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bangladesh]</th>                       <td>    0.2444</td> <td>    0.001</td> <td>  204.631</td> <td> 0.000</td> <td>    0.242</td> <td>    0.247</td>
</tr>
<tr>
  <th>C(country_standard)[T.Barbados]</th>                         <td>    0.0871</td> <td>    0.003</td> <td>   28.720</td> <td> 0.000</td> <td>    0.081</td> <td>    0.093</td>
</tr>
<tr>
  <th>C(country_standard)[T.Belarus]</th>                          <td>    0.0278</td> <td>    0.003</td> <td>    9.758</td> <td> 0.000</td> <td>    0.022</td> <td>    0.033</td>
</tr>
<tr>
  <th>C(country_standard)[T.Belgium]</th>                          <td>    0.2540</td> <td>    0.003</td> <td>   80.789</td> <td> 0.000</td> <td>    0.248</td> <td>    0.260</td>
</tr>
<tr>
  <th>C(country_standard)[T.Belize]</th>                           <td>    0.1593</td> <td>    0.001</td> <td>  158.946</td> <td> 0.000</td> <td>    0.157</td> <td>    0.161</td>
</tr>
<tr>
  <th>C(country_standard)[T.Benin]</th>                            <td>    0.3022</td> <td>    0.003</td> <td>   91.821</td> <td> 0.000</td> <td>    0.296</td> <td>    0.309</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bermuda]</th>                          <td>    0.0920</td> <td>    0.001</td> <td>   88.931</td> <td> 0.000</td> <td>    0.090</td> <td>    0.094</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bhutan]</th>                           <td>    0.2844</td> <td>    0.002</td> <td>  159.301</td> <td> 0.000</td> <td>    0.281</td> <td>    0.288</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bolivia]</th>                          <td>    0.1484</td> <td>    0.001</td> <td>  138.170</td> <td> 0.000</td> <td>    0.146</td> <td>    0.151</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bosnia & Herzegovina]</th>             <td>    0.1001</td> <td>    0.002</td> <td>   58.677</td> <td> 0.000</td> <td>    0.097</td> <td>    0.103</td>
</tr>
<tr>
  <th>C(country_standard)[T.Botswana]</th>                         <td>    0.2380</td> <td>    0.002</td> <td>  100.007</td> <td> 0.000</td> <td>    0.233</td> <td>    0.243</td>
</tr>
<tr>
  <th>C(country_standard)[T.Brazil]</th>                           <td>    0.1006</td> <td>    0.002</td> <td>   46.060</td> <td> 0.000</td> <td>    0.096</td> <td>    0.105</td>
</tr>
<tr>
  <th>C(country_standard)[T.British Virgin Islands]</th>           <td>    0.0946</td> <td>    0.002</td> <td>   44.886</td> <td> 0.000</td> <td>    0.091</td> <td>    0.099</td>
</tr>
<tr>
  <th>C(country_standard)[T.Brunei]</th>                           <td>   -0.5220</td> <td>    0.003</td> <td> -158.480</td> <td> 0.000</td> <td>   -0.528</td> <td>   -0.516</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bulgaria]</th>                         <td>    0.2274</td> <td>    0.003</td> <td>   73.936</td> <td> 0.000</td> <td>    0.221</td> <td>    0.233</td>
</tr>
<tr>
  <th>C(country_standard)[T.Burkina Faso]</th>                     <td>    0.1700</td> <td>    0.002</td> <td>   76.457</td> <td> 0.000</td> <td>    0.166</td> <td>    0.174</td>
</tr>
<tr>
  <th>C(country_standard)[T.Burundi]</th>                          <td>    0.0598</td> <td>    0.002</td> <td>   26.365</td> <td> 0.000</td> <td>    0.055</td> <td>    0.064</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cambodia]</th>                         <td>    0.2266</td> <td>    0.002</td> <td>  107.545</td> <td> 0.000</td> <td>    0.223</td> <td>    0.231</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cameroon]</th>                         <td>    0.0964</td> <td>    0.002</td> <td>   59.756</td> <td> 0.000</td> <td>    0.093</td> <td>    0.100</td>
</tr>
<tr>
  <th>C(country_standard)[T.Canada]</th>                           <td>    0.2015</td> <td>    0.002</td> <td>   99.759</td> <td> 0.000</td> <td>    0.198</td> <td>    0.205</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cape Verde]</th>                       <td>    0.4001</td> <td>    0.004</td> <td>   94.013</td> <td> 0.000</td> <td>    0.392</td> <td>    0.408</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cayman Islands]</th>                   <td>   -0.2351</td> <td>    0.001</td> <td> -159.416</td> <td> 0.000</td> <td>   -0.238</td> <td>   -0.232</td>
</tr>
<tr>
  <th>C(country_standard)[T.Central African Republic]</th>         <td>   -0.1219</td> <td>    0.001</td> <td>  -96.270</td> <td> 0.000</td> <td>   -0.124</td> <td>   -0.119</td>
</tr>
<tr>
  <th>C(country_standard)[T.Chad]</th>                             <td>   -0.1152</td> <td>    0.002</td> <td>  -52.224</td> <td> 0.000</td> <td>   -0.119</td> <td>   -0.111</td>
</tr>
<tr>
  <th>C(country_standard)[T.Channel Islands]</th>                  <td>    0.0987</td> <td>    0.006</td> <td>   15.684</td> <td> 0.000</td> <td>    0.086</td> <td>    0.111</td>
</tr>
<tr>
  <th>C(country_standard)[T.Chile]</th>                            <td>    0.3000</td> <td>    0.003</td> <td>   98.077</td> <td> 0.000</td> <td>    0.294</td> <td>    0.306</td>
</tr>
<tr>
  <th>C(country_standard)[T.China]</th>                            <td>   -0.1090</td> <td>    0.005</td> <td>  -21.107</td> <td> 0.000</td> <td>   -0.119</td> <td>   -0.099</td>
</tr>
<tr>
  <th>C(country_standard)[T.Colombia]</th>                         <td>    0.1984</td> <td>    0.001</td> <td>  141.766</td> <td> 0.000</td> <td>    0.196</td> <td>    0.201</td>
</tr>
<tr>
  <th>C(country_standard)[T.Comoros]</th>                          <td>   -0.0524</td> <td>    0.001</td> <td>  -49.445</td> <td> 0.000</td> <td>   -0.054</td> <td>   -0.050</td>
</tr>
<tr>
  <th>C(country_standard)[T.Congo - Brazzaville]</th>              <td>   -0.0171</td> <td>    0.002</td> <td>   -7.359</td> <td> 0.000</td> <td>   -0.022</td> <td>   -0.013</td>
</tr>
<tr>
  <th>C(country_standard)[T.Congo - Kinshasa]</th>                 <td>    0.2682</td> <td>    0.002</td> <td>  142.100</td> <td> 0.000</td> <td>    0.264</td> <td>    0.272</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cook Islands]</th>                     <td>    0.0946</td> <td>    0.002</td> <td>   45.092</td> <td> 0.000</td> <td>    0.091</td> <td>    0.099</td>
</tr>
<tr>
  <th>C(country_standard)[T.Costa Rica]</th>                       <td>    0.3000</td> <td>    0.004</td> <td>   73.826</td> <td> 0.000</td> <td>    0.292</td> <td>    0.308</td>
</tr>
<tr>
  <th>C(country_standard)[T.Croatia]</th>                          <td>    0.2444</td> <td>    0.003</td> <td>   79.410</td> <td> 0.000</td> <td>    0.238</td> <td>    0.250</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cuba]</th>                             <td>    0.0384</td> <td>    0.004</td> <td>   10.136</td> <td> 0.000</td> <td>    0.031</td> <td>    0.046</td>
</tr>
<tr>
  <th>C(country_standard)[T.Curaçao]</th>                          <td>    0.0946</td> <td>    0.002</td> <td>   45.171</td> <td> 0.000</td> <td>    0.091</td> <td>    0.099</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cyprus]</th>                           <td>    0.0911</td> <td>    0.002</td> <td>   45.458</td> <td> 0.000</td> <td>    0.087</td> <td>    0.095</td>
</tr>
<tr>
  <th>C(country_standard)[T.Czechia]</th>                          <td>    0.1505</td> <td>    0.002</td> <td>   61.988</td> <td> 0.000</td> <td>    0.146</td> <td>    0.155</td>
</tr>
<tr>
  <th>C(country_standard)[T.Côte d’Ivoire]</th>                    <td>    0.0611</td> <td>    0.002</td> <td>   38.088</td> <td> 0.000</td> <td>    0.058</td> <td>    0.064</td>
</tr>
<tr>
  <th>C(country_standard)[T.Denmark]</th>                          <td>    0.1333</td> <td>    0.002</td> <td>   60.235</td> <td> 0.000</td> <td>    0.129</td> <td>    0.138</td>
</tr>
<tr>
  <th>C(country_standard)[T.Djibouti]</th>                         <td>    0.0189</td> <td>    0.002</td> <td>    8.846</td> <td> 0.000</td> <td>    0.015</td> <td>    0.023</td>
</tr>
<tr>
  <th>C(country_standard)[T.Dominica]</th>                         <td>    0.1224</td> <td>    0.001</td> <td>  154.123</td> <td> 0.000</td> <td>    0.121</td> <td>    0.124</td>
</tr>
<tr>
  <th>C(country_standard)[T.Dominican Republic]</th>               <td>    0.1076</td> <td>    0.001</td> <td>   93.852</td> <td> 0.000</td> <td>    0.105</td> <td>    0.110</td>
</tr>
<tr>
  <th>C(country_standard)[T.Ecuador]</th>                          <td>    0.0930</td> <td>    0.001</td> <td>   98.260</td> <td> 0.000</td> <td>    0.091</td> <td>    0.095</td>
</tr>
<tr>
  <th>C(country_standard)[T.Egypt]</th>                            <td>    0.0184</td> <td>    0.002</td> <td>    7.597</td> <td> 0.000</td> <td>    0.014</td> <td>    0.023</td>
</tr>
<tr>
  <th>C(country_standard)[T.El Salvador]</th>                      <td>    0.2269</td> <td>    0.002</td> <td>   93.527</td> <td> 0.000</td> <td>    0.222</td> <td>    0.232</td>
</tr>
<tr>
  <th>C(country_standard)[T.Equatorial Guinea]</th>                <td>   -0.3422</td> <td>    0.004</td> <td>  -77.116</td> <td> 0.000</td> <td>   -0.351</td> <td>   -0.333</td>
</tr>
<tr>
  <th>C(country_standard)[T.Eritrea]</th>                          <td>    0.0845</td> <td>    0.004</td> <td>   20.374</td> <td> 0.000</td> <td>    0.076</td> <td>    0.093</td>
</tr>
<tr>
  <th>C(country_standard)[T.Estonia]</th>                          <td>    0.2441</td> <td>    0.003</td> <td>   73.330</td> <td> 0.000</td> <td>    0.238</td> <td>    0.251</td>
</tr>
<tr>
  <th>C(country_standard)[T.Eswatini]</th>                         <td>    0.0482</td> <td>    0.002</td> <td>   20.970</td> <td> 0.000</td> <td>    0.044</td> <td>    0.053</td>
</tr>
<tr>
  <th>C(country_standard)[T.Ethiopia]</th>                         <td>    0.2544</td> <td>    0.002</td> <td>  144.484</td> <td> 0.000</td> <td>    0.251</td> <td>    0.258</td>
</tr>
<tr>
  <th>C(country_standard)[T.Faroe Islands]</th>                    <td>    0.0919</td> <td>    0.001</td> <td>   84.809</td> <td> 0.000</td> <td>    0.090</td> <td>    0.094</td>
</tr>
<tr>
  <th>C(country_standard)[T.Fiji]</th>                             <td>    0.1670</td> <td>    0.001</td> <td>  158.728</td> <td> 0.000</td> <td>    0.165</td> <td>    0.169</td>
</tr>
<tr>
  <th>C(country_standard)[T.Finland]</th>                          <td>    0.2785</td> <td>    0.003</td> <td>  102.712</td> <td> 0.000</td> <td>    0.273</td> <td>    0.284</td>
</tr>
<tr>
  <th>C(country_standard)[T.France]</th>                           <td>    0.1305</td> <td>    0.002</td> <td>   56.270</td> <td> 0.000</td> <td>    0.126</td> <td>    0.135</td>
</tr>
<tr>
  <th>C(country_standard)[T.French Guiana]</th>                    <td>    0.0946</td> <td>    0.002</td> <td>   45.742</td> <td> 0.000</td> <td>    0.091</td> <td>    0.099</td>
</tr>
<tr>
  <th>C(country_standard)[T.French Polynesia]</th>                 <td>    0.0946</td> <td>    0.002</td> <td>   45.366</td> <td> 0.000</td> <td>    0.091</td> <td>    0.099</td>
</tr>
<tr>
  <th>C(country_standard)[T.Gabon]</th>                            <td>   -0.0829</td> <td>    0.002</td> <td>  -46.277</td> <td> 0.000</td> <td>   -0.086</td> <td>   -0.079</td>
</tr>
<tr>
  <th>C(country_standard)[T.Gambia]</th>                           <td>    0.1446</td> <td>    0.001</td> <td>  135.940</td> <td> 0.000</td> <td>    0.143</td> <td>    0.147</td>
</tr>
<tr>
  <th>C(country_standard)[T.Georgia]</th>                          <td>    0.1874</td> <td>    0.002</td> <td>   83.836</td> <td> 0.000</td> <td>    0.183</td> <td>    0.192</td>
</tr>
<tr>
  <th>C(country_standard)[T.Germany]</th>                          <td>    0.0045</td> <td>    0.002</td> <td>    2.456</td> <td> 0.014</td> <td>    0.001</td> <td>    0.008</td>
</tr>
<tr>
  <th>C(country_standard)[T.Ghana]</th>                            <td>    0.2937</td> <td>    0.004</td> <td>   75.127</td> <td> 0.000</td> <td>    0.286</td> <td>    0.301</td>
</tr>
<tr>
  <th>C(country_standard)[T.Gibraltar]</th>                        <td>    0.0920</td> <td>    0.001</td> <td>   93.599</td> <td> 0.000</td> <td>    0.090</td> <td>    0.094</td>
</tr>
<tr>
  <th>C(country_standard)[T.Greece]</th>                           <td>    0.2005</td> <td>    0.004</td> <td>   56.564</td> <td> 0.000</td> <td>    0.194</td> <td>    0.207</td>
</tr>
<tr>
  <th>C(country_standard)[T.Greenland]</th>                        <td>    0.0920</td> <td>    0.001</td> <td>   89.675</td> <td> 0.000</td> <td>    0.090</td> <td>    0.094</td>
</tr>
<tr>
  <th>C(country_standard)[T.Grenada]</th>                          <td>    0.0765</td> <td>    0.001</td> <td>  130.990</td> <td> 0.000</td> <td>    0.075</td> <td>    0.078</td>
</tr>
<tr>
  <th>C(country_standard)[T.Guam]</th>                             <td>    0.0946</td> <td>    0.002</td> <td>   45.604</td> <td> 0.000</td> <td>    0.091</td> <td>    0.099</td>
</tr>
<tr>
  <th>C(country_standard)[T.Guatemala]</th>                        <td>    0.2100</td> <td>    0.002</td> <td>  131.455</td> <td> 0.000</td> <td>    0.207</td> <td>    0.213</td>
</tr>
<tr>
  <th>C(country_standard)[T.Guernsey]</th>                         <td>    0.0594</td> <td>    0.034</td> <td>    1.733</td> <td> 0.083</td> <td>   -0.008</td> <td>    0.127</td>
</tr>
<tr>
  <th>C(country_standard)[T.Guinea]</th>                           <td>   -0.0585</td> <td>    0.001</td> <td>  -56.489</td> <td> 0.000</td> <td>   -0.061</td> <td>   -0.056</td>
</tr>
<tr>
  <th>C(country_standard)[T.Guinea-Bissau]</th>                    <td>   -0.0761</td> <td>    0.001</td> <td>  -62.197</td> <td> 0.000</td> <td>   -0.078</td> <td>   -0.074</td>
</tr>
<tr>
  <th>C(country_standard)[T.Guyana]</th>                           <td>    0.2243</td> <td>    0.003</td> <td>   85.539</td> <td> 0.000</td> <td>    0.219</td> <td>    0.229</td>
</tr>
<tr>
  <th>C(country_standard)[T.Haiti]</th>                            <td>   -0.0688</td> <td>    0.001</td> <td>  -70.450</td> <td> 0.000</td> <td>   -0.071</td> <td>   -0.067</td>
</tr>
<tr>
  <th>C(country_standard)[T.Honduras]</th>                         <td>    0.2184</td> <td>    0.001</td> <td>  201.423</td> <td> 0.000</td> <td>    0.216</td> <td>    0.220</td>
</tr>
<tr>
  <th>C(country_standard)[T.Hong Kong SAR China]</th>              <td>   -0.1401</td> <td>    0.002</td> <td>  -70.943</td> <td> 0.000</td> <td>   -0.144</td> <td>   -0.136</td>
</tr>
<tr>
  <th>C(country_standard)[T.Hungary]</th>                          <td>    0.1905</td> <td>    0.001</td> <td>  130.492</td> <td> 0.000</td> <td>    0.188</td> <td>    0.193</td>
</tr>
<tr>
  <th>C(country_standard)[T.Iceland]</th>                          <td>    0.2143</td> <td>    0.002</td> <td>   99.323</td> <td> 0.000</td> <td>    0.210</td> <td>    0.218</td>
</tr>
<tr>
  <th>C(country_standard)[T.India]</th>                            <td>    0.2222</td> <td>    0.002</td> <td>  131.835</td> <td> 0.000</td> <td>    0.219</td> <td>    0.225</td>
</tr>
<tr>
  <th>C(country_standard)[T.Indonesia]</th>                        <td>    0.1776</td> <td>    0.001</td> <td>  144.801</td> <td> 0.000</td> <td>    0.175</td> <td>    0.180</td>
</tr>
<tr>
  <th>C(country_standard)[T.Iran]</th>                             <td>   -0.0381</td> <td>    0.002</td> <td>  -16.079</td> <td> 0.000</td> <td>   -0.043</td> <td>   -0.033</td>
</tr>
<tr>
  <th>C(country_standard)[T.Iraq]</th>                             <td>   -0.1481</td> <td>    0.001</td> <td> -137.456</td> <td> 0.000</td> <td>   -0.150</td> <td>   -0.146</td>
</tr>
<tr>
  <th>C(country_standard)[T.Ireland]</th>                          <td>   -0.1848</td> <td>    0.001</td> <td> -140.009</td> <td> 0.000</td> <td>   -0.187</td> <td>   -0.182</td>
</tr>
<tr>
  <th>C(country_standard)[T.Isle of Man]</th>                      <td>    0.0920</td> <td>    0.001</td> <td>   91.600</td> <td> 0.000</td> <td>    0.090</td> <td>    0.094</td>
</tr>
<tr>
  <th>C(country_standard)[T.Israel]</th>                           <td>    0.1542</td> <td>    0.002</td> <td>   88.666</td> <td> 0.000</td> <td>    0.151</td> <td>    0.158</td>
</tr>
<tr>
  <th>C(country_standard)[T.Italy]</th>                            <td>    0.0565</td> <td>    0.003</td> <td>   19.308</td> <td> 0.000</td> <td>    0.051</td> <td>    0.062</td>
</tr>
<tr>
  <th>C(country_standard)[T.Jamaica]</th>                          <td>    0.3078</td> <td>    0.003</td> <td>   88.457</td> <td> 0.000</td> <td>    0.301</td> <td>    0.315</td>
</tr>
<tr>
  <th>C(country_standard)[T.Japan]</th>                            <td>    0.1982</td> <td>    0.003</td> <td>   77.718</td> <td> 0.000</td> <td>    0.193</td> <td>    0.203</td>
</tr>
<tr>
  <th>C(country_standard)[T.Jersey]</th>                           <td>    0.0920</td> <td>    0.001</td> <td>   90.115</td> <td> 0.000</td> <td>    0.090</td> <td>    0.094</td>
</tr>
<tr>
  <th>C(country_standard)[T.Jordan]</th>                           <td>    0.2059</td> <td>    0.001</td> <td>  209.029</td> <td> 0.000</td> <td>    0.204</td> <td>    0.208</td>
</tr>
<tr>
  <th>C(country_standard)[T.Kazakhstan]</th>                       <td>   -0.1610</td> <td>    0.003</td> <td>  -46.776</td> <td> 0.000</td> <td>   -0.168</td> <td>   -0.154</td>
</tr>
<tr>
  <th>C(country_standard)[T.Kenya]</th>                            <td>    0.2001</td> <td>    0.002</td> <td>  111.148</td> <td> 0.000</td> <td>    0.197</td> <td>    0.204</td>
</tr>
<tr>
  <th>C(country_standard)[T.Kiribati]</th>                         <td>    0.0946</td> <td>    0.002</td> <td>   45.669</td> <td> 0.000</td> <td>    0.091</td> <td>    0.099</td>
</tr>
<tr>
  <th>C(country_standard)[T.Kosovo]</th>                           <td>    0.0987</td> <td>    0.006</td> <td>   15.680</td> <td> 0.000</td> <td>    0.086</td> <td>    0.111</td>
</tr>
<tr>
  <th>C(country_standard)[T.Kuwait]</th>                           <td>   -0.4748</td> <td>    0.003</td> <td> -155.765</td> <td> 0.000</td> <td>   -0.481</td> <td>   -0.469</td>
</tr>
<tr>
  <th>C(country_standard)[T.Kyrgyzstan]</th>                       <td>    0.1073</td> <td>    0.002</td> <td>   58.614</td> <td> 0.000</td> <td>    0.104</td> <td>    0.111</td>
</tr>
<tr>
  <th>C(country_standard)[T.Laos]</th>                             <td>    0.0079</td> <td>    0.003</td> <td>    2.293</td> <td> 0.022</td> <td>    0.001</td> <td>    0.015</td>
</tr>
<tr>
  <th>C(country_standard)[T.Latvia]</th>                           <td>    0.1129</td> <td>    0.002</td> <td>   45.645</td> <td> 0.000</td> <td>    0.108</td> <td>    0.118</td>
</tr>
<tr>
  <th>C(country_standard)[T.Lebanon]</th>                          <td>   -0.0895</td> <td>    0.001</td> <td> -155.711</td> <td> 0.000</td> <td>   -0.091</td> <td>   -0.088</td>
</tr>
<tr>
  <th>C(country_standard)[T.Lesotho]</th>                          <td>    0.1856</td> <td>    0.002</td> <td>   97.556</td> <td> 0.000</td> <td>    0.182</td> <td>    0.189</td>
</tr>
<tr>
  <th>C(country_standard)[T.Liberia]</th>                          <td>    0.0038</td> <td>    0.002</td> <td>    1.648</td> <td> 0.099</td> <td>   -0.001</td> <td>    0.008</td>
</tr>
<tr>
  <th>C(country_standard)[T.Libya]</th>                            <td>   -0.1427</td> <td>    0.001</td> <td>  -95.912</td> <td> 0.000</td> <td>   -0.146</td> <td>   -0.140</td>
</tr>
<tr>
  <th>C(country_standard)[T.Liechtenstein]</th>                    <td>    0.1018</td> <td>    0.001</td> <td>  135.319</td> <td> 0.000</td> <td>    0.100</td> <td>    0.103</td>
</tr>
<tr>
  <th>C(country_standard)[T.Lithuania]</th>                        <td>    0.1853</td> <td>    0.003</td> <td>   58.810</td> <td> 0.000</td> <td>    0.179</td> <td>    0.191</td>
</tr>
<tr>
  <th>C(country_standard)[T.Luxembourg]</th>                       <td>   -0.2862</td> <td>    0.002</td> <td> -173.714</td> <td> 0.000</td> <td>   -0.289</td> <td>   -0.283</td>
</tr>
<tr>
  <th>C(country_standard)[T.Macao SAR China]</th>                  <td>    0.1002</td> <td>    0.001</td> <td>   93.163</td> <td> 0.000</td> <td>    0.098</td> <td>    0.102</td>
</tr>
<tr>
  <th>C(country_standard)[T.Madagascar]</th>                       <td>    0.1245</td> <td>    0.002</td> <td>   82.343</td> <td> 0.000</td> <td>    0.122</td> <td>    0.127</td>
</tr>
<tr>
  <th>C(country_standard)[T.Malawi]</th>                           <td>    0.2101</td> <td>    0.002</td> <td>   99.003</td> <td> 0.000</td> <td>    0.206</td> <td>    0.214</td>
</tr>
<tr>
  <th>C(country_standard)[T.Malaysia]</th>                         <td>    0.1648</td> <td>    0.001</td> <td>  116.858</td> <td> 0.000</td> <td>    0.162</td> <td>    0.168</td>
</tr>
<tr>
  <th>C(country_standard)[T.Maldives]</th>                         <td>    0.1446</td> <td>    0.001</td> <td>  202.029</td> <td> 0.000</td> <td>    0.143</td> <td>    0.146</td>
</tr>
<tr>
  <th>C(country_standard)[T.Mali]</th>                             <td>    0.1842</td> <td>    0.002</td> <td>  104.090</td> <td> 0.000</td> <td>    0.181</td> <td>    0.188</td>
</tr>
<tr>
  <th>C(country_standard)[T.Malta]</th>                            <td>    0.2502</td> <td>    0.002</td> <td>  115.820</td> <td> 0.000</td> <td>    0.246</td> <td>    0.254</td>
</tr>
<tr>
  <th>C(country_standard)[T.Marshall Islands]</th>                 <td>    0.2652</td> <td>    0.002</td> <td>  156.353</td> <td> 0.000</td> <td>    0.262</td> <td>    0.269</td>
</tr>
<tr>
  <th>C(country_standard)[T.Martinique]</th>                       <td>    0.0946</td> <td>    0.002</td> <td>   46.390</td> <td> 0.000</td> <td>    0.091</td> <td>    0.099</td>
</tr>
<tr>
  <th>C(country_standard)[T.Mauritania]</th>                       <td>    0.1760</td> <td>    0.001</td> <td>  145.805</td> <td> 0.000</td> <td>    0.174</td> <td>    0.178</td>
</tr>
<tr>
  <th>C(country_standard)[T.Mauritius]</th>                        <td>    0.3952</td> <td>    0.004</td> <td>  111.738</td> <td> 0.000</td> <td>    0.388</td> <td>    0.402</td>
</tr>
<tr>
  <th>C(country_standard)[T.Mexico]</th>                           <td>    0.1430</td> <td>    0.001</td> <td>  183.758</td> <td> 0.000</td> <td>    0.141</td> <td>    0.144</td>
</tr>
<tr>
  <th>C(country_standard)[T.Micronesia (Federated States of)]</th> <td>    0.0921</td> <td>    0.001</td> <td>   96.572</td> <td> 0.000</td> <td>    0.090</td> <td>    0.094</td>
</tr>
<tr>
  <th>C(country_standard)[T.Moldova]</th>                          <td>    0.2515</td> <td>    0.002</td> <td>  107.250</td> <td> 0.000</td> <td>    0.247</td> <td>    0.256</td>
</tr>
<tr>
  <th>C(country_standard)[T.Monaco]</th>                           <td>    0.1018</td> <td>    0.001</td> <td>  135.383</td> <td> 0.000</td> <td>    0.100</td> <td>    0.103</td>
</tr>
<tr>
  <th>C(country_standard)[T.Mongolia]</th>                         <td>    0.2317</td> <td>    0.003</td> <td>   72.345</td> <td> 0.000</td> <td>    0.225</td> <td>    0.238</td>
</tr>
<tr>
  <th>C(country_standard)[T.Montenegro]</th>                       <td>    0.2241</td> <td>    0.002</td> <td>  132.937</td> <td> 0.000</td> <td>    0.221</td> <td>    0.227</td>
</tr>
<tr>
  <th>C(country_standard)[T.Morocco]</th>                          <td>    0.1530</td> <td>    0.001</td> <td>  147.374</td> <td> 0.000</td> <td>    0.151</td> <td>    0.155</td>
</tr>
<tr>
  <th>C(country_standard)[T.Mozambique]</th>                       <td>    0.1735</td> <td>    0.002</td> <td>  113.259</td> <td> 0.000</td> <td>    0.171</td> <td>    0.177</td>
</tr>
<tr>
  <th>C(country_standard)[T.Myanmar (Burma)]</th>                  <td>    0.1056</td> <td>    0.001</td> <td>   76.848</td> <td> 0.000</td> <td>    0.103</td> <td>    0.108</td>
</tr>
<tr>
  <th>C(country_standard)[T.Namibia]</th>                          <td>    0.1787</td> <td>    0.003</td> <td>   66.161</td> <td> 0.000</td> <td>    0.173</td> <td>    0.184</td>
</tr>
<tr>
  <th>C(country_standard)[T.Nauru]</th>                            <td>    0.0946</td> <td>    0.002</td> <td>   45.289</td> <td> 0.000</td> <td>    0.091</td> <td>    0.099</td>
</tr>
<tr>
  <th>C(country_standard)[T.Nepal]</th>                            <td>    0.1827</td> <td>    0.002</td> <td>   82.957</td> <td> 0.000</td> <td>    0.178</td> <td>    0.187</td>
</tr>
<tr>
  <th>C(country_standard)[T.Netherlands]</th>                      <td>    0.0005</td> <td>    0.002</td> <td>    0.297</td> <td> 0.766</td> <td>   -0.003</td> <td>    0.004</td>
</tr>
<tr>
  <th>C(country_standard)[T.New Caledonia]</th>                    <td>    0.0946</td> <td>    0.002</td> <td>   46.190</td> <td> 0.000</td> <td>    0.091</td> <td>    0.099</td>
</tr>
<tr>
  <th>C(country_standard)[T.New Zealand]</th>                      <td>    0.3494</td> <td>    0.003</td> <td>  106.546</td> <td> 0.000</td> <td>    0.343</td> <td>    0.356</td>
</tr>
<tr>
  <th>C(country_standard)[T.Nicaragua]</th>                        <td>    0.0774</td> <td>    0.002</td> <td>   43.617</td> <td> 0.000</td> <td>    0.074</td> <td>    0.081</td>
</tr>
<tr>
  <th>C(country_standard)[T.Niger]</th>                            <td>    0.0740</td> <td>    0.002</td> <td>   31.096</td> <td> 0.000</td> <td>    0.069</td> <td>    0.079</td>
</tr>
<tr>
  <th>C(country_standard)[T.Nigeria]</th>                          <td>   -0.0073</td> <td>    0.001</td> <td>   -6.910</td> <td> 0.000</td> <td>   -0.009</td> <td>   -0.005</td>
</tr>
<tr>
  <th>C(country_standard)[T.Niue]</th>                             <td>    0.0946</td> <td>    0.002</td> <td>   46.384</td> <td> 0.000</td> <td>    0.091</td> <td>    0.099</td>
</tr>
<tr>
  <th>C(country_standard)[T.North Korea]</th>                      <td>   -0.2229</td> <td>    0.001</td> <td> -217.272</td> <td> 0.000</td> <td>   -0.225</td> <td>   -0.221</td>
</tr>
<tr>
  <th>C(country_standard)[T.North Macedonia]</th>                  <td>    0.1506</td> <td>    0.001</td> <td>  135.777</td> <td> 0.000</td> <td>    0.148</td> <td>    0.153</td>
</tr>
<tr>
  <th>C(country_standard)[T.Northern Mariana Islands]</th>         <td>    0.0946</td> <td>    0.002</td> <td>   44.992</td> <td> 0.000</td> <td>    0.091</td> <td>    0.099</td>
</tr>
<tr>
  <th>C(country_standard)[T.Norway]</th>                           <td>   -0.0352</td> <td>    0.002</td> <td>  -20.950</td> <td> 0.000</td> <td>   -0.038</td> <td>   -0.032</td>
</tr>
<tr>
  <th>C(country_standard)[T.Oman]</th>                             <td>   -0.1940</td> <td>    0.004</td> <td>  -49.574</td> <td> 0.000</td> <td>   -0.202</td> <td>   -0.186</td>
</tr>
<tr>
  <th>C(country_standard)[T.Pakistan]</th>                         <td>    0.2553</td> <td>    0.002</td> <td>  140.288</td> <td> 0.000</td> <td>    0.252</td> <td>    0.259</td>
</tr>
<tr>
  <th>C(country_standard)[T.Palau]</th>                            <td>    0.0946</td> <td>    0.002</td> <td>   46.396</td> <td> 0.000</td> <td>    0.091</td> <td>    0.099</td>
</tr>
<tr>
  <th>C(country_standard)[T.Palestinian Territories]</th>          <td>    0.2502</td> <td>    0.002</td> <td>  111.817</td> <td> 0.000</td> <td>    0.246</td> <td>    0.255</td>
</tr>
<tr>
  <th>C(country_standard)[T.Panama]</th>                           <td>    0.1207</td> <td>    0.002</td> <td>   56.789</td> <td> 0.000</td> <td>    0.117</td> <td>    0.125</td>
</tr>
<tr>
  <th>C(country_standard)[T.Papua New Guinea]</th>                 <td>    0.2263</td> <td>    0.002</td> <td>  136.684</td> <td> 0.000</td> <td>    0.223</td> <td>    0.230</td>
</tr>
<tr>
  <th>C(country_standard)[T.Paraguay]</th>                         <td>    0.1324</td> <td>    0.001</td> <td>  104.150</td> <td> 0.000</td> <td>    0.130</td> <td>    0.135</td>
</tr>
<tr>
  <th>C(country_standard)[T.Peru]</th>                             <td>    0.0919</td> <td>    0.003</td> <td>   33.007</td> <td> 0.000</td> <td>    0.086</td> <td>    0.097</td>
</tr>
<tr>
  <th>C(country_standard)[T.Philippines]</th>                      <td>    0.1486</td> <td>    0.001</td> <td>  147.934</td> <td> 0.000</td> <td>    0.147</td> <td>    0.151</td>
</tr>
<tr>
  <th>C(country_standard)[T.Poland]</th>                           <td>    0.1644</td> <td>    0.003</td> <td>   60.073</td> <td> 0.000</td> <td>    0.159</td> <td>    0.170</td>
</tr>
<tr>
  <th>C(country_standard)[T.Portugal]</th>                         <td>    0.2594</td> <td>    0.003</td> <td>   76.726</td> <td> 0.000</td> <td>    0.253</td> <td>    0.266</td>
</tr>
<tr>
  <th>C(country_standard)[T.Puerto Rico]</th>                      <td>    0.0946</td> <td>    0.002</td> <td>   46.720</td> <td> 0.000</td> <td>    0.091</td> <td>    0.099</td>
</tr>
<tr>
  <th>C(country_standard)[T.Qatar]</th>                            <td>   -0.8731</td> <td>    0.010</td> <td>  -88.751</td> <td> 0.000</td> <td>   -0.892</td> <td>   -0.854</td>
</tr>
<tr>
  <th>C(country_standard)[T.Romania]</th>                          <td>    0.1221</td> <td>    0.003</td> <td>   46.993</td> <td> 0.000</td> <td>    0.117</td> <td>    0.127</td>
</tr>
<tr>
  <th>C(country_standard)[T.Russia]</th>                           <td>   -0.2667</td> <td>    0.004</td> <td>  -67.062</td> <td> 0.000</td> <td>   -0.274</td> <td>   -0.259</td>
</tr>
<tr>
  <th>C(country_standard)[T.Rwanda]</th>                           <td>    0.2463</td> <td>    0.002</td> <td>  138.090</td> <td> 0.000</td> <td>    0.243</td> <td>    0.250</td>
</tr>
<tr>
  <th>C(country_standard)[T.Réunion]</th>                          <td>    0.0947</td> <td>    0.002</td> <td>   44.551</td> <td> 0.000</td> <td>    0.090</td> <td>    0.099</td>
</tr>
<tr>
  <th>C(country_standard)[T.Saint Martin (French part)]</th>       <td>    0.0946</td> <td>    0.002</td> <td>   45.766</td> <td> 0.000</td> <td>    0.091</td> <td>    0.099</td>
</tr>
<tr>
  <th>C(country_standard)[T.Samoa]</th>                            <td>    0.2261</td> <td>    0.001</td> <td>  156.305</td> <td> 0.000</td> <td>    0.223</td> <td>    0.229</td>
</tr>
<tr>
  <th>C(country_standard)[T.San Marino]</th>                       <td>   -0.3106</td> <td>    0.002</td> <td> -160.886</td> <td> 0.000</td> <td>   -0.314</td> <td>   -0.307</td>
</tr>
<tr>
  <th>C(country_standard)[T.Saudi Arabia]</th>                     <td>   -0.4366</td> <td>    0.007</td> <td>  -61.680</td> <td> 0.000</td> <td>   -0.450</td> <td>   -0.423</td>
</tr>
<tr>
  <th>C(country_standard)[T.Senegal]</th>                          <td>    0.2681</td> <td>    0.003</td> <td>   82.105</td> <td> 0.000</td> <td>    0.262</td> <td>    0.274</td>
</tr>
<tr>
  <th>C(country_standard)[T.Serbia]</th>                           <td>    0.2068</td> <td>    0.002</td> <td>  111.627</td> <td> 0.000</td> <td>    0.203</td> <td>    0.210</td>
</tr>
<tr>
  <th>C(country_standard)[T.Seychelles]</th>                       <td>    0.0207</td> <td>    0.000</td> <td>   46.861</td> <td> 0.000</td> <td>    0.020</td> <td>    0.022</td>
</tr>
<tr>
  <th>C(country_standard)[T.Sierra Leone]</th>                     <td>    0.0563</td> <td>    0.002</td> <td>   26.646</td> <td> 0.000</td> <td>    0.052</td> <td>    0.060</td>
</tr>
<tr>
  <th>C(country_standard)[T.Singapore]</th>                        <td>   -0.3011</td> <td>    0.005</td> <td>  -66.673</td> <td> 0.000</td> <td>   -0.310</td> <td>   -0.292</td>
</tr>
<tr>
  <th>C(country_standard)[T.Sint Maarten]</th>                     <td>    0.0946</td> <td>    0.002</td> <td>   46.639</td> <td> 0.000</td> <td>    0.091</td> <td>    0.099</td>
</tr>
<tr>
  <th>C(country_standard)[T.Slovakia]</th>                         <td>    0.3708</td> <td>    0.004</td> <td>   94.472</td> <td> 0.000</td> <td>    0.363</td> <td>    0.378</td>
</tr>
<tr>
  <th>C(country_standard)[T.Slovenia]</th>                         <td>    0.2469</td> <td>    0.003</td> <td>   80.543</td> <td> 0.000</td> <td>    0.241</td> <td>    0.253</td>
</tr>
<tr>
  <th>C(country_standard)[T.Solomon Islands]</th>                  <td>    0.3098</td> <td>    0.003</td> <td>  101.369</td> <td> 0.000</td> <td>    0.304</td> <td>    0.316</td>
</tr>
<tr>
  <th>C(country_standard)[T.Somalia]</th>                          <td>    0.0850</td> <td>    0.003</td> <td>   28.632</td> <td> 0.000</td> <td>    0.079</td> <td>    0.091</td>
</tr>
<tr>
  <th>C(country_standard)[T.South Africa]</th>                     <td>    0.2895</td> <td>    0.003</td> <td>  106.262</td> <td> 0.000</td> <td>    0.284</td> <td>    0.295</td>
</tr>
<tr>
  <th>C(country_standard)[T.South Korea]</th>                      <td>    0.2026</td> <td>    0.001</td> <td>  151.435</td> <td> 0.000</td> <td>    0.200</td> <td>    0.205</td>
</tr>
<tr>
  <th>C(country_standard)[T.South Sudan]</th>                      <td>    0.2795</td> <td>    0.003</td> <td>   92.285</td> <td> 0.000</td> <td>    0.274</td> <td>    0.285</td>
</tr>
<tr>
  <th>C(country_standard)[T.Spain]</th>                            <td>    0.1718</td> <td>    0.003</td> <td>   63.190</td> <td> 0.000</td> <td>    0.166</td> <td>    0.177</td>
</tr>
<tr>
  <th>C(country_standard)[T.Sri Lanka]</th>                        <td>    0.2290</td> <td>    0.001</td> <td>  196.426</td> <td> 0.000</td> <td>    0.227</td> <td>    0.231</td>
</tr>
<tr>
  <th>C(country_standard)[T.St. Helena]</th>                       <td>    0.0594</td> <td>    0.034</td> <td>    1.734</td> <td> 0.083</td> <td>   -0.008</td> <td>    0.127</td>
</tr>
<tr>
  <th>C(country_standard)[T.St. Kitts & Nevis]</th>                <td>    0.0390</td> <td>    0.000</td> <td>   83.928</td> <td> 0.000</td> <td>    0.038</td> <td>    0.040</td>
</tr>
<tr>
  <th>C(country_standard)[T.St. Lucia]</th>                        <td>    0.1661</td> <td>    0.001</td> <td>  150.478</td> <td> 0.000</td> <td>    0.164</td> <td>    0.168</td>
</tr>
<tr>
  <th>C(country_standard)[T.St. Vincent & Grenadines]</th>         <td>    0.1151</td> <td>    0.001</td> <td>  149.451</td> <td> 0.000</td> <td>    0.114</td> <td>    0.117</td>
</tr>
<tr>
  <th>C(country_standard)[T.Sudan]</th>                            <td>   -0.0070</td> <td>    0.002</td> <td>   -2.966</td> <td> 0.003</td> <td>   -0.012</td> <td>   -0.002</td>
</tr>
<tr>
  <th>C(country_standard)[T.Suriname]</th>                         <td>    0.1131</td> <td>    0.002</td> <td>   47.903</td> <td> 0.000</td> <td>    0.108</td> <td>    0.118</td>
</tr>
<tr>
  <th>C(country_standard)[T.Sweden]</th>                           <td>    0.1015</td> <td>    0.002</td> <td>   51.300</td> <td> 0.000</td> <td>    0.098</td> <td>    0.105</td>
</tr>
<tr>
  <th>C(country_standard)[T.Switzerland]</th>                      <td>    0.1070</td> <td>    0.002</td> <td>   53.035</td> <td> 0.000</td> <td>    0.103</td> <td>    0.111</td>
</tr>
<tr>
  <th>C(country_standard)[T.Syria]</th>                            <td>   -0.2158</td> <td>    0.005</td> <td>  -45.911</td> <td> 0.000</td> <td>   -0.225</td> <td>   -0.207</td>
</tr>
<tr>
  <th>C(country_standard)[T.São Tomé & Príncipe]</th>              <td>    0.3245</td> <td>    0.004</td> <td>   81.997</td> <td> 0.000</td> <td>    0.317</td> <td>    0.332</td>
</tr>
<tr>
  <th>C(country_standard)[T.Taiwan]</th>                           <td>    0.3434</td> <td>    0.003</td> <td>  115.438</td> <td> 0.000</td> <td>    0.338</td> <td>    0.349</td>
</tr>
<tr>
  <th>C(country_standard)[T.Tajikistan]</th>                       <td>    0.0179</td> <td>    0.003</td> <td>    6.518</td> <td> 0.000</td> <td>    0.012</td> <td>    0.023</td>
</tr>
<tr>
  <th>C(country_standard)[T.Tanzania]</th>                         <td>    0.2511</td> <td>    0.002</td> <td>  106.377</td> <td> 0.000</td> <td>    0.246</td> <td>    0.256</td>
</tr>
<tr>
  <th>C(country_standard)[T.Thailand]</th>                         <td>    0.0706</td> <td>    0.002</td> <td>   30.298</td> <td> 0.000</td> <td>    0.066</td> <td>    0.075</td>
</tr>
<tr>
  <th>C(country_standard)[T.Timor-Leste]</th>                      <td>    0.3759</td> <td>    0.004</td> <td>  106.624</td> <td> 0.000</td> <td>    0.369</td> <td>    0.383</td>
</tr>
<tr>
  <th>C(country_standard)[T.Togo]</th>                             <td>   -0.0400</td> <td>    0.001</td> <td>  -35.751</td> <td> 0.000</td> <td>   -0.042</td> <td>   -0.038</td>
</tr>
<tr>
  <th>C(country_standard)[T.Tonga]</th>                            <td>    0.0946</td> <td>    0.002</td> <td>   45.015</td> <td> 0.000</td> <td>    0.091</td> <td>    0.099</td>
</tr>
<tr>
  <th>C(country_standard)[T.Trinidad & Tobago]</th>                <td>    0.1235</td> <td>    0.002</td> <td>   53.635</td> <td> 0.000</td> <td>    0.119</td> <td>    0.128</td>
</tr>
<tr>
  <th>C(country_standard)[T.Tunisia]</th>                          <td>    0.2035</td> <td>    0.003</td> <td>   62.270</td> <td> 0.000</td> <td>    0.197</td> <td>    0.210</td>
</tr>
<tr>
  <th>C(country_standard)[T.Turkey]</th>                           <td>    0.0382</td> <td>    0.002</td> <td>   17.267</td> <td> 0.000</td> <td>    0.034</td> <td>    0.043</td>
</tr>
<tr>
  <th>C(country_standard)[T.Turkmenistan]</th>                     <td>   -0.1832</td> <td>    0.005</td> <td>  -37.624</td> <td> 0.000</td> <td>   -0.193</td> <td>   -0.174</td>
</tr>
<tr>
  <th>C(country_standard)[T.Turks & Caicos Islands]</th>           <td>    0.0919</td> <td>    0.001</td> <td>   86.450</td> <td> 0.000</td> <td>    0.090</td> <td>    0.094</td>
</tr>
<tr>
  <th>C(country_standard)[T.Tuvalu]</th>                           <td>    0.0946</td> <td>    0.002</td> <td>   45.319</td> <td> 0.000</td> <td>    0.091</td> <td>    0.099</td>
</tr>
<tr>
  <th>C(country_standard)[T.U.S. Virgin Islands]</th>              <td>    0.0946</td> <td>    0.002</td> <td>   46.007</td> <td> 0.000</td> <td>    0.091</td> <td>    0.099</td>
</tr>
<tr>
  <th>C(country_standard)[T.Uganda]</th>                           <td>    0.1592</td> <td>    0.001</td> <td>  151.656</td> <td> 0.000</td> <td>    0.157</td> <td>    0.161</td>
</tr>
<tr>
  <th>C(country_standard)[T.Ukraine]</th>                          <td>    0.0269</td> <td>    0.001</td> <td>   36.830</td> <td> 0.000</td> <td>    0.025</td> <td>    0.028</td>
</tr>
<tr>
  <th>C(country_standard)[T.United Arab Emirates]</th>             <td>   -0.4411</td> <td>    0.006</td> <td>  -70.316</td> <td> 0.000</td> <td>   -0.453</td> <td>   -0.429</td>
</tr>
<tr>
  <th>C(country_standard)[T.United Kingdom]</th>                   <td>    0.1042</td> <td>    0.002</td> <td>   48.116</td> <td> 0.000</td> <td>    0.100</td> <td>    0.108</td>
</tr>
<tr>
  <th>C(country_standard)[T.United States]</th>                    <td>   -0.2004</td> <td>    0.001</td> <td> -162.843</td> <td> 0.000</td> <td>   -0.203</td> <td>   -0.198</td>
</tr>
<tr>
  <th>C(country_standard)[T.Uruguay]</th>                          <td>    0.3318</td> <td>    0.004</td> <td>   92.006</td> <td> 0.000</td> <td>    0.325</td> <td>    0.339</td>
</tr>
<tr>
  <th>C(country_standard)[T.Uzbekistan]</th>                       <td>   -0.0236</td> <td>    0.003</td> <td>   -7.573</td> <td> 0.000</td> <td>   -0.030</td> <td>   -0.018</td>
</tr>
<tr>
  <th>C(country_standard)[T.Vanuatu]</th>                          <td>    0.3149</td> <td>    0.004</td> <td>   74.289</td> <td> 0.000</td> <td>    0.307</td> <td>    0.323</td>
</tr>
<tr>
  <th>C(country_standard)[T.Vatican City]</th>                     <td>    0.1291</td> <td>    0.038</td> <td>    3.370</td> <td> 0.001</td> <td>    0.054</td> <td>    0.204</td>
</tr>
<tr>
  <th>C(country_standard)[T.Venezuela]</th>                        <td>   -0.1046</td> <td>    0.003</td> <td>  -40.942</td> <td> 0.000</td> <td>   -0.110</td> <td>   -0.100</td>
</tr>
<tr>
  <th>C(country_standard)[T.Vietnam]</th>                          <td>    0.1315</td> <td>    0.002</td> <td>   59.755</td> <td> 0.000</td> <td>    0.127</td> <td>    0.136</td>
</tr>
<tr>
  <th>C(country_standard)[T.Western Sahara]</th>                   <td>    0.0828</td> <td>    0.010</td> <td>    8.188</td> <td> 0.000</td> <td>    0.063</td> <td>    0.103</td>
</tr>
<tr>
  <th>C(country_standard)[T.Yemen]</th>                            <td>   -0.1122</td> <td>    0.003</td> <td>  -38.907</td> <td> 0.000</td> <td>   -0.118</td> <td>   -0.107</td>
</tr>
<tr>
  <th>C(country_standard)[T.Zambia]</th>                           <td>    0.1326</td> <td>    0.001</td> <td>  112.330</td> <td> 0.000</td> <td>    0.130</td> <td>    0.135</td>
</tr>
<tr>
  <th>C(country_standard)[T.Zimbabwe]</th>                         <td>   -0.0088</td> <td>    0.001</td> <td>   -7.438</td> <td> 0.000</td> <td>   -0.011</td> <td>   -0.006</td>
</tr>
<tr>
  <th>transparency_index</th>                                      <td>   -0.0380</td> <td>    0.016</td> <td>   -2.305</td> <td> 0.021</td> <td>   -0.070</td> <td>   -0.006</td>
</tr>
<tr>
  <th>gdp</th>                                                     <td> 1.927e-14</td> <td> 1.06e-16</td> <td>  181.886</td> <td> 0.000</td> <td> 1.91e-14</td> <td> 1.95e-14</td>
</tr>
<tr>
  <th>gdp_per_capita</th>                                          <td> 1.086e-05</td> <td> 6.67e-08</td> <td>  162.698</td> <td> 0.000</td> <td> 1.07e-05</td> <td>  1.1e-05</td>
</tr>
<tr>
  <th>gini_2020</th>                                               <td>    0.8413</td> <td>    0.006</td> <td>  129.730</td> <td> 0.000</td> <td>    0.829</td> <td>    0.854</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>1069.577</td> <th>  Durbin-Watson:     </th> <td>   1.548</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>49209.336</td>
</tr>
<tr>
  <th>Skew:</th>           <td>-0.646</td>  <th>  Prob(JB):          </th> <td>    0.00</td> 
</tr>
<tr>
  <th>Kurtosis:</th>       <td>21.029</td>  <th>  Cond. No.          </th> <td>4.96e+26</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors are robust to cluster correlation (cluster)<br/>[2] The smallest eigenvalue is 5.07e-26. This might indicate that there are<br/>strong multicollinearity problems or that the design matrix is singular.




```python
reg_bugetparticipation = smf.ols('budget_participation_index ~ transparency_index + gdp + gdp_per_capita + gini_2020 + C(country_standard)', ndi_df).fit(cov_type='cluster', cov_kwds={'groups': ndi_df['country_standard']})
```


```python
reg_bugetparticipation.summary()
```

    /Users/katiacordoba/opt/anaconda3/lib/python3.7/site-packages/statsmodels/base/model.py:1832: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 232, but rank is 228
      'rank is %d' % (J, J_), ValueWarning)





<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>    <td>budget_participation_index</td> <th>  R-squared:         </th> <td>   0.052</td> 
</tr>
<tr>
  <th>Model:</th>                        <td>OLS</td>            <th>  Adj. R-squared:    </th> <td>   0.051</td> 
</tr>
<tr>
  <th>Method:</th>                  <td>Least Squares</td>       <th>  F-statistic:       </th> <td>1.904e+07</td>
</tr>
<tr>
  <th>Date:</th>                  <td>Fri, 05 Mar 2021</td>      <th>  Prob (F-statistic):</th>  <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                      <td>13:17:43</td>          <th>  Log-Likelihood:    </th> <td>  3481.8</td> 
</tr>
<tr>
  <th>No. Observations:</th>           <td>  3615</td>           <th>  AIC:               </th> <td>  -6954.</td> 
</tr>
<tr>
  <th>Df Residuals:</th>               <td>  3610</td>           <th>  BIC:               </th> <td>  -6923.</td> 
</tr>
<tr>
  <th>Df Model:</th>                   <td>     4</td>           <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>            <td>cluster</td>          <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
                             <td></td>                                <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>                                               <td>    0.1506</td> <td>    0.003</td> <td>   48.717</td> <td> 0.000</td> <td>    0.145</td> <td>    0.157</td>
</tr>
<tr>
  <th>C(country_standard)[T.Albania]</th>                          <td>   -0.0263</td> <td>    0.002</td> <td>  -14.080</td> <td> 0.000</td> <td>   -0.030</td> <td>   -0.023</td>
</tr>
<tr>
  <th>C(country_standard)[T.Algeria]</th>                          <td>   -0.0285</td> <td>    0.002</td> <td>  -18.514</td> <td> 0.000</td> <td>   -0.031</td> <td>   -0.025</td>
</tr>
<tr>
  <th>C(country_standard)[T.American Samoa]</th>                   <td>   -0.0338</td> <td>    0.001</td> <td>  -41.040</td> <td> 0.000</td> <td>   -0.035</td> <td>   -0.032</td>
</tr>
<tr>
  <th>C(country_standard)[T.Andorra]</th>                          <td>   -0.0333</td> <td>    0.000</td> <td>  -84.097</td> <td> 0.000</td> <td>   -0.034</td> <td>   -0.033</td>
</tr>
<tr>
  <th>C(country_standard)[T.Angola]</th>                           <td>   -0.0422</td> <td>    0.001</td> <td>  -38.257</td> <td> 0.000</td> <td>   -0.044</td> <td>   -0.040</td>
</tr>
<tr>
  <th>C(country_standard)[T.Anguilla]</th>                         <td>   -0.0328</td> <td>    0.000</td> <td>  -79.332</td> <td> 0.000</td> <td>   -0.034</td> <td>   -0.032</td>
</tr>
<tr>
  <th>C(country_standard)[T.Antigua & Barbuda]</th>                <td>   -0.0144</td> <td>    0.000</td> <td>  -60.591</td> <td> 0.000</td> <td>   -0.015</td> <td>   -0.014</td>
</tr>
<tr>
  <th>C(country_standard)[T.Argentina]</th>                        <td>   -0.0010</td> <td>    0.002</td> <td>   -0.656</td> <td> 0.512</td> <td>   -0.004</td> <td>    0.002</td>
</tr>
<tr>
  <th>C(country_standard)[T.Armenia]</th>                          <td>   -0.0562</td> <td>    0.001</td> <td>  -72.305</td> <td> 0.000</td> <td>   -0.058</td> <td>   -0.055</td>
</tr>
<tr>
  <th>C(country_standard)[T.Aruba]</th>                            <td>   -0.0337</td> <td>    0.001</td> <td>  -45.492</td> <td> 0.000</td> <td>   -0.035</td> <td>   -0.032</td>
</tr>
<tr>
  <th>C(country_standard)[T.Australia]</th>                        <td>    0.0108</td> <td>    0.002</td> <td>    6.011</td> <td> 0.000</td> <td>    0.007</td> <td>    0.014</td>
</tr>
<tr>
  <th>C(country_standard)[T.Austria]</th>                          <td>   -0.0056</td> <td>    0.001</td> <td>   -4.346</td> <td> 0.000</td> <td>   -0.008</td> <td>   -0.003</td>
</tr>
<tr>
  <th>C(country_standard)[T.Azerbaijan]</th>                       <td>   -0.0070</td> <td>    0.002</td> <td>   -3.294</td> <td> 0.001</td> <td>   -0.011</td> <td>   -0.003</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bahamas]</th>                          <td>   -0.0056</td> <td>    0.000</td> <td>  -12.747</td> <td> 0.000</td> <td>   -0.006</td> <td>   -0.005</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bahrain]</th>                          <td>    0.0118</td> <td>    0.003</td> <td>    3.500</td> <td> 0.000</td> <td>    0.005</td> <td>    0.018</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bangladesh]</th>                       <td>   -0.0575</td> <td>    0.000</td> <td> -139.624</td> <td> 0.000</td> <td>   -0.058</td> <td>   -0.057</td>
</tr>
<tr>
  <th>C(country_standard)[T.Barbados]</th>                         <td>   -0.0350</td> <td>    0.002</td> <td>  -17.638</td> <td> 0.000</td> <td>   -0.039</td> <td>   -0.031</td>
</tr>
<tr>
  <th>C(country_standard)[T.Belarus]</th>                          <td>   -0.0430</td> <td>    0.002</td> <td>  -23.797</td> <td> 0.000</td> <td>   -0.047</td> <td>   -0.039</td>
</tr>
<tr>
  <th>C(country_standard)[T.Belgium]</th>                          <td>   -0.0324</td> <td>    0.002</td> <td>  -15.548</td> <td> 0.000</td> <td>   -0.036</td> <td>   -0.028</td>
</tr>
<tr>
  <th>C(country_standard)[T.Belize]</th>                           <td>   -0.0354</td> <td>    0.001</td> <td>  -68.411</td> <td> 0.000</td> <td>   -0.036</td> <td>   -0.034</td>
</tr>
<tr>
  <th>C(country_standard)[T.Benin]</th>                            <td>   -0.0627</td> <td>    0.002</td> <td>  -30.214</td> <td> 0.000</td> <td>   -0.067</td> <td>   -0.059</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bermuda]</th>                          <td>   -0.0328</td> <td>    0.000</td> <td>  -78.402</td> <td> 0.000</td> <td>   -0.034</td> <td>   -0.032</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bhutan]</th>                           <td>   -0.0478</td> <td>    0.001</td> <td>  -44.722</td> <td> 0.000</td> <td>   -0.050</td> <td>   -0.046</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bolivia]</th>                          <td>   -0.0422</td> <td>    0.001</td> <td>  -80.182</td> <td> 0.000</td> <td>   -0.043</td> <td>   -0.041</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bosnia & Herzegovina]</th>             <td>   -0.0257</td> <td>    0.001</td> <td>  -26.712</td> <td> 0.000</td> <td>   -0.028</td> <td>   -0.024</td>
</tr>
<tr>
  <th>C(country_standard)[T.Botswana]</th>                         <td>   -0.0312</td> <td>    0.002</td> <td>  -20.275</td> <td> 0.000</td> <td>   -0.034</td> <td>   -0.028</td>
</tr>
<tr>
  <th>C(country_standard)[T.Brazil]</th>                           <td>    0.0384</td> <td>    0.001</td> <td>   26.963</td> <td> 0.000</td> <td>    0.036</td> <td>    0.041</td>
</tr>
<tr>
  <th>C(country_standard)[T.British Virgin Islands]</th>           <td>   -0.0337</td> <td>    0.001</td> <td>  -45.214</td> <td> 0.000</td> <td>   -0.035</td> <td>   -0.032</td>
</tr>
<tr>
  <th>C(country_standard)[T.Brunei]</th>                           <td>    0.0443</td> <td>    0.002</td> <td>   21.729</td> <td> 0.000</td> <td>    0.040</td> <td>    0.048</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bulgaria]</th>                         <td>   -0.0197</td> <td>    0.002</td> <td>   -9.923</td> <td> 0.000</td> <td>   -0.024</td> <td>   -0.016</td>
</tr>
<tr>
  <th>C(country_standard)[T.Burkina Faso]</th>                     <td>   -0.0314</td> <td>    0.001</td> <td>  -24.229</td> <td> 0.000</td> <td>   -0.034</td> <td>   -0.029</td>
</tr>
<tr>
  <th>C(country_standard)[T.Burundi]</th>                          <td>   -0.0547</td> <td>    0.001</td> <td>  -41.472</td> <td> 0.000</td> <td>   -0.057</td> <td>   -0.052</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cambodia]</th>                         <td>   -0.0460</td> <td>    0.001</td> <td>  -36.382</td> <td> 0.000</td> <td>   -0.048</td> <td>   -0.044</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cameroon]</th>                         <td>   -0.0441</td> <td>    0.001</td> <td>  -48.486</td> <td> 0.000</td> <td>   -0.046</td> <td>   -0.042</td>
</tr>
<tr>
  <th>C(country_standard)[T.Canada]</th>                           <td>    0.0231</td> <td>    0.001</td> <td>   17.529</td> <td> 0.000</td> <td>    0.020</td> <td>    0.026</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cape Verde]</th>                       <td>   -0.0576</td> <td>    0.003</td> <td>  -20.866</td> <td> 0.000</td> <td>   -0.063</td> <td>   -0.052</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cayman Islands]</th>                   <td>    0.0065</td> <td>    0.001</td> <td>    7.005</td> <td> 0.000</td> <td>    0.005</td> <td>    0.008</td>
</tr>
<tr>
  <th>C(country_standard)[T.Central African Republic]</th>         <td>   -0.0433</td> <td>    0.001</td> <td>  -69.476</td> <td> 0.000</td> <td>   -0.045</td> <td>   -0.042</td>
</tr>
<tr>
  <th>C(country_standard)[T.Chad]</th>                             <td>   -0.0460</td> <td>    0.001</td> <td>  -34.488</td> <td> 0.000</td> <td>   -0.049</td> <td>   -0.043</td>
</tr>
<tr>
  <th>C(country_standard)[T.Channel Islands]</th>                  <td>   -0.0351</td> <td>    0.002</td> <td>  -16.251</td> <td> 0.000</td> <td>   -0.039</td> <td>   -0.031</td>
</tr>
<tr>
  <th>C(country_standard)[T.Chile]</th>                            <td>    0.0371</td> <td>    0.002</td> <td>   18.393</td> <td> 0.000</td> <td>    0.033</td> <td>    0.041</td>
</tr>
<tr>
  <th>C(country_standard)[T.China]</th>                            <td>   -0.0170</td> <td>    0.003</td> <td>   -4.972</td> <td> 0.000</td> <td>   -0.024</td> <td>   -0.010</td>
</tr>
<tr>
  <th>C(country_standard)[T.Colombia]</th>                         <td>   -0.0355</td> <td>    0.001</td> <td>  -41.691</td> <td> 0.000</td> <td>   -0.037</td> <td>   -0.034</td>
</tr>
<tr>
  <th>C(country_standard)[T.Comoros]</th>                          <td>   -0.0450</td> <td>    0.000</td> <td> -102.852</td> <td> 0.000</td> <td>   -0.046</td> <td>   -0.044</td>
</tr>
<tr>
  <th>C(country_standard)[T.Congo - Brazzaville]</th>              <td>   -0.0361</td> <td>    0.001</td> <td>  -24.791</td> <td> 0.000</td> <td>   -0.039</td> <td>   -0.033</td>
</tr>
<tr>
  <th>C(country_standard)[T.Congo - Kinshasa]</th>                 <td>   -0.0443</td> <td>    0.001</td> <td>  -40.009</td> <td> 0.000</td> <td>   -0.046</td> <td>   -0.042</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cook Islands]</th>                     <td>   -0.0337</td> <td>    0.001</td> <td>  -45.123</td> <td> 0.000</td> <td>   -0.035</td> <td>   -0.032</td>
</tr>
<tr>
  <th>C(country_standard)[T.Costa Rica]</th>                       <td>    0.0189</td> <td>    0.003</td> <td>    7.088</td> <td> 0.000</td> <td>    0.014</td> <td>    0.024</td>
</tr>
<tr>
  <th>C(country_standard)[T.Croatia]</th>                          <td>    0.0140</td> <td>    0.002</td> <td>    6.980</td> <td> 0.000</td> <td>    0.010</td> <td>    0.018</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cuba]</th>                             <td>   -0.0220</td> <td>    0.002</td> <td>   -8.824</td> <td> 0.000</td> <td>   -0.027</td> <td>   -0.017</td>
</tr>
<tr>
  <th>C(country_standard)[T.Curaçao]</th>                          <td>   -0.0337</td> <td>    0.001</td> <td>  -45.292</td> <td> 0.000</td> <td>   -0.035</td> <td>   -0.032</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cyprus]</th>                           <td>   -0.0114</td> <td>    0.001</td> <td>   -8.754</td> <td> 0.000</td> <td>   -0.014</td> <td>   -0.009</td>
</tr>
<tr>
  <th>C(country_standard)[T.Czechia]</th>                          <td>    0.0076</td> <td>    0.002</td> <td>    4.762</td> <td> 0.000</td> <td>    0.004</td> <td>    0.011</td>
</tr>
<tr>
  <th>C(country_standard)[T.Côte d’Ivoire]</th>                    <td>    0.0098</td> <td>    0.001</td> <td>   11.102</td> <td> 0.000</td> <td>    0.008</td> <td>    0.012</td>
</tr>
<tr>
  <th>C(country_standard)[T.Denmark]</th>                          <td>    0.0103</td> <td>    0.001</td> <td>    7.521</td> <td> 0.000</td> <td>    0.008</td> <td>    0.013</td>
</tr>
<tr>
  <th>C(country_standard)[T.Djibouti]</th>                         <td>   -0.0453</td> <td>    0.001</td> <td>  -34.986</td> <td> 0.000</td> <td>   -0.048</td> <td>   -0.043</td>
</tr>
<tr>
  <th>C(country_standard)[T.Dominica]</th>                         <td>   -0.0297</td> <td>    0.000</td> <td>  -78.848</td> <td> 0.000</td> <td>   -0.030</td> <td>   -0.029</td>
</tr>
<tr>
  <th>C(country_standard)[T.Dominican Republic]</th>               <td>    0.0231</td> <td>    0.001</td> <td>   36.151</td> <td> 0.000</td> <td>    0.022</td> <td>    0.024</td>
</tr>
<tr>
  <th>C(country_standard)[T.Ecuador]</th>                          <td>   -0.0072</td> <td>    0.000</td> <td>  -15.713</td> <td> 0.000</td> <td>   -0.008</td> <td>   -0.006</td>
</tr>
<tr>
  <th>C(country_standard)[T.Egypt]</th>                            <td>   -0.0311</td> <td>    0.002</td> <td>  -20.031</td> <td> 0.000</td> <td>   -0.034</td> <td>   -0.028</td>
</tr>
<tr>
  <th>C(country_standard)[T.El Salvador]</th>                      <td>   -0.0180</td> <td>    0.002</td> <td>  -11.795</td> <td> 0.000</td> <td>   -0.021</td> <td>   -0.015</td>
</tr>
<tr>
  <th>C(country_standard)[T.Equatorial Guinea]</th>                <td>   -0.0065</td> <td>    0.003</td> <td>   -2.203</td> <td> 0.028</td> <td>   -0.012</td> <td>   -0.001</td>
</tr>
<tr>
  <th>C(country_standard)[T.Eritrea]</th>                          <td>   -0.0573</td> <td>    0.003</td> <td>  -21.733</td> <td> 0.000</td> <td>   -0.062</td> <td>   -0.052</td>
</tr>
<tr>
  <th>C(country_standard)[T.Estonia]</th>                          <td>   -0.0311</td> <td>    0.002</td> <td>  -14.150</td> <td> 0.000</td> <td>   -0.035</td> <td>   -0.027</td>
</tr>
<tr>
  <th>C(country_standard)[T.Eswatini]</th>                         <td>   -0.0390</td> <td>    0.001</td> <td>  -27.110</td> <td> 0.000</td> <td>   -0.042</td> <td>   -0.036</td>
</tr>
<tr>
  <th>C(country_standard)[T.Ethiopia]</th>                         <td>   -0.0648</td> <td>    0.001</td> <td>  -75.701</td> <td> 0.000</td> <td>   -0.066</td> <td>   -0.063</td>
</tr>
<tr>
  <th>C(country_standard)[T.Faroe Islands]</th>                    <td>   -0.0328</td> <td>    0.000</td> <td>  -80.715</td> <td> 0.000</td> <td>   -0.034</td> <td>   -0.032</td>
</tr>
<tr>
  <th>C(country_standard)[T.Fiji]</th>                             <td>   -0.0489</td> <td>    0.000</td> <td> -112.887</td> <td> 0.000</td> <td>   -0.050</td> <td>   -0.048</td>
</tr>
<tr>
  <th>C(country_standard)[T.Finland]</th>                          <td>    0.0497</td> <td>    0.002</td> <td>   27.940</td> <td> 0.000</td> <td>    0.046</td> <td>    0.053</td>
</tr>
<tr>
  <th>C(country_standard)[T.France]</th>                           <td>    0.0432</td> <td>    0.002</td> <td>   28.259</td> <td> 0.000</td> <td>    0.040</td> <td>    0.046</td>
</tr>
<tr>
  <th>C(country_standard)[T.French Guiana]</th>                    <td>   -0.0337</td> <td>    0.001</td> <td>  -45.456</td> <td> 0.000</td> <td>   -0.035</td> <td>   -0.032</td>
</tr>
<tr>
  <th>C(country_standard)[T.French Polynesia]</th>                 <td>   -0.0337</td> <td>    0.001</td> <td>  -45.244</td> <td> 0.000</td> <td>   -0.035</td> <td>   -0.032</td>
</tr>
<tr>
  <th>C(country_standard)[T.Gabon]</th>                            <td>   -0.0318</td> <td>    0.001</td> <td>  -28.232</td> <td> 0.000</td> <td>   -0.034</td> <td>   -0.030</td>
</tr>
<tr>
  <th>C(country_standard)[T.Gambia]</th>                           <td>   -0.0442</td> <td>    0.000</td> <td> -102.353</td> <td> 0.000</td> <td>   -0.045</td> <td>   -0.043</td>
</tr>
<tr>
  <th>C(country_standard)[T.Georgia]</th>                          <td>    0.0083</td> <td>    0.001</td> <td>    6.019</td> <td> 0.000</td> <td>    0.006</td> <td>    0.011</td>
</tr>
<tr>
  <th>C(country_standard)[T.Germany]</th>                          <td>    0.0712</td> <td>    0.001</td> <td>   63.458</td> <td> 0.000</td> <td>    0.069</td> <td>    0.073</td>
</tr>
<tr>
  <th>C(country_standard)[T.Ghana]</th>                            <td>   -0.0320</td> <td>    0.003</td> <td>  -12.710</td> <td> 0.000</td> <td>   -0.037</td> <td>   -0.027</td>
</tr>
<tr>
  <th>C(country_standard)[T.Gibraltar]</th>                        <td>   -0.0328</td> <td>    0.000</td> <td>  -77.595</td> <td> 0.000</td> <td>   -0.034</td> <td>   -0.032</td>
</tr>
<tr>
  <th>C(country_standard)[T.Greece]</th>                           <td>    0.0163</td> <td>    0.002</td> <td>    7.016</td> <td> 0.000</td> <td>    0.012</td> <td>    0.021</td>
</tr>
<tr>
  <th>C(country_standard)[T.Greenland]</th>                        <td>   -0.0328</td> <td>    0.000</td> <td>  -79.410</td> <td> 0.000</td> <td>   -0.034</td> <td>   -0.032</td>
</tr>
<tr>
  <th>C(country_standard)[T.Grenada]</th>                          <td>   -0.0240</td> <td>    0.000</td> <td> -108.138</td> <td> 0.000</td> <td>   -0.024</td> <td>   -0.024</td>
</tr>
<tr>
  <th>C(country_standard)[T.Guam]</th>                             <td>   -0.0337</td> <td>    0.001</td> <td>  -45.387</td> <td> 0.000</td> <td>   -0.035</td> <td>   -0.032</td>
</tr>
<tr>
  <th>C(country_standard)[T.Guatemala]</th>                        <td>    0.0140</td> <td>    0.001</td> <td>   15.246</td> <td> 0.000</td> <td>    0.012</td> <td>    0.016</td>
</tr>
<tr>
  <th>C(country_standard)[T.Guernsey]</th>                         <td>   -0.0226</td> <td>    0.011</td> <td>   -2.103</td> <td> 0.035</td> <td>   -0.044</td> <td>   -0.002</td>
</tr>
<tr>
  <th>C(country_standard)[T.Guinea]</th>                           <td>   -0.0498</td> <td>    0.000</td> <td> -145.215</td> <td> 0.000</td> <td>   -0.050</td> <td>   -0.049</td>
</tr>
<tr>
  <th>C(country_standard)[T.Guinea-Bissau]</th>                    <td>   -0.0555</td> <td>    0.000</td> <td> -114.473</td> <td> 0.000</td> <td>   -0.056</td> <td>   -0.055</td>
</tr>
<tr>
  <th>C(country_standard)[T.Guyana]</th>                           <td>   -0.0509</td> <td>    0.002</td> <td>  -30.800</td> <td> 0.000</td> <td>   -0.054</td> <td>   -0.048</td>
</tr>
<tr>
  <th>C(country_standard)[T.Haiti]</th>                            <td>   -0.0419</td> <td>    0.000</td> <td> -105.838</td> <td> 0.000</td> <td>   -0.043</td> <td>   -0.041</td>
</tr>
<tr>
  <th>C(country_standard)[T.Honduras]</th>                         <td>    0.0130</td> <td>    0.000</td> <td>   29.968</td> <td> 0.000</td> <td>    0.012</td> <td>    0.014</td>
</tr>
<tr>
  <th>C(country_standard)[T.Hong Kong SAR China]</th>              <td>    0.0220</td> <td>    0.001</td> <td>   18.189</td> <td> 0.000</td> <td>    0.020</td> <td>    0.024</td>
</tr>
<tr>
  <th>C(country_standard)[T.Hungary]</th>                          <td>   -0.0367</td> <td>    0.001</td> <td>  -39.959</td> <td> 0.000</td> <td>   -0.039</td> <td>   -0.035</td>
</tr>
<tr>
  <th>C(country_standard)[T.Iceland]</th>                          <td>   -0.0120</td> <td>    0.001</td> <td>   -8.506</td> <td> 0.000</td> <td>   -0.015</td> <td>   -0.009</td>
</tr>
<tr>
  <th>C(country_standard)[T.India]</th>                            <td>   -0.0327</td> <td>    0.001</td> <td>  -34.856</td> <td> 0.000</td> <td>   -0.035</td> <td>   -0.031</td>
</tr>
<tr>
  <th>C(country_standard)[T.Indonesia]</th>                        <td>    0.0351</td> <td>    0.001</td> <td>   46.528</td> <td> 0.000</td> <td>    0.034</td> <td>    0.037</td>
</tr>
<tr>
  <th>C(country_standard)[T.Iran]</th>                             <td>   -0.0290</td> <td>    0.002</td> <td>  -18.954</td> <td> 0.000</td> <td>   -0.032</td> <td>   -0.026</td>
</tr>
<tr>
  <th>C(country_standard)[T.Iraq]</th>                             <td>   -0.0485</td> <td>    0.000</td> <td> -102.089</td> <td> 0.000</td> <td>   -0.049</td> <td>   -0.048</td>
</tr>
<tr>
  <th>C(country_standard)[T.Ireland]</th>                          <td>    0.0315</td> <td>    0.001</td> <td>   62.357</td> <td> 0.000</td> <td>    0.031</td> <td>    0.032</td>
</tr>
<tr>
  <th>C(country_standard)[T.Isle of Man]</th>                      <td>   -0.0328</td> <td>    0.000</td> <td>  -78.016</td> <td> 0.000</td> <td>   -0.034</td> <td>   -0.032</td>
</tr>
<tr>
  <th>C(country_standard)[T.Israel]</th>                           <td>   -0.0132</td> <td>    0.001</td> <td>  -11.683</td> <td> 0.000</td> <td>   -0.015</td> <td>   -0.011</td>
</tr>
<tr>
  <th>C(country_standard)[T.Italy]</th>                            <td>    0.0332</td> <td>    0.002</td> <td>   17.219</td> <td> 0.000</td> <td>    0.029</td> <td>    0.037</td>
</tr>
<tr>
  <th>C(country_standard)[T.Jamaica]</th>                          <td>   -0.0465</td> <td>    0.002</td> <td>  -20.552</td> <td> 0.000</td> <td>   -0.051</td> <td>   -0.042</td>
</tr>
<tr>
  <th>C(country_standard)[T.Japan]</th>                            <td>   -0.0282</td> <td>    0.002</td> <td>  -17.382</td> <td> 0.000</td> <td>   -0.031</td> <td>   -0.025</td>
</tr>
<tr>
  <th>C(country_standard)[T.Jersey]</th>                           <td>   -0.0328</td> <td>    0.000</td> <td>  -78.030</td> <td> 0.000</td> <td>   -0.034</td> <td>   -0.032</td>
</tr>
<tr>
  <th>C(country_standard)[T.Jordan]</th>                           <td>    0.0131</td> <td>    0.000</td> <td>   38.681</td> <td> 0.000</td> <td>    0.012</td> <td>    0.014</td>
</tr>
<tr>
  <th>C(country_standard)[T.Kazakhstan]</th>                       <td>   -0.0105</td> <td>    0.002</td> <td>   -4.602</td> <td> 0.000</td> <td>   -0.015</td> <td>   -0.006</td>
</tr>
<tr>
  <th>C(country_standard)[T.Kenya]</th>                            <td>    0.0109</td> <td>    0.001</td> <td>   10.396</td> <td> 0.000</td> <td>    0.009</td> <td>    0.013</td>
</tr>
<tr>
  <th>C(country_standard)[T.Kiribati]</th>                         <td>   -0.0337</td> <td>    0.001</td> <td>  -44.923</td> <td> 0.000</td> <td>   -0.035</td> <td>   -0.032</td>
</tr>
<tr>
  <th>C(country_standard)[T.Kosovo]</th>                           <td>   -0.0351</td> <td>    0.002</td> <td>  -16.245</td> <td> 0.000</td> <td>   -0.039</td> <td>   -0.031</td>
</tr>
<tr>
  <th>C(country_standard)[T.Kuwait]</th>                           <td>    0.0333</td> <td>    0.002</td> <td>   17.347</td> <td> 0.000</td> <td>    0.030</td> <td>    0.037</td>
</tr>
<tr>
  <th>C(country_standard)[T.Kyrgyzstan]</th>                       <td>   -0.0606</td> <td>    0.001</td> <td>  -60.159</td> <td> 0.000</td> <td>   -0.063</td> <td>   -0.059</td>
</tr>
<tr>
  <th>C(country_standard)[T.Laos]</th>                             <td>   -0.0274</td> <td>    0.002</td> <td>  -12.233</td> <td> 0.000</td> <td>   -0.032</td> <td>   -0.023</td>
</tr>
<tr>
  <th>C(country_standard)[T.Latvia]</th>                           <td>   -0.0228</td> <td>    0.002</td> <td>  -13.988</td> <td> 0.000</td> <td>   -0.026</td> <td>   -0.020</td>
</tr>
<tr>
  <th>C(country_standard)[T.Lebanon]</th>                          <td>   -0.0244</td> <td>    0.000</td> <td> -116.601</td> <td> 0.000</td> <td>   -0.025</td> <td>   -0.024</td>
</tr>
<tr>
  <th>C(country_standard)[T.Lesotho]</th>                          <td>   -0.0434</td> <td>    0.001</td> <td>  -37.580</td> <td> 0.000</td> <td>   -0.046</td> <td>   -0.041</td>
</tr>
<tr>
  <th>C(country_standard)[T.Liberia]</th>                          <td>   -0.0270</td> <td>    0.001</td> <td>  -19.815</td> <td> 0.000</td> <td>   -0.030</td> <td>   -0.024</td>
</tr>
<tr>
  <th>C(country_standard)[T.Libya]</th>                            <td>   -0.0403</td> <td>    0.001</td> <td>  -46.212</td> <td> 0.000</td> <td>   -0.042</td> <td>   -0.039</td>
</tr>
<tr>
  <th>C(country_standard)[T.Liechtenstein]</th>                    <td>   -0.0334</td> <td>    0.000</td> <td>  -91.916</td> <td> 0.000</td> <td>   -0.034</td> <td>   -0.033</td>
</tr>
<tr>
  <th>C(country_standard)[T.Lithuania]</th>                        <td>   -0.0384</td> <td>    0.002</td> <td>  -18.537</td> <td> 0.000</td> <td>   -0.042</td> <td>   -0.034</td>
</tr>
<tr>
  <th>C(country_standard)[T.Luxembourg]</th>                       <td>    0.0449</td> <td>    0.001</td> <td>   78.588</td> <td> 0.000</td> <td>    0.044</td> <td>    0.046</td>
</tr>
<tr>
  <th>C(country_standard)[T.Macao SAR China]</th>                  <td>   -0.0332</td> <td>    0.000</td> <td>  -73.641</td> <td> 0.000</td> <td>   -0.034</td> <td>   -0.032</td>
</tr>
<tr>
  <th>C(country_standard)[T.Madagascar]</th>                       <td>   -0.0557</td> <td>    0.001</td> <td>  -71.755</td> <td> 0.000</td> <td>   -0.057</td> <td>   -0.054</td>
</tr>
<tr>
  <th>C(country_standard)[T.Malawi]</th>                           <td>   -0.0226</td> <td>    0.001</td> <td>  -17.820</td> <td> 0.000</td> <td>   -0.025</td> <td>   -0.020</td>
</tr>
<tr>
  <th>C(country_standard)[T.Malaysia]</th>                         <td>   -0.0093</td> <td>    0.001</td> <td>  -10.102</td> <td> 0.000</td> <td>   -0.011</td> <td>   -0.007</td>
</tr>
<tr>
  <th>C(country_standard)[T.Maldives]</th>                         <td>   -0.0374</td> <td>    0.000</td> <td> -153.394</td> <td> 0.000</td> <td>   -0.038</td> <td>   -0.037</td>
</tr>
<tr>
  <th>C(country_standard)[T.Mali]</th>                             <td>   -0.0581</td> <td>    0.001</td> <td>  -59.644</td> <td> 0.000</td> <td>   -0.060</td> <td>   -0.056</td>
</tr>
<tr>
  <th>C(country_standard)[T.Malta]</th>                            <td>   -0.0314</td> <td>    0.001</td> <td>  -22.139</td> <td> 0.000</td> <td>   -0.034</td> <td>   -0.029</td>
</tr>
<tr>
  <th>C(country_standard)[T.Marshall Islands]</th>                 <td>   -0.0528</td> <td>    0.001</td> <td>  -55.132</td> <td> 0.000</td> <td>   -0.055</td> <td>   -0.051</td>
</tr>
<tr>
  <th>C(country_standard)[T.Martinique]</th>                       <td>   -0.0337</td> <td>    0.001</td> <td>  -45.353</td> <td> 0.000</td> <td>   -0.035</td> <td>   -0.032</td>
</tr>
<tr>
  <th>C(country_standard)[T.Mauritania]</th>                       <td>   -0.0556</td> <td>    0.000</td> <td> -125.147</td> <td> 0.000</td> <td>   -0.056</td> <td>   -0.055</td>
</tr>
<tr>
  <th>C(country_standard)[T.Mauritius]</th>                        <td>   -0.0499</td> <td>    0.002</td> <td>  -21.661</td> <td> 0.000</td> <td>   -0.054</td> <td>   -0.045</td>
</tr>
<tr>
  <th>C(country_standard)[T.Mexico]</th>                           <td>    0.0038</td> <td>    0.000</td> <td>    8.751</td> <td> 0.000</td> <td>    0.003</td> <td>    0.005</td>
</tr>
<tr>
  <th>C(country_standard)[T.Micronesia (Federated States of)]</th> <td>   -0.0328</td> <td>    0.000</td> <td>  -75.841</td> <td> 0.000</td> <td>   -0.034</td> <td>   -0.032</td>
</tr>
<tr>
  <th>C(country_standard)[T.Moldova]</th>                          <td>   -0.0343</td> <td>    0.001</td> <td>  -24.757</td> <td> 0.000</td> <td>   -0.037</td> <td>   -0.032</td>
</tr>
<tr>
  <th>C(country_standard)[T.Monaco]</th>                           <td>   -0.0334</td> <td>    0.000</td> <td>  -92.249</td> <td> 0.000</td> <td>   -0.034</td> <td>   -0.033</td>
</tr>
<tr>
  <th>C(country_standard)[T.Mongolia]</th>                         <td>   -0.0262</td> <td>    0.002</td> <td>  -12.785</td> <td> 0.000</td> <td>   -0.030</td> <td>   -0.022</td>
</tr>
<tr>
  <th>C(country_standard)[T.Montenegro]</th>                       <td>   -0.0510</td> <td>    0.001</td> <td>  -51.079</td> <td> 0.000</td> <td>   -0.053</td> <td>   -0.049</td>
</tr>
<tr>
  <th>C(country_standard)[T.Morocco]</th>                          <td>   -0.0411</td> <td>    0.001</td> <td>  -81.720</td> <td> 0.000</td> <td>   -0.042</td> <td>   -0.040</td>
</tr>
<tr>
  <th>C(country_standard)[T.Mozambique]</th>                       <td>   -0.0569</td> <td>    0.001</td> <td>  -72.766</td> <td> 0.000</td> <td>   -0.058</td> <td>   -0.055</td>
</tr>
<tr>
  <th>C(country_standard)[T.Myanmar (Burma)]</th>                  <td>   -0.0677</td> <td>    0.000</td> <td> -140.954</td> <td> 0.000</td> <td>   -0.069</td> <td>   -0.067</td>
</tr>
<tr>
  <th>C(country_standard)[T.Namibia]</th>                          <td>   -0.0409</td> <td>    0.002</td> <td>  -23.471</td> <td> 0.000</td> <td>   -0.044</td> <td>   -0.037</td>
</tr>
<tr>
  <th>C(country_standard)[T.Nauru]</th>                            <td>   -0.0337</td> <td>    0.001</td> <td>  -45.214</td> <td> 0.000</td> <td>   -0.035</td> <td>   -0.032</td>
</tr>
<tr>
  <th>C(country_standard)[T.Nepal]</th>                            <td>   -0.0586</td> <td>    0.001</td> <td>  -44.772</td> <td> 0.000</td> <td>   -0.061</td> <td>   -0.056</td>
</tr>
<tr>
  <th>C(country_standard)[T.Netherlands]</th>                      <td>    0.0242</td> <td>    0.001</td> <td>   26.109</td> <td> 0.000</td> <td>    0.022</td> <td>    0.026</td>
</tr>
<tr>
  <th>C(country_standard)[T.New Caledonia]</th>                    <td>   -0.0337</td> <td>    0.001</td> <td>  -45.164</td> <td> 0.000</td> <td>   -0.035</td> <td>   -0.032</td>
</tr>
<tr>
  <th>C(country_standard)[T.New Zealand]</th>                      <td>    0.0012</td> <td>    0.002</td> <td>    0.574</td> <td> 0.566</td> <td>   -0.003</td> <td>    0.005</td>
</tr>
<tr>
  <th>C(country_standard)[T.Nicaragua]</th>                        <td>   -0.0388</td> <td>    0.001</td> <td>  -36.577</td> <td> 0.000</td> <td>   -0.041</td> <td>   -0.037</td>
</tr>
<tr>
  <th>C(country_standard)[T.Niger]</th>                            <td>   -0.0649</td> <td>    0.001</td> <td>  -46.263</td> <td> 0.000</td> <td>   -0.068</td> <td>   -0.062</td>
</tr>
<tr>
  <th>C(country_standard)[T.Nigeria]</th>                          <td>   -0.0371</td> <td>    0.001</td> <td>  -67.907</td> <td> 0.000</td> <td>   -0.038</td> <td>   -0.036</td>
</tr>
<tr>
  <th>C(country_standard)[T.Niue]</th>                             <td>   -0.0337</td> <td>    0.001</td> <td>  -44.932</td> <td> 0.000</td> <td>   -0.035</td> <td>   -0.032</td>
</tr>
<tr>
  <th>C(country_standard)[T.North Korea]</th>                      <td>   -0.0328</td> <td>    0.000</td> <td>  -80.156</td> <td> 0.000</td> <td>   -0.034</td> <td>   -0.032</td>
</tr>
<tr>
  <th>C(country_standard)[T.North Macedonia]</th>                  <td>   -0.0096</td> <td>    0.001</td> <td>  -16.120</td> <td> 0.000</td> <td>   -0.011</td> <td>   -0.008</td>
</tr>
<tr>
  <th>C(country_standard)[T.Northern Mariana Islands]</th>         <td>   -0.0337</td> <td>    0.001</td> <td>  -44.145</td> <td> 0.000</td> <td>   -0.035</td> <td>   -0.032</td>
</tr>
<tr>
  <th>C(country_standard)[T.Norway]</th>                           <td>    0.0582</td> <td>    0.001</td> <td>   65.225</td> <td> 0.000</td> <td>    0.056</td> <td>    0.060</td>
</tr>
<tr>
  <th>C(country_standard)[T.Oman]</th>                             <td>    0.0084</td> <td>    0.003</td> <td>    3.266</td> <td> 0.001</td> <td>    0.003</td> <td>    0.013</td>
</tr>
<tr>
  <th>C(country_standard)[T.Pakistan]</th>                         <td>   -0.0166</td> <td>    0.001</td> <td>  -26.410</td> <td> 0.000</td> <td>   -0.018</td> <td>   -0.015</td>
</tr>
<tr>
  <th>C(country_standard)[T.Palau]</th>                            <td>   -0.0337</td> <td>    0.001</td> <td>  -45.787</td> <td> 0.000</td> <td>   -0.035</td> <td>   -0.032</td>
</tr>
<tr>
  <th>C(country_standard)[T.Palestinian Territories]</th>          <td>   -0.0429</td> <td>    0.001</td> <td>  -30.923</td> <td> 0.000</td> <td>   -0.046</td> <td>   -0.040</td>
</tr>
<tr>
  <th>C(country_standard)[T.Panama]</th>                           <td>    0.0363</td> <td>    0.001</td> <td>   26.008</td> <td> 0.000</td> <td>    0.034</td> <td>    0.039</td>
</tr>
<tr>
  <th>C(country_standard)[T.Papua New Guinea]</th>                 <td>   -0.0473</td> <td>    0.001</td> <td>  -49.266</td> <td> 0.000</td> <td>   -0.049</td> <td>   -0.045</td>
</tr>
<tr>
  <th>C(country_standard)[T.Paraguay]</th>                         <td>    0.0222</td> <td>    0.001</td> <td>   30.949</td> <td> 0.000</td> <td>    0.021</td> <td>    0.024</td>
</tr>
<tr>
  <th>C(country_standard)[T.Peru]</th>                             <td>   -0.0068</td> <td>    0.002</td> <td>   -3.713</td> <td> 0.000</td> <td>   -0.010</td> <td>   -0.003</td>
</tr>
<tr>
  <th>C(country_standard)[T.Philippines]</th>                      <td>    0.0320</td> <td>    0.001</td> <td>   58.969</td> <td> 0.000</td> <td>    0.031</td> <td>    0.033</td>
</tr>
<tr>
  <th>C(country_standard)[T.Poland]</th>                           <td>   -0.0374</td> <td>    0.002</td> <td>  -20.875</td> <td> 0.000</td> <td>   -0.041</td> <td>   -0.034</td>
</tr>
<tr>
  <th>C(country_standard)[T.Portugal]</th>                         <td>   -0.0051</td> <td>    0.002</td> <td>   -2.303</td> <td> 0.021</td> <td>   -0.009</td> <td>   -0.001</td>
</tr>
<tr>
  <th>C(country_standard)[T.Puerto Rico]</th>                      <td>   -0.0337</td> <td>    0.001</td> <td>  -45.601</td> <td> 0.000</td> <td>   -0.035</td> <td>   -0.032</td>
</tr>
<tr>
  <th>C(country_standard)[T.Qatar]</th>                            <td>    0.0899</td> <td>    0.006</td> <td>   14.098</td> <td> 0.000</td> <td>    0.077</td> <td>    0.102</td>
</tr>
<tr>
  <th>C(country_standard)[T.Romania]</th>                          <td>   -0.0148</td> <td>    0.002</td> <td>   -8.799</td> <td> 0.000</td> <td>   -0.018</td> <td>   -0.011</td>
</tr>
<tr>
  <th>C(country_standard)[T.Russia]</th>                           <td>    0.0087</td> <td>    0.003</td> <td>    3.324</td> <td> 0.001</td> <td>    0.004</td> <td>    0.014</td>
</tr>
<tr>
  <th>C(country_standard)[T.Rwanda]</th>                           <td>   -0.0454</td> <td>    0.001</td> <td>  -43.955</td> <td> 0.000</td> <td>   -0.047</td> <td>   -0.043</td>
</tr>
<tr>
  <th>C(country_standard)[T.Réunion]</th>                          <td>   -0.0337</td> <td>    0.001</td> <td>  -45.415</td> <td> 0.000</td> <td>   -0.035</td> <td>   -0.032</td>
</tr>
<tr>
  <th>C(country_standard)[T.Saint Martin (French part)]</th>       <td>   -0.0336</td> <td>    0.001</td> <td>  -47.280</td> <td> 0.000</td> <td>   -0.035</td> <td>   -0.032</td>
</tr>
<tr>
  <th>C(country_standard)[T.Samoa]</th>                            <td>   -0.0470</td> <td>    0.001</td> <td>  -58.740</td> <td> 0.000</td> <td>   -0.049</td> <td>   -0.045</td>
</tr>
<tr>
  <th>C(country_standard)[T.San Marino]</th>                       <td>    0.0155</td> <td>    0.001</td> <td>   12.801</td> <td> 0.000</td> <td>    0.013</td> <td>    0.018</td>
</tr>
<tr>
  <th>C(country_standard)[T.Saudi Arabia]</th>                     <td>    0.0371</td> <td>    0.005</td> <td>    7.933</td> <td> 0.000</td> <td>    0.028</td> <td>    0.046</td>
</tr>
<tr>
  <th>C(country_standard)[T.Senegal]</th>                          <td>   -0.0282</td> <td>    0.002</td> <td>  -13.566</td> <td> 0.000</td> <td>   -0.032</td> <td>   -0.024</td>
</tr>
<tr>
  <th>C(country_standard)[T.Serbia]</th>                           <td>   -0.0187</td> <td>    0.001</td> <td>  -16.669</td> <td> 0.000</td> <td>   -0.021</td> <td>   -0.016</td>
</tr>
<tr>
  <th>C(country_standard)[T.Seychelles]</th>                       <td>   -0.0243</td> <td>    0.000</td> <td> -134.459</td> <td> 0.000</td> <td>   -0.025</td> <td>   -0.024</td>
</tr>
<tr>
  <th>C(country_standard)[T.Sierra Leone]</th>                     <td>   -0.0306</td> <td>    0.001</td> <td>  -25.055</td> <td> 0.000</td> <td>   -0.033</td> <td>   -0.028</td>
</tr>
<tr>
  <th>C(country_standard)[T.Singapore]</th>                        <td>    0.0587</td> <td>    0.003</td> <td>   20.759</td> <td> 0.000</td> <td>    0.053</td> <td>    0.064</td>
</tr>
<tr>
  <th>C(country_standard)[T.Sint Maarten]</th>                     <td>   -0.0337</td> <td>    0.001</td> <td>  -45.811</td> <td> 0.000</td> <td>   -0.035</td> <td>   -0.032</td>
</tr>
<tr>
  <th>C(country_standard)[T.Slovakia]</th>                         <td>   -0.0018</td> <td>    0.003</td> <td>   -0.694</td> <td> 0.488</td> <td>   -0.007</td> <td>    0.003</td>
</tr>
<tr>
  <th>C(country_standard)[T.Slovenia]</th>                         <td>   -0.0362</td> <td>    0.002</td> <td>  -17.987</td> <td> 0.000</td> <td>   -0.040</td> <td>   -0.032</td>
</tr>
<tr>
  <th>C(country_standard)[T.Solomon Islands]</th>                  <td>   -0.0629</td> <td>    0.002</td> <td>  -32.901</td> <td> 0.000</td> <td>   -0.067</td> <td>   -0.059</td>
</tr>
<tr>
  <th>C(country_standard)[T.Somalia]</th>                          <td>   -0.0240</td> <td>    0.002</td> <td>  -12.364</td> <td> 0.000</td> <td>   -0.028</td> <td>   -0.020</td>
</tr>
<tr>
  <th>C(country_standard)[T.South Africa]</th>                     <td>    0.0276</td> <td>    0.002</td> <td>   15.507</td> <td> 0.000</td> <td>    0.024</td> <td>    0.031</td>
</tr>
<tr>
  <th>C(country_standard)[T.South Korea]</th>                      <td>    0.0102</td> <td>    0.001</td> <td>   12.061</td> <td> 0.000</td> <td>    0.009</td> <td>    0.012</td>
</tr>
<tr>
  <th>C(country_standard)[T.South Sudan]</th>                      <td>   -0.0440</td> <td>    0.002</td> <td>  -22.994</td> <td> 0.000</td> <td>   -0.048</td> <td>   -0.040</td>
</tr>
<tr>
  <th>C(country_standard)[T.Spain]</th>                            <td>   -0.0265</td> <td>    0.002</td> <td>  -14.742</td> <td> 0.000</td> <td>   -0.030</td> <td>   -0.023</td>
</tr>
<tr>
  <th>C(country_standard)[T.Sri Lanka]</th>                        <td>   -0.0467</td> <td>    0.001</td> <td>  -78.850</td> <td> 0.000</td> <td>   -0.048</td> <td>   -0.046</td>
</tr>
<tr>
  <th>C(country_standard)[T.St. Helena]</th>                       <td>   -0.0226</td> <td>    0.011</td> <td>   -2.104</td> <td> 0.035</td> <td>   -0.044</td> <td>   -0.002</td>
</tr>
<tr>
  <th>C(country_standard)[T.St. Kitts & Nevis]</th>                <td>   -0.0260</td> <td>    0.000</td> <td> -158.838</td> <td> 0.000</td> <td>   -0.026</td> <td>   -0.026</td>
</tr>
<tr>
  <th>C(country_standard)[T.St. Lucia]</th>                        <td>   -0.0411</td> <td>    0.001</td> <td>  -69.413</td> <td> 0.000</td> <td>   -0.042</td> <td>   -0.040</td>
</tr>
<tr>
  <th>C(country_standard)[T.St. Vincent & Grenadines]</th>         <td>   -0.0292</td> <td>    0.000</td> <td>  -82.180</td> <td> 0.000</td> <td>   -0.030</td> <td>   -0.028</td>
</tr>
<tr>
  <th>C(country_standard)[T.Sudan]</th>                            <td>   -0.0490</td> <td>    0.001</td> <td>  -33.988</td> <td> 0.000</td> <td>   -0.052</td> <td>   -0.046</td>
</tr>
<tr>
  <th>C(country_standard)[T.Suriname]</th>                         <td>   -0.0287</td> <td>    0.002</td> <td>  -18.794</td> <td> 0.000</td> <td>   -0.032</td> <td>   -0.026</td>
</tr>
<tr>
  <th>C(country_standard)[T.Sweden]</th>                           <td>    0.0471</td> <td>    0.001</td> <td>   39.520</td> <td> 0.000</td> <td>    0.045</td> <td>    0.049</td>
</tr>
<tr>
  <th>C(country_standard)[T.Switzerland]</th>                      <td>    0.0030</td> <td>    0.001</td> <td>    2.301</td> <td> 0.021</td> <td>    0.000</td> <td>    0.005</td>
</tr>
<tr>
  <th>C(country_standard)[T.Syria]</th>                            <td>   -0.0238</td> <td>    0.003</td> <td>   -7.731</td> <td> 0.000</td> <td>   -0.030</td> <td>   -0.018</td>
</tr>
<tr>
  <th>C(country_standard)[T.São Tomé & Príncipe]</th>              <td>   -0.0684</td> <td>    0.003</td> <td>  -27.063</td> <td> 0.000</td> <td>   -0.073</td> <td>   -0.063</td>
</tr>
<tr>
  <th>C(country_standard)[T.Taiwan]</th>                           <td>   -0.0365</td> <td>    0.002</td> <td>  -18.720</td> <td> 0.000</td> <td>   -0.040</td> <td>   -0.033</td>
</tr>
<tr>
  <th>C(country_standard)[T.Tajikistan]</th>                       <td>   -0.0544</td> <td>    0.002</td> <td>  -32.397</td> <td> 0.000</td> <td>   -0.058</td> <td>   -0.051</td>
</tr>
<tr>
  <th>C(country_standard)[T.Tanzania]</th>                         <td>   -0.0659</td> <td>    0.001</td> <td>  -47.573</td> <td> 0.000</td> <td>   -0.069</td> <td>   -0.063</td>
</tr>
<tr>
  <th>C(country_standard)[T.Thailand]</th>                         <td>   -0.0111</td> <td>    0.002</td> <td>   -7.350</td> <td> 0.000</td> <td>   -0.014</td> <td>   -0.008</td>
</tr>
<tr>
  <th>C(country_standard)[T.Timor-Leste]</th>                      <td>   -0.0793</td> <td>    0.002</td> <td>  -36.307</td> <td> 0.000</td> <td>   -0.084</td> <td>   -0.075</td>
</tr>
<tr>
  <th>C(country_standard)[T.Togo]</th>                             <td>   -0.0520</td> <td>    0.000</td> <td> -125.640</td> <td> 0.000</td> <td>   -0.053</td> <td>   -0.051</td>
</tr>
<tr>
  <th>C(country_standard)[T.Tonga]</th>                            <td>   -0.0337</td> <td>    0.001</td> <td>  -44.050</td> <td> 0.000</td> <td>   -0.035</td> <td>   -0.032</td>
</tr>
<tr>
  <th>C(country_standard)[T.Trinidad & Tobago]</th>                <td>    0.0046</td> <td>    0.002</td> <td>    3.031</td> <td> 0.002</td> <td>    0.002</td> <td>    0.008</td>
</tr>
<tr>
  <th>C(country_standard)[T.Tunisia]</th>                          <td>   -0.0533</td> <td>    0.002</td> <td>  -25.375</td> <td> 0.000</td> <td>   -0.057</td> <td>   -0.049</td>
</tr>
<tr>
  <th>C(country_standard)[T.Turkey]</th>                           <td>   -0.0090</td> <td>    0.001</td> <td>   -6.217</td> <td> 0.000</td> <td>   -0.012</td> <td>   -0.006</td>
</tr>
<tr>
  <th>C(country_standard)[T.Turkmenistan]</th>                     <td>   -0.0356</td> <td>    0.003</td> <td>  -12.432</td> <td> 0.000</td> <td>   -0.041</td> <td>   -0.030</td>
</tr>
<tr>
  <th>C(country_standard)[T.Turks & Caicos Islands]</th>           <td>   -0.0328</td> <td>    0.000</td> <td>  -77.092</td> <td> 0.000</td> <td>   -0.034</td> <td>   -0.032</td>
</tr>
<tr>
  <th>C(country_standard)[T.Tuvalu]</th>                           <td>   -0.0337</td> <td>    0.001</td> <td>  -43.864</td> <td> 0.000</td> <td>   -0.035</td> <td>   -0.032</td>
</tr>
<tr>
  <th>C(country_standard)[T.U.S. Virgin Islands]</th>              <td>   -0.0337</td> <td>    0.001</td> <td>  -44.866</td> <td> 0.000</td> <td>   -0.035</td> <td>   -0.032</td>
</tr>
<tr>
  <th>C(country_standard)[T.Uganda]</th>                           <td>   -0.0514</td> <td>    0.000</td> <td> -149.155</td> <td> 0.000</td> <td>   -0.052</td> <td>   -0.051</td>
</tr>
<tr>
  <th>C(country_standard)[T.Ukraine]</th>                          <td>    0.0040</td> <td>    0.000</td> <td>   14.747</td> <td> 0.000</td> <td>    0.003</td> <td>    0.004</td>
</tr>
<tr>
  <th>C(country_standard)[T.United Arab Emirates]</th>             <td>    0.0493</td> <td>    0.004</td> <td>   12.037</td> <td> 0.000</td> <td>    0.041</td> <td>    0.057</td>
</tr>
<tr>
  <th>C(country_standard)[T.United Kingdom]</th>                   <td>    0.0525</td> <td>    0.001</td> <td>   37.326</td> <td> 0.000</td> <td>    0.050</td> <td>    0.055</td>
</tr>
<tr>
  <th>C(country_standard)[T.United States]</th>                    <td>    0.0133</td> <td>    0.001</td> <td>   21.480</td> <td> 0.000</td> <td>    0.012</td> <td>    0.015</td>
</tr>
<tr>
  <th>C(country_standard)[T.Uruguay]</th>                          <td>   -0.0410</td> <td>    0.002</td> <td>  -17.355</td> <td> 0.000</td> <td>   -0.046</td> <td>   -0.036</td>
</tr>
<tr>
  <th>C(country_standard)[T.Uzbekistan]</th>                       <td>   -0.0383</td> <td>    0.002</td> <td>  -19.019</td> <td> 0.000</td> <td>   -0.042</td> <td>   -0.034</td>
</tr>
<tr>
  <th>C(country_standard)[T.Vanuatu]</th>                          <td>   -0.0673</td> <td>    0.003</td> <td>  -24.695</td> <td> 0.000</td> <td>   -0.073</td> <td>   -0.062</td>
</tr>
<tr>
  <th>C(country_standard)[T.Vatican City]</th>                     <td>   -0.0450</td> <td>    0.013</td> <td>   -3.585</td> <td> 0.000</td> <td>   -0.070</td> <td>   -0.020</td>
</tr>
<tr>
  <th>C(country_standard)[T.Venezuela]</th>                        <td>   -0.0255</td> <td>    0.002</td> <td>  -15.431</td> <td> 0.000</td> <td>   -0.029</td> <td>   -0.022</td>
</tr>
<tr>
  <th>C(country_standard)[T.Vietnam]</th>                          <td>   -0.0360</td> <td>    0.001</td> <td>  -26.229</td> <td> 0.000</td> <td>   -0.039</td> <td>   -0.033</td>
</tr>
<tr>
  <th>C(country_standard)[T.Western Sahara]</th>                   <td>   -0.0297</td> <td>    0.003</td> <td>   -8.658</td> <td> 0.000</td> <td>   -0.036</td> <td>   -0.023</td>
</tr>
<tr>
  <th>C(country_standard)[T.Yemen]</th>                            <td>   -0.0342</td> <td>    0.002</td> <td>  -18.967</td> <td> 0.000</td> <td>   -0.038</td> <td>   -0.031</td>
</tr>
<tr>
  <th>C(country_standard)[T.Zambia]</th>                           <td>   -0.0411</td> <td>    0.001</td> <td>  -68.308</td> <td> 0.000</td> <td>   -0.042</td> <td>   -0.040</td>
</tr>
<tr>
  <th>C(country_standard)[T.Zimbabwe]</th>                         <td>   -0.0516</td> <td>    0.000</td> <td> -110.750</td> <td> 0.000</td> <td>   -0.052</td> <td>   -0.051</td>
</tr>
<tr>
  <th>transparency_index</th>                                      <td>    0.0446</td> <td>    0.011</td> <td>    4.078</td> <td> 0.000</td> <td>    0.023</td> <td>    0.066</td>
</tr>
<tr>
  <th>gdp</th>                                                     <td>-9.131e-16</td> <td> 6.35e-17</td> <td>  -14.371</td> <td> 0.000</td> <td>-1.04e-15</td> <td>-7.89e-16</td>
</tr>
<tr>
  <th>gdp_per_capita</th>                                          <td>-1.287e-06</td> <td> 4.05e-08</td> <td>  -31.772</td> <td> 0.000</td> <td>-1.37e-06</td> <td>-1.21e-06</td>
</tr>
<tr>
  <th>gini_2020</th>                                               <td>   -0.1648</td> <td>    0.004</td> <td>  -42.246</td> <td> 0.000</td> <td>   -0.172</td> <td>   -0.157</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>4974.751</td> <th>  Durbin-Watson:     </th>  <td>   2.083</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>849143.865</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 8.135</td>  <th>  Prob(JB):          </th>  <td>    0.00</td> 
</tr>
<tr>
  <th>Kurtosis:</th>       <td>76.299</td>  <th>  Cond. No.          </th>  <td>4.96e+26</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors are robust to cluster correlation (cluster)<br/>[2] The smallest eigenvalue is 5.07e-26. This might indicate that there are<br/>strong multicollinearity problems or that the design matrix is singular.




```python
df_2020 = ndi_df.loc[ndi_df['year'] == 2020]
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
      <th>trust_index</th>
      <th>corruption_index</th>
      <th>effectiveness_index</th>
      <th>budget_participation_index</th>
      <th>pandemic_dem_violation_index</th>
      <th>covid_index</th>
      <th>gdp</th>
      <th>gdp_per_capita</th>
      <th>gini_2020</th>
      <th>percap_domestic_health_expenditure</th>
      <th>median_age</th>
      <th>aged_65_older</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14</th>
      <td>Afghanistan</td>
      <td>2020</td>
      <td>0.416170</td>
      <td>0.532376</td>
      <td>0.379785</td>
      <td>0.120559</td>
      <td>0.0</td>
      <td>0.214286</td>
      <td>0.012655</td>
      <td>1.929110e+10</td>
      <td>1803.987000</td>
      <td>0.655000</td>
      <td>2.578007</td>
      <td>18.600000</td>
      <td>2.581000</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Albania</td>
      <td>2020</td>
      <td>0.659931</td>
      <td>0.532376</td>
      <td>0.744248</td>
      <td>0.517820</td>
      <td>0.0</td>
      <td>0.357143</td>
      <td>0.525529</td>
      <td>1.527918e+10</td>
      <td>11803.431000</td>
      <td>0.637000</td>
      <td>148.436569</td>
      <td>38.000000</td>
      <td>13.188000</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Algeria</td>
      <td>2020</td>
      <td>0.309386</td>
      <td>0.282815</td>
      <td>0.725630</td>
      <td>0.319232</td>
      <td>0.0</td>
      <td>0.428571</td>
      <td>0.011143</td>
      <td>1.710913e+11</td>
      <td>13913.839000</td>
      <td>0.749000</td>
      <td>168.449661</td>
      <td>29.100000</td>
      <td>6.211000</td>
    </tr>
    <tr>
      <th>76</th>
      <td>Andorra</td>
      <td>2020</td>
      <td>0.612422</td>
      <td>0.532376</td>
      <td>0.924998</td>
      <td>0.503408</td>
      <td>0.0</td>
      <td>0.336622</td>
      <td>0.924444</td>
      <td>3.154058e+09</td>
      <td>18862.965835</td>
      <td>0.728947</td>
      <td>1916.984497</td>
      <td>30.359825</td>
      <td>8.649119</td>
    </tr>
    <tr>
      <th>92</th>
      <td>Angola</td>
      <td>2020</td>
      <td>0.386187</td>
      <td>0.532376</td>
      <td>0.549063</td>
      <td>0.274033</td>
      <td>0.0</td>
      <td>0.357143</td>
      <td>0.019782</td>
      <td>8.881570e+10</td>
      <td>5819.495000</td>
      <td>0.731000</td>
      <td>36.737221</td>
      <td>16.800000</td>
      <td>2.405000</td>
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
    </tr>
    <tr>
      <th>3541</th>
      <td>Venezuela</td>
      <td>2020</td>
      <td>0.264281</td>
      <td>0.532376</td>
      <td>0.358032</td>
      <td>0.160171</td>
      <td>0.0</td>
      <td>0.928571</td>
      <td>0.010997</td>
      <td>4.823593e+11</td>
      <td>16745.022000</td>
      <td>0.743000</td>
      <td>122.942413</td>
      <td>29.000000</td>
      <td>6.614000</td>
    </tr>
    <tr>
      <th>3557</th>
      <td>Vietnam</td>
      <td>2020</td>
      <td>0.314951</td>
      <td>0.851301</td>
      <td>0.767232</td>
      <td>0.397338</td>
      <td>0.0</td>
      <td>0.214286</td>
      <td>0.293635</td>
      <td>2.619212e+11</td>
      <td>6171.884000</td>
      <td>0.761000</td>
      <td>69.108612</td>
      <td>32.600000</td>
      <td>7.150000</td>
    </tr>
    <tr>
      <th>3581</th>
      <td>Yemen</td>
      <td>2020</td>
      <td>0.187194</td>
      <td>0.005970</td>
      <td>0.313708</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.336622</td>
      <td>0.028759</td>
      <td>2.258108e+10</td>
      <td>1479.147000</td>
      <td>0.798000</td>
      <td>7.451180</td>
      <td>20.300000</td>
      <td>2.922000</td>
    </tr>
    <tr>
      <th>3597</th>
      <td>Zambia</td>
      <td>2020</td>
      <td>0.582441</td>
      <td>0.473912</td>
      <td>0.711931</td>
      <td>0.324835</td>
      <td>0.0</td>
      <td>0.428571</td>
      <td>0.399771</td>
      <td>2.330977e+10</td>
      <td>3689.251000</td>
      <td>0.798000</td>
      <td>29.700403</td>
      <td>17.700000</td>
      <td>2.480000</td>
    </tr>
    <tr>
      <th>3613</th>
      <td>Zimbabwe</td>
      <td>2020</td>
      <td>0.461171</td>
      <td>0.272614</td>
      <td>0.492241</td>
      <td>0.245736</td>
      <td>0.0</td>
      <td>0.500000</td>
      <td>0.271321</td>
      <td>2.144076e+10</td>
      <td>1899.775000</td>
      <td>0.719000</td>
      <td>39.249222</td>
      <td>19.600000</td>
      <td>2.822000</td>
    </tr>
  </tbody>
</table>
<p>193 rows × 15 columns</p>
</div>




```python
#COVID outcomes model
reg_covid = smf.ols('covid_index ~ transparency_index + gdp + gdp_per_capita + gini_2020 + percap_domestic_health_expenditure + median_age + aged_65_older', df_2020).fit()
```


```python
reg_covid.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>covid_index</td>   <th>  R-squared:         </th> <td>   0.406</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.384</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   18.09</td>
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 05 Mar 2021</td> <th>  Prob (F-statistic):</th> <td>3.18e-18</td>
</tr>
<tr>
  <th>Time:</th>                 <td>13:17:43</td>     <th>  Log-Likelihood:    </th> <td>  2.8006</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   193</td>      <th>  AIC:               </th> <td>   10.40</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   185</td>      <th>  BIC:               </th> <td>   36.50</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     7</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
                   <td></td>                     <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>                          <td>    0.0685</td> <td>    0.224</td> <td>    0.307</td> <td> 0.760</td> <td>   -0.373</td> <td>    0.510</td>
</tr>
<tr>
  <th>transparency_index</th>                 <td>    0.1659</td> <td>    0.100</td> <td>    1.666</td> <td> 0.097</td> <td>   -0.031</td> <td>    0.362</td>
</tr>
<tr>
  <th>gdp</th>                                <td> 6.084e-15</td> <td>    1e-14</td> <td>    0.606</td> <td> 0.545</td> <td>-1.37e-14</td> <td> 2.59e-14</td>
</tr>
<tr>
  <th>gdp_per_capita</th>                     <td> 4.814e-06</td> <td> 1.73e-06</td> <td>    2.778</td> <td> 0.006</td> <td>  1.4e-06</td> <td> 8.23e-06</td>
</tr>
<tr>
  <th>gini_2020</th>                          <td>   -0.3196</td> <td>    0.284</td> <td>   -1.125</td> <td> 0.262</td> <td>   -0.880</td> <td>    0.241</td>
</tr>
<tr>
  <th>percap_domestic_health_expenditure</th> <td> -2.29e-07</td> <td>  2.7e-05</td> <td>   -0.008</td> <td> 0.993</td> <td>-5.36e-05</td> <td> 5.31e-05</td>
</tr>
<tr>
  <th>median_age</th>                         <td>    0.0104</td> <td>    0.006</td> <td>    1.680</td> <td> 0.095</td> <td>   -0.002</td> <td>    0.023</td>
</tr>
<tr>
  <th>aged_65_older</th>                      <td>    0.0011</td> <td>    0.009</td> <td>    0.117</td> <td> 0.907</td> <td>   -0.017</td> <td>    0.020</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 3.407</td> <th>  Durbin-Watson:     </th> <td>   1.999</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.182</td> <th>  Jarque-Bera (JB):  </th> <td>   3.313</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.320</td> <th>  Prob(JB):          </th> <td>   0.191</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.962</td> <th>  Cond. No.          </th> <td>3.95e+13</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 3.95e+13. This might indicate that there are<br/>strong multicollinearity or other numerical problems.




```python
#Fixed effects or not, only 2020? 
#reg_pandemic_violations = smf.ols('pandemic_dem_violation_index ~ transparency_index + gdp + gdp_per_capita + gini_2020 + percap_domestic_health_expenditure + median_age + aged_65_older + C(country_standard)', ndi_df).fit(cov_type='cluster', cov_kwds={'groups': ndi_df['country_standard']})
```


```python
reg_pandemic_violations = smf.ols('pandemic_dem_violation_index ~ transparency_index + gdp + gdp_per_capita + gini_2020 + percap_domestic_health_expenditure + median_age + aged_65_older', df_2020).fit()
```


```python
reg_pandemic_violations.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>    <td>pandemic_dem_violation_index</td> <th>  R-squared:         </th> <td>   0.267</td>
</tr>
<tr>
  <th>Model:</th>                         <td>OLS</td>             <th>  Adj. R-squared:    </th> <td>   0.239</td>
</tr>
<tr>
  <th>Method:</th>                   <td>Least Squares</td>        <th>  F-statistic:       </th> <td>   9.617</td>
</tr>
<tr>
  <th>Date:</th>                   <td>Fri, 05 Mar 2021</td>       <th>  Prob (F-statistic):</th> <td>3.51e-10</td>
</tr>
<tr>
  <th>Time:</th>                       <td>13:17:43</td>           <th>  Log-Likelihood:    </th> <td>  91.696</td>
</tr>
<tr>
  <th>No. Observations:</th>            <td>   193</td>            <th>  AIC:               </th> <td>  -167.4</td>
</tr>
<tr>
  <th>Df Residuals:</th>                <td>   185</td>            <th>  BIC:               </th> <td>  -141.3</td>
</tr>
<tr>
  <th>Df Model:</th>                    <td>     7</td>            <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>            <td>nonrobust</td>          <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
                   <td></td>                     <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>                          <td>    0.5105</td> <td>    0.141</td> <td>    3.619</td> <td> 0.000</td> <td>    0.232</td> <td>    0.789</td>
</tr>
<tr>
  <th>transparency_index</th>                 <td>   -0.1998</td> <td>    0.063</td> <td>   -3.181</td> <td> 0.002</td> <td>   -0.324</td> <td>   -0.076</td>
</tr>
<tr>
  <th>gdp</th>                                <td> 1.262e-14</td> <td> 6.33e-15</td> <td>    1.993</td> <td> 0.048</td> <td>  1.3e-16</td> <td> 2.51e-14</td>
</tr>
<tr>
  <th>gdp_per_capita</th>                     <td> 1.067e-06</td> <td> 1.09e-06</td> <td>    0.976</td> <td> 0.330</td> <td>-1.09e-06</td> <td> 3.22e-06</td>
</tr>
<tr>
  <th>gini_2020</th>                          <td>   -0.0092</td> <td>    0.179</td> <td>   -0.051</td> <td> 0.959</td> <td>   -0.363</td> <td>    0.344</td>
</tr>
<tr>
  <th>percap_domestic_health_expenditure</th> <td>-5.339e-05</td> <td> 1.71e-05</td> <td>   -3.129</td> <td> 0.002</td> <td> -8.7e-05</td> <td>-1.97e-05</td>
</tr>
<tr>
  <th>median_age</th>                         <td>   -0.0012</td> <td>    0.004</td> <td>   -0.308</td> <td> 0.759</td> <td>   -0.009</td> <td>    0.007</td>
</tr>
<tr>
  <th>aged_65_older</th>                      <td>    0.0008</td> <td>    0.006</td> <td>    0.129</td> <td> 0.898</td> <td>   -0.011</td> <td>    0.012</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>44.580</td> <th>  Durbin-Watson:     </th> <td>   1.949</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  97.304</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 1.055</td> <th>  Prob(JB):          </th> <td>7.43e-22</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 5.766</td> <th>  Cond. No.          </th> <td>3.95e+13</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 3.95e+13. This might indicate that there are<br/>strong multicollinearity or other numerical problems.




```python

```
