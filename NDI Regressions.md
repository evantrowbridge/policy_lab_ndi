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
    Requirement already satisfied: pandas>=0.24 in ./opt/anaconda3/lib/python3.7/site-packages (from linearmodels) (0.25.3)
    Requirement already satisfied: Cython>=0.29.21 in ./opt/anaconda3/lib/python3.7/site-packages (from linearmodels) (0.29.22)
    Requirement already satisfied: patsy in ./opt/anaconda3/lib/python3.7/site-packages (from linearmodels) (0.5.1)
    Requirement already satisfied: mypy-extensions>=0.4 in ./opt/anaconda3/lib/python3.7/site-packages (from linearmodels) (0.4.3)
    Requirement already satisfied: property-cached>=1.6.3 in ./opt/anaconda3/lib/python3.7/site-packages (from linearmodels) (1.6.4)
    Requirement already satisfied: pyhdfe>=0.1 in ./opt/anaconda3/lib/python3.7/site-packages (from linearmodels) (0.1.0)
    Requirement already satisfied: scipy>=1.2 in ./opt/anaconda3/lib/python3.7/site-packages (from linearmodels) (1.4.1)
    Requirement already satisfied: statsmodels>=0.11 in ./opt/anaconda3/lib/python3.7/site-packages (from linearmodels) (0.11.0)
    Requirement already satisfied: numpy>=1.16 in ./opt/anaconda3/lib/python3.7/site-packages (from linearmodels) (1.18.1)
    Requirement already satisfied: python-dateutil>=2.6.1 in ./opt/anaconda3/lib/python3.7/site-packages (from pandas>=0.24->linearmodels) (2.8.1)
    Requirement already satisfied: pytz>=2017.2 in ./opt/anaconda3/lib/python3.7/site-packages (from pandas>=0.24->linearmodels) (2019.3)
    Requirement already satisfied: six in ./opt/anaconda3/lib/python3.7/site-packages (from patsy->linearmodels) (1.14.0)



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
      <td>0.0</td>
      <td>0.621583</td>
      <td>NaN</td>
      <td>0.899550</td>
      <td>0.109936</td>
      <td>0.0</td>
      <td>0.214286</td>
      <td>0.012655</td>
      <td>1.929110e+10</td>
      <td>NaN</td>
      <td>2.578007</td>
      <td>18.6</td>
      <td>2.581</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>2007</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.623198</td>
      <td>NaN</td>
      <td>0.935146</td>
      <td>0.133040</td>
      <td>0.0</td>
      <td>0.214286</td>
      <td>0.012655</td>
      <td>1.929110e+10</td>
      <td>NaN</td>
      <td>2.578007</td>
      <td>18.6</td>
      <td>2.581</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Afghanistan</td>
      <td>2008</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.629477</td>
      <td>NaN</td>
      <td>0.946890</td>
      <td>0.143647</td>
      <td>0.0</td>
      <td>0.214286</td>
      <td>0.012655</td>
      <td>1.929110e+10</td>
      <td>NaN</td>
      <td>2.578007</td>
      <td>18.6</td>
      <td>2.581</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Afghanistan</td>
      <td>2009</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.630297</td>
      <td>NaN</td>
      <td>0.923037</td>
      <td>0.144919</td>
      <td>0.0</td>
      <td>0.214286</td>
      <td>0.012655</td>
      <td>1.929110e+10</td>
      <td>NaN</td>
      <td>2.578007</td>
      <td>18.6</td>
      <td>2.581</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Afghanistan</td>
      <td>2010</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.627897</td>
      <td>NaN</td>
      <td>0.946299</td>
      <td>0.147968</td>
      <td>0.0</td>
      <td>0.214286</td>
      <td>0.012655</td>
      <td>1.929110e+10</td>
      <td>NaN</td>
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
    </tr>
    <tr>
      <th>3598</th>
      <td>Zimbabwe</td>
      <td>2017</td>
      <td>0.138089</td>
      <td>0.0</td>
      <td>0.543230</td>
      <td>NaN</td>
      <td>0.847985</td>
      <td>0.237870</td>
      <td>0.0</td>
      <td>0.500000</td>
      <td>0.271321</td>
      <td>2.144076e+10</td>
      <td>44.3</td>
      <td>39.249222</td>
      <td>19.6</td>
      <td>2.822</td>
    </tr>
    <tr>
      <th>3599</th>
      <td>Zimbabwe</td>
      <td>2018</td>
      <td>0.396885</td>
      <td>0.0</td>
      <td>0.533216</td>
      <td>NaN</td>
      <td>0.843607</td>
      <td>0.236812</td>
      <td>0.0</td>
      <td>0.500000</td>
      <td>0.271321</td>
      <td>2.144076e+10</td>
      <td>44.3</td>
      <td>39.249222</td>
      <td>19.6</td>
      <td>2.822</td>
    </tr>
    <tr>
      <th>3600</th>
      <td>Zimbabwe</td>
      <td>2019</td>
      <td>0.394583</td>
      <td>0.0</td>
      <td>0.504282</td>
      <td>NaN</td>
      <td>0.832169</td>
      <td>0.261814</td>
      <td>0.0</td>
      <td>0.500000</td>
      <td>0.271321</td>
      <td>2.144076e+10</td>
      <td>44.3</td>
      <td>39.249222</td>
      <td>19.6</td>
      <td>2.822</td>
    </tr>
    <tr>
      <th>3601</th>
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
      <td>39.249222</td>
      <td>19.6</td>
      <td>2.822</td>
    </tr>
    <tr>
      <th>3602</th>
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
      <td>39.249222</td>
      <td>19.6</td>
      <td>2.822</td>
    </tr>
  </tbody>
</table>
<p>3603 rows × 16 columns</p>
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
      <td>0.580426</td>
      <td>0.0</td>
      <td>0.621583</td>
      <td>0.409391</td>
      <td>0.899550</td>
      <td>0.109936</td>
      <td>0.0</td>
      <td>0.214286</td>
      <td>0.012655</td>
      <td>1.929110e+10</td>
      <td>38.076323</td>
      <td>2.578007</td>
      <td>18.6</td>
      <td>2.581</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>2007</td>
      <td>0.580426</td>
      <td>0.0</td>
      <td>0.623198</td>
      <td>0.409391</td>
      <td>0.935146</td>
      <td>0.133040</td>
      <td>0.0</td>
      <td>0.214286</td>
      <td>0.012655</td>
      <td>1.929110e+10</td>
      <td>38.076323</td>
      <td>2.578007</td>
      <td>18.6</td>
      <td>2.581</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Afghanistan</td>
      <td>2008</td>
      <td>0.580426</td>
      <td>0.0</td>
      <td>0.629477</td>
      <td>0.409391</td>
      <td>0.946890</td>
      <td>0.143647</td>
      <td>0.0</td>
      <td>0.214286</td>
      <td>0.012655</td>
      <td>1.929110e+10</td>
      <td>38.076323</td>
      <td>2.578007</td>
      <td>18.6</td>
      <td>2.581</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Afghanistan</td>
      <td>2009</td>
      <td>0.580426</td>
      <td>0.0</td>
      <td>0.630297</td>
      <td>0.409391</td>
      <td>0.923037</td>
      <td>0.144919</td>
      <td>0.0</td>
      <td>0.214286</td>
      <td>0.012655</td>
      <td>1.929110e+10</td>
      <td>38.076323</td>
      <td>2.578007</td>
      <td>18.6</td>
      <td>2.581</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Afghanistan</td>
      <td>2010</td>
      <td>0.580426</td>
      <td>0.0</td>
      <td>0.627897</td>
      <td>0.409391</td>
      <td>0.946299</td>
      <td>0.147968</td>
      <td>0.0</td>
      <td>0.214286</td>
      <td>0.012655</td>
      <td>1.929110e+10</td>
      <td>38.076323</td>
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
    </tr>
    <tr>
      <th>3598</th>
      <td>Zimbabwe</td>
      <td>2017</td>
      <td>0.138089</td>
      <td>0.0</td>
      <td>0.543230</td>
      <td>0.409391</td>
      <td>0.847985</td>
      <td>0.237870</td>
      <td>0.0</td>
      <td>0.500000</td>
      <td>0.271321</td>
      <td>2.144076e+10</td>
      <td>44.300000</td>
      <td>39.249222</td>
      <td>19.6</td>
      <td>2.822</td>
    </tr>
    <tr>
      <th>3599</th>
      <td>Zimbabwe</td>
      <td>2018</td>
      <td>0.396885</td>
      <td>0.0</td>
      <td>0.533216</td>
      <td>0.409391</td>
      <td>0.843607</td>
      <td>0.236812</td>
      <td>0.0</td>
      <td>0.500000</td>
      <td>0.271321</td>
      <td>2.144076e+10</td>
      <td>44.300000</td>
      <td>39.249222</td>
      <td>19.6</td>
      <td>2.822</td>
    </tr>
    <tr>
      <th>3600</th>
      <td>Zimbabwe</td>
      <td>2019</td>
      <td>0.394583</td>
      <td>0.0</td>
      <td>0.504282</td>
      <td>0.409391</td>
      <td>0.832169</td>
      <td>0.261814</td>
      <td>0.0</td>
      <td>0.500000</td>
      <td>0.271321</td>
      <td>2.144076e+10</td>
      <td>44.300000</td>
      <td>39.249222</td>
      <td>19.6</td>
      <td>2.822</td>
    </tr>
    <tr>
      <th>3601</th>
      <td>Zimbabwe</td>
      <td>2020</td>
      <td>0.580426</td>
      <td>0.0</td>
      <td>0.652288</td>
      <td>0.402851</td>
      <td>0.568511</td>
      <td>0.520235</td>
      <td>0.0</td>
      <td>0.500000</td>
      <td>0.271321</td>
      <td>2.144076e+10</td>
      <td>44.300000</td>
      <td>39.249222</td>
      <td>19.6</td>
      <td>2.822</td>
    </tr>
    <tr>
      <th>3602</th>
      <td>Zimbabwe</td>
      <td>2021</td>
      <td>0.580426</td>
      <td>0.0</td>
      <td>0.652288</td>
      <td>0.409391</td>
      <td>0.568511</td>
      <td>0.520235</td>
      <td>0.0</td>
      <td>0.500000</td>
      <td>0.271321</td>
      <td>2.144076e+10</td>
      <td>44.300000</td>
      <td>39.249222</td>
      <td>19.6</td>
      <td>2.822</td>
    </tr>
  </tbody>
</table>
<p>3603 rows × 16 columns</p>
</div>




```python
pairplot = sns.pairplot(data = ndi_df)
```


![png](output_5_0.png)



```python
#pairplot.savefig("Indices, transparency and controls pairplot.png")
```


```python
reg_accountability = smf.ols('accountability_index ~ transparency_index + gdp + gini + C(country_standard)', ndi_df).fit(cov_type='cluster', cov_kwds={'groups': ndi_df['country_standard']})
```


```python
reg_accountability.summary()
```

    /Users/katiacordoba/opt/anaconda3/lib/python3.7/site-packages/statsmodels/base/model.py:1832: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 231, but rank is 228
      'rank is %d' % (J, J_), ValueWarning)





<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>    <td>accountability_index</td> <th>  R-squared:         </th> <td>   0.841</td> 
</tr>
<tr>
  <th>Model:</th>                     <td>OLS</td>         <th>  Adj. R-squared:    </th> <td>   0.841</td> 
</tr>
<tr>
  <th>Method:</th>               <td>Least Squares</td>    <th>  F-statistic:       </th> <td>2.223e+08</td>
</tr>
<tr>
  <th>Date:</th>               <td>Fri, 05 Mar 2021</td>   <th>  Prob (F-statistic):</th>  <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                   <td>23:11:41</td>       <th>  Log-Likelihood:    </th> <td>  4430.8</td> 
</tr>
<tr>
  <th>No. Observations:</th>        <td>  3603</td>        <th>  AIC:               </th> <td>  -8852.</td> 
</tr>
<tr>
  <th>Df Residuals:</th>            <td>  3598</td>        <th>  BIC:               </th> <td>  -8821.</td> 
</tr>
<tr>
  <th>Df Model:</th>                <td>     4</td>        <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>         <td>cluster</td>       <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
                             <td></td>                                <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>                                               <td>    0.5909</td> <td>    0.006</td> <td>  105.402</td> <td> 0.000</td> <td>    0.580</td> <td>    0.602</td>
</tr>
<tr>
  <th>C(country_standard)[T.Albania]</th>                          <td>    0.0683</td> <td>    0.004</td> <td>   15.533</td> <td> 0.000</td> <td>    0.060</td> <td>    0.077</td>
</tr>
<tr>
  <th>C(country_standard)[T.Algeria]</th>                          <td>   -0.1592</td> <td>    0.000</td> <td> -511.363</td> <td> 0.000</td> <td>   -0.160</td> <td>   -0.159</td>
</tr>
<tr>
  <th>C(country_standard)[T.American Samoa]</th>                   <td>   -0.0133</td> <td>    0.004</td> <td>   -3.589</td> <td> 0.000</td> <td>   -0.021</td> <td>   -0.006</td>
</tr>
<tr>
  <th>C(country_standard)[T.Andorra]</th>                          <td>   -0.0286</td> <td>    0.006</td> <td>   -5.089</td> <td> 0.000</td> <td>   -0.040</td> <td>   -0.018</td>
</tr>
<tr>
  <th>C(country_standard)[T.Angola]</th>                           <td>   -0.1737</td> <td>    0.000</td> <td> -522.803</td> <td> 0.000</td> <td>   -0.174</td> <td>   -0.173</td>
</tr>
<tr>
  <th>C(country_standard)[T.Anguilla]</th>                         <td>   -0.0114</td> <td>    0.003</td> <td>   -3.470</td> <td> 0.001</td> <td>   -0.018</td> <td>   -0.005</td>
</tr>
<tr>
  <th>C(country_standard)[T.Antigua & Barbuda]</th>                <td>   -0.0236</td> <td>    0.005</td> <td>   -4.636</td> <td> 0.000</td> <td>   -0.034</td> <td>   -0.014</td>
</tr>
<tr>
  <th>C(country_standard)[T.Argentina]</th>                        <td>    0.0906</td> <td>    0.005</td> <td>   19.116</td> <td> 0.000</td> <td>    0.081</td> <td>    0.100</td>
</tr>
<tr>
  <th>C(country_standard)[T.Armenia]</th>                          <td>   -0.0753</td> <td>    0.002</td> <td>  -32.193</td> <td> 0.000</td> <td>   -0.080</td> <td>   -0.071</td>
</tr>
<tr>
  <th>C(country_standard)[T.Aruba]</th>                            <td>   -0.0315</td> <td>    0.005</td> <td>   -5.770</td> <td> 0.000</td> <td>   -0.042</td> <td>   -0.021</td>
</tr>
<tr>
  <th>C(country_standard)[T.Australia]</th>                        <td>    0.2166</td> <td>    0.005</td> <td>   42.892</td> <td> 0.000</td> <td>    0.207</td> <td>    0.226</td>
</tr>
<tr>
  <th>C(country_standard)[T.Austria]</th>                          <td>    0.1696</td> <td>    0.005</td> <td>   31.002</td> <td> 0.000</td> <td>    0.159</td> <td>    0.180</td>
</tr>
<tr>
  <th>C(country_standard)[T.Azerbaijan]</th>                       <td>   -0.2924</td> <td>    0.000</td> <td> -603.368</td> <td> 0.000</td> <td>   -0.293</td> <td>   -0.291</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bahamas]</th>                          <td>   -0.0260</td> <td>    0.005</td> <td>   -4.869</td> <td> 0.000</td> <td>   -0.036</td> <td>   -0.016</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bahrain]</th>                          <td>   -0.2233</td> <td>    0.000</td> <td> -548.489</td> <td> 0.000</td> <td>   -0.224</td> <td>   -0.222</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bangladesh]</th>                       <td>   -0.1249</td> <td>    0.002</td> <td>  -60.061</td> <td> 0.000</td> <td>   -0.129</td> <td>   -0.121</td>
</tr>
<tr>
  <th>C(country_standard)[T.Barbados]</th>                         <td>    0.1176</td> <td>    0.006</td> <td>   21.271</td> <td> 0.000</td> <td>    0.107</td> <td>    0.128</td>
</tr>
<tr>
  <th>C(country_standard)[T.Belarus]</th>                          <td>   -0.2785</td> <td>    0.001</td> <td> -490.356</td> <td> 0.000</td> <td>   -0.280</td> <td>   -0.277</td>
</tr>
<tr>
  <th>C(country_standard)[T.Belgium]</th>                          <td>    0.1947</td> <td>    0.005</td> <td>   36.043</td> <td> 0.000</td> <td>    0.184</td> <td>    0.205</td>
</tr>
<tr>
  <th>C(country_standard)[T.Belize]</th>                           <td>   -0.0231</td> <td>    0.005</td> <td>   -4.587</td> <td> 0.000</td> <td>   -0.033</td> <td>   -0.013</td>
</tr>
<tr>
  <th>C(country_standard)[T.Benin]</th>                            <td>    0.1050</td> <td>    0.005</td> <td>   22.749</td> <td> 0.000</td> <td>    0.096</td> <td>    0.114</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bermuda]</th>                          <td>   -0.0114</td> <td>    0.003</td> <td>   -3.450</td> <td> 0.001</td> <td>   -0.018</td> <td>   -0.005</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bhutan]</th>                           <td>    0.0332</td> <td>    0.003</td> <td>   12.306</td> <td> 0.000</td> <td>    0.028</td> <td>    0.038</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bolivia]</th>                          <td>    0.0056</td> <td>    0.003</td> <td>    2.054</td> <td> 0.040</td> <td>    0.000</td> <td>    0.011</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bosnia & Herzegovina]</th>             <td>    0.0235</td> <td>    0.003</td> <td>    9.135</td> <td> 0.000</td> <td>    0.018</td> <td>    0.029</td>
</tr>
<tr>
  <th>C(country_standard)[T.Botswana]</th>                         <td>    0.1119</td> <td>    0.005</td> <td>   22.095</td> <td> 0.000</td> <td>    0.102</td> <td>    0.122</td>
</tr>
<tr>
  <th>C(country_standard)[T.Brazil]</th>                           <td>    0.1542</td> <td>    0.004</td> <td>   37.150</td> <td> 0.000</td> <td>    0.146</td> <td>    0.162</td>
</tr>
<tr>
  <th>C(country_standard)[T.British Virgin Islands]</th>           <td>   -0.0089</td> <td>    0.003</td> <td>   -2.646</td> <td> 0.008</td> <td>   -0.016</td> <td>   -0.002</td>
</tr>
<tr>
  <th>C(country_standard)[T.Brunei]</th>                           <td>    0.0211</td> <td>    0.000</td> <td>   52.099</td> <td> 0.000</td> <td>    0.020</td> <td>    0.022</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bulgaria]</th>                         <td>    0.1092</td> <td>    0.005</td> <td>   22.473</td> <td> 0.000</td> <td>    0.100</td> <td>    0.119</td>
</tr>
<tr>
  <th>C(country_standard)[T.Burkina Faso]</th>                     <td>    0.0526</td> <td>    0.003</td> <td>   20.265</td> <td> 0.000</td> <td>    0.048</td> <td>    0.058</td>
</tr>
<tr>
  <th>C(country_standard)[T.Burundi]</th>                          <td>   -0.1805</td> <td>    0.000</td> <td> -482.507</td> <td> 0.000</td> <td>   -0.181</td> <td>   -0.180</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cambodia]</th>                         <td>   -0.1724</td> <td>    0.000</td> <td> -574.413</td> <td> 0.000</td> <td>   -0.173</td> <td>   -0.172</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cameroon]</th>                         <td>   -0.0965</td> <td>    0.000</td> <td> -315.150</td> <td> 0.000</td> <td>   -0.097</td> <td>   -0.096</td>
</tr>
<tr>
  <th>C(country_standard)[T.Canada]</th>                           <td>    0.1532</td> <td>    0.005</td> <td>   31.028</td> <td> 0.000</td> <td>    0.144</td> <td>    0.163</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cape Verde]</th>                       <td>    0.1681</td> <td>    0.005</td> <td>   30.983</td> <td> 0.000</td> <td>    0.157</td> <td>    0.179</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cayman Islands]</th>                   <td>   -0.0124</td> <td>    0.004</td> <td>   -3.194</td> <td> 0.001</td> <td>   -0.020</td> <td>   -0.005</td>
</tr>
<tr>
  <th>C(country_standard)[T.Central African Republic]</th>         <td>   -0.0361</td> <td>    0.000</td> <td>  -98.518</td> <td> 0.000</td> <td>   -0.037</td> <td>   -0.035</td>
</tr>
<tr>
  <th>C(country_standard)[T.Chad]</th>                             <td>   -0.2009</td> <td>    0.000</td> <td> -543.207</td> <td> 0.000</td> <td>   -0.202</td> <td>   -0.200</td>
</tr>
<tr>
  <th>C(country_standard)[T.Channel Islands]</th>                  <td>   -0.0069</td> <td>    0.003</td> <td>   -2.413</td> <td> 0.016</td> <td>   -0.012</td> <td>   -0.001</td>
</tr>
<tr>
  <th>C(country_standard)[T.Chile]</th>                            <td>    0.2184</td> <td>    0.005</td> <td>   40.637</td> <td> 0.000</td> <td>    0.208</td> <td>    0.229</td>
</tr>
<tr>
  <th>C(country_standard)[T.China]</th>                            <td>   -0.4224</td> <td>    0.008</td> <td>  -55.919</td> <td> 0.000</td> <td>   -0.437</td> <td>   -0.408</td>
</tr>
<tr>
  <th>C(country_standard)[T.Colombia]</th>                         <td>    0.0874</td> <td>    0.003</td> <td>   31.763</td> <td> 0.000</td> <td>    0.082</td> <td>    0.093</td>
</tr>
<tr>
  <th>C(country_standard)[T.Comoros]</th>                          <td>   -0.0172</td> <td>    0.002</td> <td>   -7.003</td> <td> 0.000</td> <td>   -0.022</td> <td>   -0.012</td>
</tr>
<tr>
  <th>C(country_standard)[T.Congo - Brazzaville]</th>              <td>   -0.1909</td> <td>    0.000</td> <td> -648.537</td> <td> 0.000</td> <td>   -0.191</td> <td>   -0.190</td>
</tr>
<tr>
  <th>C(country_standard)[T.Congo - Kinshasa]</th>                 <td>   -0.1125</td> <td>    0.000</td> <td> -275.549</td> <td> 0.000</td> <td>   -0.113</td> <td>   -0.112</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cook Islands]</th>                     <td>   -0.0065</td> <td>    0.003</td> <td>   -2.064</td> <td> 0.039</td> <td>   -0.013</td> <td>   -0.000</td>
</tr>
<tr>
  <th>C(country_standard)[T.Costa Rica]</th>                       <td>    0.2423</td> <td>    0.006</td> <td>   43.344</td> <td> 0.000</td> <td>    0.231</td> <td>    0.253</td>
</tr>
<tr>
  <th>C(country_standard)[T.Croatia]</th>                          <td>    0.1137</td> <td>    0.005</td> <td>   23.169</td> <td> 0.000</td> <td>    0.104</td> <td>    0.123</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cuba]</th>                             <td>   -0.3115</td> <td>    0.001</td> <td> -584.416</td> <td> 0.000</td> <td>   -0.313</td> <td>   -0.310</td>
</tr>
<tr>
  <th>C(country_standard)[T.Curaçao]</th>                          <td>   -0.0089</td> <td>    0.003</td> <td>   -2.653</td> <td> 0.008</td> <td>   -0.015</td> <td>   -0.002</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cyprus]</th>                           <td>    0.1344</td> <td>    0.005</td> <td>   24.958</td> <td> 0.000</td> <td>    0.124</td> <td>    0.145</td>
</tr>
<tr>
  <th>C(country_standard)[T.Czechia]</th>                          <td>    0.1607</td> <td>    0.005</td> <td>   31.072</td> <td> 0.000</td> <td>    0.151</td> <td>    0.171</td>
</tr>
<tr>
  <th>C(country_standard)[T.Côte d’Ivoire]</th>                    <td>   -0.0066</td> <td>    0.002</td> <td>   -2.794</td> <td> 0.005</td> <td>   -0.011</td> <td>   -0.002</td>
</tr>
<tr>
  <th>C(country_standard)[T.Denmark]</th>                          <td>    0.2512</td> <td>    0.006</td> <td>   44.174</td> <td> 0.000</td> <td>    0.240</td> <td>    0.262</td>
</tr>
<tr>
  <th>C(country_standard)[T.Djibouti]</th>                         <td>   -0.1838</td> <td>    0.000</td> <td> -460.442</td> <td> 0.000</td> <td>   -0.185</td> <td>   -0.183</td>
</tr>
<tr>
  <th>C(country_standard)[T.Dominica]</th>                         <td>   -0.0260</td> <td>    0.005</td> <td>   -4.857</td> <td> 0.000</td> <td>   -0.036</td> <td>   -0.015</td>
</tr>
<tr>
  <th>C(country_standard)[T.Dominican Republic]</th>               <td>   -0.0184</td> <td>    0.004</td> <td>   -5.157</td> <td> 0.000</td> <td>   -0.025</td> <td>   -0.011</td>
</tr>
<tr>
  <th>C(country_standard)[T.Ecuador]</th>                          <td>   -0.0141</td> <td>    0.003</td> <td>   -5.360</td> <td> 0.000</td> <td>   -0.019</td> <td>   -0.009</td>
</tr>
<tr>
  <th>C(country_standard)[T.Egypt]</th>                            <td>   -0.1666</td> <td>    0.000</td> <td> -543.712</td> <td> 0.000</td> <td>   -0.167</td> <td>   -0.166</td>
</tr>
<tr>
  <th>C(country_standard)[T.El Salvador]</th>                      <td>    0.0223</td> <td>    0.004</td> <td>    5.075</td> <td> 0.000</td> <td>    0.014</td> <td>    0.031</td>
</tr>
<tr>
  <th>C(country_standard)[T.Equatorial Guinea]</th>                <td>   -0.2761</td> <td>    0.001</td> <td> -333.998</td> <td> 0.000</td> <td>   -0.278</td> <td>   -0.274</td>
</tr>
<tr>
  <th>C(country_standard)[T.Eritrea]</th>                          <td>   -0.5189</td> <td>    0.001</td> <td> -512.237</td> <td> 0.000</td> <td>   -0.521</td> <td>   -0.517</td>
</tr>
<tr>
  <th>C(country_standard)[T.Estonia]</th>                          <td>    0.2102</td> <td>    0.006</td> <td>   38.155</td> <td> 0.000</td> <td>    0.199</td> <td>    0.221</td>
</tr>
<tr>
  <th>C(country_standard)[T.Eswatini]</th>                         <td>   -0.1812</td> <td>    0.000</td> <td> -620.201</td> <td> 0.000</td> <td>   -0.182</td> <td>   -0.181</td>
</tr>
<tr>
  <th>C(country_standard)[T.Ethiopia]</th>                         <td>   -0.1963</td> <td>    0.000</td> <td> -497.445</td> <td> 0.000</td> <td>   -0.197</td> <td>   -0.196</td>
</tr>
<tr>
  <th>C(country_standard)[T.Faroe Islands]</th>                    <td>   -0.0071</td> <td>    0.003</td> <td>   -2.517</td> <td> 0.012</td> <td>   -0.013</td> <td>   -0.002</td>
</tr>
<tr>
  <th>C(country_standard)[T.Fiji]</th>                             <td>   -0.1628</td> <td>    0.003</td> <td>  -63.158</td> <td> 0.000</td> <td>   -0.168</td> <td>   -0.158</td>
</tr>
<tr>
  <th>C(country_standard)[T.Finland]</th>                          <td>    0.2185</td> <td>    0.006</td> <td>   38.331</td> <td> 0.000</td> <td>    0.207</td> <td>    0.230</td>
</tr>
<tr>
  <th>C(country_standard)[T.France]</th>                           <td>    0.1593</td> <td>    0.004</td> <td>   37.884</td> <td> 0.000</td> <td>    0.151</td> <td>    0.168</td>
</tr>
<tr>
  <th>C(country_standard)[T.French Guiana]</th>                    <td>   -0.0309</td> <td>    0.005</td> <td>   -5.720</td> <td> 0.000</td> <td>   -0.041</td> <td>   -0.020</td>
</tr>
<tr>
  <th>C(country_standard)[T.French Polynesia]</th>                 <td>   -0.0088</td> <td>    0.003</td> <td>   -2.654</td> <td> 0.008</td> <td>   -0.015</td> <td>   -0.002</td>
</tr>
<tr>
  <th>C(country_standard)[T.Gabon]</th>                            <td>   -0.0729</td> <td>    0.000</td> <td> -227.636</td> <td> 0.000</td> <td>   -0.073</td> <td>   -0.072</td>
</tr>
<tr>
  <th>C(country_standard)[T.Gambia]</th>                           <td>   -0.1569</td> <td>    0.001</td> <td> -190.631</td> <td> 0.000</td> <td>   -0.158</td> <td>   -0.155</td>
</tr>
<tr>
  <th>C(country_standard)[T.Georgia]</th>                          <td>    0.0599</td> <td>    0.003</td> <td>   20.525</td> <td> 0.000</td> <td>    0.054</td> <td>    0.066</td>
</tr>
<tr>
  <th>C(country_standard)[T.Germany]</th>                          <td>    0.2242</td> <td>    0.004</td> <td>   58.230</td> <td> 0.000</td> <td>    0.217</td> <td>    0.232</td>
</tr>
<tr>
  <th>C(country_standard)[T.Ghana]</th>                            <td>    0.1250</td> <td>    0.005</td> <td>   24.945</td> <td> 0.000</td> <td>    0.115</td> <td>    0.135</td>
</tr>
<tr>
  <th>C(country_standard)[T.Gibraltar]</th>                        <td>   -0.0072</td> <td>    0.003</td> <td>   -2.526</td> <td> 0.012</td> <td>   -0.013</td> <td>   -0.002</td>
</tr>
<tr>
  <th>C(country_standard)[T.Greece]</th>                           <td>    0.1958</td> <td>    0.005</td> <td>   38.786</td> <td> 0.000</td> <td>    0.186</td> <td>    0.206</td>
</tr>
<tr>
  <th>C(country_standard)[T.Greenland]</th>                        <td>   -0.0290</td> <td>    0.005</td> <td>   -5.609</td> <td> 0.000</td> <td>   -0.039</td> <td>   -0.019</td>
</tr>
<tr>
  <th>C(country_standard)[T.Grenada]</th>                          <td>   -0.0245</td> <td>    0.005</td> <td>   -4.723</td> <td> 0.000</td> <td>   -0.035</td> <td>   -0.014</td>
</tr>
<tr>
  <th>C(country_standard)[T.Guam]</th>                             <td>   -0.0123</td> <td>    0.004</td> <td>   -3.397</td> <td> 0.001</td> <td>   -0.019</td> <td>   -0.005</td>
</tr>
<tr>
  <th>C(country_standard)[T.Guatemala]</th>                        <td>    0.0563</td> <td>    0.003</td> <td>   22.382</td> <td> 0.000</td> <td>    0.051</td> <td>    0.061</td>
</tr>
<tr>
  <th>C(country_standard)[T.Guernsey]</th>                         <td>   -0.0072</td> <td>    0.003</td> <td>   -2.552</td> <td> 0.011</td> <td>   -0.013</td> <td>   -0.002</td>
</tr>
<tr>
  <th>C(country_standard)[T.Guinea]</th>                           <td>   -0.1387</td> <td>    0.002</td> <td>  -70.388</td> <td> 0.000</td> <td>   -0.143</td> <td>   -0.135</td>
</tr>
<tr>
  <th>C(country_standard)[T.Guinea-Bissau]</th>                    <td>   -0.0513</td> <td>    0.002</td> <td>  -30.815</td> <td> 0.000</td> <td>   -0.055</td> <td>   -0.048</td>
</tr>
<tr>
  <th>C(country_standard)[T.Guyana]</th>                           <td>   -0.0036</td> <td>    0.005</td> <td>   -0.760</td> <td> 0.447</td> <td>   -0.013</td> <td>    0.006</td>
</tr>
<tr>
  <th>C(country_standard)[T.Haiti]</th>                            <td>   -0.0160</td> <td>    0.002</td> <td>   -7.601</td> <td> 0.000</td> <td>   -0.020</td> <td>   -0.012</td>
</tr>
<tr>
  <th>C(country_standard)[T.Honduras]</th>                         <td>   -0.0686</td> <td>    0.002</td> <td>  -27.947</td> <td> 0.000</td> <td>   -0.073</td> <td>   -0.064</td>
</tr>
<tr>
  <th>C(country_standard)[T.Hong Kong SAR China]</th>              <td>    0.0271</td> <td>    0.003</td> <td>    9.017</td> <td> 0.000</td> <td>    0.021</td> <td>    0.033</td>
</tr>
<tr>
  <th>C(country_standard)[T.Hungary]</th>                          <td>    0.0607</td> <td>    0.004</td> <td>   13.864</td> <td> 0.000</td> <td>    0.052</td> <td>    0.069</td>
</tr>
<tr>
  <th>C(country_standard)[T.Iceland]</th>                          <td>    0.1626</td> <td>    0.006</td> <td>   28.760</td> <td> 0.000</td> <td>    0.151</td> <td>    0.174</td>
</tr>
<tr>
  <th>C(country_standard)[T.India]</th>                            <td>    0.0483</td> <td>    0.003</td> <td>   17.754</td> <td> 0.000</td> <td>    0.043</td> <td>    0.054</td>
</tr>
<tr>
  <th>C(country_standard)[T.Indonesia]</th>                        <td>    0.1051</td> <td>    0.003</td> <td>   41.377</td> <td> 0.000</td> <td>    0.100</td> <td>    0.110</td>
</tr>
<tr>
  <th>C(country_standard)[T.Iran]</th>                             <td>   -0.1198</td> <td>    0.001</td> <td> -197.596</td> <td> 0.000</td> <td>   -0.121</td> <td>   -0.119</td>
</tr>
<tr>
  <th>C(country_standard)[T.Iraq]</th>                             <td>    0.0232</td> <td>    0.000</td> <td>   66.038</td> <td> 0.000</td> <td>    0.022</td> <td>    0.024</td>
</tr>
<tr>
  <th>C(country_standard)[T.Ireland]</th>                          <td>    0.1852</td> <td>    0.005</td> <td>   33.876</td> <td> 0.000</td> <td>    0.174</td> <td>    0.196</td>
</tr>
<tr>
  <th>C(country_standard)[T.Isle of Man]</th>                      <td>   -0.0072</td> <td>    0.003</td> <td>   -2.526</td> <td> 0.012</td> <td>   -0.013</td> <td>   -0.002</td>
</tr>
<tr>
  <th>C(country_standard)[T.Israel]</th>                           <td>    0.1259</td> <td>    0.005</td> <td>   25.353</td> <td> 0.000</td> <td>    0.116</td> <td>    0.136</td>
</tr>
<tr>
  <th>C(country_standard)[T.Italy]</th>                            <td>    0.1995</td> <td>    0.004</td> <td>   44.847</td> <td> 0.000</td> <td>    0.191</td> <td>    0.208</td>
</tr>
<tr>
  <th>C(country_standard)[T.Jamaica]</th>                          <td>    0.1383</td> <td>    0.005</td> <td>   27.295</td> <td> 0.000</td> <td>    0.128</td> <td>    0.148</td>
</tr>
<tr>
  <th>C(country_standard)[T.Japan]</th>                            <td>    0.1557</td> <td>    0.003</td> <td>   53.336</td> <td> 0.000</td> <td>    0.150</td> <td>    0.161</td>
</tr>
<tr>
  <th>C(country_standard)[T.Jersey]</th>                           <td>   -0.0290</td> <td>    0.005</td> <td>   -5.602</td> <td> 0.000</td> <td>   -0.039</td> <td>   -0.019</td>
</tr>
<tr>
  <th>C(country_standard)[T.Jordan]</th>                           <td>   -0.0414</td> <td>    0.001</td> <td>  -35.180</td> <td> 0.000</td> <td>   -0.044</td> <td>   -0.039</td>
</tr>
<tr>
  <th>C(country_standard)[T.Kazakhstan]</th>                       <td>   -0.1993</td> <td>    0.000</td> <td> -467.480</td> <td> 0.000</td> <td>   -0.200</td> <td>   -0.198</td>
</tr>
<tr>
  <th>C(country_standard)[T.Kenya]</th>                            <td>    0.0448</td> <td>    0.003</td> <td>   17.622</td> <td> 0.000</td> <td>    0.040</td> <td>    0.050</td>
</tr>
<tr>
  <th>C(country_standard)[T.Kiribati]</th>                         <td>   -0.0329</td> <td>    0.006</td> <td>   -5.863</td> <td> 0.000</td> <td>   -0.044</td> <td>   -0.022</td>
</tr>
<tr>
  <th>C(country_standard)[T.Kosovo]</th>                           <td>   -0.0069</td> <td>    0.003</td> <td>   -2.412</td> <td> 0.016</td> <td>   -0.012</td> <td>   -0.001</td>
</tr>
<tr>
  <th>C(country_standard)[T.Kuwait]</th>                           <td>    0.0080</td> <td>    0.002</td> <td>    3.731</td> <td> 0.000</td> <td>    0.004</td> <td>    0.012</td>
</tr>
<tr>
  <th>C(country_standard)[T.Kyrgyzstan]</th>                       <td>   -0.0440</td> <td>    0.002</td> <td>  -19.507</td> <td> 0.000</td> <td>   -0.048</td> <td>   -0.040</td>
</tr>
<tr>
  <th>C(country_standard)[T.Laos]</th>                             <td>   -0.2898</td> <td>    0.001</td> <td> -429.165</td> <td> 0.000</td> <td>   -0.291</td> <td>   -0.288</td>
</tr>
<tr>
  <th>C(country_standard)[T.Latvia]</th>                           <td>    0.1353</td> <td>    0.005</td> <td>   25.849</td> <td> 0.000</td> <td>    0.125</td> <td>    0.146</td>
</tr>
<tr>
  <th>C(country_standard)[T.Lebanon]</th>                          <td>   -0.0202</td> <td>    0.002</td> <td>   -8.930</td> <td> 0.000</td> <td>   -0.025</td> <td>   -0.016</td>
</tr>
<tr>
  <th>C(country_standard)[T.Lesotho]</th>                          <td>    0.0374</td> <td>    0.004</td> <td>   10.570</td> <td> 0.000</td> <td>    0.030</td> <td>    0.044</td>
</tr>
<tr>
  <th>C(country_standard)[T.Liberia]</th>                          <td>    0.1025</td> <td>    0.003</td> <td>   39.792</td> <td> 0.000</td> <td>    0.097</td> <td>    0.108</td>
</tr>
<tr>
  <th>C(country_standard)[T.Libya]</th>                            <td>   -0.1837</td> <td>    0.000</td> <td> -485.276</td> <td> 0.000</td> <td>   -0.184</td> <td>   -0.183</td>
</tr>
<tr>
  <th>C(country_standard)[T.Liechtenstein]</th>                    <td>   -0.0297</td> <td>    0.006</td> <td>   -5.174</td> <td> 0.000</td> <td>   -0.041</td> <td>   -0.018</td>
</tr>
<tr>
  <th>C(country_standard)[T.Lithuania]</th>                        <td>    0.1703</td> <td>    0.005</td> <td>   31.757</td> <td> 0.000</td> <td>    0.160</td> <td>    0.181</td>
</tr>
<tr>
  <th>C(country_standard)[T.Luxembourg]</th>                       <td>    0.1780</td> <td>    0.006</td> <td>   30.378</td> <td> 0.000</td> <td>    0.167</td> <td>    0.190</td>
</tr>
<tr>
  <th>C(country_standard)[T.Macao SAR China]</th>                  <td>   -0.0012</td> <td>    0.003</td> <td>   -0.458</td> <td> 0.647</td> <td>   -0.006</td> <td>    0.004</td>
</tr>
<tr>
  <th>C(country_standard)[T.Madagascar]</th>                       <td>   -0.0732</td> <td>    0.002</td> <td>  -30.798</td> <td> 0.000</td> <td>   -0.078</td> <td>   -0.069</td>
</tr>
<tr>
  <th>C(country_standard)[T.Malawi]</th>                           <td>    0.0611</td> <td>    0.003</td> <td>   22.344</td> <td> 0.000</td> <td>    0.056</td> <td>    0.066</td>
</tr>
<tr>
  <th>C(country_standard)[T.Malaysia]</th>                         <td>   -0.1347</td> <td>    0.002</td> <td>  -57.581</td> <td> 0.000</td> <td>   -0.139</td> <td>   -0.130</td>
</tr>
<tr>
  <th>C(country_standard)[T.Maldives]</th>                         <td>   -0.0338</td> <td>    0.002</td> <td>  -15.252</td> <td> 0.000</td> <td>   -0.038</td> <td>   -0.029</td>
</tr>
<tr>
  <th>C(country_standard)[T.Mali]</th>                             <td>    0.0346</td> <td>    0.002</td> <td>   15.577</td> <td> 0.000</td> <td>    0.030</td> <td>    0.039</td>
</tr>
<tr>
  <th>C(country_standard)[T.Malta]</th>                            <td>    0.0563</td> <td>    0.005</td> <td>   10.293</td> <td> 0.000</td> <td>    0.046</td> <td>    0.067</td>
</tr>
<tr>
  <th>C(country_standard)[T.Marshall Islands]</th>                 <td>   -0.0279</td> <td>    0.006</td> <td>   -5.026</td> <td> 0.000</td> <td>   -0.039</td> <td>   -0.017</td>
</tr>
<tr>
  <th>C(country_standard)[T.Martinique]</th>                       <td>   -0.0112</td> <td>    0.004</td> <td>   -3.193</td> <td> 0.001</td> <td>   -0.018</td> <td>   -0.004</td>
</tr>
<tr>
  <th>C(country_standard)[T.Mauritania]</th>                       <td>   -0.0711</td> <td>    0.000</td> <td> -149.490</td> <td> 0.000</td> <td>   -0.072</td> <td>   -0.070</td>
</tr>
<tr>
  <th>C(country_standard)[T.Mauritius]</th>                        <td>    0.1520</td> <td>    0.005</td> <td>   28.856</td> <td> 0.000</td> <td>    0.142</td> <td>    0.162</td>
</tr>
<tr>
  <th>C(country_standard)[T.Mexico]</th>                           <td>    0.0676</td> <td>    0.002</td> <td>   30.605</td> <td> 0.000</td> <td>    0.063</td> <td>    0.072</td>
</tr>
<tr>
  <th>C(country_standard)[T.Micronesia (Federated States of)]</th> <td>   -0.0255</td> <td>    0.005</td> <td>   -5.300</td> <td> 0.000</td> <td>   -0.035</td> <td>   -0.016</td>
</tr>
<tr>
  <th>C(country_standard)[T.Moldova]</th>                          <td>    0.0324</td> <td>    0.003</td> <td>   12.387</td> <td> 0.000</td> <td>    0.027</td> <td>    0.037</td>
</tr>
<tr>
  <th>C(country_standard)[T.Monaco]</th>                           <td>   -0.0253</td> <td>    0.005</td> <td>   -4.805</td> <td> 0.000</td> <td>   -0.036</td> <td>   -0.015</td>
</tr>
<tr>
  <th>C(country_standard)[T.Mongolia]</th>                         <td>    0.0814</td> <td>    0.005</td> <td>   17.332</td> <td> 0.000</td> <td>    0.072</td> <td>    0.091</td>
</tr>
<tr>
  <th>C(country_standard)[T.Montenegro]</th>                       <td>    0.0036</td> <td>    0.004</td> <td>    1.016</td> <td> 0.310</td> <td>   -0.003</td> <td>    0.011</td>
</tr>
<tr>
  <th>C(country_standard)[T.Morocco]</th>                          <td>    0.0141</td> <td>    0.002</td> <td>    6.596</td> <td> 0.000</td> <td>    0.010</td> <td>    0.018</td>
</tr>
<tr>
  <th>C(country_standard)[T.Mozambique]</th>                       <td>   -0.0305</td> <td>    0.003</td> <td>  -11.812</td> <td> 0.000</td> <td>   -0.036</td> <td>   -0.025</td>
</tr>
<tr>
  <th>C(country_standard)[T.Myanmar (Burma)]</th>                  <td>   -0.1733</td> <td>    0.001</td> <td> -290.938</td> <td> 0.000</td> <td>   -0.174</td> <td>   -0.172</td>
</tr>
<tr>
  <th>C(country_standard)[T.Namibia]</th>                          <td>    0.0795</td> <td>    0.005</td> <td>   15.392</td> <td> 0.000</td> <td>    0.069</td> <td>    0.090</td>
</tr>
<tr>
  <th>C(country_standard)[T.Nauru]</th>                            <td>   -0.0305</td> <td>    0.005</td> <td>   -5.696</td> <td> 0.000</td> <td>   -0.041</td> <td>   -0.020</td>
</tr>
<tr>
  <th>C(country_standard)[T.Nepal]</th>                            <td>    0.0666</td> <td>    0.002</td> <td>   27.856</td> <td> 0.000</td> <td>    0.062</td> <td>    0.071</td>
</tr>
<tr>
  <th>C(country_standard)[T.Netherlands]</th>                      <td>    0.2125</td> <td>    0.005</td> <td>   39.366</td> <td> 0.000</td> <td>    0.202</td> <td>    0.223</td>
</tr>
<tr>
  <th>C(country_standard)[T.New Caledonia]</th>                    <td>   -0.0089</td> <td>    0.003</td> <td>   -2.647</td> <td> 0.008</td> <td>   -0.016</td> <td>   -0.002</td>
</tr>
<tr>
  <th>C(country_standard)[T.New Zealand]</th>                      <td>    0.2057</td> <td>    0.006</td> <td>   35.276</td> <td> 0.000</td> <td>    0.194</td> <td>    0.217</td>
</tr>
<tr>
  <th>C(country_standard)[T.Nicaragua]</th>                        <td>   -0.1595</td> <td>    0.002</td> <td>  -88.953</td> <td> 0.000</td> <td>   -0.163</td> <td>   -0.156</td>
</tr>
<tr>
  <th>C(country_standard)[T.Niger]</th>                            <td>    0.0486</td> <td>    0.002</td> <td>   20.381</td> <td> 0.000</td> <td>    0.044</td> <td>    0.053</td>
</tr>
<tr>
  <th>C(country_standard)[T.Nigeria]</th>                          <td>    0.0239</td> <td>    0.002</td> <td>   11.449</td> <td> 0.000</td> <td>    0.020</td> <td>    0.028</td>
</tr>
<tr>
  <th>C(country_standard)[T.Niue]</th>                             <td>   -0.0065</td> <td>    0.003</td> <td>   -2.065</td> <td> 0.039</td> <td>   -0.013</td> <td>   -0.000</td>
</tr>
<tr>
  <th>C(country_standard)[T.North Korea]</th>                      <td>   -0.4298</td> <td>    0.002</td> <td> -262.924</td> <td> 0.000</td> <td>   -0.433</td> <td>   -0.427</td>
</tr>
<tr>
  <th>C(country_standard)[T.North Macedonia]</th>                  <td>    0.0183</td> <td>    0.003</td> <td>    6.945</td> <td> 0.000</td> <td>    0.013</td> <td>    0.023</td>
</tr>
<tr>
  <th>C(country_standard)[T.Northern Mariana Islands]</th>         <td>   -0.0089</td> <td>    0.003</td> <td>   -2.652</td> <td> 0.008</td> <td>   -0.015</td> <td>   -0.002</td>
</tr>
<tr>
  <th>C(country_standard)[T.Norway]</th>                           <td>    0.2426</td> <td>    0.006</td> <td>   42.178</td> <td> 0.000</td> <td>    0.231</td> <td>    0.254</td>
</tr>
<tr>
  <th>C(country_standard)[T.Oman]</th>                             <td>   -0.1630</td> <td>    0.000</td> <td> -547.413</td> <td> 0.000</td> <td>   -0.164</td> <td>   -0.162</td>
</tr>
<tr>
  <th>C(country_standard)[T.Pakistan]</th>                         <td>    0.0063</td> <td>    0.000</td> <td>   13.561</td> <td> 0.000</td> <td>    0.005</td> <td>    0.007</td>
</tr>
<tr>
  <th>C(country_standard)[T.Palau]</th>                            <td>   -0.0347</td> <td>    0.006</td> <td>   -5.986</td> <td> 0.000</td> <td>   -0.046</td> <td>   -0.023</td>
</tr>
<tr>
  <th>C(country_standard)[T.Palestinian Territories]</th>          <td>   -0.2158</td> <td>    0.000</td> <td> -536.008</td> <td> 0.000</td> <td>   -0.217</td> <td>   -0.215</td>
</tr>
<tr>
  <th>C(country_standard)[T.Panama]</th>                           <td>    0.0893</td> <td>    0.005</td> <td>   17.630</td> <td> 0.000</td> <td>    0.079</td> <td>    0.099</td>
</tr>
<tr>
  <th>C(country_standard)[T.Papua New Guinea]</th>                 <td>    0.0380</td> <td>    0.003</td> <td>   13.487</td> <td> 0.000</td> <td>    0.033</td> <td>    0.044</td>
</tr>
<tr>
  <th>C(country_standard)[T.Paraguay]</th>                         <td>    0.0483</td> <td>    0.003</td> <td>   17.471</td> <td> 0.000</td> <td>    0.043</td> <td>    0.054</td>
</tr>
<tr>
  <th>C(country_standard)[T.Peru]</th>                             <td>    0.1270</td> <td>    0.005</td> <td>   27.273</td> <td> 0.000</td> <td>    0.118</td> <td>    0.136</td>
</tr>
<tr>
  <th>C(country_standard)[T.Philippines]</th>                      <td>    0.0509</td> <td>    0.003</td> <td>   18.949</td> <td> 0.000</td> <td>    0.046</td> <td>    0.056</td>
</tr>
<tr>
  <th>C(country_standard)[T.Poland]</th>                           <td>    0.1514</td> <td>    0.005</td> <td>   30.472</td> <td> 0.000</td> <td>    0.142</td> <td>    0.161</td>
</tr>
<tr>
  <th>C(country_standard)[T.Portugal]</th>                         <td>    0.2024</td> <td>    0.005</td> <td>   37.507</td> <td> 0.000</td> <td>    0.192</td> <td>    0.213</td>
</tr>
<tr>
  <th>C(country_standard)[T.Puerto Rico]</th>                      <td>   -0.0232</td> <td>    0.005</td> <td>   -5.023</td> <td> 0.000</td> <td>   -0.032</td> <td>   -0.014</td>
</tr>
<tr>
  <th>C(country_standard)[T.Qatar]</th>                            <td>   -0.3999</td> <td>    0.000</td> <td>-1254.104</td> <td> 0.000</td> <td>   -0.401</td> <td>   -0.399</td>
</tr>
<tr>
  <th>C(country_standard)[T.Romania]</th>                          <td>    0.0632</td> <td>    0.005</td> <td>   13.195</td> <td> 0.000</td> <td>    0.054</td> <td>    0.073</td>
</tr>
<tr>
  <th>C(country_standard)[T.Russia]</th>                           <td>   -0.1604</td> <td>    0.001</td> <td> -182.303</td> <td> 0.000</td> <td>   -0.162</td> <td>   -0.159</td>
</tr>
<tr>
  <th>C(country_standard)[T.Rwanda]</th>                           <td>   -0.1495</td> <td>    0.000</td> <td> -513.907</td> <td> 0.000</td> <td>   -0.150</td> <td>   -0.149</td>
</tr>
<tr>
  <th>C(country_standard)[T.Réunion]</th>                          <td>   -0.0143</td> <td>    0.004</td> <td>   -3.765</td> <td> 0.000</td> <td>   -0.022</td> <td>   -0.007</td>
</tr>
<tr>
  <th>C(country_standard)[T.Saint Martin (French part)]</th>       <td>   -0.0088</td> <td>    0.003</td> <td>   -2.658</td> <td> 0.008</td> <td>   -0.015</td> <td>   -0.002</td>
</tr>
<tr>
  <th>C(country_standard)[T.Samoa]</th>                            <td>   -0.0236</td> <td>    0.005</td> <td>   -4.609</td> <td> 0.000</td> <td>   -0.034</td> <td>   -0.014</td>
</tr>
<tr>
  <th>C(country_standard)[T.San Marino]</th>                       <td>   -0.0282</td> <td>    0.006</td> <td>   -5.048</td> <td> 0.000</td> <td>   -0.039</td> <td>   -0.017</td>
</tr>
<tr>
  <th>C(country_standard)[T.Saudi Arabia]</th>                     <td>   -0.3996</td> <td>    0.001</td> <td> -384.334</td> <td> 0.000</td> <td>   -0.402</td> <td>   -0.398</td>
</tr>
<tr>
  <th>C(country_standard)[T.Senegal]</th>                          <td>    0.0903</td> <td>    0.005</td> <td>   19.936</td> <td> 0.000</td> <td>    0.081</td> <td>    0.099</td>
</tr>
<tr>
  <th>C(country_standard)[T.Serbia]</th>                           <td>    0.0287</td> <td>    0.004</td> <td>    6.872</td> <td> 0.000</td> <td>    0.021</td> <td>    0.037</td>
</tr>
<tr>
  <th>C(country_standard)[T.Seychelles]</th>                       <td>    0.0005</td> <td>    0.003</td> <td>    0.166</td> <td> 0.868</td> <td>   -0.005</td> <td>    0.006</td>
</tr>
<tr>
  <th>C(country_standard)[T.Sierra Leone]</th>                     <td>   -0.0006</td> <td>    0.003</td> <td>   -0.234</td> <td> 0.815</td> <td>   -0.006</td> <td>    0.005</td>
</tr>
<tr>
  <th>C(country_standard)[T.Singapore]</th>                        <td>   -0.0673</td> <td>    0.002</td> <td>  -27.039</td> <td> 0.000</td> <td>   -0.072</td> <td>   -0.062</td>
</tr>
<tr>
  <th>C(country_standard)[T.Sint Maarten]</th>                     <td>   -0.0089</td> <td>    0.003</td> <td>   -2.653</td> <td> 0.008</td> <td>   -0.015</td> <td>   -0.002</td>
</tr>
<tr>
  <th>C(country_standard)[T.Slovakia]</th>                         <td>    0.1378</td> <td>    0.005</td> <td>   26.451</td> <td> 0.000</td> <td>    0.128</td> <td>    0.148</td>
</tr>
<tr>
  <th>C(country_standard)[T.Slovenia]</th>                         <td>    0.1649</td> <td>    0.005</td> <td>   31.267</td> <td> 0.000</td> <td>    0.155</td> <td>    0.175</td>
</tr>
<tr>
  <th>C(country_standard)[T.Solomon Islands]</th>                  <td>    0.0324</td> <td>    0.004</td> <td>    8.262</td> <td> 0.000</td> <td>    0.025</td> <td>    0.040</td>
</tr>
<tr>
  <th>C(country_standard)[T.Somalia]</th>                          <td>   -0.1278</td> <td>    0.001</td> <td> -118.499</td> <td> 0.000</td> <td>   -0.130</td> <td>   -0.126</td>
</tr>
<tr>
  <th>C(country_standard)[T.South Africa]</th>                     <td>    0.1261</td> <td>    0.005</td> <td>   24.537</td> <td> 0.000</td> <td>    0.116</td> <td>    0.136</td>
</tr>
<tr>
  <th>C(country_standard)[T.South Korea]</th>                      <td>    0.1381</td> <td>    0.003</td> <td>   41.439</td> <td> 0.000</td> <td>    0.132</td> <td>    0.145</td>
</tr>
<tr>
  <th>C(country_standard)[T.South Sudan]</th>                      <td>   -0.1707</td> <td>    0.001</td> <td> -295.798</td> <td> 0.000</td> <td>   -0.172</td> <td>   -0.170</td>
</tr>
<tr>
  <th>C(country_standard)[T.Spain]</th>                            <td>    0.1846</td> <td>    0.005</td> <td>   38.974</td> <td> 0.000</td> <td>    0.175</td> <td>    0.194</td>
</tr>
<tr>
  <th>C(country_standard)[T.Sri Lanka]</th>                        <td>   -0.0632</td> <td>    0.002</td> <td>  -25.547</td> <td> 0.000</td> <td>   -0.068</td> <td>   -0.058</td>
</tr>
<tr>
  <th>C(country_standard)[T.St. Helena]</th>                       <td>   -0.0072</td> <td>    0.003</td> <td>   -2.550</td> <td> 0.011</td> <td>   -0.013</td> <td>   -0.002</td>
</tr>
<tr>
  <th>C(country_standard)[T.St. Kitts & Nevis]</th>                <td>   -0.0265</td> <td>    0.005</td> <td>   -4.909</td> <td> 0.000</td> <td>   -0.037</td> <td>   -0.016</td>
</tr>
<tr>
  <th>C(country_standard)[T.St. Lucia]</th>                        <td>   -0.0230</td> <td>    0.006</td> <td>   -4.136</td> <td> 0.000</td> <td>   -0.034</td> <td>   -0.012</td>
</tr>
<tr>
  <th>C(country_standard)[T.St. Vincent & Grenadines]</th>         <td>   -0.0267</td> <td>    0.005</td> <td>   -4.923</td> <td> 0.000</td> <td>   -0.037</td> <td>   -0.016</td>
</tr>
<tr>
  <th>C(country_standard)[T.Sudan]</th>                            <td>   -0.2074</td> <td>    0.001</td> <td> -280.176</td> <td> 0.000</td> <td>   -0.209</td> <td>   -0.206</td>
</tr>
<tr>
  <th>C(country_standard)[T.Suriname]</th>                         <td>    0.1180</td> <td>    0.005</td> <td>   24.150</td> <td> 0.000</td> <td>    0.108</td> <td>    0.128</td>
</tr>
<tr>
  <th>C(country_standard)[T.Sweden]</th>                           <td>    0.2306</td> <td>    0.006</td> <td>   41.016</td> <td> 0.000</td> <td>    0.220</td> <td>    0.242</td>
</tr>
<tr>
  <th>C(country_standard)[T.Switzerland]</th>                      <td>    0.2159</td> <td>    0.006</td> <td>   38.960</td> <td> 0.000</td> <td>    0.205</td> <td>    0.227</td>
</tr>
<tr>
  <th>C(country_standard)[T.Syria]</th>                            <td>   -0.3837</td> <td>    0.001</td> <td> -375.030</td> <td> 0.000</td> <td>   -0.386</td> <td>   -0.382</td>
</tr>
<tr>
  <th>C(country_standard)[T.São Tomé & Príncipe]</th>              <td>    0.1036</td> <td>    0.005</td> <td>   20.999</td> <td> 0.000</td> <td>    0.094</td> <td>    0.113</td>
</tr>
<tr>
  <th>C(country_standard)[T.Taiwan]</th>                           <td>    0.1213</td> <td>    0.005</td> <td>   23.626</td> <td> 0.000</td> <td>    0.111</td> <td>    0.131</td>
</tr>
<tr>
  <th>C(country_standard)[T.Tajikistan]</th>                       <td>   -0.2733</td> <td>    0.001</td> <td> -453.268</td> <td> 0.000</td> <td>   -0.274</td> <td>   -0.272</td>
</tr>
<tr>
  <th>C(country_standard)[T.Tanzania]</th>                         <td>    0.0779</td> <td>    0.003</td> <td>   30.958</td> <td> 0.000</td> <td>    0.073</td> <td>    0.083</td>
</tr>
<tr>
  <th>C(country_standard)[T.Thailand]</th>                         <td>   -0.0888</td> <td>    0.001</td> <td> -131.510</td> <td> 0.000</td> <td>   -0.090</td> <td>   -0.087</td>
</tr>
<tr>
  <th>C(country_standard)[T.Timor-Leste]</th>                      <td>    0.0397</td> <td>    0.004</td> <td>   11.191</td> <td> 0.000</td> <td>    0.033</td> <td>    0.047</td>
</tr>
<tr>
  <th>C(country_standard)[T.Togo]</th>                             <td>   -0.0770</td> <td>    0.002</td> <td>  -36.000</td> <td> 0.000</td> <td>   -0.081</td> <td>   -0.073</td>
</tr>
<tr>
  <th>C(country_standard)[T.Tonga]</th>                            <td>   -0.0290</td> <td>    0.005</td> <td>   -5.577</td> <td> 0.000</td> <td>   -0.039</td> <td>   -0.019</td>
</tr>
<tr>
  <th>C(country_standard)[T.Trinidad & Tobago]</th>                <td>    0.1308</td> <td>    0.005</td> <td>   26.166</td> <td> 0.000</td> <td>    0.121</td> <td>    0.141</td>
</tr>
<tr>
  <th>C(country_standard)[T.Tunisia]</th>                          <td>   -0.0210</td> <td>    0.004</td> <td>   -4.973</td> <td> 0.000</td> <td>   -0.029</td> <td>   -0.013</td>
</tr>
<tr>
  <th>C(country_standard)[T.Turkey]</th>                           <td>   -0.0422</td> <td>    0.001</td> <td>  -32.271</td> <td> 0.000</td> <td>   -0.045</td> <td>   -0.040</td>
</tr>
<tr>
  <th>C(country_standard)[T.Turkmenistan]</th>                     <td>   -0.3967</td> <td>    0.001</td> <td> -301.169</td> <td> 0.000</td> <td>   -0.399</td> <td>   -0.394</td>
</tr>
<tr>
  <th>C(country_standard)[T.Turks & Caicos Islands]</th>           <td>   -0.0072</td> <td>    0.003</td> <td>   -2.529</td> <td> 0.011</td> <td>   -0.013</td> <td>   -0.002</td>
</tr>
<tr>
  <th>C(country_standard)[T.Tuvalu]</th>                           <td>   -0.0329</td> <td>    0.006</td> <td>   -5.869</td> <td> 0.000</td> <td>   -0.044</td> <td>   -0.022</td>
</tr>
<tr>
  <th>C(country_standard)[T.U.S. Virgin Islands]</th>              <td>   -0.0123</td> <td>    0.004</td> <td>   -3.390</td> <td> 0.001</td> <td>   -0.019</td> <td>   -0.005</td>
</tr>
<tr>
  <th>C(country_standard)[T.Uganda]</th>                           <td>   -0.0204</td> <td>    0.001</td> <td>  -17.435</td> <td> 0.000</td> <td>   -0.023</td> <td>   -0.018</td>
</tr>
<tr>
  <th>C(country_standard)[T.Ukraine]</th>                          <td>   -0.0108</td> <td>    0.002</td> <td>   -4.344</td> <td> 0.000</td> <td>   -0.016</td> <td>   -0.006</td>
</tr>
<tr>
  <th>C(country_standard)[T.United Arab Emirates]</th>             <td>   -0.2211</td> <td>    0.000</td> <td> -518.491</td> <td> 0.000</td> <td>   -0.222</td> <td>   -0.220</td>
</tr>
<tr>
  <th>C(country_standard)[T.United Kingdom]</th>                   <td>    0.1734</td> <td>    0.004</td> <td>   40.586</td> <td> 0.000</td> <td>    0.165</td> <td>    0.182</td>
</tr>
<tr>
  <th>C(country_standard)[T.United States]</th>                    <td>    0.0821</td> <td>    0.005</td> <td>   16.056</td> <td> 0.000</td> <td>    0.072</td> <td>    0.092</td>
</tr>
<tr>
  <th>C(country_standard)[T.Uruguay]</th>                          <td>    0.1993</td> <td>    0.006</td> <td>   35.965</td> <td> 0.000</td> <td>    0.188</td> <td>    0.210</td>
</tr>
<tr>
  <th>C(country_standard)[T.Uzbekistan]</th>                       <td>   -0.3173</td> <td>    0.001</td> <td> -411.243</td> <td> 0.000</td> <td>   -0.319</td> <td>   -0.316</td>
</tr>
<tr>
  <th>C(country_standard)[T.Vanuatu]</th>                          <td>    0.1155</td> <td>    0.005</td> <td>   22.887</td> <td> 0.000</td> <td>    0.106</td> <td>    0.125</td>
</tr>
<tr>
  <th>C(country_standard)[T.Vatican City]</th>                     <td>   -0.0076</td> <td>    0.003</td> <td>   -2.622</td> <td> 0.009</td> <td>   -0.013</td> <td>   -0.002</td>
</tr>
<tr>
  <th>C(country_standard)[T.Venezuela]</th>                        <td>   -0.1877</td> <td>    0.001</td> <td> -262.029</td> <td> 0.000</td> <td>   -0.189</td> <td>   -0.186</td>
</tr>
<tr>
  <th>C(country_standard)[T.Vietnam]</th>                          <td>   -0.1527</td> <td>    0.001</td> <td> -294.647</td> <td> 0.000</td> <td>   -0.154</td> <td>   -0.152</td>
</tr>
<tr>
  <th>C(country_standard)[T.Western Sahara]</th>                   <td>    0.0702</td> <td>    0.006</td> <td>   12.764</td> <td> 0.000</td> <td>    0.059</td> <td>    0.081</td>
</tr>
<tr>
  <th>C(country_standard)[T.Yemen]</th>                            <td>   -0.1667</td> <td>    0.001</td> <td> -310.055</td> <td> 0.000</td> <td>   -0.168</td> <td>   -0.166</td>
</tr>
<tr>
  <th>C(country_standard)[T.Zambia]</th>                           <td>    0.0395</td> <td>    0.003</td> <td>   14.481</td> <td> 0.000</td> <td>    0.034</td> <td>    0.045</td>
</tr>
<tr>
  <th>C(country_standard)[T.Zimbabwe]</th>                         <td>   -0.1241</td> <td>    0.001</td> <td> -147.966</td> <td> 0.000</td> <td>   -0.126</td> <td>   -0.122</td>
</tr>
<tr>
  <th>transparency_index</th>                                      <td>    0.1337</td> <td>    0.014</td> <td>    9.320</td> <td> 0.000</td> <td>    0.106</td> <td>    0.162</td>
</tr>
<tr>
  <th>gdp</th>                                                     <td> 5.285e-15</td> <td> 4.96e-16</td> <td>   10.644</td> <td> 0.000</td> <td> 4.31e-15</td> <td> 6.26e-15</td>
</tr>
<tr>
  <th>gini</th>                                                    <td>   -0.0003</td> <td> 9.09e-06</td> <td>  -32.828</td> <td> 0.000</td> <td>   -0.000</td> <td>   -0.000</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>546.380</td> <th>  Durbin-Watson:     </th> <td>   0.774</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>7374.499</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.243</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 9.992</td>  <th>  Cond. No.          </th> <td>9.45e+26</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors are robust to cluster correlation (cluster)<br/>[2] The smallest eigenvalue is 1.39e-26. This might indicate that there are<br/>strong multicollinearity problems or that the design matrix is singular.




```python
reg_corruption = smf.ols('corruption_index ~ transparency_index + gdp + gini + C(country_standard)', ndi_df).fit(cov_type='cluster', cov_kwds={'groups': ndi_df['country_standard']})
```


```python
reg_corruption.summary()
```

    /Users/katiacordoba/opt/anaconda3/lib/python3.7/site-packages/statsmodels/base/model.py:1832: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 231, but rank is 229
      'rank is %d' % (J, J_), ValueWarning)





<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>    <td>corruption_index</td> <th>  R-squared:         </th> <td>   0.854</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.853</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>1.035e+09</td>
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 05 Mar 2021</td> <th>  Prob (F-statistic):</th>  <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>23:10:57</td>     <th>  Log-Likelihood:    </th> <td>  4029.3</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>  3603</td>      <th>  AIC:               </th> <td>  -8049.</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  3598</td>      <th>  BIC:               </th> <td>  -8018.</td> 
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
  <th>Intercept</th>                                               <td>    0.6102</td> <td>    0.007</td> <td>   90.371</td> <td> 0.000</td> <td>    0.597</td> <td>    0.623</td>
</tr>
<tr>
  <th>C(country_standard)[T.Albania]</th>                          <td>   -0.1145</td> <td>    0.017</td> <td>   -6.755</td> <td> 0.000</td> <td>   -0.148</td> <td>   -0.081</td>
</tr>
<tr>
  <th>C(country_standard)[T.Algeria]</th>                          <td>   -0.1047</td> <td>    0.001</td> <td>  -71.170</td> <td> 0.000</td> <td>   -0.108</td> <td>   -0.102</td>
</tr>
<tr>
  <th>C(country_standard)[T.American Samoa]</th>                   <td>   -0.4552</td> <td>    0.008</td> <td>  -58.215</td> <td> 0.000</td> <td>   -0.471</td> <td>   -0.440</td>
</tr>
<tr>
  <th>C(country_standard)[T.Andorra]</th>                          <td>   -0.5156</td> <td>    0.007</td> <td>  -77.058</td> <td> 0.000</td> <td>   -0.529</td> <td>   -0.502</td>
</tr>
<tr>
  <th>C(country_standard)[T.Angola]</th>                           <td>   -0.1581</td> <td>    0.002</td> <td> -102.490</td> <td> 0.000</td> <td>   -0.161</td> <td>   -0.155</td>
</tr>
<tr>
  <th>C(country_standard)[T.Anguilla]</th>                         <td>   -0.5256</td> <td>    0.004</td> <td> -125.808</td> <td> 0.000</td> <td>   -0.534</td> <td>   -0.517</td>
</tr>
<tr>
  <th>C(country_standard)[T.Antigua & Barbuda]</th>                <td>   -0.4510</td> <td>    0.006</td> <td>  -73.557</td> <td> 0.000</td> <td>   -0.463</td> <td>   -0.439</td>
</tr>
<tr>
  <th>C(country_standard)[T.Argentina]</th>                        <td>   -0.1982</td> <td>    0.006</td> <td>  -34.835</td> <td> 0.000</td> <td>   -0.209</td> <td>   -0.187</td>
</tr>
<tr>
  <th>C(country_standard)[T.Armenia]</th>                          <td>   -0.1483</td> <td>    0.003</td> <td>  -48.290</td> <td> 0.000</td> <td>   -0.154</td> <td>   -0.142</td>
</tr>
<tr>
  <th>C(country_standard)[T.Aruba]</th>                            <td>   -0.5158</td> <td>    0.007</td> <td>  -72.037</td> <td> 0.000</td> <td>   -0.530</td> <td>   -0.502</td>
</tr>
<tr>
  <th>C(country_standard)[T.Australia]</th>                        <td>   -0.5408</td> <td>    0.006</td> <td>  -90.333</td> <td> 0.000</td> <td>   -0.553</td> <td>   -0.529</td>
</tr>
<tr>
  <th>C(country_standard)[T.Austria]</th>                          <td>   -0.4928</td> <td>    0.007</td> <td>  -75.564</td> <td> 0.000</td> <td>   -0.506</td> <td>   -0.480</td>
</tr>
<tr>
  <th>C(country_standard)[T.Azerbaijan]</th>                       <td>   -0.1132</td> <td>    0.001</td> <td>  -76.670</td> <td> 0.000</td> <td>   -0.116</td> <td>   -0.110</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bahamas]</th>                          <td>   -0.5187</td> <td>    0.006</td> <td>  -80.978</td> <td> 0.000</td> <td>   -0.531</td> <td>   -0.506</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bahrain]</th>                          <td>   -0.3372</td> <td>    0.001</td> <td> -232.635</td> <td> 0.000</td> <td>   -0.340</td> <td>   -0.334</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bangladesh]</th>                       <td>   -0.0269</td> <td>    0.003</td> <td>   -9.641</td> <td> 0.000</td> <td>   -0.032</td> <td>   -0.021</td>
</tr>
<tr>
  <th>C(country_standard)[T.Barbados]</th>                         <td>   -0.5438</td> <td>    0.007</td> <td>  -82.203</td> <td> 0.000</td> <td>   -0.557</td> <td>   -0.531</td>
</tr>
<tr>
  <th>C(country_standard)[T.Belarus]</th>                          <td>   -0.1092</td> <td>    0.002</td> <td>  -67.491</td> <td> 0.000</td> <td>   -0.112</td> <td>   -0.106</td>
</tr>
<tr>
  <th>C(country_standard)[T.Belgium]</th>                          <td>   -0.4505</td> <td>    0.006</td> <td>  -70.169</td> <td> 0.000</td> <td>   -0.463</td> <td>   -0.438</td>
</tr>
<tr>
  <th>C(country_standard)[T.Belize]</th>                           <td>   -0.2221</td> <td>    0.006</td> <td>  -36.341</td> <td> 0.000</td> <td>   -0.234</td> <td>   -0.210</td>
</tr>
<tr>
  <th>C(country_standard)[T.Benin]</th>                            <td>   -0.2511</td> <td>    0.006</td> <td>  -44.906</td> <td> 0.000</td> <td>   -0.262</td> <td>   -0.240</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bermuda]</th>                          <td>   -0.5256</td> <td>    0.004</td> <td> -125.979</td> <td> 0.000</td> <td>   -0.534</td> <td>   -0.517</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bhutan]</th>                           <td>   -0.4932</td> <td>    0.004</td> <td> -139.947</td> <td> 0.000</td> <td>   -0.500</td> <td>   -0.486</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bolivia]</th>                          <td>   -0.2041</td> <td>    0.003</td> <td>  -58.345</td> <td> 0.000</td> <td>   -0.211</td> <td>   -0.197</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bosnia & Herzegovina]</th>             <td>   -0.1751</td> <td>    0.003</td> <td>  -52.297</td> <td> 0.000</td> <td>   -0.182</td> <td>   -0.169</td>
</tr>
<tr>
  <th>C(country_standard)[T.Botswana]</th>                         <td>   -0.5827</td> <td>    0.006</td> <td>  -95.588</td> <td> 0.000</td> <td>   -0.595</td> <td>   -0.571</td>
</tr>
<tr>
  <th>C(country_standard)[T.Brazil]</th>                           <td>   -0.2697</td> <td>    0.005</td> <td>  -54.960</td> <td> 0.000</td> <td>   -0.279</td> <td>   -0.260</td>
</tr>
<tr>
  <th>C(country_standard)[T.British Virgin Islands]</th>           <td>   -0.2676</td> <td>    0.009</td> <td>  -31.420</td> <td> 0.000</td> <td>   -0.284</td> <td>   -0.251</td>
</tr>
<tr>
  <th>C(country_standard)[T.Brunei]</th>                           <td>   -0.4417</td> <td>    0.001</td> <td> -303.771</td> <td> 0.000</td> <td>   -0.445</td> <td>   -0.439</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bulgaria]</th>                         <td>   -0.2488</td> <td>    0.006</td> <td>  -42.316</td> <td> 0.000</td> <td>   -0.260</td> <td>   -0.237</td>
</tr>
<tr>
  <th>C(country_standard)[T.Burkina Faso]</th>                     <td>   -0.2056</td> <td>    0.003</td> <td>  -61.455</td> <td> 0.000</td> <td>   -0.212</td> <td>   -0.199</td>
</tr>
<tr>
  <th>C(country_standard)[T.Burundi]</th>                          <td>   -0.0666</td> <td>    0.001</td> <td>  -46.562</td> <td> 0.000</td> <td>   -0.069</td> <td>   -0.064</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cambodia]</th>                         <td>   -0.0681</td> <td>    0.001</td> <td>  -48.274</td> <td> 0.000</td> <td>   -0.071</td> <td>   -0.065</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cameroon]</th>                         <td>   -0.1679</td> <td>    0.001</td> <td> -119.362</td> <td> 0.000</td> <td>   -0.171</td> <td>   -0.165</td>
</tr>
<tr>
  <th>C(country_standard)[T.Canada]</th>                           <td>   -0.5210</td> <td>    0.006</td> <td>  -88.100</td> <td> 0.000</td> <td>   -0.533</td> <td>   -0.509</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cape Verde]</th>                       <td>   -0.4633</td> <td>    0.007</td> <td>  -70.868</td> <td> 0.000</td> <td>   -0.476</td> <td>   -0.450</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cayman Islands]</th>                   <td>   -0.4744</td> <td>    0.005</td> <td>  -99.996</td> <td> 0.000</td> <td>   -0.484</td> <td>   -0.465</td>
</tr>
<tr>
  <th>C(country_standard)[T.Central African Republic]</th>         <td>   -0.2443</td> <td>    0.001</td> <td> -172.755</td> <td> 0.000</td> <td>   -0.247</td> <td>   -0.241</td>
</tr>
<tr>
  <th>C(country_standard)[T.Chad]</th>                             <td>   -0.0864</td> <td>    0.001</td> <td>  -61.101</td> <td> 0.000</td> <td>   -0.089</td> <td>   -0.084</td>
</tr>
<tr>
  <th>C(country_standard)[T.Channel Islands]</th>                  <td>   -0.2582</td> <td>    0.004</td> <td>  -62.161</td> <td> 0.000</td> <td>   -0.266</td> <td>   -0.250</td>
</tr>
<tr>
  <th>C(country_standard)[T.Chile]</th>                            <td>   -0.5730</td> <td>    0.006</td> <td>  -89.387</td> <td> 0.000</td> <td>   -0.586</td> <td>   -0.560</td>
</tr>
<tr>
  <th>C(country_standard)[T.China]</th>                            <td>    0.5208</td> <td>    0.009</td> <td>   56.880</td> <td> 0.000</td> <td>    0.503</td> <td>    0.539</td>
</tr>
<tr>
  <th>C(country_standard)[T.Colombia]</th>                         <td>   -0.3092</td> <td>    0.003</td> <td>  -89.889</td> <td> 0.000</td> <td>   -0.316</td> <td>   -0.303</td>
</tr>
<tr>
  <th>C(country_standard)[T.Comoros]</th>                          <td>   -0.1883</td> <td>    0.003</td> <td>  -59.221</td> <td> 0.000</td> <td>   -0.195</td> <td>   -0.182</td>
</tr>
<tr>
  <th>C(country_standard)[T.Congo - Brazzaville]</th>              <td>   -0.1648</td> <td>    0.001</td> <td> -117.480</td> <td> 0.000</td> <td>   -0.168</td> <td>   -0.162</td>
</tr>
<tr>
  <th>C(country_standard)[T.Congo - Kinshasa]</th>                 <td>   -0.0772</td> <td>    0.001</td> <td>  -54.028</td> <td> 0.000</td> <td>   -0.080</td> <td>   -0.074</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cook Islands]</th>                     <td>   -0.2726</td> <td>    0.009</td> <td>  -30.500</td> <td> 0.000</td> <td>   -0.290</td> <td>   -0.255</td>
</tr>
<tr>
  <th>C(country_standard)[T.Costa Rica]</th>                       <td>   -0.4660</td> <td>    0.007</td> <td>  -69.928</td> <td> 0.000</td> <td>   -0.479</td> <td>   -0.453</td>
</tr>
<tr>
  <th>C(country_standard)[T.Croatia]</th>                          <td>   -0.2236</td> <td>    0.006</td> <td>  -37.752</td> <td> 0.000</td> <td>   -0.235</td> <td>   -0.212</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cuba]</th>                             <td>   -0.3519</td> <td>    0.002</td> <td> -229.588</td> <td> 0.000</td> <td>   -0.355</td> <td>   -0.349</td>
</tr>
<tr>
  <th>C(country_standard)[T.Curaçao]</th>                          <td>   -0.2677</td> <td>    0.009</td> <td>  -31.177</td> <td> 0.000</td> <td>   -0.285</td> <td>   -0.251</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cyprus]</th>                           <td>   -0.3950</td> <td>    0.006</td> <td>  -61.053</td> <td> 0.000</td> <td>   -0.408</td> <td>   -0.382</td>
</tr>
<tr>
  <th>C(country_standard)[T.Czechia]</th>                          <td>   -0.2225</td> <td>    0.006</td> <td>  -35.770</td> <td> 0.000</td> <td>   -0.235</td> <td>   -0.210</td>
</tr>
<tr>
  <th>C(country_standard)[T.Côte d’Ivoire]</th>                    <td>   -0.1541</td> <td>    0.003</td> <td>  -50.239</td> <td> 0.000</td> <td>   -0.160</td> <td>   -0.148</td>
</tr>
<tr>
  <th>C(country_standard)[T.Denmark]</th>                          <td>   -0.6346</td> <td>    0.007</td> <td>  -93.933</td> <td> 0.000</td> <td>   -0.648</td> <td>   -0.621</td>
</tr>
<tr>
  <th>C(country_standard)[T.Djibouti]</th>                         <td>   -0.2349</td> <td>    0.001</td> <td> -164.122</td> <td> 0.000</td> <td>   -0.238</td> <td>   -0.232</td>
</tr>
<tr>
  <th>C(country_standard)[T.Dominica]</th>                         <td>   -0.3964</td> <td>    0.006</td> <td>  -61.598</td> <td> 0.000</td> <td>   -0.409</td> <td>   -0.384</td>
</tr>
<tr>
  <th>C(country_standard)[T.Dominican Republic]</th>               <td>   -0.1697</td> <td>    0.004</td> <td>  -38.846</td> <td> 0.000</td> <td>   -0.178</td> <td>   -0.161</td>
</tr>
<tr>
  <th>C(country_standard)[T.Ecuador]</th>                          <td>   -0.2126</td> <td>    0.003</td> <td>  -64.096</td> <td> 0.000</td> <td>   -0.219</td> <td>   -0.206</td>
</tr>
<tr>
  <th>C(country_standard)[T.Egypt]</th>                            <td>   -0.1157</td> <td>    0.001</td> <td>  -81.769</td> <td> 0.000</td> <td>   -0.118</td> <td>   -0.113</td>
</tr>
<tr>
  <th>C(country_standard)[T.El Salvador]</th>                      <td>   -0.1993</td> <td>    0.005</td> <td>  -37.270</td> <td> 0.000</td> <td>   -0.210</td> <td>   -0.189</td>
</tr>
<tr>
  <th>C(country_standard)[T.Equatorial Guinea]</th>                <td>   -0.0050</td> <td>    0.002</td> <td>   -2.954</td> <td> 0.003</td> <td>   -0.008</td> <td>   -0.002</td>
</tr>
<tr>
  <th>C(country_standard)[T.Eritrea]</th>                          <td>   -0.1452</td> <td>    0.002</td> <td>  -79.112</td> <td> 0.000</td> <td>   -0.149</td> <td>   -0.142</td>
</tr>
<tr>
  <th>C(country_standard)[T.Estonia]</th>                          <td>   -0.4339</td> <td>    0.007</td> <td>  -66.021</td> <td> 0.000</td> <td>   -0.447</td> <td>   -0.421</td>
</tr>
<tr>
  <th>C(country_standard)[T.Eswatini]</th>                         <td>   -0.3911</td> <td>    0.001</td> <td> -301.925</td> <td> 0.000</td> <td>   -0.394</td> <td>   -0.389</td>
</tr>
<tr>
  <th>C(country_standard)[T.Ethiopia]</th>                         <td>   -0.1683</td> <td>    0.001</td> <td> -115.380</td> <td> 0.000</td> <td>   -0.171</td> <td>   -0.165</td>
</tr>
<tr>
  <th>C(country_standard)[T.Faroe Islands]</th>                    <td>   -0.2595</td> <td>    0.004</td> <td>  -71.125</td> <td> 0.000</td> <td>   -0.267</td> <td>   -0.252</td>
</tr>
<tr>
  <th>C(country_standard)[T.Fiji]</th>                             <td>   -0.2846</td> <td>    0.003</td> <td>  -83.991</td> <td> 0.000</td> <td>   -0.291</td> <td>   -0.278</td>
</tr>
<tr>
  <th>C(country_standard)[T.Finland]</th>                          <td>   -0.6110</td> <td>    0.007</td> <td>  -90.180</td> <td> 0.000</td> <td>   -0.624</td> <td>   -0.598</td>
</tr>
<tr>
  <th>C(country_standard)[T.France]</th>                           <td>   -0.3452</td> <td>    0.005</td> <td>  -66.604</td> <td> 0.000</td> <td>   -0.355</td> <td>   -0.335</td>
</tr>
<tr>
  <th>C(country_standard)[T.French Guiana]</th>                    <td>   -0.4685</td> <td>    0.007</td> <td>  -65.463</td> <td> 0.000</td> <td>   -0.482</td> <td>   -0.454</td>
</tr>
<tr>
  <th>C(country_standard)[T.French Polynesia]</th>                 <td>   -0.2677</td> <td>    0.009</td> <td>  -31.131</td> <td> 0.000</td> <td>   -0.285</td> <td>   -0.251</td>
</tr>
<tr>
  <th>C(country_standard)[T.Gabon]</th>                            <td>   -0.1500</td> <td>    0.001</td> <td> -104.319</td> <td> 0.000</td> <td>   -0.153</td> <td>   -0.147</td>
</tr>
<tr>
  <th>C(country_standard)[T.Gambia]</th>                           <td>   -0.1567</td> <td>    0.002</td> <td>  -92.068</td> <td> 0.000</td> <td>   -0.160</td> <td>   -0.153</td>
</tr>
<tr>
  <th>C(country_standard)[T.Georgia]</th>                          <td>   -0.3339</td> <td>    0.004</td> <td>  -88.707</td> <td> 0.000</td> <td>   -0.341</td> <td>   -0.327</td>
</tr>
<tr>
  <th>C(country_standard)[T.Germany]</th>                          <td>   -0.3702</td> <td>    0.005</td> <td>  -75.384</td> <td> 0.000</td> <td>   -0.380</td> <td>   -0.361</td>
</tr>
<tr>
  <th>C(country_standard)[T.Ghana]</th>                            <td>   -0.2981</td> <td>    0.006</td> <td>  -48.913</td> <td> 0.000</td> <td>   -0.310</td> <td>   -0.286</td>
</tr>
<tr>
  <th>C(country_standard)[T.Gibraltar]</th>                        <td>   -0.2595</td> <td>    0.004</td> <td>  -71.160</td> <td> 0.000</td> <td>   -0.267</td> <td>   -0.252</td>
</tr>
<tr>
  <th>C(country_standard)[T.Greece]</th>                           <td>   -0.2263</td> <td>    0.006</td> <td>  -37.115</td> <td> 0.000</td> <td>   -0.238</td> <td>   -0.214</td>
</tr>
<tr>
  <th>C(country_standard)[T.Greenland]</th>                        <td>   -0.4375</td> <td>    0.006</td> <td>  -70.173</td> <td> 0.000</td> <td>   -0.450</td> <td>   -0.425</td>
</tr>
<tr>
  <th>C(country_standard)[T.Grenada]</th>                          <td>   -0.3508</td> <td>    0.006</td> <td>  -55.756</td> <td> 0.000</td> <td>   -0.363</td> <td>   -0.339</td>
</tr>
<tr>
  <th>C(country_standard)[T.Guam]</th>                             <td>   -0.4900</td> <td>    0.008</td> <td>  -60.228</td> <td> 0.000</td> <td>   -0.506</td> <td>   -0.474</td>
</tr>
<tr>
  <th>C(country_standard)[T.Guatemala]</th>                        <td>   -0.2292</td> <td>    0.003</td> <td>  -71.496</td> <td> 0.000</td> <td>   -0.235</td> <td>   -0.223</td>
</tr>
<tr>
  <th>C(country_standard)[T.Guernsey]</th>                         <td>   -0.2597</td> <td>    0.004</td> <td>  -71.668</td> <td> 0.000</td> <td>   -0.267</td> <td>   -0.253</td>
</tr>
<tr>
  <th>C(country_standard)[T.Guinea]</th>                           <td>   -0.0449</td> <td>    0.003</td> <td>  -16.438</td> <td> 0.000</td> <td>   -0.050</td> <td>   -0.040</td>
</tr>
<tr>
  <th>C(country_standard)[T.Guinea-Bissau]</th>                    <td>   -0.1428</td> <td>    0.002</td> <td>  -59.981</td> <td> 0.000</td> <td>   -0.147</td> <td>   -0.138</td>
</tr>
<tr>
  <th>C(country_standard)[T.Guyana]</th>                           <td>   -0.1581</td> <td>    0.006</td> <td>  -27.918</td> <td> 0.000</td> <td>   -0.169</td> <td>   -0.147</td>
</tr>
<tr>
  <th>C(country_standard)[T.Haiti]</th>                            <td>   -0.0670</td> <td>    0.003</td> <td>  -23.917</td> <td> 0.000</td> <td>   -0.072</td> <td>   -0.061</td>
</tr>
<tr>
  <th>C(country_standard)[T.Honduras]</th>                         <td>   -0.2432</td> <td>    0.003</td> <td>  -76.581</td> <td> 0.000</td> <td>   -0.249</td> <td>   -0.237</td>
</tr>
<tr>
  <th>C(country_standard)[T.Hong Kong SAR China]</th>              <td>   -0.6175</td> <td>    0.004</td> <td> -164.699</td> <td> 0.000</td> <td>   -0.625</td> <td>   -0.610</td>
</tr>
<tr>
  <th>C(country_standard)[T.Hungary]</th>                          <td>   -0.2677</td> <td>    0.005</td> <td>  -50.460</td> <td> 0.000</td> <td>   -0.278</td> <td>   -0.257</td>
</tr>
<tr>
  <th>C(country_standard)[T.Iceland]</th>                          <td>   -0.5549</td> <td>    0.007</td> <td>  -81.390</td> <td> 0.000</td> <td>   -0.568</td> <td>   -0.542</td>
</tr>
<tr>
  <th>C(country_standard)[T.India]</th>                            <td>   -0.0312</td> <td>    0.003</td> <td>   -9.521</td> <td> 0.000</td> <td>   -0.038</td> <td>   -0.025</td>
</tr>
<tr>
  <th>C(country_standard)[T.Indonesia]</th>                        <td>   -0.1092</td> <td>    0.003</td> <td>  -34.217</td> <td> 0.000</td> <td>   -0.115</td> <td>   -0.103</td>
</tr>
<tr>
  <th>C(country_standard)[T.Iran]</th>                             <td>   -0.1559</td> <td>    0.001</td> <td> -107.239</td> <td> 0.000</td> <td>   -0.159</td> <td>   -0.153</td>
</tr>
<tr>
  <th>C(country_standard)[T.Iraq]</th>                             <td>    0.0457</td> <td>    0.001</td> <td>   31.244</td> <td> 0.000</td> <td>    0.043</td> <td>    0.049</td>
</tr>
<tr>
  <th>C(country_standard)[T.Ireland]</th>                          <td>   -0.5176</td> <td>    0.007</td> <td>  -79.343</td> <td> 0.000</td> <td>   -0.530</td> <td>   -0.505</td>
</tr>
<tr>
  <th>C(country_standard)[T.Isle of Man]</th>                      <td>   -0.2595</td> <td>    0.004</td> <td>  -71.161</td> <td> 0.000</td> <td>   -0.267</td> <td>   -0.252</td>
</tr>
<tr>
  <th>C(country_standard)[T.Israel]</th>                           <td>   -0.4286</td> <td>    0.006</td> <td>  -72.073</td> <td> 0.000</td> <td>   -0.440</td> <td>   -0.417</td>
</tr>
<tr>
  <th>C(country_standard)[T.Italy]</th>                            <td>   -0.1778</td> <td>    0.006</td> <td>  -31.533</td> <td> 0.000</td> <td>   -0.189</td> <td>   -0.167</td>
</tr>
<tr>
  <th>C(country_standard)[T.Jamaica]</th>                          <td>   -0.2211</td> <td>    0.006</td> <td>  -36.282</td> <td> 0.000</td> <td>   -0.233</td> <td>   -0.209</td>
</tr>
<tr>
  <th>C(country_standard)[T.Japan]</th>                            <td>   -0.2506</td> <td>    0.004</td> <td>  -62.043</td> <td> 0.000</td> <td>   -0.259</td> <td>   -0.243</td>
</tr>
<tr>
  <th>C(country_standard)[T.Jersey]</th>                           <td>   -0.3988</td> <td>    0.006</td> <td>  -65.017</td> <td> 0.000</td> <td>   -0.411</td> <td>   -0.387</td>
</tr>
<tr>
  <th>C(country_standard)[T.Jordan]</th>                           <td>   -0.3024</td> <td>    0.002</td> <td> -154.768</td> <td> 0.000</td> <td>   -0.306</td> <td>   -0.299</td>
</tr>
<tr>
  <th>C(country_standard)[T.Kazakhstan]</th>                       <td>   -0.0439</td> <td>    0.002</td> <td>  -29.109</td> <td> 0.000</td> <td>   -0.047</td> <td>   -0.041</td>
</tr>
<tr>
  <th>C(country_standard)[T.Kenya]</th>                            <td>   -0.1163</td> <td>    0.003</td> <td>  -35.897</td> <td> 0.000</td> <td>   -0.123</td> <td>   -0.110</td>
</tr>
<tr>
  <th>C(country_standard)[T.Kiribati]</th>                         <td>   -0.2672</td> <td>    0.007</td> <td>  -36.735</td> <td> 0.000</td> <td>   -0.281</td> <td>   -0.253</td>
</tr>
<tr>
  <th>C(country_standard)[T.Kosovo]</th>                           <td>   -0.2582</td> <td>    0.004</td> <td>  -62.089</td> <td> 0.000</td> <td>   -0.266</td> <td>   -0.250</td>
</tr>
<tr>
  <th>C(country_standard)[T.Kuwait]</th>                           <td>   -0.2955</td> <td>    0.003</td> <td> -104.888</td> <td> 0.000</td> <td>   -0.301</td> <td>   -0.290</td>
</tr>
<tr>
  <th>C(country_standard)[T.Kyrgyzstan]</th>                       <td>    0.0172</td> <td>    0.003</td> <td>    5.698</td> <td> 0.000</td> <td>    0.011</td> <td>    0.023</td>
</tr>
<tr>
  <th>C(country_standard)[T.Laos]</th>                             <td>   -0.0898</td> <td>    0.002</td> <td>  -56.368</td> <td> 0.000</td> <td>   -0.093</td> <td>   -0.087</td>
</tr>
<tr>
  <th>C(country_standard)[T.Latvia]</th>                           <td>   -0.3232</td> <td>    0.006</td> <td>  -51.713</td> <td> 0.000</td> <td>   -0.335</td> <td>   -0.311</td>
</tr>
<tr>
  <th>C(country_standard)[T.Lebanon]</th>                          <td>   -0.0552</td> <td>    0.003</td> <td>  -18.410</td> <td> 0.000</td> <td>   -0.061</td> <td>   -0.049</td>
</tr>
<tr>
  <th>C(country_standard)[T.Lesotho]</th>                          <td>   -0.3464</td> <td>    0.004</td> <td>  -79.848</td> <td> 0.000</td> <td>   -0.355</td> <td>   -0.338</td>
</tr>
<tr>
  <th>C(country_standard)[T.Liberia]</th>                          <td>   -0.1424</td> <td>    0.003</td> <td>  -42.288</td> <td> 0.000</td> <td>   -0.149</td> <td>   -0.136</td>
</tr>
<tr>
  <th>C(country_standard)[T.Libya]</th>                            <td>   -0.0383</td> <td>    0.001</td> <td>  -26.698</td> <td> 0.000</td> <td>   -0.041</td> <td>   -0.035</td>
</tr>
<tr>
  <th>C(country_standard)[T.Liechtenstein]</th>                    <td>   -0.6163</td> <td>    0.007</td> <td>  -91.195</td> <td> 0.000</td> <td>   -0.630</td> <td>   -0.603</td>
</tr>
<tr>
  <th>C(country_standard)[T.Lithuania]</th>                        <td>   -0.3468</td> <td>    0.006</td> <td>  -54.269</td> <td> 0.000</td> <td>   -0.359</td> <td>   -0.334</td>
</tr>
<tr>
  <th>C(country_standard)[T.Luxembourg]</th>                       <td>   -0.6322</td> <td>    0.007</td> <td>  -90.947</td> <td> 0.000</td> <td>   -0.646</td> <td>   -0.619</td>
</tr>
<tr>
  <th>C(country_standard)[T.Macao SAR China]</th>                  <td>   -0.4020</td> <td>    0.003</td> <td> -115.145</td> <td> 0.000</td> <td>   -0.409</td> <td>   -0.395</td>
</tr>
<tr>
  <th>C(country_standard)[T.Madagascar]</th>                       <td>   -0.1988</td> <td>    0.003</td> <td>  -64.232</td> <td> 0.000</td> <td>   -0.205</td> <td>   -0.193</td>
</tr>
<tr>
  <th>C(country_standard)[T.Malawi]</th>                           <td>   -0.2295</td> <td>    0.003</td> <td>  -66.607</td> <td> 0.000</td> <td>   -0.236</td> <td>   -0.223</td>
</tr>
<tr>
  <th>C(country_standard)[T.Malaysia]</th>                         <td>   -0.3316</td> <td>    0.003</td> <td> -109.439</td> <td> 0.000</td> <td>   -0.338</td> <td>   -0.326</td>
</tr>
<tr>
  <th>C(country_standard)[T.Maldives]</th>                         <td>   -0.1067</td> <td>    0.003</td> <td>  -35.748</td> <td> 0.000</td> <td>   -0.113</td> <td>   -0.101</td>
</tr>
<tr>
  <th>C(country_standard)[T.Mali]</th>                             <td>   -0.1201</td> <td>    0.003</td> <td>  -39.437</td> <td> 0.000</td> <td>   -0.126</td> <td>   -0.114</td>
</tr>
<tr>
  <th>C(country_standard)[T.Malta]</th>                            <td>   -0.3409</td> <td>    0.007</td> <td>  -51.231</td> <td> 0.000</td> <td>   -0.354</td> <td>   -0.328</td>
</tr>
<tr>
  <th>C(country_standard)[T.Marshall Islands]</th>                 <td>   -0.2120</td> <td>    0.007</td> <td>  -31.820</td> <td> 0.000</td> <td>   -0.225</td> <td>   -0.199</td>
</tr>
<tr>
  <th>C(country_standard)[T.Martinique]</th>                       <td>   -0.4913</td> <td>    0.008</td> <td>  -59.313</td> <td> 0.000</td> <td>   -0.507</td> <td>   -0.475</td>
</tr>
<tr>
  <th>C(country_standard)[T.Mauritania]</th>                       <td>   -0.1113</td> <td>    0.002</td> <td>  -73.470</td> <td> 0.000</td> <td>   -0.114</td> <td>   -0.108</td>
</tr>
<tr>
  <th>C(country_standard)[T.Mauritius]</th>                        <td>   -0.3369</td> <td>    0.006</td> <td>  -52.927</td> <td> 0.000</td> <td>   -0.349</td> <td>   -0.324</td>
</tr>
<tr>
  <th>C(country_standard)[T.Mexico]</th>                           <td>   -0.1762</td> <td>    0.003</td> <td>  -60.936</td> <td> 0.000</td> <td>   -0.182</td> <td>   -0.171</td>
</tr>
<tr>
  <th>C(country_standard)[T.Micronesia (Federated States of)]</th> <td>   -0.2770</td> <td>    0.006</td> <td>  -47.670</td> <td> 0.000</td> <td>   -0.288</td> <td>   -0.266</td>
</tr>
<tr>
  <th>C(country_standard)[T.Moldova]</th>                          <td>   -0.0485</td> <td>    0.003</td> <td>  -13.945</td> <td> 0.000</td> <td>   -0.055</td> <td>   -0.042</td>
</tr>
<tr>
  <th>C(country_standard)[T.Monaco]</th>                           <td>   -0.2648</td> <td>    0.006</td> <td>  -41.823</td> <td> 0.000</td> <td>   -0.277</td> <td>   -0.252</td>
</tr>
<tr>
  <th>C(country_standard)[T.Mongolia]</th>                         <td>   -0.1168</td> <td>    0.006</td> <td>  -20.440</td> <td> 0.000</td> <td>   -0.128</td> <td>   -0.106</td>
</tr>
<tr>
  <th>C(country_standard)[T.Montenegro]</th>                       <td>   -0.2592</td> <td>    0.004</td> <td>  -59.054</td> <td> 0.000</td> <td>   -0.268</td> <td>   -0.251</td>
</tr>
<tr>
  <th>C(country_standard)[T.Morocco]</th>                          <td>   -0.2379</td> <td>    0.003</td> <td>  -84.174</td> <td> 0.000</td> <td>   -0.243</td> <td>   -0.232</td>
</tr>
<tr>
  <th>C(country_standard)[T.Mozambique]</th>                       <td>   -0.2895</td> <td>    0.003</td> <td>  -88.403</td> <td> 0.000</td> <td>   -0.296</td> <td>   -0.283</td>
</tr>
<tr>
  <th>C(country_standard)[T.Myanmar (Burma)]</th>                  <td>    0.0037</td> <td>    0.002</td> <td>    2.356</td> <td> 0.018</td> <td>    0.001</td> <td>    0.007</td>
</tr>
<tr>
  <th>C(country_standard)[T.Namibia]</th>                          <td>   -0.5116</td> <td>    0.006</td> <td>  -82.510</td> <td> 0.000</td> <td>   -0.524</td> <td>   -0.499</td>
</tr>
<tr>
  <th>C(country_standard)[T.Nauru]</th>                            <td>   -0.1642</td> <td>    0.007</td> <td>  -23.402</td> <td> 0.000</td> <td>   -0.178</td> <td>   -0.150</td>
</tr>
<tr>
  <th>C(country_standard)[T.Nepal]</th>                            <td>   -0.0974</td> <td>    0.003</td> <td>  -30.961</td> <td> 0.000</td> <td>   -0.104</td> <td>   -0.091</td>
</tr>
<tr>
  <th>C(country_standard)[T.Netherlands]</th>                      <td>   -0.5425</td> <td>    0.006</td> <td>  -84.580</td> <td> 0.000</td> <td>   -0.555</td> <td>   -0.530</td>
</tr>
<tr>
  <th>C(country_standard)[T.New Caledonia]</th>                    <td>   -0.2677</td> <td>    0.009</td> <td>  -31.249</td> <td> 0.000</td> <td>   -0.284</td> <td>   -0.251</td>
</tr>
<tr>
  <th>C(country_standard)[T.New Zealand]</th>                      <td>   -0.7141</td> <td>    0.007</td> <td> -103.563</td> <td> 0.000</td> <td>   -0.728</td> <td>   -0.701</td>
</tr>
<tr>
  <th>C(country_standard)[T.Nicaragua]</th>                        <td>   -0.1962</td> <td>    0.002</td> <td>  -79.996</td> <td> 0.000</td> <td>   -0.201</td> <td>   -0.191</td>
</tr>
<tr>
  <th>C(country_standard)[T.Niger]</th>                            <td>   -0.1296</td> <td>    0.003</td> <td>  -40.503</td> <td> 0.000</td> <td>   -0.136</td> <td>   -0.123</td>
</tr>
<tr>
  <th>C(country_standard)[T.Nigeria]</th>                          <td>   -0.0346</td> <td>    0.003</td> <td>  -12.426</td> <td> 0.000</td> <td>   -0.040</td> <td>   -0.029</td>
</tr>
<tr>
  <th>C(country_standard)[T.Niue]</th>                             <td>   -0.2317</td> <td>    0.009</td> <td>  -25.870</td> <td> 0.000</td> <td>   -0.249</td> <td>   -0.214</td>
</tr>
<tr>
  <th>C(country_standard)[T.North Korea]</th>                      <td>   -0.0935</td> <td>    0.002</td> <td>  -42.959</td> <td> 0.000</td> <td>   -0.098</td> <td>   -0.089</td>
</tr>
<tr>
  <th>C(country_standard)[T.North Macedonia]</th>                  <td>   -0.2115</td> <td>    0.003</td> <td>  -61.007</td> <td> 0.000</td> <td>   -0.218</td> <td>   -0.205</td>
</tr>
<tr>
  <th>C(country_standard)[T.Northern Mariana Islands]</th>         <td>   -0.2677</td> <td>    0.009</td> <td>  -31.282</td> <td> 0.000</td> <td>   -0.284</td> <td>   -0.251</td>
</tr>
<tr>
  <th>C(country_standard)[T.Norway]</th>                           <td>   -0.5690</td> <td>    0.007</td> <td>  -83.608</td> <td> 0.000</td> <td>   -0.582</td> <td>   -0.556</td>
</tr>
<tr>
  <th>C(country_standard)[T.Oman]</th>                             <td>   -0.3627</td> <td>    0.001</td> <td> -257.018</td> <td> 0.000</td> <td>   -0.365</td> <td>   -0.360</td>
</tr>
<tr>
  <th>C(country_standard)[T.Pakistan]</th>                         <td>   -0.0780</td> <td>    0.001</td> <td>  -53.634</td> <td> 0.000</td> <td>   -0.081</td> <td>   -0.075</td>
</tr>
<tr>
  <th>C(country_standard)[T.Palau]</th>                            <td>   -0.1419</td> <td>    0.007</td> <td>  -19.862</td> <td> 0.000</td> <td>   -0.156</td> <td>   -0.128</td>
</tr>
<tr>
  <th>C(country_standard)[T.Palestinian Territories]</th>          <td>   -0.2260</td> <td>    0.001</td> <td> -152.863</td> <td> 0.000</td> <td>   -0.229</td> <td>   -0.223</td>
</tr>
<tr>
  <th>C(country_standard)[T.Panama]</th>                           <td>   -0.2857</td> <td>    0.006</td> <td>  -46.035</td> <td> 0.000</td> <td>   -0.298</td> <td>   -0.274</td>
</tr>
<tr>
  <th>C(country_standard)[T.Papua New Guinea]</th>                 <td>   -0.1153</td> <td>    0.004</td> <td>  -31.909</td> <td> 0.000</td> <td>   -0.122</td> <td>   -0.108</td>
</tr>
<tr>
  <th>C(country_standard)[T.Paraguay]</th>                         <td>   -0.1733</td> <td>    0.003</td> <td>  -49.764</td> <td> 0.000</td> <td>   -0.180</td> <td>   -0.167</td>
</tr>
<tr>
  <th>C(country_standard)[T.Peru]</th>                             <td>   -0.2275</td> <td>    0.006</td> <td>  -40.138</td> <td> 0.000</td> <td>   -0.239</td> <td>   -0.216</td>
</tr>
<tr>
  <th>C(country_standard)[T.Philippines]</th>                      <td>   -0.2004</td> <td>    0.003</td> <td>  -58.915</td> <td> 0.000</td> <td>   -0.207</td> <td>   -0.194</td>
</tr>
<tr>
  <th>C(country_standard)[T.Poland]</th>                           <td>   -0.2841</td> <td>    0.006</td> <td>  -47.809</td> <td> 0.000</td> <td>   -0.296</td> <td>   -0.272</td>
</tr>
<tr>
  <th>C(country_standard)[T.Portugal]</th>                         <td>   -0.4132</td> <td>    0.006</td> <td>  -63.595</td> <td> 0.000</td> <td>   -0.426</td> <td>   -0.400</td>
</tr>
<tr>
  <th>C(country_standard)[T.Puerto Rico]</th>                      <td>   -0.3243</td> <td>    0.007</td> <td>  -44.841</td> <td> 0.000</td> <td>   -0.338</td> <td>   -0.310</td>
</tr>
<tr>
  <th>C(country_standard)[T.Qatar]</th>                            <td>   -0.5139</td> <td>    0.001</td> <td> -370.072</td> <td> 0.000</td> <td>   -0.517</td> <td>   -0.511</td>
</tr>
<tr>
  <th>C(country_standard)[T.Romania]</th>                          <td>   -0.2168</td> <td>    0.006</td> <td>  -37.595</td> <td> 0.000</td> <td>   -0.228</td> <td>   -0.206</td>
</tr>
<tr>
  <th>C(country_standard)[T.Russia]</th>                           <td>   -0.0272</td> <td>    0.002</td> <td>  -16.098</td> <td> 0.000</td> <td>   -0.031</td> <td>   -0.024</td>
</tr>
<tr>
  <th>C(country_standard)[T.Rwanda]</th>                           <td>   -0.4338</td> <td>    0.001</td> <td> -316.687</td> <td> 0.000</td> <td>   -0.436</td> <td>   -0.431</td>
</tr>
<tr>
  <th>C(country_standard)[T.Réunion]</th>                          <td>   -0.4287</td> <td>    0.008</td> <td>  -54.039</td> <td> 0.000</td> <td>   -0.444</td> <td>   -0.413</td>
</tr>
<tr>
  <th>C(country_standard)[T.Saint Martin (French part)]</th>       <td>   -0.2677</td> <td>    0.009</td> <td>  -31.152</td> <td> 0.000</td> <td>   -0.285</td> <td>   -0.251</td>
</tr>
<tr>
  <th>C(country_standard)[T.Samoa]</th>                            <td>   -0.3298</td> <td>    0.006</td> <td>  -53.648</td> <td> 0.000</td> <td>   -0.342</td> <td>   -0.318</td>
</tr>
<tr>
  <th>C(country_standard)[T.San Marino]</th>                       <td>   -0.2621</td> <td>    0.007</td> <td>  -39.056</td> <td> 0.000</td> <td>   -0.275</td> <td>   -0.249</td>
</tr>
<tr>
  <th>C(country_standard)[T.Saudi Arabia]</th>                     <td>   -0.2894</td> <td>    0.002</td> <td> -167.260</td> <td> 0.000</td> <td>   -0.293</td> <td>   -0.286</td>
</tr>
<tr>
  <th>C(country_standard)[T.Senegal]</th>                          <td>   -0.2393</td> <td>    0.005</td> <td>  -43.942</td> <td> 0.000</td> <td>   -0.250</td> <td>   -0.229</td>
</tr>
<tr>
  <th>C(country_standard)[T.Serbia]</th>                           <td>   -0.1980</td> <td>    0.005</td> <td>  -37.935</td> <td> 0.000</td> <td>   -0.208</td> <td>   -0.188</td>
</tr>
<tr>
  <th>C(country_standard)[T.Seychelles]</th>                       <td>   -0.4682</td> <td>    0.004</td> <td> -120.002</td> <td> 0.000</td> <td>   -0.476</td> <td>   -0.461</td>
</tr>
<tr>
  <th>C(country_standard)[T.Sierra Leone]</th>                     <td>   -0.1048</td> <td>    0.004</td> <td>  -28.027</td> <td> 0.000</td> <td>   -0.112</td> <td>   -0.097</td>
</tr>
<tr>
  <th>C(country_standard)[T.Singapore]</th>                        <td>   -0.7016</td> <td>    0.003</td> <td> -216.077</td> <td> 0.000</td> <td>   -0.708</td> <td>   -0.695</td>
</tr>
<tr>
  <th>C(country_standard)[T.Sint Maarten]</th>                     <td>   -0.2677</td> <td>    0.009</td> <td>  -31.187</td> <td> 0.000</td> <td>   -0.285</td> <td>   -0.251</td>
</tr>
<tr>
  <th>C(country_standard)[T.Slovakia]</th>                         <td>   -0.2052</td> <td>    0.006</td> <td>  -32.292</td> <td> 0.000</td> <td>   -0.218</td> <td>   -0.193</td>
</tr>
<tr>
  <th>C(country_standard)[T.Slovenia]</th>                         <td>   -0.3224</td> <td>    0.006</td> <td>  -50.687</td> <td> 0.000</td> <td>   -0.335</td> <td>   -0.310</td>
</tr>
<tr>
  <th>C(country_standard)[T.Solomon Islands]</th>                  <td>   -0.2156</td> <td>    0.005</td> <td>  -45.123</td> <td> 0.000</td> <td>   -0.225</td> <td>   -0.206</td>
</tr>
<tr>
  <th>C(country_standard)[T.Somalia]</th>                          <td>    0.0585</td> <td>    0.002</td> <td>   32.354</td> <td> 0.000</td> <td>    0.055</td> <td>    0.062</td>
</tr>
<tr>
  <th>C(country_standard)[T.South Africa]</th>                     <td>   -0.4725</td> <td>    0.006</td> <td>  -76.465</td> <td> 0.000</td> <td>   -0.485</td> <td>   -0.460</td>
</tr>
<tr>
  <th>C(country_standard)[T.South Korea]</th>                      <td>   -0.1180</td> <td>    0.004</td> <td>  -29.237</td> <td> 0.000</td> <td>   -0.126</td> <td>   -0.110</td>
</tr>
<tr>
  <th>C(country_standard)[T.South Sudan]</th>                      <td>   -0.1723</td> <td>    0.001</td> <td> -118.662</td> <td> 0.000</td> <td>   -0.175</td> <td>   -0.169</td>
</tr>
<tr>
  <th>C(country_standard)[T.Spain]</th>                            <td>   -0.3428</td> <td>    0.006</td> <td>  -60.438</td> <td> 0.000</td> <td>   -0.354</td> <td>   -0.332</td>
</tr>
<tr>
  <th>C(country_standard)[T.Sri Lanka]</th>                        <td>   -0.2361</td> <td>    0.003</td> <td>  -71.644</td> <td> 0.000</td> <td>   -0.243</td> <td>   -0.230</td>
</tr>
<tr>
  <th>C(country_standard)[T.St. Helena]</th>                       <td>   -0.2597</td> <td>    0.004</td> <td>  -71.177</td> <td> 0.000</td> <td>   -0.267</td> <td>   -0.253</td>
</tr>
<tr>
  <th>C(country_standard)[T.St. Kitts & Nevis]</th>                <td>   -0.4111</td> <td>    0.007</td> <td>  -62.498</td> <td> 0.000</td> <td>   -0.424</td> <td>   -0.398</td>
</tr>
<tr>
  <th>C(country_standard)[T.St. Lucia]</th>                        <td>   -0.5639</td> <td>    0.007</td> <td>  -85.377</td> <td> 0.000</td> <td>   -0.577</td> <td>   -0.551</td>
</tr>
<tr>
  <th>C(country_standard)[T.St. Vincent & Grenadines]</th>         <td>   -0.4382</td> <td>    0.006</td> <td>  -67.875</td> <td> 0.000</td> <td>   -0.451</td> <td>   -0.426</td>
</tr>
<tr>
  <th>C(country_standard)[T.Sudan]</th>                            <td>   -0.0019</td> <td>    0.002</td> <td>   -1.148</td> <td> 0.251</td> <td>   -0.005</td> <td>    0.001</td>
</tr>
<tr>
  <th>C(country_standard)[T.Suriname]</th>                         <td>   -0.2229</td> <td>    0.006</td> <td>  -37.467</td> <td> 0.000</td> <td>   -0.235</td> <td>   -0.211</td>
</tr>
<tr>
  <th>C(country_standard)[T.Sweden]</th>                           <td>   -0.5987</td> <td>    0.007</td> <td>  -89.817</td> <td> 0.000</td> <td>   -0.612</td> <td>   -0.586</td>
</tr>
<tr>
  <th>C(country_standard)[T.Switzerland]</th>                      <td>   -0.5988</td> <td>    0.007</td> <td>  -91.222</td> <td> 0.000</td> <td>   -0.612</td> <td>   -0.586</td>
</tr>
<tr>
  <th>C(country_standard)[T.Syria]</th>                            <td>   -0.0300</td> <td>    0.002</td> <td>  -16.896</td> <td> 0.000</td> <td>   -0.033</td> <td>   -0.027</td>
</tr>
<tr>
  <th>C(country_standard)[T.São Tomé & Príncipe]</th>              <td>   -0.3998</td> <td>    0.006</td> <td>  -67.043</td> <td> 0.000</td> <td>   -0.412</td> <td>   -0.388</td>
</tr>
<tr>
  <th>C(country_standard)[T.Taiwan]</th>                           <td>   -0.3962</td> <td>    0.006</td> <td>  -64.800</td> <td> 0.000</td> <td>   -0.408</td> <td>   -0.384</td>
</tr>
<tr>
  <th>C(country_standard)[T.Tajikistan]</th>                       <td>   -0.0485</td> <td>    0.002</td> <td>  -30.267</td> <td> 0.000</td> <td>   -0.052</td> <td>   -0.045</td>
</tr>
<tr>
  <th>C(country_standard)[T.Tanzania]</th>                         <td>   -0.2053</td> <td>    0.003</td> <td>  -63.294</td> <td> 0.000</td> <td>   -0.212</td> <td>   -0.199</td>
</tr>
<tr>
  <th>C(country_standard)[T.Thailand]</th>                         <td>   -0.1851</td> <td>    0.002</td> <td> -123.100</td> <td> 0.000</td> <td>   -0.188</td> <td>   -0.182</td>
</tr>
<tr>
  <th>C(country_standard)[T.Timor-Leste]</th>                      <td>   -0.0751</td> <td>    0.004</td> <td>  -17.146</td> <td> 0.000</td> <td>   -0.084</td> <td>   -0.066</td>
</tr>
<tr>
  <th>C(country_standard)[T.Togo]</th>                             <td>   -0.1610</td> <td>    0.003</td> <td>  -55.391</td> <td> 0.000</td> <td>   -0.167</td> <td>   -0.155</td>
</tr>
<tr>
  <th>C(country_standard)[T.Tonga]</th>                            <td>   -0.1310</td> <td>    0.007</td> <td>  -17.769</td> <td> 0.000</td> <td>   -0.146</td> <td>   -0.117</td>
</tr>
<tr>
  <th>C(country_standard)[T.Trinidad & Tobago]</th>                <td>   -0.2255</td> <td>    0.006</td> <td>  -36.571</td> <td> 0.000</td> <td>   -0.238</td> <td>   -0.213</td>
</tr>
<tr>
  <th>C(country_standard)[T.Tunisia]</th>                          <td>   -0.1946</td> <td>    0.005</td> <td>  -36.315</td> <td> 0.000</td> <td>   -0.205</td> <td>   -0.184</td>
</tr>
<tr>
  <th>C(country_standard)[T.Turkey]</th>                           <td>   -0.2854</td> <td>    0.002</td> <td> -147.503</td> <td> 0.000</td> <td>   -0.289</td> <td>   -0.282</td>
</tr>
<tr>
  <th>C(country_standard)[T.Turkmenistan]</th>                     <td>   -0.0135</td> <td>    0.002</td> <td>   -6.337</td> <td> 0.000</td> <td>   -0.018</td> <td>   -0.009</td>
</tr>
<tr>
  <th>C(country_standard)[T.Turks & Caicos Islands]</th>           <td>   -0.2596</td> <td>    0.004</td> <td>  -71.603</td> <td> 0.000</td> <td>   -0.267</td> <td>   -0.252</td>
</tr>
<tr>
  <th>C(country_standard)[T.Tuvalu]</th>                           <td>   -0.2083</td> <td>    0.007</td> <td>  -28.379</td> <td> 0.000</td> <td>   -0.223</td> <td>   -0.194</td>
</tr>
<tr>
  <th>C(country_standard)[T.U.S. Virgin Islands]</th>              <td>   -0.4312</td> <td>    0.008</td> <td>  -52.316</td> <td> 0.000</td> <td>   -0.447</td> <td>   -0.415</td>
</tr>
<tr>
  <th>C(country_standard)[T.Uganda]</th>                           <td>   -0.1513</td> <td>    0.002</td> <td>  -72.751</td> <td> 0.000</td> <td>   -0.155</td> <td>   -0.147</td>
</tr>
<tr>
  <th>C(country_standard)[T.Ukraine]</th>                          <td>    0.0007</td> <td>    0.003</td> <td>    0.216</td> <td> 0.829</td> <td>   -0.006</td> <td>    0.007</td>
</tr>
<tr>
  <th>C(country_standard)[T.United Arab Emirates]</th>             <td>   -0.4712</td> <td>    0.002</td> <td> -265.496</td> <td> 0.000</td> <td>   -0.475</td> <td>   -0.468</td>
</tr>
<tr>
  <th>C(country_standard)[T.United Kingdom]</th>                   <td>   -0.4337</td> <td>    0.005</td> <td>  -86.023</td> <td> 0.000</td> <td>   -0.444</td> <td>   -0.424</td>
</tr>
<tr>
  <th>C(country_standard)[T.United States]</th>                    <td>    0.5691</td> <td>    0.007</td> <td>   86.428</td> <td> 0.000</td> <td>    0.556</td> <td>    0.582</td>
</tr>
<tr>
  <th>C(country_standard)[T.Uruguay]</th>                          <td>   -0.5376</td> <td>    0.007</td> <td>  -81.573</td> <td> 0.000</td> <td>   -0.551</td> <td>   -0.525</td>
</tr>
<tr>
  <th>C(country_standard)[T.Uzbekistan]</th>                       <td>   -0.0754</td> <td>    0.002</td> <td>  -46.065</td> <td> 0.000</td> <td>   -0.079</td> <td>   -0.072</td>
</tr>
<tr>
  <th>C(country_standard)[T.Vanuatu]</th>                          <td>   -0.2958</td> <td>    0.006</td> <td>  -48.682</td> <td> 0.000</td> <td>   -0.308</td> <td>   -0.284</td>
</tr>
<tr>
  <th>C(country_standard)[T.Vatican City]</th>                     <td>   -0.2614</td> <td>    0.004</td> <td>  -73.738</td> <td> 0.000</td> <td>   -0.268</td> <td>   -0.254</td>
</tr>
<tr>
  <th>C(country_standard)[T.Venezuela]</th>                        <td>   -0.0966</td> <td>    0.001</td> <td>  -66.173</td> <td> 0.000</td> <td>   -0.099</td> <td>   -0.094</td>
</tr>
<tr>
  <th>C(country_standard)[T.Vietnam]</th>                          <td>   -0.1642</td> <td>    0.001</td> <td> -109.944</td> <td> 0.000</td> <td>   -0.167</td> <td>   -0.161</td>
</tr>
<tr>
  <th>C(country_standard)[T.Western Sahara]</th>                   <td>   -0.3364</td> <td>    0.007</td> <td>  -51.302</td> <td> 0.000</td> <td>   -0.349</td> <td>   -0.324</td>
</tr>
<tr>
  <th>C(country_standard)[T.Yemen]</th>                            <td>   -0.0520</td> <td>    0.002</td> <td>  -34.184</td> <td> 0.000</td> <td>   -0.055</td> <td>   -0.049</td>
</tr>
<tr>
  <th>C(country_standard)[T.Zambia]</th>                           <td>   -0.3616</td> <td>    0.003</td> <td> -105.416</td> <td> 0.000</td> <td>   -0.368</td> <td>   -0.355</td>
</tr>
<tr>
  <th>C(country_standard)[T.Zimbabwe]</th>                         <td>   -0.0957</td> <td>    0.002</td> <td>  -57.861</td> <td> 0.000</td> <td>   -0.099</td> <td>   -0.092</td>
</tr>
<tr>
  <th>transparency_index</th>                                      <td>   -0.1307</td> <td>    0.017</td> <td>   -7.808</td> <td> 0.000</td> <td>   -0.164</td> <td>   -0.098</td>
</tr>
<tr>
  <th>gdp</th>                                                     <td>-5.318e-14</td> <td> 6.16e-16</td> <td>  -86.309</td> <td> 0.000</td> <td>-5.44e-14</td> <td> -5.2e-14</td>
</tr>
<tr>
  <th>gini</th>                                                    <td>    0.0084</td> <td> 1.48e-05</td> <td>  567.678</td> <td> 0.000</td> <td>    0.008</td> <td>    0.008</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>827.049</td> <th>  Durbin-Watson:     </th> <td>   0.991</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>6118.243</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.892</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 9.130</td>  <th>  Cond. No.          </th> <td>9.45e+26</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors are robust to cluster correlation (cluster)<br/>[2] The smallest eigenvalue is 1.39e-26. This might indicate that there are<br/>strong multicollinearity problems or that the design matrix is singular.




```python
reg_trust = smf.ols('trust_index ~ transparency_index + gdp + gini + C(country_standard)', ndi_df).fit(cov_type='cluster', cov_kwds={'groups': ndi_df['country_standard']})
```


```python
reg_trust.summary()
```

    /Users/katiacordoba/opt/anaconda3/lib/python3.7/site-packages/statsmodels/base/model.py:1832: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 231, but rank is 228
      'rank is %d' % (J, J_), ValueWarning)





<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>trust_index</td>   <th>  R-squared:         </th>  <td>   0.108</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.107</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>3.171e+11</td>
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 05 Mar 2021</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>23:12:07</td>     <th>  Log-Likelihood:    </th>  <td>  6684.0</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>  3603</td>      <th>  AIC:               </th> <td>-1.336e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  3598</td>      <th>  BIC:               </th> <td>-1.333e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     4</td>      <th>                     </th>      <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>       <td>cluster</td>     <th>                     </th>      <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
                             <td></td>                                <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>                                               <td>    0.3530</td> <td>    0.002</td> <td>  189.659</td> <td> 0.000</td> <td>    0.349</td> <td>    0.357</td>
</tr>
<tr>
  <th>C(country_standard)[T.Albania]</th>                          <td>    0.0055</td> <td>    0.001</td> <td>    6.202</td> <td> 0.000</td> <td>    0.004</td> <td>    0.007</td>
</tr>
<tr>
  <th>C(country_standard)[T.Algeria]</th>                          <td>    0.0102</td> <td>  1.6e-05</td> <td>  639.502</td> <td> 0.000</td> <td>    0.010</td> <td>    0.010</td>
</tr>
<tr>
  <th>C(country_standard)[T.American Samoa]</th>                   <td>   -0.0023</td> <td>    0.001</td> <td>   -2.080</td> <td> 0.038</td> <td>   -0.005</td> <td>   -0.000</td>
</tr>
<tr>
  <th>C(country_standard)[T.Andorra]</th>                          <td>   -0.0018</td> <td>    0.002</td> <td>   -0.965</td> <td> 0.335</td> <td>   -0.005</td> <td>    0.002</td>
</tr>
<tr>
  <th>C(country_standard)[T.Angola]</th>                           <td>   -0.0187</td> <td> 2.76e-05</td> <td> -677.622</td> <td> 0.000</td> <td>   -0.019</td> <td>   -0.019</td>
</tr>
<tr>
  <th>C(country_standard)[T.Anguilla]</th>                         <td>   -0.0023</td> <td>    0.001</td> <td>   -2.078</td> <td> 0.038</td> <td>   -0.004</td> <td>   -0.000</td>
</tr>
<tr>
  <th>C(country_standard)[T.Antigua & Barbuda]</th>                <td>   -0.0027</td> <td>    0.002</td> <td>   -1.609</td> <td> 0.108</td> <td>   -0.006</td> <td>    0.001</td>
</tr>
<tr>
  <th>C(country_standard)[T.Argentina]</th>                        <td>   -0.0460</td> <td>    0.002</td> <td>  -29.256</td> <td> 0.000</td> <td>   -0.049</td> <td>   -0.043</td>
</tr>
<tr>
  <th>C(country_standard)[T.Armenia]</th>                          <td>   -0.0057</td> <td>    0.001</td> <td>   -7.374</td> <td> 0.000</td> <td>   -0.007</td> <td>   -0.004</td>
</tr>
<tr>
  <th>C(country_standard)[T.Aruba]</th>                            <td>   -0.0034</td> <td>    0.002</td> <td>   -1.901</td> <td> 0.057</td> <td>   -0.007</td> <td>    0.000</td>
</tr>
<tr>
  <th>C(country_standard)[T.Australia]</th>                        <td>    0.0023</td> <td>    0.002</td> <td>    1.393</td> <td> 0.164</td> <td>   -0.001</td> <td>    0.006</td>
</tr>
<tr>
  <th>C(country_standard)[T.Austria]</th>                          <td>    0.0083</td> <td>    0.002</td> <td>    4.586</td> <td> 0.000</td> <td>    0.005</td> <td>    0.012</td>
</tr>
<tr>
  <th>C(country_standard)[T.Azerbaijan]</th>                       <td>    0.0128</td> <td>    0.000</td> <td>   98.853</td> <td> 0.000</td> <td>    0.013</td> <td>    0.013</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bahamas]</th>                          <td>   -0.0029</td> <td>    0.002</td> <td>   -1.620</td> <td> 0.105</td> <td>   -0.006</td> <td>    0.001</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bahrain]</th>                          <td> 9.805e-05</td> <td> 9.12e-05</td> <td>    1.076</td> <td> 0.282</td> <td>-8.06e-05</td> <td>    0.000</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bangladesh]</th>                       <td>    0.0236</td> <td>    0.001</td> <td>   34.558</td> <td> 0.000</td> <td>    0.022</td> <td>    0.025</td>
</tr>
<tr>
  <th>C(country_standard)[T.Barbados]</th>                         <td>   -0.0030</td> <td>    0.002</td> <td>   -1.609</td> <td> 0.108</td> <td>   -0.007</td> <td>    0.001</td>
</tr>
<tr>
  <th>C(country_standard)[T.Belarus]</th>                          <td>    0.0216</td> <td>    0.000</td> <td>  138.223</td> <td> 0.000</td> <td>    0.021</td> <td>    0.022</td>
</tr>
<tr>
  <th>C(country_standard)[T.Belgium]</th>                          <td>    0.0115</td> <td>    0.002</td> <td>    6.429</td> <td> 0.000</td> <td>    0.008</td> <td>    0.015</td>
</tr>
<tr>
  <th>C(country_standard)[T.Belize]</th>                           <td>   -0.0027</td> <td>    0.002</td> <td>   -1.607</td> <td> 0.108</td> <td>   -0.006</td> <td>    0.001</td>
</tr>
<tr>
  <th>C(country_standard)[T.Benin]</th>                            <td>   -0.0161</td> <td>    0.002</td> <td>  -10.534</td> <td> 0.000</td> <td>   -0.019</td> <td>   -0.013</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bermuda]</th>                          <td>   -0.0023</td> <td>    0.001</td> <td>   -2.076</td> <td> 0.038</td> <td>   -0.004</td> <td>   -0.000</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bhutan]</th>                           <td>   -0.0005</td> <td>    0.001</td> <td>   -0.539</td> <td> 0.590</td> <td>   -0.002</td> <td>    0.001</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bolivia]</th>                          <td>   -0.0212</td> <td>    0.001</td> <td>  -23.533</td> <td> 0.000</td> <td>   -0.023</td> <td>   -0.019</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bosnia & Herzegovina]</th>             <td>    0.0057</td> <td>    0.001</td> <td>    6.764</td> <td> 0.000</td> <td>    0.004</td> <td>    0.007</td>
</tr>
<tr>
  <th>C(country_standard)[T.Botswana]</th>                         <td>   -0.0241</td> <td>    0.002</td> <td>  -14.323</td> <td> 0.000</td> <td>   -0.027</td> <td>   -0.021</td>
</tr>
<tr>
  <th>C(country_standard)[T.Brazil]</th>                           <td>   -0.0568</td> <td>    0.001</td> <td>  -41.202</td> <td> 0.000</td> <td>   -0.060</td> <td>   -0.054</td>
</tr>
<tr>
  <th>C(country_standard)[T.British Virgin Islands]</th>           <td>   -0.0021</td> <td>    0.001</td> <td>   -2.165</td> <td> 0.030</td> <td>   -0.004</td> <td>   -0.000</td>
</tr>
<tr>
  <th>C(country_standard)[T.Brunei]</th>                           <td>   -0.0002</td> <td> 9.12e-05</td> <td>   -1.702</td> <td> 0.089</td> <td>   -0.000</td> <td> 2.35e-05</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bulgaria]</th>                         <td>   -0.0155</td> <td>    0.002</td> <td>   -9.607</td> <td> 0.000</td> <td>   -0.019</td> <td>   -0.012</td>
</tr>
<tr>
  <th>C(country_standard)[T.Burkina Faso]</th>                     <td>    0.0011</td> <td>    0.001</td> <td>    1.292</td> <td> 0.196</td> <td>   -0.001</td> <td>    0.003</td>
</tr>
<tr>
  <th>C(country_standard)[T.Burundi]</th>                          <td>   -0.0009</td> <td> 7.38e-05</td> <td>  -11.524</td> <td> 0.000</td> <td>   -0.001</td> <td>   -0.001</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cambodia]</th>                         <td>  1.61e-06</td> <td> 1.95e-05</td> <td>    0.083</td> <td> 0.934</td> <td>-3.65e-05</td> <td> 3.98e-05</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cameroon]</th>                         <td>   -0.0120</td> <td> 2.73e-05</td> <td> -441.532</td> <td> 0.000</td> <td>   -0.012</td> <td>   -0.012</td>
</tr>
<tr>
  <th>C(country_standard)[T.Canada]</th>                           <td>    0.0034</td> <td>    0.002</td> <td>    2.069</td> <td> 0.039</td> <td>    0.000</td> <td>    0.007</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cape Verde]</th>                       <td>   -0.0090</td> <td>    0.002</td> <td>   -4.980</td> <td> 0.000</td> <td>   -0.012</td> <td>   -0.005</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cayman Islands]</th>                   <td>   -0.0021</td> <td>    0.001</td> <td>   -1.610</td> <td> 0.107</td> <td>   -0.005</td> <td>    0.000</td>
</tr>
<tr>
  <th>C(country_standard)[T.Central African Republic]</th>         <td>   -0.0255</td> <td> 7.58e-05</td> <td> -337.069</td> <td> 0.000</td> <td>   -0.026</td> <td>   -0.025</td>
</tr>
<tr>
  <th>C(country_standard)[T.Chad]</th>                             <td>   -0.0072</td> <td> 7.76e-05</td> <td>  -93.018</td> <td> 0.000</td> <td>   -0.007</td> <td>   -0.007</td>
</tr>
<tr>
  <th>C(country_standard)[T.Channel Islands]</th>                  <td>   -0.0020</td> <td>    0.001</td> <td>   -2.153</td> <td> 0.031</td> <td>   -0.004</td> <td>   -0.000</td>
</tr>
<tr>
  <th>C(country_standard)[T.Chile]</th>                            <td>   -0.0362</td> <td>    0.002</td> <td>  -20.302</td> <td> 0.000</td> <td>   -0.040</td> <td>   -0.033</td>
</tr>
<tr>
  <th>C(country_standard)[T.China]</th>                            <td>    0.0551</td> <td>    0.003</td> <td>   21.949</td> <td> 0.000</td> <td>    0.050</td> <td>    0.060</td>
</tr>
<tr>
  <th>C(country_standard)[T.Colombia]</th>                         <td>   -0.0460</td> <td>    0.001</td> <td>  -50.512</td> <td> 0.000</td> <td>   -0.048</td> <td>   -0.044</td>
</tr>
<tr>
  <th>C(country_standard)[T.Comoros]</th>                          <td>   -0.0114</td> <td>    0.001</td> <td>  -14.136</td> <td> 0.000</td> <td>   -0.013</td> <td>   -0.010</td>
</tr>
<tr>
  <th>C(country_standard)[T.Congo - Brazzaville]</th>              <td>   -0.0152</td> <td>  1.4e-05</td> <td>-1087.488</td> <td> 0.000</td> <td>   -0.015</td> <td>   -0.015</td>
</tr>
<tr>
  <th>C(country_standard)[T.Congo - Kinshasa]</th>                 <td>   -0.0055</td> <td> 9.73e-05</td> <td>  -56.913</td> <td> 0.000</td> <td>   -0.006</td> <td>   -0.005</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cook Islands]</th>                     <td>   -0.0020</td> <td>    0.001</td> <td>   -2.207</td> <td> 0.027</td> <td>   -0.004</td> <td>   -0.000</td>
</tr>
<tr>
  <th>C(country_standard)[T.Costa Rica]</th>                       <td>   -0.0170</td> <td>    0.002</td> <td>   -9.154</td> <td> 0.000</td> <td>   -0.021</td> <td>   -0.013</td>
</tr>
<tr>
  <th>C(country_standard)[T.Croatia]</th>                          <td>    0.0081</td> <td>    0.002</td> <td>    4.969</td> <td> 0.000</td> <td>    0.005</td> <td>    0.011</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cuba]</th>                             <td>    0.0001</td> <td>    0.000</td> <td>    0.910</td> <td> 0.363</td> <td>   -0.000</td> <td>    0.000</td>
</tr>
<tr>
  <th>C(country_standard)[T.Curaçao]</th>                          <td>   -0.0021</td> <td>    0.001</td> <td>   -2.158</td> <td> 0.031</td> <td>   -0.004</td> <td>   -0.000</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cyprus]</th>                           <td>    0.0181</td> <td>    0.002</td> <td>   10.118</td> <td> 0.000</td> <td>    0.015</td> <td>    0.022</td>
</tr>
<tr>
  <th>C(country_standard)[T.Czechia]</th>                          <td>    0.0155</td> <td>    0.002</td> <td>    9.005</td> <td> 0.000</td> <td>    0.012</td> <td>    0.019</td>
</tr>
<tr>
  <th>C(country_standard)[T.Côte d’Ivoire]</th>                    <td>   -0.0061</td> <td>    0.001</td> <td>   -7.811</td> <td> 0.000</td> <td>   -0.008</td> <td>   -0.005</td>
</tr>
<tr>
  <th>C(country_standard)[T.Denmark]</th>                          <td>    0.0097</td> <td>    0.002</td> <td>    5.162</td> <td> 0.000</td> <td>    0.006</td> <td>    0.013</td>
</tr>
<tr>
  <th>C(country_standard)[T.Djibouti]</th>                         <td>   -0.0048</td> <td> 9.17e-05</td> <td>  -52.331</td> <td> 0.000</td> <td>   -0.005</td> <td>   -0.005</td>
</tr>
<tr>
  <th>C(country_standard)[T.Dominica]</th>                         <td>   -0.0029</td> <td>    0.002</td> <td>   -1.606</td> <td> 0.108</td> <td>   -0.006</td> <td>    0.001</td>
</tr>
<tr>
  <th>C(country_standard)[T.Dominican Republic]</th>               <td>   -0.0099</td> <td>    0.001</td> <td>   -8.361</td> <td> 0.000</td> <td>   -0.012</td> <td>   -0.008</td>
</tr>
<tr>
  <th>C(country_standard)[T.Ecuador]</th>                          <td>   -0.0255</td> <td>    0.001</td> <td>  -29.433</td> <td> 0.000</td> <td>   -0.027</td> <td>   -0.024</td>
</tr>
<tr>
  <th>C(country_standard)[T.Egypt]</th>                            <td>   -0.0148</td> <td> 2.67e-05</td> <td> -554.330</td> <td> 0.000</td> <td>   -0.015</td> <td>   -0.015</td>
</tr>
<tr>
  <th>C(country_standard)[T.El Salvador]</th>                      <td>   -0.0031</td> <td>    0.001</td> <td>   -2.130</td> <td> 0.033</td> <td>   -0.006</td> <td>   -0.000</td>
</tr>
<tr>
  <th>C(country_standard)[T.Equatorial Guinea]</th>                <td>    0.0004</td> <td>    0.000</td> <td>    1.580</td> <td> 0.114</td> <td>-9.74e-05</td> <td>    0.001</td>
</tr>
<tr>
  <th>C(country_standard)[T.Eritrea]</th>                          <td>    0.0005</td> <td>    0.000</td> <td>    1.622</td> <td> 0.105</td> <td>   -0.000</td> <td>    0.001</td>
</tr>
<tr>
  <th>C(country_standard)[T.Estonia]</th>                          <td>    0.0101</td> <td>    0.002</td> <td>    5.513</td> <td> 0.000</td> <td>    0.006</td> <td>    0.014</td>
</tr>
<tr>
  <th>C(country_standard)[T.Eswatini]</th>                         <td>   -0.0231</td> <td>  3.8e-05</td> <td> -608.907</td> <td> 0.000</td> <td>   -0.023</td> <td>   -0.023</td>
</tr>
<tr>
  <th>C(country_standard)[T.Ethiopia]</th>                         <td>    0.0029</td> <td> 8.65e-05</td> <td>   33.273</td> <td> 0.000</td> <td>    0.003</td> <td>    0.003</td>
</tr>
<tr>
  <th>C(country_standard)[T.Faroe Islands]</th>                    <td>   -0.0020</td> <td>    0.001</td> <td>   -2.155</td> <td> 0.031</td> <td>   -0.004</td> <td>   -0.000</td>
</tr>
<tr>
  <th>C(country_standard)[T.Fiji]</th>                             <td>    0.0006</td> <td>    0.001</td> <td>    0.669</td> <td> 0.504</td> <td>   -0.001</td> <td>    0.002</td>
</tr>
<tr>
  <th>C(country_standard)[T.Finland]</th>                          <td>    0.0116</td> <td>    0.002</td> <td>    6.155</td> <td> 0.000</td> <td>    0.008</td> <td>    0.015</td>
</tr>
<tr>
  <th>C(country_standard)[T.France]</th>                           <td>   -0.0045</td> <td>    0.001</td> <td>   -3.192</td> <td> 0.001</td> <td>   -0.007</td> <td>   -0.002</td>
</tr>
<tr>
  <th>C(country_standard)[T.French Guiana]</th>                    <td>   -0.0034</td> <td>    0.002</td> <td>   -1.906</td> <td> 0.057</td> <td>   -0.007</td> <td> 9.51e-05</td>
</tr>
<tr>
  <th>C(country_standard)[T.French Polynesia]</th>                 <td>   -0.0021</td> <td>    0.001</td> <td>   -2.157</td> <td> 0.031</td> <td>   -0.004</td> <td>   -0.000</td>
</tr>
<tr>
  <th>C(country_standard)[T.Gabon]</th>                            <td> 3.614e-05</td> <td> 3.75e-05</td> <td>    0.963</td> <td> 0.335</td> <td>-3.74e-05</td> <td>    0.000</td>
</tr>
<tr>
  <th>C(country_standard)[T.Gambia]</th>                           <td>    0.0027</td> <td>    0.000</td> <td>   10.486</td> <td> 0.000</td> <td>    0.002</td> <td>    0.003</td>
</tr>
<tr>
  <th>C(country_standard)[T.Georgia]</th>                          <td>   -0.0172</td> <td>    0.001</td> <td>  -17.859</td> <td> 0.000</td> <td>   -0.019</td> <td>   -0.015</td>
</tr>
<tr>
  <th>C(country_standard)[T.Germany]</th>                          <td>    0.0079</td> <td>    0.001</td> <td>    6.180</td> <td> 0.000</td> <td>    0.005</td> <td>    0.010</td>
</tr>
<tr>
  <th>C(country_standard)[T.Ghana]</th>                            <td>    0.0091</td> <td>    0.002</td> <td>    5.487</td> <td> 0.000</td> <td>    0.006</td> <td>    0.012</td>
</tr>
<tr>
  <th>C(country_standard)[T.Gibraltar]</th>                        <td>   -0.0020</td> <td>    0.001</td> <td>   -2.156</td> <td> 0.031</td> <td>   -0.004</td> <td>   -0.000</td>
</tr>
<tr>
  <th>C(country_standard)[T.Greece]</th>                           <td>   -0.0054</td> <td>    0.002</td> <td>   -3.238</td> <td> 0.001</td> <td>   -0.009</td> <td>   -0.002</td>
</tr>
<tr>
  <th>C(country_standard)[T.Greenland]</th>                        <td>   -0.0033</td> <td>    0.002</td> <td>   -1.907</td> <td> 0.056</td> <td>   -0.007</td> <td> 9.05e-05</td>
</tr>
<tr>
  <th>C(country_standard)[T.Grenada]</th>                          <td>   -0.0028</td> <td>    0.002</td> <td>   -1.607</td> <td> 0.108</td> <td>   -0.006</td> <td>    0.001</td>
</tr>
<tr>
  <th>C(country_standard)[T.Guam]</th>                             <td>   -0.0023</td> <td>    0.001</td> <td>   -2.092</td> <td> 0.036</td> <td>   -0.004</td> <td>   -0.000</td>
</tr>
<tr>
  <th>C(country_standard)[T.Guatemala]</th>                        <td>   -0.0365</td> <td>    0.001</td> <td>  -43.958</td> <td> 0.000</td> <td>   -0.038</td> <td>   -0.035</td>
</tr>
<tr>
  <th>C(country_standard)[T.Guernsey]</th>                         <td>   -0.0020</td> <td>    0.001</td> <td>   -2.157</td> <td> 0.031</td> <td>   -0.004</td> <td>   -0.000</td>
</tr>
<tr>
  <th>C(country_standard)[T.Guinea]</th>                           <td>    0.0051</td> <td>    0.001</td> <td>    7.875</td> <td> 0.000</td> <td>    0.004</td> <td>    0.006</td>
</tr>
<tr>
  <th>C(country_standard)[T.Guinea-Bissau]</th>                    <td>   -0.0186</td> <td>    0.001</td> <td>  -34.301</td> <td> 0.000</td> <td>   -0.020</td> <td>   -0.018</td>
</tr>
<tr>
  <th>C(country_standard)[T.Guyana]</th>                           <td>   -0.0025</td> <td>    0.002</td> <td>   -1.610</td> <td> 0.107</td> <td>   -0.006</td> <td>    0.001</td>
</tr>
<tr>
  <th>C(country_standard)[T.Haiti]</th>                            <td>   -0.0252</td> <td>    0.001</td> <td>  -36.278</td> <td> 0.000</td> <td>   -0.027</td> <td>   -0.024</td>
</tr>
<tr>
  <th>C(country_standard)[T.Honduras]</th>                         <td>   -0.0210</td> <td>    0.001</td> <td>  -25.958</td> <td> 0.000</td> <td>   -0.023</td> <td>   -0.019</td>
</tr>
<tr>
  <th>C(country_standard)[T.Hong Kong SAR China]</th>              <td>    0.0137</td> <td>    0.001</td> <td>   13.775</td> <td> 0.000</td> <td>    0.012</td> <td>    0.016</td>
</tr>
<tr>
  <th>C(country_standard)[T.Hungary]</th>                          <td>   -0.0031</td> <td>    0.001</td> <td>   -2.153</td> <td> 0.031</td> <td>   -0.006</td> <td>   -0.000</td>
</tr>
<tr>
  <th>C(country_standard)[T.Iceland]</th>                          <td>    0.0128</td> <td>    0.002</td> <td>    6.815</td> <td> 0.000</td> <td>    0.009</td> <td>    0.016</td>
</tr>
<tr>
  <th>C(country_standard)[T.India]</th>                            <td>    0.0172</td> <td>    0.001</td> <td>   19.039</td> <td> 0.000</td> <td>    0.015</td> <td>    0.019</td>
</tr>
<tr>
  <th>C(country_standard)[T.Indonesia]</th>                        <td>    0.0162</td> <td>    0.001</td> <td>   19.310</td> <td> 0.000</td> <td>    0.015</td> <td>    0.018</td>
</tr>
<tr>
  <th>C(country_standard)[T.Iran]</th>                             <td>    0.0103</td> <td>    0.000</td> <td>   57.367</td> <td> 0.000</td> <td>    0.010</td> <td>    0.011</td>
</tr>
<tr>
  <th>C(country_standard)[T.Iraq]</th>                             <td>   -0.0050</td> <td> 5.85e-05</td> <td>  -85.683</td> <td> 0.000</td> <td>   -0.005</td> <td>   -0.005</td>
</tr>
<tr>
  <th>C(country_standard)[T.Ireland]</th>                          <td>    0.0041</td> <td>    0.002</td> <td>    2.242</td> <td> 0.025</td> <td>    0.001</td> <td>    0.008</td>
</tr>
<tr>
  <th>C(country_standard)[T.Isle of Man]</th>                      <td>   -0.0020</td> <td>    0.001</td> <td>   -2.156</td> <td> 0.031</td> <td>   -0.004</td> <td>   -0.000</td>
</tr>
<tr>
  <th>C(country_standard)[T.Israel]</th>                           <td>   -0.0044</td> <td>    0.002</td> <td>   -2.651</td> <td> 0.008</td> <td>   -0.008</td> <td>   -0.001</td>
</tr>
<tr>
  <th>C(country_standard)[T.Italy]</th>                            <td>   -0.0015</td> <td>    0.001</td> <td>   -1.012</td> <td> 0.312</td> <td>   -0.004</td> <td>    0.001</td>
</tr>
<tr>
  <th>C(country_standard)[T.Jamaica]</th>                          <td>   -0.0027</td> <td>    0.002</td> <td>   -1.617</td> <td> 0.106</td> <td>   -0.006</td> <td>    0.001</td>
</tr>
<tr>
  <th>C(country_standard)[T.Japan]</th>                            <td>    0.0044</td> <td>    0.001</td> <td>    4.590</td> <td> 0.000</td> <td>    0.003</td> <td>    0.006</td>
</tr>
<tr>
  <th>C(country_standard)[T.Jersey]</th>                           <td>   -0.0033</td> <td>    0.002</td> <td>   -1.906</td> <td> 0.057</td> <td>   -0.007</td> <td>  9.2e-05</td>
</tr>
<tr>
  <th>C(country_standard)[T.Jordan]</th>                           <td>    0.0270</td> <td>    0.000</td> <td>   71.413</td> <td> 0.000</td> <td>    0.026</td> <td>    0.028</td>
</tr>
<tr>
  <th>C(country_standard)[T.Kazakhstan]</th>                       <td>    0.0362</td> <td> 9.89e-05</td> <td>  366.119</td> <td> 0.000</td> <td>    0.036</td> <td>    0.036</td>
</tr>
<tr>
  <th>C(country_standard)[T.Kenya]</th>                            <td>   -0.0053</td> <td>    0.001</td> <td>   -6.280</td> <td> 0.000</td> <td>   -0.007</td> <td>   -0.004</td>
</tr>
<tr>
  <th>C(country_standard)[T.Kiribati]</th>                         <td>   -0.0035</td> <td>    0.002</td> <td>   -1.897</td> <td> 0.058</td> <td>   -0.007</td> <td>    0.000</td>
</tr>
<tr>
  <th>C(country_standard)[T.Kosovo]</th>                           <td>   -0.0020</td> <td>    0.001</td> <td>   -2.153</td> <td> 0.031</td> <td>   -0.004</td> <td>   -0.000</td>
</tr>
<tr>
  <th>C(country_standard)[T.Kuwait]</th>                           <td>    0.0116</td> <td>    0.001</td> <td>   16.477</td> <td> 0.000</td> <td>    0.010</td> <td>    0.013</td>
</tr>
<tr>
  <th>C(country_standard)[T.Kyrgyzstan]</th>                       <td>    0.0118</td> <td>    0.001</td> <td>   15.829</td> <td> 0.000</td> <td>    0.010</td> <td>    0.013</td>
</tr>
<tr>
  <th>C(country_standard)[T.Laos]</th>                             <td>    0.0027</td> <td>    0.000</td> <td>   13.159</td> <td> 0.000</td> <td>    0.002</td> <td>    0.003</td>
</tr>
<tr>
  <th>C(country_standard)[T.Latvia]</th>                           <td>    0.0006</td> <td>    0.002</td> <td>    0.373</td> <td> 0.709</td> <td>   -0.003</td> <td>    0.004</td>
</tr>
<tr>
  <th>C(country_standard)[T.Lebanon]</th>                          <td>   -0.0143</td> <td>    0.001</td> <td>  -19.176</td> <td> 0.000</td> <td>   -0.016</td> <td>   -0.013</td>
</tr>
<tr>
  <th>C(country_standard)[T.Lesotho]</th>                          <td>   -0.0115</td> <td>    0.001</td> <td>   -9.779</td> <td> 0.000</td> <td>   -0.014</td> <td>   -0.009</td>
</tr>
<tr>
  <th>C(country_standard)[T.Liberia]</th>                          <td>    0.0025</td> <td>    0.001</td> <td>    2.981</td> <td> 0.003</td> <td>    0.001</td> <td>    0.004</td>
</tr>
<tr>
  <th>C(country_standard)[T.Libya]</th>                            <td>   -0.0063</td> <td> 8.14e-05</td> <td>  -77.553</td> <td> 0.000</td> <td>   -0.006</td> <td>   -0.006</td>
</tr>
<tr>
  <th>C(country_standard)[T.Liechtenstein]</th>                    <td>   -0.0031</td> <td>    0.002</td> <td>   -1.611</td> <td> 0.107</td> <td>   -0.007</td> <td>    0.001</td>
</tr>
<tr>
  <th>C(country_standard)[T.Lithuania]</th>                        <td>   -0.0018</td> <td>    0.002</td> <td>   -1.027</td> <td> 0.305</td> <td>   -0.005</td> <td>    0.002</td>
</tr>
<tr>
  <th>C(country_standard)[T.Luxembourg]</th>                       <td>    0.0013</td> <td>    0.002</td> <td>    0.644</td> <td> 0.519</td> <td>   -0.003</td> <td>    0.005</td>
</tr>
<tr>
  <th>C(country_standard)[T.Macao SAR China]</th>                  <td>    0.0097</td> <td>    0.001</td> <td>   11.043</td> <td> 0.000</td> <td>    0.008</td> <td>    0.011</td>
</tr>
<tr>
  <th>C(country_standard)[T.Madagascar]</th>                       <td>   -0.0076</td> <td>    0.001</td> <td>   -9.719</td> <td> 0.000</td> <td>   -0.009</td> <td>   -0.006</td>
</tr>
<tr>
  <th>C(country_standard)[T.Malawi]</th>                           <td>   -0.0108</td> <td>    0.001</td> <td>  -11.893</td> <td> 0.000</td> <td>   -0.013</td> <td>   -0.009</td>
</tr>
<tr>
  <th>C(country_standard)[T.Malaysia]</th>                         <td>    0.0274</td> <td>    0.001</td> <td>   35.567</td> <td> 0.000</td> <td>    0.026</td> <td>    0.029</td>
</tr>
<tr>
  <th>C(country_standard)[T.Maldives]</th>                         <td>    0.0083</td> <td>    0.001</td> <td>   11.461</td> <td> 0.000</td> <td>    0.007</td> <td>    0.010</td>
</tr>
<tr>
  <th>C(country_standard)[T.Mali]</th>                             <td>    0.0179</td> <td>    0.001</td> <td>   24.509</td> <td> 0.000</td> <td>    0.016</td> <td>    0.019</td>
</tr>
<tr>
  <th>C(country_standard)[T.Malta]</th>                            <td>    0.0095</td> <td>    0.002</td> <td>    5.233</td> <td> 0.000</td> <td>    0.006</td> <td>    0.013</td>
</tr>
<tr>
  <th>C(country_standard)[T.Marshall Islands]</th>                 <td>   -0.0030</td> <td>    0.002</td> <td>   -1.607</td> <td> 0.108</td> <td>   -0.007</td> <td>    0.001</td>
</tr>
<tr>
  <th>C(country_standard)[T.Martinique]</th>                       <td>   -0.0022</td> <td>    0.001</td> <td>   -2.110</td> <td> 0.035</td> <td>   -0.004</td> <td>   -0.000</td>
</tr>
<tr>
  <th>C(country_standard)[T.Mauritania]</th>                       <td>    0.0075</td> <td>    0.000</td> <td>   62.069</td> <td> 0.000</td> <td>    0.007</td> <td>    0.008</td>
</tr>
<tr>
  <th>C(country_standard)[T.Mauritius]</th>                        <td>   -0.0010</td> <td>    0.002</td> <td>   -0.590</td> <td> 0.555</td> <td>   -0.004</td> <td>    0.002</td>
</tr>
<tr>
  <th>C(country_standard)[T.Mexico]</th>                           <td>   -0.0437</td> <td>    0.001</td> <td>  -60.070</td> <td> 0.000</td> <td>   -0.045</td> <td>   -0.042</td>
</tr>
<tr>
  <th>C(country_standard)[T.Micronesia (Federated States of)]</th> <td>   -0.0031</td> <td>    0.002</td> <td>   -1.926</td> <td> 0.054</td> <td>   -0.006</td> <td> 5.43e-05</td>
</tr>
<tr>
  <th>C(country_standard)[T.Moldova]</th>                          <td>    0.0057</td> <td>    0.001</td> <td>    6.629</td> <td> 0.000</td> <td>    0.004</td> <td>    0.007</td>
</tr>
<tr>
  <th>C(country_standard)[T.Monaco]</th>                           <td>   -0.0028</td> <td>    0.002</td> <td>   -1.610</td> <td> 0.107</td> <td>   -0.006</td> <td>    0.001</td>
</tr>
<tr>
  <th>C(country_standard)[T.Mongolia]</th>                         <td>    0.0050</td> <td>    0.002</td> <td>    3.227</td> <td> 0.001</td> <td>    0.002</td> <td>    0.008</td>
</tr>
<tr>
  <th>C(country_standard)[T.Montenegro]</th>                       <td>   -0.0032</td> <td>    0.001</td> <td>   -2.714</td> <td> 0.007</td> <td>   -0.005</td> <td>   -0.001</td>
</tr>
<tr>
  <th>C(country_standard)[T.Morocco]</th>                          <td>    0.0060</td> <td>    0.001</td> <td>    8.557</td> <td> 0.000</td> <td>    0.005</td> <td>    0.007</td>
</tr>
<tr>
  <th>C(country_standard)[T.Mozambique]</th>                       <td>   -0.0237</td> <td>    0.001</td> <td>  -27.879</td> <td> 0.000</td> <td>   -0.025</td> <td>   -0.022</td>
</tr>
<tr>
  <th>C(country_standard)[T.Myanmar (Burma)]</th>                  <td>    0.0256</td> <td>    0.000</td> <td>  148.229</td> <td> 0.000</td> <td>    0.025</td> <td>    0.026</td>
</tr>
<tr>
  <th>C(country_standard)[T.Namibia]</th>                          <td>   -0.0323</td> <td>    0.002</td> <td>  -18.830</td> <td> 0.000</td> <td>   -0.036</td> <td>   -0.029</td>
</tr>
<tr>
  <th>C(country_standard)[T.Nauru]</th>                            <td>   -0.0033</td> <td>    0.002</td> <td>   -1.905</td> <td> 0.057</td> <td>   -0.007</td> <td> 9.71e-05</td>
</tr>
<tr>
  <th>C(country_standard)[T.Nepal]</th>                            <td>    0.0061</td> <td>    0.001</td> <td>    7.750</td> <td> 0.000</td> <td>    0.005</td> <td>    0.008</td>
</tr>
<tr>
  <th>C(country_standard)[T.Netherlands]</th>                      <td>    0.0024</td> <td>    0.002</td> <td>    1.313</td> <td> 0.189</td> <td>   -0.001</td> <td>    0.006</td>
</tr>
<tr>
  <th>C(country_standard)[T.New Caledonia]</th>                    <td>   -0.0021</td> <td>    0.001</td> <td>   -2.164</td> <td> 0.030</td> <td>   -0.004</td> <td>   -0.000</td>
</tr>
<tr>
  <th>C(country_standard)[T.New Zealand]</th>                      <td>    0.0093</td> <td>    0.002</td> <td>    4.827</td> <td> 0.000</td> <td>    0.006</td> <td>    0.013</td>
</tr>
<tr>
  <th>C(country_standard)[T.Nicaragua]</th>                        <td>   -0.0255</td> <td>    0.001</td> <td>  -43.477</td> <td> 0.000</td> <td>   -0.027</td> <td>   -0.024</td>
</tr>
<tr>
  <th>C(country_standard)[T.Niger]</th>                            <td>    0.0040</td> <td>    0.001</td> <td>    5.122</td> <td> 0.000</td> <td>    0.002</td> <td>    0.006</td>
</tr>
<tr>
  <th>C(country_standard)[T.Nigeria]</th>                          <td>   -0.0041</td> <td>    0.001</td> <td>   -5.971</td> <td> 0.000</td> <td>   -0.005</td> <td>   -0.003</td>
</tr>
<tr>
  <th>C(country_standard)[T.Niue]</th>                             <td>   -0.0019</td> <td>    0.001</td> <td>   -2.203</td> <td> 0.028</td> <td>   -0.004</td> <td>   -0.000</td>
</tr>
<tr>
  <th>C(country_standard)[T.North Korea]</th>                      <td>    0.0003</td> <td>    0.001</td> <td>    0.646</td> <td> 0.519</td> <td>   -0.001</td> <td>    0.001</td>
</tr>
<tr>
  <th>C(country_standard)[T.North Macedonia]</th>                  <td>    0.0040</td> <td>    0.001</td> <td>    4.662</td> <td> 0.000</td> <td>    0.002</td> <td>    0.006</td>
</tr>
<tr>
  <th>C(country_standard)[T.Northern Mariana Islands]</th>         <td>   -0.0021</td> <td>    0.001</td> <td>   -2.160</td> <td> 0.031</td> <td>   -0.004</td> <td>   -0.000</td>
</tr>
<tr>
  <th>C(country_standard)[T.Norway]</th>                           <td>    0.0220</td> <td>    0.002</td> <td>   11.525</td> <td> 0.000</td> <td>    0.018</td> <td>    0.026</td>
</tr>
<tr>
  <th>C(country_standard)[T.Oman]</th>                             <td>-6.396e-05</td> <td> 1.48e-05</td> <td>   -4.321</td> <td> 0.000</td> <td> -9.3e-05</td> <td> -3.5e-05</td>
</tr>
<tr>
  <th>C(country_standard)[T.Pakistan]</th>                         <td>    0.0084</td> <td>    0.000</td> <td>   70.357</td> <td> 0.000</td> <td>    0.008</td> <td>    0.009</td>
</tr>
<tr>
  <th>C(country_standard)[T.Palau]</th>                            <td>   -0.0036</td> <td>    0.002</td> <td>   -1.882</td> <td> 0.060</td> <td>   -0.007</td> <td>    0.000</td>
</tr>
<tr>
  <th>C(country_standard)[T.Palestinian Territories]</th>          <td>    0.0018</td> <td> 8.92e-05</td> <td>   20.046</td> <td> 0.000</td> <td>    0.002</td> <td>    0.002</td>
</tr>
<tr>
  <th>C(country_standard)[T.Panama]</th>                           <td>   -0.0184</td> <td>    0.002</td> <td>  -10.942</td> <td> 0.000</td> <td>   -0.022</td> <td>   -0.015</td>
</tr>
<tr>
  <th>C(country_standard)[T.Papua New Guinea]</th>                 <td>   -0.0069</td> <td>    0.001</td> <td>   -7.396</td> <td> 0.000</td> <td>   -0.009</td> <td>   -0.005</td>
</tr>
<tr>
  <th>C(country_standard)[T.Paraguay]</th>                         <td>   -0.0129</td> <td>    0.001</td> <td>  -14.120</td> <td> 0.000</td> <td>   -0.015</td> <td>   -0.011</td>
</tr>
<tr>
  <th>C(country_standard)[T.Peru]</th>                             <td>   -0.0749</td> <td>    0.002</td> <td>  -48.568</td> <td> 0.000</td> <td>   -0.078</td> <td>   -0.072</td>
</tr>
<tr>
  <th>C(country_standard)[T.Philippines]</th>                      <td>    0.0144</td> <td>    0.001</td> <td>   16.232</td> <td> 0.000</td> <td>    0.013</td> <td>    0.016</td>
</tr>
<tr>
  <th>C(country_standard)[T.Poland]</th>                           <td>   -0.0024</td> <td>    0.002</td> <td>   -1.434</td> <td> 0.151</td> <td>   -0.006</td> <td>    0.001</td>
</tr>
<tr>
  <th>C(country_standard)[T.Portugal]</th>                         <td>   -0.0015</td> <td>    0.002</td> <td>   -0.830</td> <td> 0.406</td> <td>   -0.005</td> <td>    0.002</td>
</tr>
<tr>
  <th>C(country_standard)[T.Puerto Rico]</th>                      <td>   -0.0216</td> <td>    0.001</td> <td>  -14.476</td> <td> 0.000</td> <td>   -0.024</td> <td>   -0.019</td>
</tr>
<tr>
  <th>C(country_standard)[T.Qatar]</th>                            <td>    0.0301</td> <td> 4.58e-05</td> <td>  657.488</td> <td> 0.000</td> <td>    0.030</td> <td>    0.030</td>
</tr>
<tr>
  <th>C(country_standard)[T.Romania]</th>                          <td>   -0.0293</td> <td>    0.002</td> <td>  -18.434</td> <td> 0.000</td> <td>   -0.032</td> <td>   -0.026</td>
</tr>
<tr>
  <th>C(country_standard)[T.Russia]</th>                           <td>   -0.0125</td> <td>    0.000</td> <td>  -45.128</td> <td> 0.000</td> <td>   -0.013</td> <td>   -0.012</td>
</tr>
<tr>
  <th>C(country_standard)[T.Rwanda]</th>                           <td>    0.0224</td> <td> 1.44e-05</td> <td> 1562.570</td> <td> 0.000</td> <td>    0.022</td> <td>    0.022</td>
</tr>
<tr>
  <th>C(country_standard)[T.Réunion]</th>                          <td>   -0.0024</td> <td>    0.001</td> <td>   -2.062</td> <td> 0.039</td> <td>   -0.005</td> <td>   -0.000</td>
</tr>
<tr>
  <th>C(country_standard)[T.Saint Martin (French part)]</th>       <td>   -0.0021</td> <td>    0.001</td> <td>   -2.154</td> <td> 0.031</td> <td>   -0.004</td> <td>   -0.000</td>
</tr>
<tr>
  <th>C(country_standard)[T.Samoa]</th>                            <td>   -0.0036</td> <td>    0.002</td> <td>   -2.121</td> <td> 0.034</td> <td>   -0.007</td> <td>   -0.000</td>
</tr>
<tr>
  <th>C(country_standard)[T.San Marino]</th>                       <td>   -0.0030</td> <td>    0.002</td> <td>   -1.608</td> <td> 0.108</td> <td>   -0.007</td> <td>    0.001</td>
</tr>
<tr>
  <th>C(country_standard)[T.Saudi Arabia]</th>                     <td>   -0.0003</td> <td>    0.000</td> <td>   -0.938</td> <td> 0.348</td> <td>   -0.001</td> <td>    0.000</td>
</tr>
<tr>
  <th>C(country_standard)[T.Senegal]</th>                          <td>   -0.0056</td> <td>    0.002</td> <td>   -3.701</td> <td> 0.000</td> <td>   -0.009</td> <td>   -0.003</td>
</tr>
<tr>
  <th>C(country_standard)[T.Serbia]</th>                           <td>   -0.0262</td> <td>    0.001</td> <td>  -18.874</td> <td> 0.000</td> <td>   -0.029</td> <td>   -0.023</td>
</tr>
<tr>
  <th>C(country_standard)[T.Seychelles]</th>                       <td>   -0.0138</td> <td>    0.001</td> <td>  -14.238</td> <td> 0.000</td> <td>   -0.016</td> <td>   -0.012</td>
</tr>
<tr>
  <th>C(country_standard)[T.Sierra Leone]</th>                     <td>    0.0019</td> <td>    0.001</td> <td>    2.081</td> <td> 0.037</td> <td>    0.000</td> <td>    0.004</td>
</tr>
<tr>
  <th>C(country_standard)[T.Singapore]</th>                        <td>    0.0152</td> <td>    0.001</td> <td>   18.505</td> <td> 0.000</td> <td>    0.014</td> <td>    0.017</td>
</tr>
<tr>
  <th>C(country_standard)[T.Sint Maarten]</th>                     <td>   -0.0021</td> <td>    0.001</td> <td>   -2.158</td> <td> 0.031</td> <td>   -0.004</td> <td>   -0.000</td>
</tr>
<tr>
  <th>C(country_standard)[T.Slovakia]</th>                         <td>    0.0152</td> <td>    0.002</td> <td>    8.787</td> <td> 0.000</td> <td>    0.012</td> <td>    0.019</td>
</tr>
<tr>
  <th>C(country_standard)[T.Slovenia]</th>                         <td>   -0.0009</td> <td>    0.002</td> <td>   -0.500</td> <td> 0.617</td> <td>   -0.004</td> <td>    0.003</td>
</tr>
<tr>
  <th>C(country_standard)[T.Solomon Islands]</th>                  <td>   -0.0007</td> <td>    0.001</td> <td>   -0.554</td> <td> 0.579</td> <td>   -0.003</td> <td>    0.002</td>
</tr>
<tr>
  <th>C(country_standard)[T.Somalia]</th>                          <td> 4.632e-05</td> <td>    0.000</td> <td>    0.133</td> <td> 0.894</td> <td>   -0.001</td> <td>    0.001</td>
</tr>
<tr>
  <th>C(country_standard)[T.South Africa]</th>                     <td>   -0.0262</td> <td>    0.002</td> <td>  -15.377</td> <td> 0.000</td> <td>   -0.030</td> <td>   -0.023</td>
</tr>
<tr>
  <th>C(country_standard)[T.South Korea]</th>                      <td>    0.0089</td> <td>    0.001</td> <td>    8.105</td> <td> 0.000</td> <td>    0.007</td> <td>    0.011</td>
</tr>
<tr>
  <th>C(country_standard)[T.South Sudan]</th>                      <td>   -0.0113</td> <td>    0.000</td> <td>  -70.029</td> <td> 0.000</td> <td>   -0.012</td> <td>   -0.011</td>
</tr>
<tr>
  <th>C(country_standard)[T.Spain]</th>                            <td>   -0.0052</td> <td>    0.002</td> <td>   -3.329</td> <td> 0.001</td> <td>   -0.008</td> <td>   -0.002</td>
</tr>
<tr>
  <th>C(country_standard)[T.Sri Lanka]</th>                        <td>   -0.0038</td> <td>    0.001</td> <td>   -4.678</td> <td> 0.000</td> <td>   -0.005</td> <td>   -0.002</td>
</tr>
<tr>
  <th>C(country_standard)[T.St. Helena]</th>                       <td>   -0.0020</td> <td>    0.001</td> <td>   -2.157</td> <td> 0.031</td> <td>   -0.004</td> <td>   -0.000</td>
</tr>
<tr>
  <th>C(country_standard)[T.St. Kitts & Nevis]</th>                <td>   -0.0029</td> <td>    0.002</td> <td>   -1.607</td> <td> 0.108</td> <td>   -0.006</td> <td>    0.001</td>
</tr>
<tr>
  <th>C(country_standard)[T.St. Lucia]</th>                        <td>   -0.0214</td> <td>    0.002</td> <td>  -11.587</td> <td> 0.000</td> <td>   -0.025</td> <td>   -0.018</td>
</tr>
<tr>
  <th>C(country_standard)[T.St. Vincent & Grenadines]</th>         <td>   -0.0029</td> <td>    0.002</td> <td>   -1.608</td> <td> 0.108</td> <td>   -0.006</td> <td>    0.001</td>
</tr>
<tr>
  <th>C(country_standard)[T.Sudan]</th>                            <td>    0.0058</td> <td>    0.000</td> <td>   25.666</td> <td> 0.000</td> <td>    0.005</td> <td>    0.006</td>
</tr>
<tr>
  <th>C(country_standard)[T.Suriname]</th>                         <td>   -0.0026</td> <td>    0.002</td> <td>   -1.608</td> <td> 0.108</td> <td>   -0.006</td> <td>    0.001</td>
</tr>
<tr>
  <th>C(country_standard)[T.Sweden]</th>                           <td>    0.0244</td> <td>    0.002</td> <td>   13.108</td> <td> 0.000</td> <td>    0.021</td> <td>    0.028</td>
</tr>
<tr>
  <th>C(country_standard)[T.Switzerland]</th>                      <td>    0.0131</td> <td>    0.002</td> <td>    7.147</td> <td> 0.000</td> <td>    0.010</td> <td>    0.017</td>
</tr>
<tr>
  <th>C(country_standard)[T.Syria]</th>                            <td> 1.866e-05</td> <td>    0.000</td> <td>    0.057</td> <td> 0.955</td> <td>   -0.001</td> <td>    0.001</td>
</tr>
<tr>
  <th>C(country_standard)[T.São Tomé & Príncipe]</th>              <td>   -0.0282</td> <td>    0.002</td> <td>  -17.253</td> <td> 0.000</td> <td>   -0.031</td> <td>   -0.025</td>
</tr>
<tr>
  <th>C(country_standard)[T.Taiwan]</th>                           <td>   -0.0183</td> <td>    0.002</td> <td>  -10.715</td> <td> 0.000</td> <td>   -0.022</td> <td>   -0.015</td>
</tr>
<tr>
  <th>C(country_standard)[T.Tajikistan]</th>                       <td>    0.0345</td> <td>    0.000</td> <td>  197.469</td> <td> 0.000</td> <td>    0.034</td> <td>    0.035</td>
</tr>
<tr>
  <th>C(country_standard)[T.Tanzania]</th>                         <td>   -0.0048</td> <td>    0.001</td> <td>   -5.785</td> <td> 0.000</td> <td>   -0.006</td> <td>   -0.003</td>
</tr>
<tr>
  <th>C(country_standard)[T.Thailand]</th>                         <td>    0.0226</td> <td>    0.000</td> <td>  109.585</td> <td> 0.000</td> <td>    0.022</td> <td>    0.023</td>
</tr>
<tr>
  <th>C(country_standard)[T.Timor-Leste]</th>                      <td>    0.0113</td> <td>    0.001</td> <td>    9.601</td> <td> 0.000</td> <td>    0.009</td> <td>    0.014</td>
</tr>
<tr>
  <th>C(country_standard)[T.Togo]</th>                             <td>   -0.0082</td> <td>    0.001</td> <td>  -11.654</td> <td> 0.000</td> <td>   -0.010</td> <td>   -0.007</td>
</tr>
<tr>
  <th>C(country_standard)[T.Tonga]</th>                            <td>   -0.0033</td> <td>    0.002</td> <td>   -1.917</td> <td> 0.055</td> <td>   -0.007</td> <td> 7.34e-05</td>
</tr>
<tr>
  <th>C(country_standard)[T.Trinidad & Tobago]</th>                <td>   -0.0169</td> <td>    0.002</td> <td>  -10.228</td> <td> 0.000</td> <td>   -0.020</td> <td>   -0.014</td>
</tr>
<tr>
  <th>C(country_standard)[T.Tunisia]</th>                          <td>   -0.0260</td> <td>    0.001</td> <td>  -18.947</td> <td> 0.000</td> <td>   -0.029</td> <td>   -0.023</td>
</tr>
<tr>
  <th>C(country_standard)[T.Turkey]</th>                           <td>    0.0247</td> <td>    0.000</td> <td>   58.832</td> <td> 0.000</td> <td>    0.024</td> <td>    0.026</td>
</tr>
<tr>
  <th>C(country_standard)[T.Turkmenistan]</th>                     <td>    0.0006</td> <td>    0.000</td> <td>    1.514</td> <td> 0.130</td> <td>   -0.000</td> <td>    0.001</td>
</tr>
<tr>
  <th>C(country_standard)[T.Turks & Caicos Islands]</th>           <td>   -0.0020</td> <td>    0.001</td> <td>   -2.155</td> <td> 0.031</td> <td>   -0.004</td> <td>   -0.000</td>
</tr>
<tr>
  <th>C(country_standard)[T.Tuvalu]</th>                           <td>   -0.0035</td> <td>    0.002</td> <td>   -1.893</td> <td> 0.058</td> <td>   -0.007</td> <td>    0.000</td>
</tr>
<tr>
  <th>C(country_standard)[T.U.S. Virgin Islands]</th>              <td>   -0.0023</td> <td>    0.001</td> <td>   -2.094</td> <td> 0.036</td> <td>   -0.004</td> <td>   -0.000</td>
</tr>
<tr>
  <th>C(country_standard)[T.Uganda]</th>                           <td>   -0.0073</td> <td>    0.000</td> <td>  -19.620</td> <td> 0.000</td> <td>   -0.008</td> <td>   -0.007</td>
</tr>
<tr>
  <th>C(country_standard)[T.Ukraine]</th>                          <td>   -0.0237</td> <td>    0.001</td> <td>  -28.884</td> <td> 0.000</td> <td>   -0.025</td> <td>   -0.022</td>
</tr>
<tr>
  <th>C(country_standard)[T.United Arab Emirates]</th>             <td>    0.0075</td> <td>  8.3e-05</td> <td>   90.448</td> <td> 0.000</td> <td>    0.007</td> <td>    0.008</td>
</tr>
<tr>
  <th>C(country_standard)[T.United Kingdom]</th>                   <td>   -0.0007</td> <td>    0.001</td> <td>   -0.526</td> <td> 0.599</td> <td>   -0.004</td> <td>    0.002</td>
</tr>
<tr>
  <th>C(country_standard)[T.United States]</th>                    <td>   -0.0344</td> <td>    0.002</td> <td>  -20.428</td> <td> 0.000</td> <td>   -0.038</td> <td>   -0.031</td>
</tr>
<tr>
  <th>C(country_standard)[T.Uruguay]</th>                          <td>   -0.0009</td> <td>    0.002</td> <td>   -0.511</td> <td> 0.609</td> <td>   -0.005</td> <td>    0.003</td>
</tr>
<tr>
  <th>C(country_standard)[T.Uzbekistan]</th>                       <td>    0.0339</td> <td>    0.000</td> <td>  142.425</td> <td> 0.000</td> <td>    0.033</td> <td>    0.034</td>
</tr>
<tr>
  <th>C(country_standard)[T.Vanuatu]</th>                          <td>   -0.0020</td> <td>    0.002</td> <td>   -1.207</td> <td> 0.227</td> <td>   -0.005</td> <td>    0.001</td>
</tr>
<tr>
  <th>C(country_standard)[T.Vatican City]</th>                     <td>   -0.0020</td> <td>    0.001</td> <td>   -2.129</td> <td> 0.033</td> <td>   -0.004</td> <td>   -0.000</td>
</tr>
<tr>
  <th>C(country_standard)[T.Venezuela]</th>                        <td>   -0.0133</td> <td>    0.000</td> <td>  -59.985</td> <td> 0.000</td> <td>   -0.014</td> <td>   -0.013</td>
</tr>
<tr>
  <th>C(country_standard)[T.Vietnam]</th>                          <td>    0.0653</td> <td>    0.000</td> <td>  453.839</td> <td> 0.000</td> <td>    0.065</td> <td>    0.066</td>
</tr>
<tr>
  <th>C(country_standard)[T.Western Sahara]</th>                   <td>    0.0024</td> <td>    0.002</td> <td>    1.329</td> <td> 0.184</td> <td>   -0.001</td> <td>    0.006</td>
</tr>
<tr>
  <th>C(country_standard)[T.Yemen]</th>                            <td>   -0.0128</td> <td>    0.000</td> <td>  -85.465</td> <td> 0.000</td> <td>   -0.013</td> <td>   -0.012</td>
</tr>
<tr>
  <th>C(country_standard)[T.Zambia]</th>                           <td>   -0.0274</td> <td>    0.001</td> <td>  -30.481</td> <td> 0.000</td> <td>   -0.029</td> <td>   -0.026</td>
</tr>
<tr>
  <th>C(country_standard)[T.Zimbabwe]</th>                         <td>   -0.0069</td> <td>    0.000</td> <td>  -26.254</td> <td> 0.000</td> <td>   -0.007</td> <td>   -0.006</td>
</tr>
<tr>
  <th>transparency_index</th>                                      <td>    0.0077</td> <td>    0.005</td> <td>    1.610</td> <td> 0.107</td> <td>   -0.002</td> <td>    0.017</td>
</tr>
<tr>
  <th>gdp</th>                                                     <td> 8.212e-16</td> <td> 1.64e-16</td> <td>    4.997</td> <td> 0.000</td> <td> 4.99e-16</td> <td> 1.14e-15</td>
</tr>
<tr>
  <th>gini</th>                                                    <td>    0.0014</td> <td>  2.9e-06</td> <td>  484.916</td> <td> 0.000</td> <td>    0.001</td> <td>    0.001</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>2267.441</td> <th>  Durbin-Watson:     </th>  <td>   2.197</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>449251.183</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 1.926</td>  <th>  Prob(JB):          </th>  <td>    0.00</td> 
</tr>
<tr>
  <th>Kurtosis:</th>       <td>57.568</td>  <th>  Cond. No.          </th>  <td>9.45e+26</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors are robust to cluster correlation (cluster)<br/>[2] The smallest eigenvalue is 1.39e-26. This might indicate that there are<br/>strong multicollinearity problems or that the design matrix is singular.




```python
reg_effectiveness = smf.ols('effectiveness_index ~ transparency_index + gdp + gini + C(country_standard)', ndi_df).fit(cov_type='cluster', cov_kwds={'groups': ndi_df['country_standard']})
```


```python
reg_effectiveness.summary()
```

    /Users/katiacordoba/opt/anaconda3/lib/python3.7/site-packages/statsmodels/base/model.py:1832: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 231, but rank is 229
      'rank is %d' % (J, J_), ValueWarning)





<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>    <td>effectiveness_index</td> <th>  R-squared:         </th> <td>   0.861</td> 
</tr>
<tr>
  <th>Model:</th>                    <td>OLS</td>         <th>  Adj. R-squared:    </th> <td>   0.861</td> 
</tr>
<tr>
  <th>Method:</th>              <td>Least Squares</td>    <th>  F-statistic:       </th> <td>3.614e+07</td>
</tr>
<tr>
  <th>Date:</th>              <td>Fri, 05 Mar 2021</td>   <th>  Prob (F-statistic):</th>  <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                  <td>23:12:23</td>       <th>  Log-Likelihood:    </th> <td>  4259.2</td> 
</tr>
<tr>
  <th>No. Observations:</th>       <td>  3603</td>        <th>  AIC:               </th> <td>  -8508.</td> 
</tr>
<tr>
  <th>Df Residuals:</th>           <td>  3598</td>        <th>  BIC:               </th> <td>  -8477.</td> 
</tr>
<tr>
  <th>Df Model:</th>               <td>     4</td>        <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>        <td>cluster</td>       <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
                             <td></td>                                <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>                                               <td>    0.3032</td> <td>    0.006</td> <td>   47.118</td> <td> 0.000</td> <td>    0.291</td> <td>    0.316</td>
</tr>
<tr>
  <th>C(country_standard)[T.Albania]</th>                          <td>    0.2224</td> <td>    0.017</td> <td>   13.137</td> <td> 0.000</td> <td>    0.189</td> <td>    0.256</td>
</tr>
<tr>
  <th>C(country_standard)[T.Algeria]</th>                          <td>    0.0925</td> <td>    0.001</td> <td>   63.137</td> <td> 0.000</td> <td>    0.090</td> <td>    0.095</td>
</tr>
<tr>
  <th>C(country_standard)[T.American Samoa]</th>                   <td>    0.3677</td> <td>    0.008</td> <td>   47.292</td> <td> 0.000</td> <td>    0.352</td> <td>    0.383</td>
</tr>
<tr>
  <th>C(country_standard)[T.Andorra]</th>                          <td>    0.2693</td> <td>    0.006</td> <td>   42.053</td> <td> 0.000</td> <td>    0.257</td> <td>    0.282</td>
</tr>
<tr>
  <th>C(country_standard)[T.Angola]</th>                           <td>    0.1733</td> <td>    0.002</td> <td>  114.854</td> <td> 0.000</td> <td>    0.170</td> <td>    0.176</td>
</tr>
<tr>
  <th>C(country_standard)[T.Anguilla]</th>                         <td>    0.5094</td> <td>    0.004</td> <td>  127.360</td> <td> 0.000</td> <td>    0.502</td> <td>    0.517</td>
</tr>
<tr>
  <th>C(country_standard)[T.Antigua & Barbuda]</th>                <td>    0.3274</td> <td>    0.006</td> <td>   55.992</td> <td> 0.000</td> <td>    0.316</td> <td>    0.339</td>
</tr>
<tr>
  <th>C(country_standard)[T.Argentina]</th>                        <td>    0.2663</td> <td>    0.005</td> <td>   49.091</td> <td> 0.000</td> <td>    0.256</td> <td>    0.277</td>
</tr>
<tr>
  <th>C(country_standard)[T.Armenia]</th>                          <td>    0.1933</td> <td>    0.003</td> <td>   65.414</td> <td> 0.000</td> <td>    0.187</td> <td>    0.199</td>
</tr>
<tr>
  <th>C(country_standard)[T.Aruba]</th>                            <td>    0.4826</td> <td>    0.007</td> <td>   71.728</td> <td> 0.000</td> <td>    0.469</td> <td>    0.496</td>
</tr>
<tr>
  <th>C(country_standard)[T.Australia]</th>                        <td>    0.4996</td> <td>    0.006</td> <td>   87.483</td> <td> 0.000</td> <td>    0.488</td> <td>    0.511</td>
</tr>
<tr>
  <th>C(country_standard)[T.Austria]</th>                          <td>    0.4864</td> <td>    0.006</td> <td>   78.168</td> <td> 0.000</td> <td>    0.474</td> <td>    0.499</td>
</tr>
<tr>
  <th>C(country_standard)[T.Azerbaijan]</th>                       <td>    0.1267</td> <td>    0.001</td> <td>   86.682</td> <td> 0.000</td> <td>    0.124</td> <td>    0.130</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bahamas]</th>                          <td>    0.4272</td> <td>    0.006</td> <td>   69.268</td> <td> 0.000</td> <td>    0.415</td> <td>    0.439</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bahrain]</th>                          <td>    0.2717</td> <td>    0.001</td> <td>  187.984</td> <td> 0.000</td> <td>    0.269</td> <td>    0.274</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bangladesh]</th>                       <td>    0.1993</td> <td>    0.003</td> <td>   73.827</td> <td> 0.000</td> <td>    0.194</td> <td>    0.205</td>
</tr>
<tr>
  <th>C(country_standard)[T.Barbados]</th>                         <td>    0.4864</td> <td>    0.006</td> <td>   76.270</td> <td> 0.000</td> <td>    0.474</td> <td>    0.499</td>
</tr>
<tr>
  <th>C(country_standard)[T.Belarus]</th>                          <td>    0.1039</td> <td>    0.002</td> <td>   64.792</td> <td> 0.000</td> <td>    0.101</td> <td>    0.107</td>
</tr>
<tr>
  <th>C(country_standard)[T.Belgium]</th>                          <td>    0.4746</td> <td>    0.006</td> <td>   77.178</td> <td> 0.000</td> <td>    0.463</td> <td>    0.487</td>
</tr>
<tr>
  <th>C(country_standard)[T.Belize]</th>                           <td>    0.1793</td> <td>    0.006</td> <td>   30.519</td> <td> 0.000</td> <td>    0.168</td> <td>    0.191</td>
</tr>
<tr>
  <th>C(country_standard)[T.Benin]</th>                            <td>    0.3088</td> <td>    0.005</td> <td>   57.667</td> <td> 0.000</td> <td>    0.298</td> <td>    0.319</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bermuda]</th>                          <td>    0.4750</td> <td>    0.004</td> <td>  118.569</td> <td> 0.000</td> <td>    0.467</td> <td>    0.483</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bhutan]</th>                           <td>    0.3487</td> <td>    0.003</td> <td>  102.982</td> <td> 0.000</td> <td>    0.342</td> <td>    0.355</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bolivia]</th>                          <td>    0.2615</td> <td>    0.003</td> <td>   77.621</td> <td> 0.000</td> <td>    0.255</td> <td>    0.268</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bosnia & Herzegovina]</th>             <td>    0.1312</td> <td>    0.003</td> <td>   40.722</td> <td> 0.000</td> <td>    0.125</td> <td>    0.138</td>
</tr>
<tr>
  <th>C(country_standard)[T.Botswana]</th>                         <td>    0.4749</td> <td>    0.006</td> <td>   81.436</td> <td> 0.000</td> <td>    0.463</td> <td>    0.486</td>
</tr>
<tr>
  <th>C(country_standard)[T.Brazil]</th>                           <td>    0.3065</td> <td>    0.005</td> <td>   65.343</td> <td> 0.000</td> <td>    0.297</td> <td>    0.316</td>
</tr>
<tr>
  <th>C(country_standard)[T.British Virgin Islands]</th>           <td>    0.2731</td> <td>    0.008</td> <td>   32.461</td> <td> 0.000</td> <td>    0.257</td> <td>    0.290</td>
</tr>
<tr>
  <th>C(country_standard)[T.Brunei]</th>                           <td>    0.4987</td> <td>    0.001</td> <td>  344.314</td> <td> 0.000</td> <td>    0.496</td> <td>    0.502</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bulgaria]</th>                         <td>    0.3295</td> <td>    0.006</td> <td>   58.957</td> <td> 0.000</td> <td>    0.319</td> <td>    0.340</td>
</tr>
<tr>
  <th>C(country_standard)[T.Burkina Faso]</th>                     <td>    0.1429</td> <td>    0.003</td> <td>   44.366</td> <td> 0.000</td> <td>    0.137</td> <td>    0.149</td>
</tr>
<tr>
  <th>C(country_standard)[T.Burundi]</th>                          <td>    0.0739</td> <td>    0.001</td> <td>   51.967</td> <td> 0.000</td> <td>    0.071</td> <td>    0.077</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cambodia]</th>                         <td>    0.2798</td> <td>    0.001</td> <td>  199.082</td> <td> 0.000</td> <td>    0.277</td> <td>    0.283</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cameroon]</th>                         <td>    0.2080</td> <td>    0.001</td> <td>  148.243</td> <td> 0.000</td> <td>    0.205</td> <td>    0.211</td>
</tr>
<tr>
  <th>C(country_standard)[T.Canada]</th>                           <td>    0.5122</td> <td>    0.006</td> <td>   90.536</td> <td> 0.000</td> <td>    0.501</td> <td>    0.523</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cape Verde]</th>                       <td>    0.4250</td> <td>    0.006</td> <td>   68.194</td> <td> 0.000</td> <td>    0.413</td> <td>    0.437</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cayman Islands]</th>                   <td>    0.5074</td> <td>    0.005</td> <td>  111.917</td> <td> 0.000</td> <td>    0.498</td> <td>    0.516</td>
</tr>
<tr>
  <th>C(country_standard)[T.Central African Republic]</th>         <td>    0.0442</td> <td>    0.001</td> <td>   31.686</td> <td> 0.000</td> <td>    0.041</td> <td>    0.047</td>
</tr>
<tr>
  <th>C(country_standard)[T.Chad]</th>                             <td>   -0.0222</td> <td>    0.001</td> <td>  -15.772</td> <td> 0.000</td> <td>   -0.025</td> <td>   -0.019</td>
</tr>
<tr>
  <th>C(country_standard)[T.Channel Islands]</th>                  <td>    0.2638</td> <td>    0.004</td> <td>   65.670</td> <td> 0.000</td> <td>    0.256</td> <td>    0.272</td>
</tr>
<tr>
  <th>C(country_standard)[T.Chile]</th>                            <td>    0.5473</td> <td>    0.006</td> <td>   89.608</td> <td> 0.000</td> <td>    0.535</td> <td>    0.559</td>
</tr>
<tr>
  <th>C(country_standard)[T.China]</th>                            <td>   -0.4700</td> <td>    0.009</td> <td>  -54.167</td> <td> 0.000</td> <td>   -0.487</td> <td>   -0.453</td>
</tr>
<tr>
  <th>C(country_standard)[T.Colombia]</th>                         <td>    0.3878</td> <td>    0.003</td> <td>  117.117</td> <td> 0.000</td> <td>    0.381</td> <td>    0.394</td>
</tr>
<tr>
  <th>C(country_standard)[T.Comoros]</th>                          <td>    0.0509</td> <td>    0.003</td> <td>   16.602</td> <td> 0.000</td> <td>    0.045</td> <td>    0.057</td>
</tr>
<tr>
  <th>C(country_standard)[T.Congo - Brazzaville]</th>              <td>    0.1494</td> <td>    0.001</td> <td>  106.795</td> <td> 0.000</td> <td>    0.147</td> <td>    0.152</td>
</tr>
<tr>
  <th>C(country_standard)[T.Congo - Kinshasa]</th>                 <td>   -0.0578</td> <td>    0.001</td> <td>  -40.605</td> <td> 0.000</td> <td>   -0.061</td> <td>   -0.055</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cook Islands]</th>                     <td>    0.2209</td> <td>    0.009</td> <td>   25.165</td> <td> 0.000</td> <td>    0.204</td> <td>    0.238</td>
</tr>
<tr>
  <th>C(country_standard)[T.Costa Rica]</th>                       <td>    0.4607</td> <td>    0.006</td> <td>   72.364</td> <td> 0.000</td> <td>    0.448</td> <td>    0.473</td>
</tr>
<tr>
  <th>C(country_standard)[T.Croatia]</th>                          <td>    0.3360</td> <td>    0.006</td> <td>   59.378</td> <td> 0.000</td> <td>    0.325</td> <td>    0.347</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cuba]</th>                             <td>    0.2670</td> <td>    0.002</td> <td>  176.378</td> <td> 0.000</td> <td>    0.264</td> <td>    0.270</td>
</tr>
<tr>
  <th>C(country_standard)[T.Curaçao]</th>                          <td>    0.2731</td> <td>    0.008</td> <td>   32.498</td> <td> 0.000</td> <td>    0.257</td> <td>    0.290</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cyprus]</th>                           <td>    0.4169</td> <td>    0.006</td> <td>   67.451</td> <td> 0.000</td> <td>    0.405</td> <td>    0.429</td>
</tr>
<tr>
  <th>C(country_standard)[T.Czechia]</th>                          <td>    0.3790</td> <td>    0.006</td> <td>   63.831</td> <td> 0.000</td> <td>    0.367</td> <td>    0.391</td>
</tr>
<tr>
  <th>C(country_standard)[T.Côte d’Ivoire]</th>                    <td>    0.1132</td> <td>    0.003</td> <td>   38.274</td> <td> 0.000</td> <td>    0.107</td> <td>    0.119</td>
</tr>
<tr>
  <th>C(country_standard)[T.Denmark]</th>                          <td>    0.5930</td> <td>    0.006</td> <td>   91.763</td> <td> 0.000</td> <td>    0.580</td> <td>    0.606</td>
</tr>
<tr>
  <th>C(country_standard)[T.Djibouti]</th>                         <td>    0.1021</td> <td>    0.001</td> <td>   71.947</td> <td> 0.000</td> <td>    0.099</td> <td>    0.105</td>
</tr>
<tr>
  <th>C(country_standard)[T.Dominica]</th>                         <td>    0.3241</td> <td>    0.006</td> <td>   52.161</td> <td> 0.000</td> <td>    0.312</td> <td>    0.336</td>
</tr>
<tr>
  <th>C(country_standard)[T.Dominican Republic]</th>               <td>    0.2657</td> <td>    0.004</td> <td>   63.575</td> <td> 0.000</td> <td>    0.258</td> <td>    0.274</td>
</tr>
<tr>
  <th>C(country_standard)[T.Ecuador]</th>                          <td>    0.2472</td> <td>    0.003</td> <td>   77.496</td> <td> 0.000</td> <td>    0.241</td> <td>    0.253</td>
</tr>
<tr>
  <th>C(country_standard)[T.Egypt]</th>                            <td>    0.1439</td> <td>    0.001</td> <td>  102.189</td> <td> 0.000</td> <td>    0.141</td> <td>    0.147</td>
</tr>
<tr>
  <th>C(country_standard)[T.El Salvador]</th>                      <td>    0.2893</td> <td>    0.005</td> <td>   56.741</td> <td> 0.000</td> <td>    0.279</td> <td>    0.299</td>
</tr>
<tr>
  <th>C(country_standard)[T.Equatorial Guinea]</th>                <td>    0.0124</td> <td>    0.002</td> <td>    7.499</td> <td> 0.000</td> <td>    0.009</td> <td>    0.016</td>
</tr>
<tr>
  <th>C(country_standard)[T.Eritrea]</th>                          <td>    0.0670</td> <td>    0.002</td> <td>   37.624</td> <td> 0.000</td> <td>    0.063</td> <td>    0.070</td>
</tr>
<tr>
  <th>C(country_standard)[T.Estonia]</th>                          <td>    0.4486</td> <td>    0.006</td> <td>   70.774</td> <td> 0.000</td> <td>    0.436</td> <td>    0.461</td>
</tr>
<tr>
  <th>C(country_standard)[T.Eswatini]</th>                         <td>    0.2297</td> <td>    0.001</td> <td>  178.446</td> <td> 0.000</td> <td>    0.227</td> <td>    0.232</td>
</tr>
<tr>
  <th>C(country_standard)[T.Ethiopia]</th>                         <td>    0.1939</td> <td>    0.001</td> <td>  134.180</td> <td> 0.000</td> <td>    0.191</td> <td>    0.197</td>
</tr>
<tr>
  <th>C(country_standard)[T.Faroe Islands]</th>                    <td>    0.2650</td> <td>    0.004</td> <td>   75.588</td> <td> 0.000</td> <td>    0.258</td> <td>    0.272</td>
</tr>
<tr>
  <th>C(country_standard)[T.Fiji]</th>                             <td>    0.2190</td> <td>    0.003</td> <td>   67.354</td> <td> 0.000</td> <td>    0.213</td> <td>    0.225</td>
</tr>
<tr>
  <th>C(country_standard)[T.Finland]</th>                          <td>    0.5898</td> <td>    0.006</td> <td>   90.844</td> <td> 0.000</td> <td>    0.577</td> <td>    0.603</td>
</tr>
<tr>
  <th>C(country_standard)[T.France]</th>                           <td>    0.3254</td> <td>    0.005</td> <td>   65.815</td> <td> 0.000</td> <td>    0.316</td> <td>    0.335</td>
</tr>
<tr>
  <th>C(country_standard)[T.French Guiana]</th>                    <td>    0.4788</td> <td>    0.007</td> <td>   68.780</td> <td> 0.000</td> <td>    0.465</td> <td>    0.492</td>
</tr>
<tr>
  <th>C(country_standard)[T.French Polynesia]</th>                 <td>    0.2731</td> <td>    0.008</td> <td>   32.508</td> <td> 0.000</td> <td>    0.257</td> <td>    0.290</td>
</tr>
<tr>
  <th>C(country_standard)[T.Gabon]</th>                            <td>    0.1201</td> <td>    0.001</td> <td>   83.892</td> <td> 0.000</td> <td>    0.117</td> <td>    0.123</td>
</tr>
<tr>
  <th>C(country_standard)[T.Gambia]</th>                           <td>    0.2015</td> <td>    0.002</td> <td>  119.756</td> <td> 0.000</td> <td>    0.198</td> <td>    0.205</td>
</tr>
<tr>
  <th>C(country_standard)[T.Georgia]</th>                          <td>    0.2298</td> <td>    0.004</td> <td>   63.510</td> <td> 0.000</td> <td>    0.223</td> <td>    0.237</td>
</tr>
<tr>
  <th>C(country_standard)[T.Germany]</th>                          <td>    0.3239</td> <td>    0.005</td> <td>   69.823</td> <td> 0.000</td> <td>    0.315</td> <td>    0.333</td>
</tr>
<tr>
  <th>C(country_standard)[T.Ghana]</th>                            <td>    0.2905</td> <td>    0.006</td> <td>   50.082</td> <td> 0.000</td> <td>    0.279</td> <td>    0.302</td>
</tr>
<tr>
  <th>C(country_standard)[T.Gibraltar]</th>                        <td>    0.2650</td> <td>    0.004</td> <td>   75.568</td> <td> 0.000</td> <td>    0.258</td> <td>    0.272</td>
</tr>
<tr>
  <th>C(country_standard)[T.Greece]</th>                           <td>    0.3306</td> <td>    0.006</td> <td>   56.601</td> <td> 0.000</td> <td>    0.319</td> <td>    0.342</td>
</tr>
<tr>
  <th>C(country_standard)[T.Greenland]</th>                        <td>    0.3826</td> <td>    0.006</td> <td>   63.413</td> <td> 0.000</td> <td>    0.371</td> <td>    0.394</td>
</tr>
<tr>
  <th>C(country_standard)[T.Grenada]</th>                          <td>    0.2813</td> <td>    0.006</td> <td>   46.341</td> <td> 0.000</td> <td>    0.269</td> <td>    0.293</td>
</tr>
<tr>
  <th>C(country_standard)[T.Guam]</th>                             <td>    0.3064</td> <td>    0.008</td> <td>   38.669</td> <td> 0.000</td> <td>    0.291</td> <td>    0.322</td>
</tr>
<tr>
  <th>C(country_standard)[T.Guatemala]</th>                        <td>    0.3215</td> <td>    0.003</td> <td>  104.123</td> <td> 0.000</td> <td>    0.315</td> <td>    0.328</td>
</tr>
<tr>
  <th>C(country_standard)[T.Guernsey]</th>                         <td>    0.2653</td> <td>    0.003</td> <td>   76.113</td> <td> 0.000</td> <td>    0.258</td> <td>    0.272</td>
</tr>
<tr>
  <th>C(country_standard)[T.Guinea]</th>                           <td>   -0.0252</td> <td>    0.003</td> <td>   -9.562</td> <td> 0.000</td> <td>   -0.030</td> <td>   -0.020</td>
</tr>
<tr>
  <th>C(country_standard)[T.Guinea-Bissau]</th>                    <td>    0.0064</td> <td>    0.002</td> <td>    2.769</td> <td> 0.006</td> <td>    0.002</td> <td>    0.011</td>
</tr>
<tr>
  <th>C(country_standard)[T.Guyana]</th>                           <td>    0.2729</td> <td>    0.005</td> <td>   50.462</td> <td> 0.000</td> <td>    0.262</td> <td>    0.283</td>
</tr>
<tr>
  <th>C(country_standard)[T.Haiti]</th>                            <td>    0.0418</td> <td>    0.003</td> <td>   15.506</td> <td> 0.000</td> <td>    0.037</td> <td>    0.047</td>
</tr>
<tr>
  <th>C(country_standard)[T.Honduras]</th>                         <td>    0.3175</td> <td>    0.003</td> <td>  103.884</td> <td> 0.000</td> <td>    0.312</td> <td>    0.324</td>
</tr>
<tr>
  <th>C(country_standard)[T.Hong Kong SAR China]</th>              <td>    0.4623</td> <td>    0.004</td> <td>  128.548</td> <td> 0.000</td> <td>    0.455</td> <td>    0.469</td>
</tr>
<tr>
  <th>C(country_standard)[T.Hungary]</th>                          <td>    0.3445</td> <td>    0.005</td> <td>   67.974</td> <td> 0.000</td> <td>    0.335</td> <td>    0.354</td>
</tr>
<tr>
  <th>C(country_standard)[T.Iceland]</th>                          <td>    0.5626</td> <td>    0.006</td> <td>   86.577</td> <td> 0.000</td> <td>    0.550</td> <td>    0.575</td>
</tr>
<tr>
  <th>C(country_standard)[T.India]</th>                            <td>    0.2233</td> <td>    0.003</td> <td>   70.972</td> <td> 0.000</td> <td>    0.217</td> <td>    0.229</td>
</tr>
<tr>
  <th>C(country_standard)[T.Indonesia]</th>                        <td>    0.3110</td> <td>    0.003</td> <td>  101.512</td> <td> 0.000</td> <td>    0.305</td> <td>    0.317</td>
</tr>
<tr>
  <th>C(country_standard)[T.Iran]</th>                             <td>    0.1750</td> <td>    0.001</td> <td>  122.551</td> <td> 0.000</td> <td>    0.172</td> <td>    0.178</td>
</tr>
<tr>
  <th>C(country_standard)[T.Iraq]</th>                             <td>   -0.0574</td> <td>    0.001</td> <td>  -39.432</td> <td> 0.000</td> <td>   -0.060</td> <td>   -0.055</td>
</tr>
<tr>
  <th>C(country_standard)[T.Ireland]</th>                          <td>    0.4923</td> <td>    0.006</td> <td>   79.026</td> <td> 0.000</td> <td>    0.480</td> <td>    0.505</td>
</tr>
<tr>
  <th>C(country_standard)[T.Isle of Man]</th>                      <td>    0.2650</td> <td>    0.004</td> <td>   75.587</td> <td> 0.000</td> <td>    0.258</td> <td>    0.272</td>
</tr>
<tr>
  <th>C(country_standard)[T.Israel]</th>                           <td>    0.4771</td> <td>    0.006</td> <td>   84.128</td> <td> 0.000</td> <td>    0.466</td> <td>    0.488</td>
</tr>
<tr>
  <th>C(country_standard)[T.Italy]</th>                            <td>    0.2499</td> <td>    0.006</td> <td>   45.344</td> <td> 0.000</td> <td>    0.239</td> <td>    0.261</td>
</tr>
<tr>
  <th>C(country_standard)[T.Jamaica]</th>                          <td>    0.3782</td> <td>    0.006</td> <td>   65.193</td> <td> 0.000</td> <td>    0.367</td> <td>    0.390</td>
</tr>
<tr>
  <th>C(country_standard)[T.Japan]</th>                            <td>    0.2521</td> <td>    0.004</td> <td>   63.968</td> <td> 0.000</td> <td>    0.244</td> <td>    0.260</td>
</tr>
<tr>
  <th>C(country_standard)[T.Jersey]</th>                           <td>    0.4086</td> <td>    0.006</td> <td>   69.928</td> <td> 0.000</td> <td>    0.397</td> <td>    0.420</td>
</tr>
<tr>
  <th>C(country_standard)[T.Jordan]</th>                           <td>    0.2529</td> <td>    0.002</td> <td>  132.225</td> <td> 0.000</td> <td>    0.249</td> <td>    0.257</td>
</tr>
<tr>
  <th>C(country_standard)[T.Kazakhstan]</th>                       <td>    0.1128</td> <td>    0.001</td> <td>   75.347</td> <td> 0.000</td> <td>    0.110</td> <td>    0.116</td>
</tr>
<tr>
  <th>C(country_standard)[T.Kenya]</th>                            <td>    0.2406</td> <td>    0.003</td> <td>   77.227</td> <td> 0.000</td> <td>    0.234</td> <td>    0.247</td>
</tr>
<tr>
  <th>C(country_standard)[T.Kiribati]</th>                         <td>    0.1226</td> <td>    0.007</td> <td>   17.534</td> <td> 0.000</td> <td>    0.109</td> <td>    0.136</td>
</tr>
<tr>
  <th>C(country_standard)[T.Kosovo]</th>                           <td>    0.2638</td> <td>    0.004</td> <td>   65.616</td> <td> 0.000</td> <td>    0.256</td> <td>    0.272</td>
</tr>
<tr>
  <th>C(country_standard)[T.Kuwait]</th>                           <td>    0.2535</td> <td>    0.003</td> <td>   93.172</td> <td> 0.000</td> <td>    0.248</td> <td>    0.259</td>
</tr>
<tr>
  <th>C(country_standard)[T.Kyrgyzstan]</th>                       <td>    0.0651</td> <td>    0.003</td> <td>   22.337</td> <td> 0.000</td> <td>    0.059</td> <td>    0.071</td>
</tr>
<tr>
  <th>C(country_standard)[T.Laos]</th>                             <td>    0.1630</td> <td>    0.002</td> <td>  103.908</td> <td> 0.000</td> <td>    0.160</td> <td>    0.166</td>
</tr>
<tr>
  <th>C(country_standard)[T.Latvia]</th>                           <td>    0.3655</td> <td>    0.006</td> <td>   61.276</td> <td> 0.000</td> <td>    0.354</td> <td>    0.377</td>
</tr>
<tr>
  <th>C(country_standard)[T.Lebanon]</th>                          <td>    0.1129</td> <td>    0.003</td> <td>   39.102</td> <td> 0.000</td> <td>    0.107</td> <td>    0.119</td>
</tr>
<tr>
  <th>C(country_standard)[T.Lesotho]</th>                          <td>    0.2911</td> <td>    0.004</td> <td>   69.983</td> <td> 0.000</td> <td>    0.283</td> <td>    0.299</td>
</tr>
<tr>
  <th>C(country_standard)[T.Liberia]</th>                          <td>    0.0049</td> <td>    0.003</td> <td>    1.522</td> <td> 0.128</td> <td>   -0.001</td> <td>    0.011</td>
</tr>
<tr>
  <th>C(country_standard)[T.Libya]</th>                            <td>    0.0360</td> <td>    0.001</td> <td>   25.273</td> <td> 0.000</td> <td>    0.033</td> <td>    0.039</td>
</tr>
<tr>
  <th>C(country_standard)[T.Liechtenstein]</th>                    <td>    0.5866</td> <td>    0.007</td> <td>   90.069</td> <td> 0.000</td> <td>    0.574</td> <td>    0.599</td>
</tr>
<tr>
  <th>C(country_standard)[T.Lithuania]</th>                        <td>    0.3840</td> <td>    0.006</td> <td>   62.670</td> <td> 0.000</td> <td>    0.372</td> <td>    0.396</td>
</tr>
<tr>
  <th>C(country_standard)[T.Luxembourg]</th>                       <td>    0.5884</td> <td>    0.007</td> <td>   88.654</td> <td> 0.000</td> <td>    0.575</td> <td>    0.601</td>
</tr>
<tr>
  <th>C(country_standard)[T.Macao SAR China]</th>                  <td>    0.5286</td> <td>    0.003</td> <td>  157.407</td> <td> 0.000</td> <td>    0.522</td> <td>    0.535</td>
</tr>
<tr>
  <th>C(country_standard)[T.Madagascar]</th>                       <td>    0.1548</td> <td>    0.003</td> <td>   51.940</td> <td> 0.000</td> <td>    0.149</td> <td>    0.161</td>
</tr>
<tr>
  <th>C(country_standard)[T.Malawi]</th>                           <td>    0.2577</td> <td>    0.003</td> <td>   77.830</td> <td> 0.000</td> <td>    0.251</td> <td>    0.264</td>
</tr>
<tr>
  <th>C(country_standard)[T.Malaysia]</th>                         <td>    0.4776</td> <td>    0.003</td> <td>  163.400</td> <td> 0.000</td> <td>    0.472</td> <td>    0.483</td>
</tr>
<tr>
  <th>C(country_standard)[T.Maldives]</th>                         <td>    0.2224</td> <td>    0.003</td> <td>   77.264</td> <td> 0.000</td> <td>    0.217</td> <td>    0.228</td>
</tr>
<tr>
  <th>C(country_standard)[T.Mali]</th>                             <td>    0.1798</td> <td>    0.003</td> <td>   61.524</td> <td> 0.000</td> <td>    0.174</td> <td>    0.186</td>
</tr>
<tr>
  <th>C(country_standard)[T.Malta]</th>                            <td>    0.4711</td> <td>    0.006</td> <td>   74.645</td> <td> 0.000</td> <td>    0.459</td> <td>    0.483</td>
</tr>
<tr>
  <th>C(country_standard)[T.Marshall Islands]</th>                 <td>    0.0043</td> <td>    0.006</td> <td>    0.661</td> <td> 0.508</td> <td>   -0.008</td> <td>    0.017</td>
</tr>
<tr>
  <th>C(country_standard)[T.Martinique]</th>                       <td>    0.4411</td> <td>    0.008</td> <td>   54.474</td> <td> 0.000</td> <td>    0.425</td> <td>    0.457</td>
</tr>
<tr>
  <th>C(country_standard)[T.Mauritania]</th>                       <td>    0.1721</td> <td>    0.002</td> <td>  114.178</td> <td> 0.000</td> <td>    0.169</td> <td>    0.175</td>
</tr>
<tr>
  <th>C(country_standard)[T.Mauritius]</th>                        <td>    0.4869</td> <td>    0.006</td> <td>   80.836</td> <td> 0.000</td> <td>    0.475</td> <td>    0.499</td>
</tr>
<tr>
  <th>C(country_standard)[T.Mexico]</th>                           <td>    0.3337</td> <td>    0.003</td> <td>  120.165</td> <td> 0.000</td> <td>    0.328</td> <td>    0.339</td>
</tr>
<tr>
  <th>C(country_standard)[T.Micronesia (Federated States of)]</th> <td>    0.1514</td> <td>    0.006</td> <td>   26.716</td> <td> 0.000</td> <td>    0.140</td> <td>    0.163</td>
</tr>
<tr>
  <th>C(country_standard)[T.Moldova]</th>                          <td>    0.1752</td> <td>    0.003</td> <td>   52.174</td> <td> 0.000</td> <td>    0.169</td> <td>    0.182</td>
</tr>
<tr>
  <th>C(country_standard)[T.Monaco]</th>                           <td>    0.2719</td> <td>    0.006</td> <td>   45.004</td> <td> 0.000</td> <td>    0.260</td> <td>    0.284</td>
</tr>
<tr>
  <th>C(country_standard)[T.Mongolia]</th>                         <td>    0.2465</td> <td>    0.005</td> <td>   45.077</td> <td> 0.000</td> <td>    0.236</td> <td>    0.257</td>
</tr>
<tr>
  <th>C(country_standard)[T.Montenegro]</th>                       <td>    0.3084</td> <td>    0.004</td> <td>   73.416</td> <td> 0.000</td> <td>    0.300</td> <td>    0.317</td>
</tr>
<tr>
  <th>C(country_standard)[T.Morocco]</th>                          <td>    0.2584</td> <td>    0.003</td> <td>   94.698</td> <td> 0.000</td> <td>    0.253</td> <td>    0.264</td>
</tr>
<tr>
  <th>C(country_standard)[T.Mozambique]</th>                       <td>    0.2437</td> <td>    0.003</td> <td>   77.569</td> <td> 0.000</td> <td>    0.238</td> <td>    0.250</td>
</tr>
<tr>
  <th>C(country_standard)[T.Myanmar (Burma)]</th>                  <td>    0.0455</td> <td>    0.002</td> <td>   28.944</td> <td> 0.000</td> <td>    0.042</td> <td>    0.049</td>
</tr>
<tr>
  <th>C(country_standard)[T.Namibia]</th>                          <td>    0.3702</td> <td>    0.006</td> <td>   62.508</td> <td> 0.000</td> <td>    0.359</td> <td>    0.382</td>
</tr>
<tr>
  <th>C(country_standard)[T.Nauru]</th>                            <td>    0.1547</td> <td>    0.007</td> <td>   22.636</td> <td> 0.000</td> <td>    0.141</td> <td>    0.168</td>
</tr>
<tr>
  <th>C(country_standard)[T.Nepal]</th>                            <td>    0.1642</td> <td>    0.003</td> <td>   54.105</td> <td> 0.000</td> <td>    0.158</td> <td>    0.170</td>
</tr>
<tr>
  <th>C(country_standard)[T.Netherlands]</th>                      <td>    0.5135</td> <td>    0.006</td> <td>   83.980</td> <td> 0.000</td> <td>    0.502</td> <td>    0.525</td>
</tr>
<tr>
  <th>C(country_standard)[T.New Caledonia]</th>                    <td>    0.2731</td> <td>    0.008</td> <td>   32.400</td> <td> 0.000</td> <td>    0.257</td> <td>    0.290</td>
</tr>
<tr>
  <th>C(country_standard)[T.New Zealand]</th>                      <td>    0.5999</td> <td>    0.007</td> <td>   91.173</td> <td> 0.000</td> <td>    0.587</td> <td>    0.613</td>
</tr>
<tr>
  <th>C(country_standard)[T.Nicaragua]</th>                        <td>    0.2053</td> <td>    0.002</td> <td>   86.394</td> <td> 0.000</td> <td>    0.201</td> <td>    0.210</td>
</tr>
<tr>
  <th>C(country_standard)[T.Niger]</th>                            <td>    0.0373</td> <td>    0.003</td> <td>   12.141</td> <td> 0.000</td> <td>    0.031</td> <td>    0.043</td>
</tr>
<tr>
  <th>C(country_standard)[T.Nigeria]</th>                          <td>    0.0921</td> <td>    0.003</td> <td>   34.228</td> <td> 0.000</td> <td>    0.087</td> <td>    0.097</td>
</tr>
<tr>
  <th>C(country_standard)[T.Niue]</th>                             <td>    0.2012</td> <td>    0.009</td> <td>   22.995</td> <td> 0.000</td> <td>    0.184</td> <td>    0.218</td>
</tr>
<tr>
  <th>C(country_standard)[T.North Korea]</th>                      <td>    0.0105</td> <td>    0.002</td> <td>    4.467</td> <td> 0.000</td> <td>    0.006</td> <td>    0.015</td>
</tr>
<tr>
  <th>C(country_standard)[T.North Macedonia]</th>                  <td>    0.2584</td> <td>    0.003</td> <td>   77.318</td> <td> 0.000</td> <td>    0.252</td> <td>    0.265</td>
</tr>
<tr>
  <th>C(country_standard)[T.Northern Mariana Islands]</th>         <td>    0.2731</td> <td>    0.008</td> <td>   32.550</td> <td> 0.000</td> <td>    0.257</td> <td>    0.290</td>
</tr>
<tr>
  <th>C(country_standard)[T.Norway]</th>                           <td>    0.5759</td> <td>    0.007</td> <td>   88.431</td> <td> 0.000</td> <td>    0.563</td> <td>    0.589</td>
</tr>
<tr>
  <th>C(country_standard)[T.Oman]</th>                             <td>    0.2805</td> <td>    0.001</td> <td>  199.498</td> <td> 0.000</td> <td>    0.278</td> <td>    0.283</td>
</tr>
<tr>
  <th>C(country_standard)[T.Pakistan]</th>                         <td>    0.2356</td> <td>    0.001</td> <td>  162.923</td> <td> 0.000</td> <td>    0.233</td> <td>    0.238</td>
</tr>
<tr>
  <th>C(country_standard)[T.Palau]</th>                            <td>    0.1590</td> <td>    0.007</td> <td>   22.852</td> <td> 0.000</td> <td>    0.145</td> <td>    0.173</td>
</tr>
<tr>
  <th>C(country_standard)[T.Palestinian Territories]</th>          <td>    0.0549</td> <td>    0.001</td> <td>   37.367</td> <td> 0.000</td> <td>    0.052</td> <td>    0.058</td>
</tr>
<tr>
  <th>C(country_standard)[T.Panama]</th>                           <td>    0.3970</td> <td>    0.006</td> <td>   67.344</td> <td> 0.000</td> <td>    0.385</td> <td>    0.409</td>
</tr>
<tr>
  <th>C(country_standard)[T.Papua New Guinea]</th>                 <td>    0.2969</td> <td>    0.003</td> <td>   85.536</td> <td> 0.000</td> <td>    0.290</td> <td>    0.304</td>
</tr>
<tr>
  <th>C(country_standard)[T.Paraguay]</th>                         <td>    0.2769</td> <td>    0.003</td> <td>   82.645</td> <td> 0.000</td> <td>    0.270</td> <td>    0.283</td>
</tr>
<tr>
  <th>C(country_standard)[T.Peru]</th>                             <td>    0.2444</td> <td>    0.005</td> <td>   45.541</td> <td> 0.000</td> <td>    0.234</td> <td>    0.255</td>
</tr>
<tr>
  <th>C(country_standard)[T.Philippines]</th>                      <td>    0.3140</td> <td>    0.003</td> <td>   96.041</td> <td> 0.000</td> <td>    0.308</td> <td>    0.320</td>
</tr>
<tr>
  <th>C(country_standard)[T.Poland]</th>                           <td>    0.3061</td> <td>    0.006</td> <td>   53.741</td> <td> 0.000</td> <td>    0.295</td> <td>    0.317</td>
</tr>
<tr>
  <th>C(country_standard)[T.Portugal]</th>                         <td>    0.4354</td> <td>    0.006</td> <td>   70.246</td> <td> 0.000</td> <td>    0.423</td> <td>    0.448</td>
</tr>
<tr>
  <th>C(country_standard)[T.Puerto Rico]</th>                      <td>    0.3175</td> <td>    0.007</td> <td>   45.592</td> <td> 0.000</td> <td>    0.304</td> <td>    0.331</td>
</tr>
<tr>
  <th>C(country_standard)[T.Qatar]</th>                            <td>    0.3254</td> <td>    0.001</td> <td>  235.496</td> <td> 0.000</td> <td>    0.323</td> <td>    0.328</td>
</tr>
<tr>
  <th>C(country_standard)[T.Romania]</th>                          <td>    0.3331</td> <td>    0.006</td> <td>   60.299</td> <td> 0.000</td> <td>    0.322</td> <td>    0.344</td>
</tr>
<tr>
  <th>C(country_standard)[T.Russia]</th>                           <td>    0.0905</td> <td>    0.002</td> <td>   55.402</td> <td> 0.000</td> <td>    0.087</td> <td>    0.094</td>
</tr>
<tr>
  <th>C(country_standard)[T.Rwanda]</th>                           <td>    0.3162</td> <td>    0.001</td> <td>  231.914</td> <td> 0.000</td> <td>    0.313</td> <td>    0.319</td>
</tr>
<tr>
  <th>C(country_standard)[T.Réunion]</th>                          <td>    0.4799</td> <td>    0.008</td> <td>   62.341</td> <td> 0.000</td> <td>    0.465</td> <td>    0.495</td>
</tr>
<tr>
  <th>C(country_standard)[T.Saint Martin (French part)]</th>       <td>    0.2730</td> <td>    0.008</td> <td>   32.728</td> <td> 0.000</td> <td>    0.257</td> <td>    0.289</td>
</tr>
<tr>
  <th>C(country_standard)[T.Samoa]</th>                            <td>    0.3243</td> <td>    0.006</td> <td>   55.277</td> <td> 0.000</td> <td>    0.313</td> <td>    0.336</td>
</tr>
<tr>
  <th>C(country_standard)[T.San Marino]</th>                       <td>    0.2693</td> <td>    0.006</td> <td>   42.041</td> <td> 0.000</td> <td>    0.257</td> <td>    0.282</td>
</tr>
<tr>
  <th>C(country_standard)[T.Saudi Arabia]</th>                     <td>    0.1848</td> <td>    0.002</td> <td>  110.051</td> <td> 0.000</td> <td>    0.181</td> <td>    0.188</td>
</tr>
<tr>
  <th>C(country_standard)[T.Senegal]</th>                          <td>    0.2621</td> <td>    0.005</td> <td>   50.275</td> <td> 0.000</td> <td>    0.252</td> <td>    0.272</td>
</tr>
<tr>
  <th>C(country_standard)[T.Serbia]</th>                           <td>    0.2714</td> <td>    0.005</td> <td>   54.385</td> <td> 0.000</td> <td>    0.262</td> <td>    0.281</td>
</tr>
<tr>
  <th>C(country_standard)[T.Seychelles]</th>                       <td>    0.3799</td> <td>    0.004</td> <td>  101.090</td> <td> 0.000</td> <td>    0.373</td> <td>    0.387</td>
</tr>
<tr>
  <th>C(country_standard)[T.Sierra Leone]</th>                     <td>    0.0341</td> <td>    0.004</td> <td>    9.503</td> <td> 0.000</td> <td>    0.027</td> <td>    0.041</td>
</tr>
<tr>
  <th>C(country_standard)[T.Singapore]</th>                        <td>    0.5907</td> <td>    0.003</td> <td>  189.069</td> <td> 0.000</td> <td>    0.585</td> <td>    0.597</td>
</tr>
<tr>
  <th>C(country_standard)[T.Sint Maarten]</th>                     <td>    0.2731</td> <td>    0.008</td> <td>   32.537</td> <td> 0.000</td> <td>    0.257</td> <td>    0.290</td>
</tr>
<tr>
  <th>C(country_standard)[T.Slovakia]</th>                         <td>    0.3883</td> <td>    0.006</td> <td>   64.372</td> <td> 0.000</td> <td>    0.376</td> <td>    0.400</td>
</tr>
<tr>
  <th>C(country_standard)[T.Slovenia]</th>                         <td>    0.4102</td> <td>    0.006</td> <td>   67.672</td> <td> 0.000</td> <td>    0.398</td> <td>    0.422</td>
</tr>
<tr>
  <th>C(country_standard)[T.Solomon Islands]</th>                  <td>    0.0999</td> <td>    0.005</td> <td>   21.899</td> <td> 0.000</td> <td>    0.091</td> <td>    0.109</td>
</tr>
<tr>
  <th>C(country_standard)[T.Somalia]</th>                          <td>   -0.1092</td> <td>    0.002</td> <td>  -62.176</td> <td> 0.000</td> <td>   -0.113</td> <td>   -0.106</td>
</tr>
<tr>
  <th>C(country_standard)[T.South Africa]</th>                     <td>    0.5164</td> <td>    0.006</td> <td>   88.222</td> <td> 0.000</td> <td>    0.505</td> <td>    0.528</td>
</tr>
<tr>
  <th>C(country_standard)[T.South Korea]</th>                      <td>    0.4120</td> <td>    0.004</td> <td>  106.941</td> <td> 0.000</td> <td>    0.404</td> <td>    0.420</td>
</tr>
<tr>
  <th>C(country_standard)[T.South Sudan]</th>                      <td>    0.1069</td> <td>    0.001</td> <td>   75.626</td> <td> 0.000</td> <td>    0.104</td> <td>    0.110</td>
</tr>
<tr>
  <th>C(country_standard)[T.Spain]</th>                            <td>    0.3830</td> <td>    0.005</td> <td>   71.027</td> <td> 0.000</td> <td>    0.372</td> <td>    0.394</td>
</tr>
<tr>
  <th>C(country_standard)[T.Sri Lanka]</th>                        <td>    0.3182</td> <td>    0.003</td> <td>   99.722</td> <td> 0.000</td> <td>    0.312</td> <td>    0.324</td>
</tr>
<tr>
  <th>C(country_standard)[T.St. Helena]</th>                       <td>    0.2653</td> <td>    0.004</td> <td>   75.604</td> <td> 0.000</td> <td>    0.258</td> <td>    0.272</td>
</tr>
<tr>
  <th>C(country_standard)[T.St. Kitts & Nevis]</th>                <td>    0.3711</td> <td>    0.006</td> <td>   59.166</td> <td> 0.000</td> <td>    0.359</td> <td>    0.383</td>
</tr>
<tr>
  <th>C(country_standard)[T.St. Lucia]</th>                        <td>    0.4061</td> <td>    0.006</td> <td>   64.364</td> <td> 0.000</td> <td>    0.394</td> <td>    0.419</td>
</tr>
<tr>
  <th>C(country_standard)[T.St. Vincent & Grenadines]</th>         <td>    0.3622</td> <td>    0.006</td> <td>   58.918</td> <td> 0.000</td> <td>    0.350</td> <td>    0.374</td>
</tr>
<tr>
  <th>C(country_standard)[T.Sudan]</th>                            <td>    0.0386</td> <td>    0.002</td> <td>   23.742</td> <td> 0.000</td> <td>    0.035</td> <td>    0.042</td>
</tr>
<tr>
  <th>C(country_standard)[T.Suriname]</th>                         <td>    0.3072</td> <td>    0.006</td> <td>   54.242</td> <td> 0.000</td> <td>    0.296</td> <td>    0.318</td>
</tr>
<tr>
  <th>C(country_standard)[T.Sweden]</th>                           <td>    0.5818</td> <td>    0.006</td> <td>   91.272</td> <td> 0.000</td> <td>    0.569</td> <td>    0.594</td>
</tr>
<tr>
  <th>C(country_standard)[T.Switzerland]</th>                      <td>    0.5759</td> <td>    0.006</td> <td>   91.910</td> <td> 0.000</td> <td>    0.564</td> <td>    0.588</td>
</tr>
<tr>
  <th>C(country_standard)[T.Syria]</th>                            <td>    0.0070</td> <td>    0.002</td> <td>    4.005</td> <td> 0.000</td> <td>    0.004</td> <td>    0.010</td>
</tr>
<tr>
  <th>C(country_standard)[T.São Tomé & Príncipe]</th>              <td>    0.2165</td> <td>    0.006</td> <td>   37.531</td> <td> 0.000</td> <td>    0.205</td> <td>    0.228</td>
</tr>
<tr>
  <th>C(country_standard)[T.Taiwan]</th>                           <td>    0.4724</td> <td>    0.006</td> <td>   81.010</td> <td> 0.000</td> <td>    0.461</td> <td>    0.484</td>
</tr>
<tr>
  <th>C(country_standard)[T.Tajikistan]</th>                       <td>    0.0158</td> <td>    0.002</td> <td>    9.967</td> <td> 0.000</td> <td>    0.013</td> <td>    0.019</td>
</tr>
<tr>
  <th>C(country_standard)[T.Tanzania]</th>                         <td>    0.2232</td> <td>    0.003</td> <td>   71.606</td> <td> 0.000</td> <td>    0.217</td> <td>    0.229</td>
</tr>
<tr>
  <th>C(country_standard)[T.Thailand]</th>                         <td>    0.3207</td> <td>    0.001</td> <td>  214.850</td> <td> 0.000</td> <td>    0.318</td> <td>    0.324</td>
</tr>
<tr>
  <th>C(country_standard)[T.Timor-Leste]</th>                      <td>    0.3995</td> <td>    0.004</td> <td>   95.197</td> <td> 0.000</td> <td>    0.391</td> <td>    0.408</td>
</tr>
<tr>
  <th>C(country_standard)[T.Togo]</th>                             <td>    0.0138</td> <td>    0.003</td> <td>    4.922</td> <td> 0.000</td> <td>    0.008</td> <td>    0.019</td>
</tr>
<tr>
  <th>C(country_standard)[T.Tonga]</th>                            <td>    0.1872</td> <td>    0.007</td> <td>   26.724</td> <td> 0.000</td> <td>    0.174</td> <td>    0.201</td>
</tr>
<tr>
  <th>C(country_standard)[T.Trinidad & Tobago]</th>                <td>    0.3835</td> <td>    0.006</td> <td>   65.892</td> <td> 0.000</td> <td>    0.372</td> <td>    0.395</td>
</tr>
<tr>
  <th>C(country_standard)[T.Tunisia]</th>                          <td>    0.2420</td> <td>    0.005</td> <td>   47.726</td> <td> 0.000</td> <td>    0.232</td> <td>    0.252</td>
</tr>
<tr>
  <th>C(country_standard)[T.Turkey]</th>                           <td>    0.3470</td> <td>    0.002</td> <td>  184.056</td> <td> 0.000</td> <td>    0.343</td> <td>    0.351</td>
</tr>
<tr>
  <th>C(country_standard)[T.Turkmenistan]</th>                     <td>    0.0002</td> <td>    0.002</td> <td>    0.120</td> <td> 0.904</td> <td>   -0.004</td> <td>    0.004</td>
</tr>
<tr>
  <th>C(country_standard)[T.Turks & Caicos Islands]</th>           <td>    0.2651</td> <td>    0.003</td> <td>   75.977</td> <td> 0.000</td> <td>    0.258</td> <td>    0.272</td>
</tr>
<tr>
  <th>C(country_standard)[T.Tuvalu]</th>                           <td>    0.1037</td> <td>    0.007</td> <td>   15.300</td> <td> 0.000</td> <td>    0.090</td> <td>    0.117</td>
</tr>
<tr>
  <th>C(country_standard)[T.U.S. Virgin Islands]</th>              <td>    0.5125</td> <td>    0.008</td> <td>   64.023</td> <td> 0.000</td> <td>    0.497</td> <td>    0.528</td>
</tr>
<tr>
  <th>C(country_standard)[T.Uganda]</th>                           <td>    0.2080</td> <td>    0.002</td> <td>  100.683</td> <td> 0.000</td> <td>    0.204</td> <td>    0.212</td>
</tr>
<tr>
  <th>C(country_standard)[T.Ukraine]</th>                          <td>    0.1544</td> <td>    0.003</td> <td>   48.661</td> <td> 0.000</td> <td>    0.148</td> <td>    0.161</td>
</tr>
<tr>
  <th>C(country_standard)[T.United Arab Emirates]</th>             <td>    0.3181</td> <td>    0.002</td> <td>  180.106</td> <td> 0.000</td> <td>    0.315</td> <td>    0.322</td>
</tr>
<tr>
  <th>C(country_standard)[T.United Kingdom]</th>                   <td>    0.3584</td> <td>    0.005</td> <td>   74.188</td> <td> 0.000</td> <td>    0.349</td> <td>    0.368</td>
</tr>
<tr>
  <th>C(country_standard)[T.United States]</th>                    <td>   -0.6660</td> <td>    0.006</td> <td> -104.092</td> <td> 0.000</td> <td>   -0.679</td> <td>   -0.653</td>
</tr>
<tr>
  <th>C(country_standard)[T.Uruguay]</th>                          <td>    0.4818</td> <td>    0.006</td> <td>   76.495</td> <td> 0.000</td> <td>    0.469</td> <td>    0.494</td>
</tr>
<tr>
  <th>C(country_standard)[T.Uzbekistan]</th>                       <td>    0.0848</td> <td>    0.002</td> <td>   52.744</td> <td> 0.000</td> <td>    0.082</td> <td>    0.088</td>
</tr>
<tr>
  <th>C(country_standard)[T.Vanuatu]</th>                          <td>    0.1851</td> <td>    0.006</td> <td>   31.842</td> <td> 0.000</td> <td>    0.174</td> <td>    0.197</td>
</tr>
<tr>
  <th>C(country_standard)[T.Vatican City]</th>                     <td>    0.2669</td> <td>    0.003</td> <td>   79.062</td> <td> 0.000</td> <td>    0.260</td> <td>    0.273</td>
</tr>
<tr>
  <th>C(country_standard)[T.Venezuela]</th>                        <td>    0.1388</td> <td>    0.001</td> <td>   96.218</td> <td> 0.000</td> <td>    0.136</td> <td>    0.142</td>
</tr>
<tr>
  <th>C(country_standard)[T.Vietnam]</th>                          <td>    0.2323</td> <td>    0.001</td> <td>  157.092</td> <td> 0.000</td> <td>    0.229</td> <td>    0.235</td>
</tr>
<tr>
  <th>C(country_standard)[T.Western Sahara]</th>                   <td>    0.3410</td> <td>    0.006</td> <td>   54.715</td> <td> 0.000</td> <td>    0.329</td> <td>    0.353</td>
</tr>
<tr>
  <th>C(country_standard)[T.Yemen]</th>                            <td>    0.0143</td> <td>    0.002</td> <td>    9.530</td> <td> 0.000</td> <td>    0.011</td> <td>    0.017</td>
</tr>
<tr>
  <th>C(country_standard)[T.Zambia]</th>                           <td>    0.3027</td> <td>    0.003</td> <td>   91.887</td> <td> 0.000</td> <td>    0.296</td> <td>    0.309</td>
</tr>
<tr>
  <th>C(country_standard)[T.Zimbabwe]</th>                         <td>    0.0551</td> <td>    0.002</td> <td>   33.819</td> <td> 0.000</td> <td>    0.052</td> <td>    0.058</td>
</tr>
<tr>
  <th>transparency_index</th>                                      <td>    0.1290</td> <td>    0.016</td> <td>    8.109</td> <td> 0.000</td> <td>    0.098</td> <td>    0.160</td>
</tr>
<tr>
  <th>gdp</th>                                                     <td> 5.608e-14</td> <td> 5.87e-16</td> <td>   95.547</td> <td> 0.000</td> <td> 5.49e-14</td> <td> 5.72e-14</td>
</tr>
<tr>
  <th>gini</th>                                                    <td>   -0.0040</td> <td> 1.46e-05</td> <td> -271.060</td> <td> 0.000</td> <td>   -0.004</td> <td>   -0.004</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>447.260</td> <th>  Durbin-Watson:     </th> <td>   0.876</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>4691.375</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.129</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 8.584</td>  <th>  Cond. No.          </th> <td>9.45e+26</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors are robust to cluster correlation (cluster)<br/>[2] The smallest eigenvalue is 1.39e-26. This might indicate that there are<br/>strong multicollinearity problems or that the design matrix is singular.




```python
reg_bugetparticipation = smf.ols('budget_participation_index ~ transparency_index + gdp + gini + C(country_standard)', ndi_df).fit(cov_type='cluster', cov_kwds={'groups': ndi_df['country_standard']})
```


```python
reg_bugetparticipation.summary()
```

    /Users/katiacordoba/opt/anaconda3/lib/python3.7/site-packages/statsmodels/base/model.py:1832: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 231, but rank is 228
      'rank is %d' % (J, J_), ValueWarning)





<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>    <td>budget_participation_index</td> <th>  R-squared:         </th> <td>   0.051</td> 
</tr>
<tr>
  <th>Model:</th>                        <td>OLS</td>            <th>  Adj. R-squared:    </th> <td>   0.050</td> 
</tr>
<tr>
  <th>Method:</th>                  <td>Least Squares</td>       <th>  F-statistic:       </th> <td>1.415e+06</td>
</tr>
<tr>
  <th>Date:</th>                  <td>Fri, 05 Mar 2021</td>      <th>  Prob (F-statistic):</th>  <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                      <td>23:12:36</td>          <th>  Log-Likelihood:    </th> <td>  3464.1</td> 
</tr>
<tr>
  <th>No. Observations:</th>           <td>  3603</td>           <th>  AIC:               </th> <td>  -6918.</td> 
</tr>
<tr>
  <th>Df Residuals:</th>               <td>  3598</td>           <th>  BIC:               </th> <td>  -6887.</td> 
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
  <th>Intercept</th>                                               <td>    0.0442</td> <td>    0.003</td> <td>   13.039</td> <td> 0.000</td> <td>    0.038</td> <td>    0.051</td>
</tr>
<tr>
  <th>C(country_standard)[T.Albania]</th>                          <td>   -0.0319</td> <td>    0.004</td> <td>   -7.996</td> <td> 0.000</td> <td>   -0.040</td> <td>   -0.024</td>
</tr>
<tr>
  <th>C(country_standard)[T.Algeria]</th>                          <td>   -0.0595</td> <td>    0.000</td> <td> -184.507</td> <td> 0.000</td> <td>   -0.060</td> <td>   -0.059</td>
</tr>
<tr>
  <th>C(country_standard)[T.American Samoa]</th>                   <td>   -0.0677</td> <td>    0.003</td> <td>  -26.787</td> <td> 0.000</td> <td>   -0.073</td> <td>   -0.063</td>
</tr>
<tr>
  <th>C(country_standard)[T.Andorra]</th>                          <td>   -0.0734</td> <td>    0.003</td> <td>  -21.611</td> <td> 0.000</td> <td>   -0.080</td> <td>   -0.067</td>
</tr>
<tr>
  <th>C(country_standard)[T.Angola]</th>                           <td>   -0.0648</td> <td>    0.000</td> <td> -187.895</td> <td> 0.000</td> <td>   -0.065</td> <td>   -0.064</td>
</tr>
<tr>
  <th>C(country_standard)[T.Anguilla]</th>                         <td>   -0.0660</td> <td>    0.002</td> <td>  -32.829</td> <td> 0.000</td> <td>   -0.070</td> <td>   -0.062</td>
</tr>
<tr>
  <th>C(country_standard)[T.Antigua & Barbuda]</th>                <td>   -0.0723</td> <td>    0.003</td> <td>  -23.487</td> <td> 0.000</td> <td>   -0.078</td> <td>   -0.066</td>
</tr>
<tr>
  <th>C(country_standard)[T.Argentina]</th>                        <td>   -0.0385</td> <td>    0.003</td> <td>  -13.395</td> <td> 0.000</td> <td>   -0.044</td> <td>   -0.033</td>
</tr>
<tr>
  <th>C(country_standard)[T.Armenia]</th>                          <td>   -0.0662</td> <td>    0.001</td> <td>  -46.259</td> <td> 0.000</td> <td>   -0.069</td> <td>   -0.063</td>
</tr>
<tr>
  <th>C(country_standard)[T.Aruba]</th>                            <td>   -0.0708</td> <td>    0.003</td> <td>  -21.197</td> <td> 0.000</td> <td>   -0.077</td> <td>   -0.064</td>
</tr>
<tr>
  <th>C(country_standard)[T.Australia]</th>                        <td>   -0.0322</td> <td>    0.003</td> <td>  -10.520</td> <td> 0.000</td> <td>   -0.038</td> <td>   -0.026</td>
</tr>
<tr>
  <th>C(country_standard)[T.Austria]</th>                          <td>   -0.0689</td> <td>    0.003</td> <td>  -20.817</td> <td> 0.000</td> <td>   -0.075</td> <td>   -0.062</td>
</tr>
<tr>
  <th>C(country_standard)[T.Azerbaijan]</th>                       <td>   -0.0304</td> <td>    0.000</td> <td>  -77.727</td> <td> 0.000</td> <td>   -0.031</td> <td>   -0.030</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bahamas]</th>                          <td>   -0.0727</td> <td>    0.003</td> <td>  -22.527</td> <td> 0.000</td> <td>   -0.079</td> <td>   -0.066</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bahrain]</th>                          <td>   -0.0619</td> <td>    0.000</td> <td> -177.319</td> <td> 0.000</td> <td>   -0.063</td> <td>   -0.061</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bangladesh]</th>                       <td>   -0.0637</td> <td>    0.001</td> <td>  -50.069</td> <td> 0.000</td> <td>   -0.066</td> <td>   -0.061</td>
</tr>
<tr>
  <th>C(country_standard)[T.Barbados]</th>                         <td>   -0.0731</td> <td>    0.003</td> <td>  -21.875</td> <td> 0.000</td> <td>   -0.080</td> <td>   -0.067</td>
</tr>
<tr>
  <th>C(country_standard)[T.Belarus]</th>                          <td>   -0.0588</td> <td>    0.000</td> <td> -133.495</td> <td> 0.000</td> <td>   -0.060</td> <td>   -0.058</td>
</tr>
<tr>
  <th>C(country_standard)[T.Belgium]</th>                          <td>   -0.0677</td> <td>    0.003</td> <td>  -20.735</td> <td> 0.000</td> <td>   -0.074</td> <td>   -0.061</td>
</tr>
<tr>
  <th>C(country_standard)[T.Belize]</th>                           <td>   -0.0722</td> <td>    0.003</td> <td>  -23.624</td> <td> 0.000</td> <td>   -0.078</td> <td>   -0.066</td>
</tr>
<tr>
  <th>C(country_standard)[T.Benin]</th>                            <td>   -0.0732</td> <td>    0.003</td> <td>  -26.247</td> <td> 0.000</td> <td>   -0.079</td> <td>   -0.068</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bermuda]</th>                          <td>   -0.0660</td> <td>    0.002</td> <td>  -32.859</td> <td> 0.000</td> <td>   -0.070</td> <td>   -0.062</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bhutan]</th>                           <td>   -0.0675</td> <td>    0.002</td> <td>  -40.869</td> <td> 0.000</td> <td>   -0.071</td> <td>   -0.064</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bolivia]</th>                          <td>   -0.0683</td> <td>    0.002</td> <td>  -41.049</td> <td> 0.000</td> <td>   -0.072</td> <td>   -0.065</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bosnia & Herzegovina]</th>             <td>   -0.0351</td> <td>    0.002</td> <td>  -22.313</td> <td> 0.000</td> <td>   -0.038</td> <td>   -0.032</td>
</tr>
<tr>
  <th>C(country_standard)[T.Botswana]</th>                         <td>   -0.0751</td> <td>    0.003</td> <td>  -24.537</td> <td> 0.000</td> <td>   -0.081</td> <td>   -0.069</td>
</tr>
<tr>
  <th>C(country_standard)[T.Brazil]</th>                           <td>   -0.0007</td> <td>    0.003</td> <td>   -0.273</td> <td> 0.785</td> <td>   -0.006</td> <td>    0.004</td>
</tr>
<tr>
  <th>C(country_standard)[T.British Virgin Islands]</th>           <td>   -0.0669</td> <td>    0.002</td> <td>  -27.615</td> <td> 0.000</td> <td>   -0.072</td> <td>   -0.062</td>
</tr>
<tr>
  <th>C(country_standard)[T.Brunei]</th>                           <td>   -0.0631</td> <td>    0.000</td> <td> -180.391</td> <td> 0.000</td> <td>   -0.064</td> <td>   -0.062</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bulgaria]</th>                         <td>   -0.0406</td> <td>    0.003</td> <td>  -13.758</td> <td> 0.000</td> <td>   -0.046</td> <td>   -0.035</td>
</tr>
<tr>
  <th>C(country_standard)[T.Burkina Faso]</th>                     <td>   -0.0356</td> <td>    0.002</td> <td>  -22.504</td> <td> 0.000</td> <td>   -0.039</td> <td>   -0.033</td>
</tr>
<tr>
  <th>C(country_standard)[T.Burundi]</th>                          <td>   -0.0632</td> <td>    0.000</td> <td> -188.873</td> <td> 0.000</td> <td>   -0.064</td> <td>   -0.062</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cambodia]</th>                         <td>   -0.0624</td> <td>    0.000</td> <td> -201.365</td> <td> 0.000</td> <td>   -0.063</td> <td>   -0.062</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cameroon]</th>                         <td>   -0.0642</td> <td>    0.000</td> <td> -207.085</td> <td> 0.000</td> <td>   -0.065</td> <td>   -0.064</td>
</tr>
<tr>
  <th>C(country_standard)[T.Canada]</th>                           <td>   -0.0300</td> <td>    0.003</td> <td>  -10.079</td> <td> 0.000</td> <td>   -0.036</td> <td>   -0.024</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cape Verde]</th>                       <td>   -0.0737</td> <td>    0.003</td> <td>  -22.493</td> <td> 0.000</td> <td>   -0.080</td> <td>   -0.067</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cayman Islands]</th>                   <td>   -0.0700</td> <td>    0.002</td> <td>  -29.693</td> <td> 0.000</td> <td>   -0.075</td> <td>   -0.065</td>
</tr>
<tr>
  <th>C(country_standard)[T.Central African Republic]</th>         <td>   -0.0666</td> <td>    0.000</td> <td> -201.854</td> <td> 0.000</td> <td>   -0.067</td> <td>   -0.066</td>
</tr>
<tr>
  <th>C(country_standard)[T.Chad]</th>                             <td>   -0.0632</td> <td>    0.000</td> <td> -188.323</td> <td> 0.000</td> <td>   -0.064</td> <td>   -0.063</td>
</tr>
<tr>
  <th>C(country_standard)[T.Channel Islands]</th>                  <td>   -0.0649</td> <td>    0.002</td> <td>  -36.526</td> <td> 0.000</td> <td>   -0.068</td> <td>   -0.061</td>
</tr>
<tr>
  <th>C(country_standard)[T.Chile]</th>                            <td>   -0.0100</td> <td>    0.003</td> <td>   -3.044</td> <td> 0.002</td> <td>   -0.016</td> <td>   -0.004</td>
</tr>
<tr>
  <th>C(country_standard)[T.China]</th>                            <td>    0.0343</td> <td>    0.005</td> <td>    7.522</td> <td> 0.000</td> <td>    0.025</td> <td>    0.043</td>
</tr>
<tr>
  <th>C(country_standard)[T.Colombia]</th>                         <td>   -0.0684</td> <td>    0.002</td> <td>  -40.928</td> <td> 0.000</td> <td>   -0.072</td> <td>   -0.065</td>
</tr>
<tr>
  <th>C(country_standard)[T.Comoros]</th>                          <td>   -0.0687</td> <td>    0.001</td> <td>  -45.820</td> <td> 0.000</td> <td>   -0.072</td> <td>   -0.066</td>
</tr>
<tr>
  <th>C(country_standard)[T.Congo - Brazzaville]</th>              <td>   -0.0647</td> <td>    0.000</td> <td> -210.880</td> <td> 0.000</td> <td>   -0.065</td> <td>   -0.064</td>
</tr>
<tr>
  <th>C(country_standard)[T.Congo - Kinshasa]</th>                 <td>   -0.0626</td> <td>    0.000</td> <td> -177.674</td> <td> 0.000</td> <td>   -0.063</td> <td>   -0.062</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cook Islands]</th>                     <td>   -0.0665</td> <td>    0.002</td> <td>  -27.846</td> <td> 0.000</td> <td>   -0.071</td> <td>   -0.062</td>
</tr>
<tr>
  <th>C(country_standard)[T.Costa Rica]</th>                       <td>   -0.0124</td> <td>    0.003</td> <td>   -3.630</td> <td> 0.000</td> <td>   -0.019</td> <td>   -0.006</td>
</tr>
<tr>
  <th>C(country_standard)[T.Croatia]</th>                          <td>   -0.0075</td> <td>    0.003</td> <td>   -2.516</td> <td> 0.012</td> <td>   -0.013</td> <td>   -0.002</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cuba]</th>                             <td>   -0.0612</td> <td>    0.000</td> <td> -146.400</td> <td> 0.000</td> <td>   -0.062</td> <td>   -0.060</td>
</tr>
<tr>
  <th>C(country_standard)[T.Curaçao]</th>                          <td>   -0.0669</td> <td>    0.002</td> <td>  -27.608</td> <td> 0.000</td> <td>   -0.072</td> <td>   -0.062</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cyprus]</th>                           <td>   -0.0714</td> <td>    0.003</td> <td>  -21.923</td> <td> 0.000</td> <td>   -0.078</td> <td>   -0.065</td>
</tr>
<tr>
  <th>C(country_standard)[T.Czechia]</th>                          <td>   -0.0371</td> <td>    0.003</td> <td>  -11.803</td> <td> 0.000</td> <td>   -0.043</td> <td>   -0.031</td>
</tr>
<tr>
  <th>C(country_standard)[T.Côte d’Ivoire]</th>                    <td>   -0.0051</td> <td>    0.001</td> <td>   -3.487</td> <td> 0.000</td> <td>   -0.008</td> <td>   -0.002</td>
</tr>
<tr>
  <th>C(country_standard)[T.Denmark]</th>                          <td>   -0.0696</td> <td>    0.003</td> <td>  -20.245</td> <td> 0.000</td> <td>   -0.076</td> <td>   -0.063</td>
</tr>
<tr>
  <th>C(country_standard)[T.Djibouti]</th>                         <td>   -0.0628</td> <td>    0.000</td> <td> -180.360</td> <td> 0.000</td> <td>   -0.063</td> <td>   -0.062</td>
</tr>
<tr>
  <th>C(country_standard)[T.Dominica]</th>                         <td>   -0.0728</td> <td>    0.003</td> <td>  -22.490</td> <td> 0.000</td> <td>   -0.079</td> <td>   -0.066</td>
</tr>
<tr>
  <th>C(country_standard)[T.Dominican Republic]</th>               <td>   -0.0075</td> <td>    0.002</td> <td>   -3.494</td> <td> 0.000</td> <td>   -0.012</td> <td>   -0.003</td>
</tr>
<tr>
  <th>C(country_standard)[T.Ecuador]</th>                          <td>   -0.0372</td> <td>    0.002</td> <td>  -23.318</td> <td> 0.000</td> <td>   -0.040</td> <td>   -0.034</td>
</tr>
<tr>
  <th>C(country_standard)[T.Egypt]</th>                            <td>   -0.0595</td> <td>    0.000</td> <td> -190.728</td> <td> 0.000</td> <td>   -0.060</td> <td>   -0.059</td>
</tr>
<tr>
  <th>C(country_standard)[T.El Salvador]</th>                      <td>   -0.0396</td> <td>    0.003</td> <td>  -14.967</td> <td> 0.000</td> <td>   -0.045</td> <td>   -0.034</td>
</tr>
<tr>
  <th>C(country_standard)[T.Equatorial Guinea]</th>                <td>   -0.0612</td> <td>    0.001</td> <td> -109.288</td> <td> 0.000</td> <td>   -0.062</td> <td>   -0.060</td>
</tr>
<tr>
  <th>C(country_standard)[T.Eritrea]</th>                          <td>   -0.0608</td> <td>    0.001</td> <td>  -91.220</td> <td> 0.000</td> <td>   -0.062</td> <td>   -0.059</td>
</tr>
<tr>
  <th>C(country_standard)[T.Estonia]</th>                          <td>   -0.0712</td> <td>    0.003</td> <td>  -21.410</td> <td> 0.000</td> <td>   -0.078</td> <td>   -0.065</td>
</tr>
<tr>
  <th>C(country_standard)[T.Eswatini]</th>                         <td>   -0.0657</td> <td>    0.000</td> <td> -226.755</td> <td> 0.000</td> <td>   -0.066</td> <td>   -0.065</td>
</tr>
<tr>
  <th>C(country_standard)[T.Ethiopia]</th>                         <td>   -0.0610</td> <td>    0.000</td> <td> -175.183</td> <td> 0.000</td> <td>   -0.062</td> <td>   -0.060</td>
</tr>
<tr>
  <th>C(country_standard)[T.Faroe Islands]</th>                    <td>   -0.0651</td> <td>    0.002</td> <td>  -37.560</td> <td> 0.000</td> <td>   -0.069</td> <td>   -0.062</td>
</tr>
<tr>
  <th>C(country_standard)[T.Fiji]</th>                             <td>   -0.0671</td> <td>    0.002</td> <td>  -42.557</td> <td> 0.000</td> <td>   -0.070</td> <td>   -0.064</td>
</tr>
<tr>
  <th>C(country_standard)[T.Finland]</th>                          <td>   -0.0073</td> <td>    0.003</td> <td>   -2.093</td> <td> 0.036</td> <td>   -0.014</td> <td>   -0.000</td>
</tr>
<tr>
  <th>C(country_standard)[T.France]</th>                           <td>    0.0085</td> <td>    0.003</td> <td>    3.404</td> <td> 0.001</td> <td>    0.004</td> <td>    0.013</td>
</tr>
<tr>
  <th>C(country_standard)[T.French Guiana]</th>                    <td>   -0.0707</td> <td>    0.003</td> <td>  -21.402</td> <td> 0.000</td> <td>   -0.077</td> <td>   -0.064</td>
</tr>
<tr>
  <th>C(country_standard)[T.French Polynesia]</th>                 <td>   -0.0669</td> <td>    0.002</td> <td>  -27.613</td> <td> 0.000</td> <td>   -0.072</td> <td>   -0.062</td>
</tr>
<tr>
  <th>C(country_standard)[T.Gabon]</th>                            <td>   -0.0627</td> <td>    0.000</td> <td> -196.408</td> <td> 0.000</td> <td>   -0.063</td> <td>   -0.062</td>
</tr>
<tr>
  <th>C(country_standard)[T.Gambia]</th>                           <td>   -0.0636</td> <td>    0.001</td> <td> -114.501</td> <td> 0.000</td> <td>   -0.065</td> <td>   -0.063</td>
</tr>
<tr>
  <th>C(country_standard)[T.Georgia]</th>                          <td>   -0.0051</td> <td>    0.002</td> <td>   -2.875</td> <td> 0.004</td> <td>   -0.009</td> <td>   -0.002</td>
</tr>
<tr>
  <th>C(country_standard)[T.Germany]</th>                          <td>    0.0157</td> <td>    0.002</td> <td>    6.936</td> <td> 0.000</td> <td>    0.011</td> <td>    0.020</td>
</tr>
<tr>
  <th>C(country_standard)[T.Ghana]</th>                            <td>   -0.0415</td> <td>    0.003</td> <td>  -13.604</td> <td> 0.000</td> <td>   -0.048</td> <td>   -0.036</td>
</tr>
<tr>
  <th>C(country_standard)[T.Gibraltar]</th>                        <td>   -0.0651</td> <td>    0.002</td> <td>  -37.557</td> <td> 0.000</td> <td>   -0.069</td> <td>   -0.062</td>
</tr>
<tr>
  <th>C(country_standard)[T.Greece]</th>                           <td>   -0.0078</td> <td>    0.003</td> <td>   -2.516</td> <td> 0.012</td> <td>   -0.014</td> <td>   -0.002</td>
</tr>
<tr>
  <th>C(country_standard)[T.Greenland]</th>                        <td>   -0.0695</td> <td>    0.003</td> <td>  -22.099</td> <td> 0.000</td> <td>   -0.076</td> <td>   -0.063</td>
</tr>
<tr>
  <th>C(country_standard)[T.Grenada]</th>                          <td>   -0.0724</td> <td>    0.003</td> <td>  -23.016</td> <td> 0.000</td> <td>   -0.079</td> <td>   -0.066</td>
</tr>
<tr>
  <th>C(country_standard)[T.Guam]</th>                             <td>   -0.0675</td> <td>    0.003</td> <td>  -26.956</td> <td> 0.000</td> <td>   -0.072</td> <td>   -0.063</td>
</tr>
<tr>
  <th>C(country_standard)[T.Guatemala]</th>                        <td>   -0.0065</td> <td>    0.002</td> <td>   -4.246</td> <td> 0.000</td> <td>   -0.009</td> <td>   -0.003</td>
</tr>
<tr>
  <th>C(country_standard)[T.Guernsey]</th>                         <td>   -0.0654</td> <td>    0.002</td> <td>  -37.859</td> <td> 0.000</td> <td>   -0.069</td> <td>   -0.062</td>
</tr>
<tr>
  <th>C(country_standard)[T.Guinea]</th>                           <td>   -0.0653</td> <td>    0.001</td> <td>  -53.643</td> <td> 0.000</td> <td>   -0.068</td> <td>   -0.063</td>
</tr>
<tr>
  <th>C(country_standard)[T.Guinea-Bissau]</th>                    <td>   -0.0682</td> <td>    0.001</td> <td>  -66.307</td> <td> 0.000</td> <td>   -0.070</td> <td>   -0.066</td>
</tr>
<tr>
  <th>C(country_standard)[T.Guyana]</th>                           <td>   -0.0715</td> <td>    0.003</td> <td>  -25.190</td> <td> 0.000</td> <td>   -0.077</td> <td>   -0.066</td>
</tr>
<tr>
  <th>C(country_standard)[T.Haiti]</th>                            <td>   -0.0671</td> <td>    0.001</td> <td>  -51.909</td> <td> 0.000</td> <td>   -0.070</td> <td>   -0.065</td>
</tr>
<tr>
  <th>C(country_standard)[T.Honduras]</th>                         <td>   -0.0074</td> <td>    0.001</td> <td>   -4.967</td> <td> 0.000</td> <td>   -0.010</td> <td>   -0.004</td>
</tr>
<tr>
  <th>C(country_standard)[T.Hong Kong SAR China]</th>              <td>   -0.0662</td> <td>    0.002</td> <td>  -36.248</td> <td> 0.000</td> <td>   -0.070</td> <td>   -0.063</td>
</tr>
<tr>
  <th>C(country_standard)[T.Hungary]</th>                          <td>   -0.0685</td> <td>    0.003</td> <td>  -25.836</td> <td> 0.000</td> <td>   -0.074</td> <td>   -0.063</td>
</tr>
<tr>
  <th>C(country_standard)[T.Iceland]</th>                          <td>   -0.0709</td> <td>    0.003</td> <td>  -20.728</td> <td> 0.000</td> <td>   -0.078</td> <td>   -0.064</td>
</tr>
<tr>
  <th>C(country_standard)[T.India]</th>                            <td>   -0.0509</td> <td>    0.002</td> <td>  -30.917</td> <td> 0.000</td> <td>   -0.054</td> <td>   -0.048</td>
</tr>
<tr>
  <th>C(country_standard)[T.Indonesia]</th>                        <td>    0.0016</td> <td>    0.002</td> <td>    1.022</td> <td> 0.307</td> <td>   -0.001</td> <td>    0.005</td>
</tr>
<tr>
  <th>C(country_standard)[T.Iran]</th>                             <td>   -0.0595</td> <td>    0.000</td> <td> -136.316</td> <td> 0.000</td> <td>   -0.060</td> <td>   -0.059</td>
</tr>
<tr>
  <th>C(country_standard)[T.Iraq]</th>                             <td>   -0.0592</td> <td>    0.000</td> <td> -176.840</td> <td> 0.000</td> <td>   -0.060</td> <td>   -0.059</td>
</tr>
<tr>
  <th>C(country_standard)[T.Ireland]</th>                          <td>   -0.0697</td> <td>    0.003</td> <td>  -21.113</td> <td> 0.000</td> <td>   -0.076</td> <td>   -0.063</td>
</tr>
<tr>
  <th>C(country_standard)[T.Isle of Man]</th>                      <td>   -0.0651</td> <td>    0.002</td> <td>  -37.558</td> <td> 0.000</td> <td>   -0.069</td> <td>   -0.062</td>
</tr>
<tr>
  <th>C(country_standard)[T.Israel]</th>                           <td>   -0.0700</td> <td>    0.003</td> <td>  -23.291</td> <td> 0.000</td> <td>   -0.076</td> <td>   -0.064</td>
</tr>
<tr>
  <th>C(country_standard)[T.Italy]</th>                            <td>    0.0029</td> <td>    0.003</td> <td>    1.105</td> <td> 0.269</td> <td>   -0.002</td> <td>    0.008</td>
</tr>
<tr>
  <th>C(country_standard)[T.Jamaica]</th>                          <td>   -0.0722</td> <td>    0.003</td> <td>  -23.562</td> <td> 0.000</td> <td>   -0.078</td> <td>   -0.066</td>
</tr>
<tr>
  <th>C(country_standard)[T.Japan]</th>                            <td>   -0.0382</td> <td>    0.002</td> <td>  -21.475</td> <td> 0.000</td> <td>   -0.042</td> <td>   -0.035</td>
</tr>
<tr>
  <th>C(country_standard)[T.Jersey]</th>                           <td>   -0.0695</td> <td>    0.003</td> <td>  -22.101</td> <td> 0.000</td> <td>   -0.076</td> <td>   -0.063</td>
</tr>
<tr>
  <th>C(country_standard)[T.Jordan]</th>                           <td>   -0.0012</td> <td>    0.001</td> <td>   -1.574</td> <td> 0.115</td> <td>   -0.003</td> <td>    0.000</td>
</tr>
<tr>
  <th>C(country_standard)[T.Kazakhstan]</th>                       <td>   -0.0589</td> <td>    0.000</td> <td> -160.163</td> <td> 0.000</td> <td>   -0.060</td> <td>   -0.058</td>
</tr>
<tr>
  <th>C(country_standard)[T.Kenya]</th>                            <td>   -0.0050</td> <td>    0.002</td> <td>   -3.226</td> <td> 0.001</td> <td>   -0.008</td> <td>   -0.002</td>
</tr>
<tr>
  <th>C(country_standard)[T.Kiribati]</th>                         <td>   -0.0711</td> <td>    0.003</td> <td>  -20.813</td> <td> 0.000</td> <td>   -0.078</td> <td>   -0.064</td>
</tr>
<tr>
  <th>C(country_standard)[T.Kosovo]</th>                           <td>   -0.0649</td> <td>    0.002</td> <td>  -36.519</td> <td> 0.000</td> <td>   -0.068</td> <td>   -0.061</td>
</tr>
<tr>
  <th>C(country_standard)[T.Kuwait]</th>                           <td>   -0.0659</td> <td>    0.001</td> <td>  -50.431</td> <td> 0.000</td> <td>   -0.068</td> <td>   -0.063</td>
</tr>
<tr>
  <th>C(country_standard)[T.Kyrgyzstan]</th>                       <td>   -0.0647</td> <td>    0.001</td> <td>  -46.669</td> <td> 0.000</td> <td>   -0.067</td> <td>   -0.062</td>
</tr>
<tr>
  <th>C(country_standard)[T.Laos]</th>                             <td>   -0.0610</td> <td>    0.000</td> <td> -126.234</td> <td> 0.000</td> <td>   -0.062</td> <td>   -0.060</td>
</tr>
<tr>
  <th>C(country_standard)[T.Latvia]</th>                           <td>   -0.0719</td> <td>    0.003</td> <td>  -22.748</td> <td> 0.000</td> <td>   -0.078</td> <td>   -0.066</td>
</tr>
<tr>
  <th>C(country_standard)[T.Lebanon]</th>                          <td>   -0.0653</td> <td>    0.001</td> <td>  -47.040</td> <td> 0.000</td> <td>   -0.068</td> <td>   -0.063</td>
</tr>
<tr>
  <th>C(country_standard)[T.Lesotho]</th>                          <td>   -0.0707</td> <td>    0.002</td> <td>  -32.973</td> <td> 0.000</td> <td>   -0.075</td> <td>   -0.066</td>
</tr>
<tr>
  <th>C(country_standard)[T.Liberia]</th>                          <td>   -0.0356</td> <td>    0.002</td> <td>  -22.599</td> <td> 0.000</td> <td>   -0.039</td> <td>   -0.033</td>
</tr>
<tr>
  <th>C(country_standard)[T.Libya]</th>                            <td>   -0.0628</td> <td>    0.000</td> <td> -184.536</td> <td> 0.000</td> <td>   -0.063</td> <td>   -0.062</td>
</tr>
<tr>
  <th>C(country_standard)[T.Liechtenstein]</th>                    <td>   -0.0735</td> <td>    0.003</td> <td>  -21.203</td> <td> 0.000</td> <td>   -0.080</td> <td>   -0.067</td>
</tr>
<tr>
  <th>C(country_standard)[T.Lithuania]</th>                        <td>   -0.0723</td> <td>    0.003</td> <td>  -22.316</td> <td> 0.000</td> <td>   -0.079</td> <td>   -0.066</td>
</tr>
<tr>
  <th>C(country_standard)[T.Luxembourg]</th>                       <td>   -0.0728</td> <td>    0.004</td> <td>  -20.559</td> <td> 0.000</td> <td>   -0.080</td> <td>   -0.066</td>
</tr>
<tr>
  <th>C(country_standard)[T.Macao SAR China]</th>                  <td>   -0.0672</td> <td>    0.002</td> <td>  -41.373</td> <td> 0.000</td> <td>   -0.070</td> <td>   -0.064</td>
</tr>
<tr>
  <th>C(country_standard)[T.Madagascar]</th>                       <td>   -0.0679</td> <td>    0.001</td> <td>  -46.675</td> <td> 0.000</td> <td>   -0.071</td> <td>   -0.065</td>
</tr>
<tr>
  <th>C(country_standard)[T.Malawi]</th>                           <td>   -0.0379</td> <td>    0.002</td> <td>  -22.767</td> <td> 0.000</td> <td>   -0.041</td> <td>   -0.035</td>
</tr>
<tr>
  <th>C(country_standard)[T.Malaysia]</th>                         <td>   -0.0655</td> <td>    0.001</td> <td>  -45.858</td> <td> 0.000</td> <td>   -0.068</td> <td>   -0.063</td>
</tr>
<tr>
  <th>C(country_standard)[T.Maldives]</th>                         <td>   -0.0654</td> <td>    0.001</td> <td>  -48.017</td> <td> 0.000</td> <td>   -0.068</td> <td>   -0.063</td>
</tr>
<tr>
  <th>C(country_standard)[T.Mali]</th>                             <td>   -0.0656</td> <td>    0.001</td> <td>  -47.829</td> <td> 0.000</td> <td>   -0.068</td> <td>   -0.063</td>
</tr>
<tr>
  <th>C(country_standard)[T.Malta]</th>                            <td>   -0.0711</td> <td>    0.003</td> <td>  -21.441</td> <td> 0.000</td> <td>   -0.078</td> <td>   -0.065</td>
</tr>
<tr>
  <th>C(country_standard)[T.Marshall Islands]</th>                 <td>   -0.0732</td> <td>    0.003</td> <td>  -21.755</td> <td> 0.000</td> <td>   -0.080</td> <td>   -0.067</td>
</tr>
<tr>
  <th>C(country_standard)[T.Martinique]</th>                       <td>   -0.0673</td> <td>    0.002</td> <td>  -27.188</td> <td> 0.000</td> <td>   -0.072</td> <td>   -0.062</td>
</tr>
<tr>
  <th>C(country_standard)[T.Mauritania]</th>                       <td>   -0.0622</td> <td>    0.000</td> <td> -161.409</td> <td> 0.000</td> <td>   -0.063</td> <td>   -0.061</td>
</tr>
<tr>
  <th>C(country_standard)[T.Mauritius]</th>                        <td>   -0.0723</td> <td>    0.003</td> <td>  -22.687</td> <td> 0.000</td> <td>   -0.078</td> <td>   -0.066</td>
</tr>
<tr>
  <th>C(country_standard)[T.Mexico]</th>                           <td>   -0.0299</td> <td>    0.001</td> <td>  -22.192</td> <td> 0.000</td> <td>   -0.033</td> <td>   -0.027</td>
</tr>
<tr>
  <th>C(country_standard)[T.Micronesia (Federated States of)]</th> <td>   -0.0688</td> <td>    0.003</td> <td>  -23.517</td> <td> 0.000</td> <td>   -0.075</td> <td>   -0.063</td>
</tr>
<tr>
  <th>C(country_standard)[T.Moldova]</th>                          <td>   -0.0336</td> <td>    0.002</td> <td>  -21.009</td> <td> 0.000</td> <td>   -0.037</td> <td>   -0.031</td>
</tr>
<tr>
  <th>C(country_standard)[T.Monaco]</th>                           <td>   -0.0726</td> <td>    0.003</td> <td>  -22.792</td> <td> 0.000</td> <td>   -0.079</td> <td>   -0.066</td>
</tr>
<tr>
  <th>C(country_standard)[T.Mongolia]</th>                         <td>   -0.0391</td> <td>    0.003</td> <td>  -13.658</td> <td> 0.000</td> <td>   -0.045</td> <td>   -0.033</td>
</tr>
<tr>
  <th>C(country_standard)[T.Montenegro]</th>                       <td>   -0.0695</td> <td>    0.002</td> <td>  -32.330</td> <td> 0.000</td> <td>   -0.074</td> <td>   -0.065</td>
</tr>
<tr>
  <th>C(country_standard)[T.Morocco]</th>                          <td>   -0.0663</td> <td>    0.001</td> <td>  -50.569</td> <td> 0.000</td> <td>   -0.069</td> <td>   -0.064</td>
</tr>
<tr>
  <th>C(country_standard)[T.Mozambique]</th>                       <td>   -0.0706</td> <td>    0.002</td> <td>  -44.976</td> <td> 0.000</td> <td>   -0.074</td> <td>   -0.068</td>
</tr>
<tr>
  <th>C(country_standard)[T.Myanmar (Burma)]</th>                  <td>   -0.0617</td> <td>    0.000</td> <td> -138.120</td> <td> 0.000</td> <td>   -0.063</td> <td>   -0.061</td>
</tr>
<tr>
  <th>C(country_standard)[T.Namibia]</th>                          <td>   -0.0765</td> <td>    0.003</td> <td>  -24.542</td> <td> 0.000</td> <td>   -0.083</td> <td>   -0.070</td>
</tr>
<tr>
  <th>C(country_standard)[T.Nauru]</th>                            <td>   -0.0707</td> <td>    0.003</td> <td>  -21.497</td> <td> 0.000</td> <td>   -0.077</td> <td>   -0.064</td>
</tr>
<tr>
  <th>C(country_standard)[T.Nepal]</th>                            <td>   -0.0659</td> <td>    0.001</td> <td>  -44.936</td> <td> 0.000</td> <td>   -0.069</td> <td>   -0.063</td>
</tr>
<tr>
  <th>C(country_standard)[T.Netherlands]</th>                      <td>   -0.0657</td> <td>    0.003</td> <td>  -20.163</td> <td> 0.000</td> <td>   -0.072</td> <td>   -0.059</td>
</tr>
<tr>
  <th>C(country_standard)[T.New Caledonia]</th>                    <td>   -0.0669</td> <td>    0.002</td> <td>  -27.604</td> <td> 0.000</td> <td>   -0.072</td> <td>   -0.062</td>
</tr>
<tr>
  <th>C(country_standard)[T.New Zealand]</th>                      <td>   -0.0413</td> <td>    0.004</td> <td>  -11.676</td> <td> 0.000</td> <td>   -0.048</td> <td>   -0.034</td>
</tr>
<tr>
  <th>C(country_standard)[T.Nicaragua]</th>                        <td>   -0.0675</td> <td>    0.001</td> <td>  -61.407</td> <td> 0.000</td> <td>   -0.070</td> <td>   -0.065</td>
</tr>
<tr>
  <th>C(country_standard)[T.Niger]</th>                            <td>   -0.0662</td> <td>    0.001</td> <td>  -45.122</td> <td> 0.000</td> <td>   -0.069</td> <td>   -0.063</td>
</tr>
<tr>
  <th>C(country_standard)[T.Nigeria]</th>                          <td>   -0.0634</td> <td>    0.001</td> <td>  -49.399</td> <td> 0.000</td> <td>   -0.066</td> <td>   -0.061</td>
</tr>
<tr>
  <th>C(country_standard)[T.Niue]</th>                             <td>   -0.0665</td> <td>    0.002</td> <td>  -27.813</td> <td> 0.000</td> <td>   -0.071</td> <td>   -0.062</td>
</tr>
<tr>
  <th>C(country_standard)[T.North Korea]</th>                      <td>   -0.0569</td> <td>    0.001</td> <td>  -55.898</td> <td> 0.000</td> <td>   -0.059</td> <td>   -0.055</td>
</tr>
<tr>
  <th>C(country_standard)[T.North Macedonia]</th>                  <td>   -0.0354</td> <td>    0.002</td> <td>  -21.992</td> <td> 0.000</td> <td>   -0.039</td> <td>   -0.032</td>
</tr>
<tr>
  <th>C(country_standard)[T.Northern Mariana Islands]</th>         <td>   -0.0669</td> <td>    0.002</td> <td>  -27.646</td> <td> 0.000</td> <td>   -0.072</td> <td>   -0.062</td>
</tr>
<tr>
  <th>C(country_standard)[T.Norway]</th>                           <td>   -0.0377</td> <td>    0.003</td> <td>  -10.807</td> <td> 0.000</td> <td>   -0.045</td> <td>   -0.031</td>
</tr>
<tr>
  <th>C(country_standard)[T.Oman]</th>                             <td>   -0.0622</td> <td>    0.000</td> <td> -201.545</td> <td> 0.000</td> <td>   -0.063</td> <td>   -0.062</td>
</tr>
<tr>
  <th>C(country_standard)[T.Pakistan]</th>                         <td>   -0.0191</td> <td>    0.000</td> <td>  -51.411</td> <td> 0.000</td> <td>   -0.020</td> <td>   -0.018</td>
</tr>
<tr>
  <th>C(country_standard)[T.Palau]</th>                            <td>   -0.0714</td> <td>    0.004</td> <td>  -20.266</td> <td> 0.000</td> <td>   -0.078</td> <td>   -0.064</td>
</tr>
<tr>
  <th>C(country_standard)[T.Palestinian Territories]</th>          <td>   -0.0612</td> <td>    0.000</td> <td> -172.125</td> <td> 0.000</td> <td>   -0.062</td> <td>   -0.060</td>
</tr>
<tr>
  <th>C(country_standard)[T.Panama]</th>                           <td>   -0.0115</td> <td>    0.003</td> <td>   -3.686</td> <td> 0.000</td> <td>   -0.018</td> <td>   -0.005</td>
</tr>
<tr>
  <th>C(country_standard)[T.Papua New Guinea]</th>                 <td>   -0.0685</td> <td>    0.002</td> <td>  -39.832</td> <td> 0.000</td> <td>   -0.072</td> <td>   -0.065</td>
</tr>
<tr>
  <th>C(country_standard)[T.Paraguay]</th>                         <td>   -0.0068</td> <td>    0.002</td> <td>   -4.026</td> <td> 0.000</td> <td>   -0.010</td> <td>   -0.003</td>
</tr>
<tr>
  <th>C(country_standard)[T.Peru]</th>                             <td>   -0.0398</td> <td>    0.003</td> <td>  -14.062</td> <td> 0.000</td> <td>   -0.045</td> <td>   -0.034</td>
</tr>
<tr>
  <th>C(country_standard)[T.Philippines]</th>                      <td>   -0.0043</td> <td>    0.002</td> <td>   -2.622</td> <td> 0.009</td> <td>   -0.007</td> <td>   -0.001</td>
</tr>
<tr>
  <th>C(country_standard)[T.Poland]</th>                           <td>   -0.0670</td> <td>    0.003</td> <td>  -22.255</td> <td> 0.000</td> <td>   -0.073</td> <td>   -0.061</td>
</tr>
<tr>
  <th>C(country_standard)[T.Portugal]</th>                         <td>   -0.0393</td> <td>    0.003</td> <td>  -11.970</td> <td> 0.000</td> <td>   -0.046</td> <td>   -0.033</td>
</tr>
<tr>
  <th>C(country_standard)[T.Puerto Rico]</th>                      <td>   -0.0694</td> <td>    0.003</td> <td>  -23.794</td> <td> 0.000</td> <td>   -0.075</td> <td>   -0.064</td>
</tr>
<tr>
  <th>C(country_standard)[T.Qatar]</th>                            <td>   -0.0614</td> <td>    0.000</td> <td> -196.384</td> <td> 0.000</td> <td>   -0.062</td> <td>   -0.061</td>
</tr>
<tr>
  <th>C(country_standard)[T.Romania]</th>                          <td>   -0.0386</td> <td>    0.003</td> <td>  -13.266</td> <td> 0.000</td> <td>   -0.044</td> <td>   -0.033</td>
</tr>
<tr>
  <th>C(country_standard)[T.Russia]</th>                           <td>   -0.0513</td> <td>    0.001</td> <td>  -90.846</td> <td> 0.000</td> <td>   -0.052</td> <td>   -0.050</td>
</tr>
<tr>
  <th>C(country_standard)[T.Rwanda]</th>                           <td>   -0.0637</td> <td>    0.000</td> <td> -212.499</td> <td> 0.000</td> <td>   -0.064</td> <td>   -0.063</td>
</tr>
<tr>
  <th>C(country_standard)[T.Réunion]</th>                          <td>   -0.0678</td> <td>    0.003</td> <td>  -26.433</td> <td> 0.000</td> <td>   -0.073</td> <td>   -0.063</td>
</tr>
<tr>
  <th>C(country_standard)[T.Saint Martin (French part)]</th>       <td>   -0.0669</td> <td>    0.002</td> <td>  -27.586</td> <td> 0.000</td> <td>   -0.072</td> <td>   -0.062</td>
</tr>
<tr>
  <th>C(country_standard)[T.Samoa]</th>                            <td>   -0.0724</td> <td>    0.003</td> <td>  -23.396</td> <td> 0.000</td> <td>   -0.079</td> <td>   -0.066</td>
</tr>
<tr>
  <th>C(country_standard)[T.San Marino]</th>                       <td>   -0.0732</td> <td>    0.003</td> <td>  -21.697</td> <td> 0.000</td> <td>   -0.080</td> <td>   -0.067</td>
</tr>
<tr>
  <th>C(country_standard)[T.Saudi Arabia]</th>                     <td>   -0.0561</td> <td>    0.001</td> <td>  -83.788</td> <td> 0.000</td> <td>   -0.057</td> <td>   -0.055</td>
</tr>
<tr>
  <th>C(country_standard)[T.Senegal]</th>                          <td>   -0.0403</td> <td>    0.003</td> <td>  -14.757</td> <td> 0.000</td> <td>   -0.046</td> <td>   -0.035</td>
</tr>
<tr>
  <th>C(country_standard)[T.Serbia]</th>                           <td>   -0.0385</td> <td>    0.003</td> <td>  -15.266</td> <td> 0.000</td> <td>   -0.043</td> <td>   -0.034</td>
</tr>
<tr>
  <th>C(country_standard)[T.Seychelles]</th>                       <td>   -0.0697</td> <td>    0.002</td> <td>  -38.689</td> <td> 0.000</td> <td>   -0.073</td> <td>   -0.066</td>
</tr>
<tr>
  <th>C(country_standard)[T.Sierra Leone]</th>                     <td>   -0.0359</td> <td>    0.002</td> <td>  -21.265</td> <td> 0.000</td> <td>   -0.039</td> <td>   -0.033</td>
</tr>
<tr>
  <th>C(country_standard)[T.Singapore]</th>                        <td>   -0.0651</td> <td>    0.002</td> <td>  -42.742</td> <td> 0.000</td> <td>   -0.068</td> <td>   -0.062</td>
</tr>
<tr>
  <th>C(country_standard)[T.Sint Maarten]</th>                     <td>   -0.0669</td> <td>    0.002</td> <td>  -27.551</td> <td> 0.000</td> <td>   -0.072</td> <td>   -0.062</td>
</tr>
<tr>
  <th>C(country_standard)[T.Slovakia]</th>                         <td>   -0.0067</td> <td>    0.003</td> <td>   -2.106</td> <td> 0.035</td> <td>   -0.013</td> <td>   -0.000</td>
</tr>
<tr>
  <th>C(country_standard)[T.Slovenia]</th>                         <td>   -0.0695</td> <td>    0.003</td> <td>  -21.795</td> <td> 0.000</td> <td>   -0.076</td> <td>   -0.063</td>
</tr>
<tr>
  <th>C(country_standard)[T.Solomon Islands]</th>                  <td>   -0.0700</td> <td>    0.002</td> <td>  -29.411</td> <td> 0.000</td> <td>   -0.075</td> <td>   -0.065</td>
</tr>
<tr>
  <th>C(country_standard)[T.Somalia]</th>                          <td>   -0.0579</td> <td>    0.001</td> <td>  -83.125</td> <td> 0.000</td> <td>   -0.059</td> <td>   -0.057</td>
</tr>
<tr>
  <th>C(country_standard)[T.South Africa]</th>                     <td>   -0.0128</td> <td>    0.003</td> <td>   -4.089</td> <td> 0.000</td> <td>   -0.019</td> <td>   -0.007</td>
</tr>
<tr>
  <th>C(country_standard)[T.South Korea]</th>                      <td>   -0.0148</td> <td>    0.002</td> <td>   -7.332</td> <td> 0.000</td> <td>   -0.019</td> <td>   -0.011</td>
</tr>
<tr>
  <th>C(country_standard)[T.South Sudan]</th>                      <td>   -0.0634</td> <td>    0.000</td> <td> -151.391</td> <td> 0.000</td> <td>   -0.064</td> <td>   -0.063</td>
</tr>
<tr>
  <th>C(country_standard)[T.Spain]</th>                            <td>   -0.0629</td> <td>    0.003</td> <td>  -21.972</td> <td> 0.000</td> <td>   -0.069</td> <td>   -0.057</td>
</tr>
<tr>
  <th>C(country_standard)[T.Sri Lanka]</th>                        <td>   -0.0670</td> <td>    0.002</td> <td>  -44.146</td> <td> 0.000</td> <td>   -0.070</td> <td>   -0.064</td>
</tr>
<tr>
  <th>C(country_standard)[T.St. Helena]</th>                       <td>   -0.0654</td> <td>    0.002</td> <td>  -37.829</td> <td> 0.000</td> <td>   -0.069</td> <td>   -0.062</td>
</tr>
<tr>
  <th>C(country_standard)[T.St. Kitts & Nevis]</th>                <td>   -0.0729</td> <td>    0.003</td> <td>  -22.253</td> <td> 0.000</td> <td>   -0.079</td> <td>   -0.066</td>
</tr>
<tr>
  <th>C(country_standard)[T.St. Lucia]</th>                        <td>   -0.0759</td> <td>    0.003</td> <td>  -22.624</td> <td> 0.000</td> <td>   -0.082</td> <td>   -0.069</td>
</tr>
<tr>
  <th>C(country_standard)[T.St. Vincent & Grenadines]</th>         <td>   -0.0731</td> <td>    0.003</td> <td>  -22.314</td> <td> 0.000</td> <td>   -0.079</td> <td>   -0.067</td>
</tr>
<tr>
  <th>C(country_standard)[T.Sudan]</th>                            <td>   -0.0604</td> <td>    0.001</td> <td> -116.464</td> <td> 0.000</td> <td>   -0.061</td> <td>   -0.059</td>
</tr>
<tr>
  <th>C(country_standard)[T.Suriname]</th>                         <td>   -0.0718</td> <td>    0.003</td> <td>  -24.284</td> <td> 0.000</td> <td>   -0.078</td> <td>   -0.066</td>
</tr>
<tr>
  <th>C(country_standard)[T.Sweden]</th>                           <td>   -0.0371</td> <td>    0.003</td> <td>  -10.883</td> <td> 0.000</td> <td>   -0.044</td> <td>   -0.030</td>
</tr>
<tr>
  <th>C(country_standard)[T.Switzerland]</th>                      <td>   -0.0681</td> <td>    0.003</td> <td>  -20.338</td> <td> 0.000</td> <td>   -0.075</td> <td>   -0.062</td>
</tr>
<tr>
  <th>C(country_standard)[T.Syria]</th>                            <td>   -0.0580</td> <td>    0.001</td> <td>  -86.786</td> <td> 0.000</td> <td>   -0.059</td> <td>   -0.057</td>
</tr>
<tr>
  <th>C(country_standard)[T.São Tomé & Príncipe]</th>              <td>   -0.0755</td> <td>    0.003</td> <td>  -25.337</td> <td> 0.000</td> <td>   -0.081</td> <td>   -0.070</td>
</tr>
<tr>
  <th>C(country_standard)[T.Taiwan]</th>                           <td>   -0.0696</td> <td>    0.003</td> <td>  -22.437</td> <td> 0.000</td> <td>   -0.076</td> <td>   -0.064</td>
</tr>
<tr>
  <th>C(country_standard)[T.Tajikistan]</th>                       <td>   -0.0607</td> <td>    0.000</td> <td> -133.301</td> <td> 0.000</td> <td>   -0.062</td> <td>   -0.060</td>
</tr>
<tr>
  <th>C(country_standard)[T.Tanzania]</th>                         <td>   -0.0676</td> <td>    0.002</td> <td>  -44.054</td> <td> 0.000</td> <td>   -0.071</td> <td>   -0.065</td>
</tr>
<tr>
  <th>C(country_standard)[T.Thailand]</th>                         <td>   -0.0604</td> <td>    0.000</td> <td> -128.114</td> <td> 0.000</td> <td>   -0.061</td> <td>   -0.059</td>
</tr>
<tr>
  <th>C(country_standard)[T.Timor-Leste]</th>                      <td>   -0.0675</td> <td>    0.002</td> <td>  -31.358</td> <td> 0.000</td> <td>   -0.072</td> <td>   -0.063</td>
</tr>
<tr>
  <th>C(country_standard)[T.Togo]</th>                             <td>   -0.0676</td> <td>    0.001</td> <td>  -51.408</td> <td> 0.000</td> <td>   -0.070</td> <td>   -0.065</td>
</tr>
<tr>
  <th>C(country_standard)[T.Tonga]</th>                            <td>   -0.0704</td> <td>    0.003</td> <td>  -21.945</td> <td> 0.000</td> <td>   -0.077</td> <td>   -0.064</td>
</tr>
<tr>
  <th>C(country_standard)[T.Trinidad & Tobago]</th>                <td>   -0.0406</td> <td>    0.003</td> <td>  -13.304</td> <td> 0.000</td> <td>   -0.047</td> <td>   -0.035</td>
</tr>
<tr>
  <th>C(country_standard)[T.Tunisia]</th>                          <td>   -0.0690</td> <td>    0.003</td> <td>  -27.211</td> <td> 0.000</td> <td>   -0.074</td> <td>   -0.064</td>
</tr>
<tr>
  <th>C(country_standard)[T.Turkey]</th>                           <td>   -0.0614</td> <td>    0.001</td> <td>  -75.644</td> <td> 0.000</td> <td>   -0.063</td> <td>   -0.060</td>
</tr>
<tr>
  <th>C(country_standard)[T.Turkmenistan]</th>                     <td>   -0.0600</td> <td>    0.001</td> <td>  -71.531</td> <td> 0.000</td> <td>   -0.062</td> <td>   -0.058</td>
</tr>
<tr>
  <th>C(country_standard)[T.Turks & Caicos Islands]</th>           <td>   -0.0652</td> <td>    0.002</td> <td>  -37.605</td> <td> 0.000</td> <td>   -0.069</td> <td>   -0.062</td>
</tr>
<tr>
  <th>C(country_standard)[T.Tuvalu]</th>                           <td>   -0.0711</td> <td>    0.003</td> <td>  -20.760</td> <td> 0.000</td> <td>   -0.078</td> <td>   -0.064</td>
</tr>
<tr>
  <th>C(country_standard)[T.U.S. Virgin Islands]</th>              <td>   -0.0675</td> <td>    0.003</td> <td>  -26.870</td> <td> 0.000</td> <td>   -0.072</td> <td>   -0.063</td>
</tr>
<tr>
  <th>C(country_standard)[T.Uganda]</th>                           <td>   -0.0654</td> <td>    0.001</td> <td>  -86.487</td> <td> 0.000</td> <td>   -0.067</td> <td>   -0.064</td>
</tr>
<tr>
  <th>C(country_standard)[T.Ukraine]</th>                          <td>   -0.0327</td> <td>    0.002</td> <td>  -21.396</td> <td> 0.000</td> <td>   -0.036</td> <td>   -0.030</td>
</tr>
<tr>
  <th>C(country_standard)[T.United Arab Emirates]</th>             <td>   -0.0588</td> <td>    0.000</td> <td> -149.216</td> <td> 0.000</td> <td>   -0.060</td> <td>   -0.058</td>
</tr>
<tr>
  <th>C(country_standard)[T.United Kingdom]</th>                   <td>    0.0087</td> <td>    0.003</td> <td>    3.324</td> <td> 0.001</td> <td>    0.004</td> <td>    0.014</td>
</tr>
<tr>
  <th>C(country_standard)[T.United States]</th>                    <td>    0.1009</td> <td>    0.003</td> <td>   32.494</td> <td> 0.000</td> <td>    0.095</td> <td>    0.107</td>
</tr>
<tr>
  <th>C(country_standard)[T.Uruguay]</th>                          <td>   -0.0732</td> <td>    0.003</td> <td>  -21.852</td> <td> 0.000</td> <td>   -0.080</td> <td>   -0.067</td>
</tr>
<tr>
  <th>C(country_standard)[T.Uzbekistan]</th>                       <td>   -0.0610</td> <td>    0.001</td> <td> -114.758</td> <td> 0.000</td> <td>   -0.062</td> <td>   -0.060</td>
</tr>
<tr>
  <th>C(country_standard)[T.Vanuatu]</th>                          <td>   -0.0721</td> <td>    0.003</td> <td>  -23.658</td> <td> 0.000</td> <td>   -0.078</td> <td>   -0.066</td>
</tr>
<tr>
  <th>C(country_standard)[T.Vatican City]</th>                     <td>   -0.0655</td> <td>    0.002</td> <td>  -37.575</td> <td> 0.000</td> <td>   -0.069</td> <td>   -0.062</td>
</tr>
<tr>
  <th>C(country_standard)[T.Venezuela]</th>                        <td>   -0.0629</td> <td>    0.000</td> <td> -130.495</td> <td> 0.000</td> <td>   -0.064</td> <td>   -0.062</td>
</tr>
<tr>
  <th>C(country_standard)[T.Vietnam]</th>                          <td>   -0.0599</td> <td>    0.000</td> <td> -149.426</td> <td> 0.000</td> <td>   -0.061</td> <td>   -0.059</td>
</tr>
<tr>
  <th>C(country_standard)[T.Western Sahara]</th>                   <td>   -0.0494</td> <td>    0.003</td> <td>  -14.838</td> <td> 0.000</td> <td>   -0.056</td> <td>   -0.043</td>
</tr>
<tr>
  <th>C(country_standard)[T.Yemen]</th>                            <td>   -0.0614</td> <td>    0.000</td> <td> -148.331</td> <td> 0.000</td> <td>   -0.062</td> <td>   -0.061</td>
</tr>
<tr>
  <th>C(country_standard)[T.Zambia]</th>                           <td>   -0.0714</td> <td>    0.002</td> <td>  -43.162</td> <td> 0.000</td> <td>   -0.075</td> <td>   -0.068</td>
</tr>
<tr>
  <th>C(country_standard)[T.Zimbabwe]</th>                         <td>   -0.0653</td> <td>    0.001</td> <td> -116.591</td> <td> 0.000</td> <td>   -0.066</td> <td>   -0.064</td>
</tr>
<tr>
  <th>transparency_index</th>                                      <td>    0.0275</td> <td>    0.009</td> <td>    3.180</td> <td> 0.001</td> <td>    0.011</td> <td>    0.044</td>
</tr>
<tr>
  <th>gdp</th>                                                     <td>-6.692e-15</td> <td>    3e-16</td> <td>  -22.279</td> <td> 0.000</td> <td>-7.28e-15</td> <td> -6.1e-15</td>
</tr>
<tr>
  <th>gini</th>                                                    <td>    0.0002</td> <td> 5.69e-06</td> <td>   38.402</td> <td> 0.000</td> <td>    0.000</td> <td>    0.000</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>4954.314</td> <th>  Durbin-Watson:     </th>  <td>   2.074</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>841951.461</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 8.124</td>  <th>  Prob(JB):          </th>  <td>    0.00</td> 
</tr>
<tr>
  <th>Kurtosis:</th>       <td>76.105</td>  <th>  Cond. No.          </th>  <td>9.45e+26</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors are robust to cluster correlation (cluster)<br/>[2] The smallest eigenvalue is 1.39e-26. This might indicate that there are<br/>strong multicollinearity problems or that the design matrix is singular.




```python
reg_buget_transparency = smf.ols('budget_transparency_index ~ transparency_index + gdp + gini + C(country_standard)', ndi_df).fit(cov_type='cluster', cov_kwds={'groups': ndi_df['country_standard']})
```


```python
reg_buget_transparency.summary()
```

    /Users/katiacordoba/opt/anaconda3/lib/python3.7/site-packages/statsmodels/base/model.py:1832: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 231, but rank is 229
      'rank is %d' % (J, J_), ValueWarning)





<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>    <td>budget_transparency_index</td> <th>  R-squared:         </th> <td>   0.051</td> 
</tr>
<tr>
  <th>Model:</th>                       <td>OLS</td>            <th>  Adj. R-squared:    </th> <td>   0.050</td> 
</tr>
<tr>
  <th>Method:</th>                 <td>Least Squares</td>       <th>  F-statistic:       </th> <td>5.540e+06</td>
</tr>
<tr>
  <th>Date:</th>                 <td>Fri, 05 Mar 2021</td>      <th>  Prob (F-statistic):</th>  <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                     <td>23:13:19</td>          <th>  Log-Likelihood:    </th> <td>  3369.6</td> 
</tr>
<tr>
  <th>No. Observations:</th>          <td>  3603</td>           <th>  AIC:               </th> <td>  -6729.</td> 
</tr>
<tr>
  <th>Df Residuals:</th>              <td>  3598</td>           <th>  BIC:               </th> <td>  -6698.</td> 
</tr>
<tr>
  <th>Df Model:</th>                  <td>     4</td>           <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>           <td>cluster</td>          <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
                             <td></td>                                <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>                                               <td>    0.0047</td> <td>    0.004</td> <td>    1.290</td> <td> 0.197</td> <td>   -0.002</td> <td>    0.012</td>
</tr>
<tr>
  <th>C(country_standard)[T.Albania]</th>                          <td>    0.0133</td> <td>    0.002</td> <td>    6.655</td> <td> 0.000</td> <td>    0.009</td> <td>    0.017</td>
</tr>
<tr>
  <th>C(country_standard)[T.Algeria]</th>                          <td>   -0.0211</td> <td> 9.69e-05</td> <td> -218.128</td> <td> 0.000</td> <td>   -0.021</td> <td>   -0.021</td>
</tr>
<tr>
  <th>C(country_standard)[T.American Samoa]</th>                   <td>   -0.0317</td> <td>    0.002</td> <td>  -14.062</td> <td> 0.000</td> <td>   -0.036</td> <td>   -0.027</td>
</tr>
<tr>
  <th>C(country_standard)[T.Andorra]</th>                          <td>   -0.0378</td> <td>    0.004</td> <td>  -10.338</td> <td> 0.000</td> <td>   -0.045</td> <td>   -0.031</td>
</tr>
<tr>
  <th>C(country_standard)[T.Angola]</th>                           <td>   -0.0210</td> <td>    0.000</td> <td> -179.234</td> <td> 0.000</td> <td>   -0.021</td> <td>   -0.021</td>
</tr>
<tr>
  <th>C(country_standard)[T.Anguilla]</th>                         <td>   -0.0311</td> <td>    0.002</td> <td>  -14.579</td> <td> 0.000</td> <td>   -0.035</td> <td>   -0.027</td>
</tr>
<tr>
  <th>C(country_standard)[T.Antigua & Barbuda]</th>                <td>   -0.0362</td> <td>    0.003</td> <td>  -10.929</td> <td> 0.000</td> <td>   -0.043</td> <td>   -0.030</td>
</tr>
<tr>
  <th>C(country_standard)[T.Argentina]</th>                        <td>    0.0062</td> <td>    0.003</td> <td>    2.026</td> <td> 0.043</td> <td>    0.000</td> <td>    0.012</td>
</tr>
<tr>
  <th>C(country_standard)[T.Armenia]</th>                          <td>    0.0137</td> <td>    0.002</td> <td>    9.088</td> <td> 0.000</td> <td>    0.011</td> <td>    0.017</td>
</tr>
<tr>
  <th>C(country_standard)[T.Aruba]</th>                            <td>   -0.0376</td> <td>    0.004</td> <td>  -10.693</td> <td> 0.000</td> <td>   -0.045</td> <td>   -0.031</td>
</tr>
<tr>
  <th>C(country_standard)[T.Australia]</th>                        <td>    0.0252</td> <td>    0.003</td> <td>    7.670</td> <td> 0.000</td> <td>    0.019</td> <td>    0.032</td>
</tr>
<tr>
  <th>C(country_standard)[T.Austria]</th>                          <td>   -0.0378</td> <td>    0.004</td> <td>  -10.638</td> <td> 0.000</td> <td>   -0.045</td> <td>   -0.031</td>
</tr>
<tr>
  <th>C(country_standard)[T.Azerbaijan]</th>                       <td>    0.0219</td> <td>    0.000</td> <td>   81.331</td> <td> 0.000</td> <td>    0.021</td> <td>    0.022</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bahamas]</th>                          <td>   -0.0369</td> <td>    0.003</td> <td>  -10.645</td> <td> 0.000</td> <td>   -0.044</td> <td>   -0.030</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bahrain]</th>                          <td>   -0.0201</td> <td>    0.000</td> <td> -101.785</td> <td> 0.000</td> <td>   -0.021</td> <td>   -0.020</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bangladesh]</th>                       <td>   -0.0274</td> <td>    0.001</td> <td>  -20.529</td> <td> 0.000</td> <td>   -0.030</td> <td>   -0.025</td>
</tr>
<tr>
  <th>C(country_standard)[T.Barbados]</th>                         <td>   -0.0375</td> <td>    0.004</td> <td>  -10.435</td> <td> 0.000</td> <td>   -0.045</td> <td>   -0.030</td>
</tr>
<tr>
  <th>C(country_standard)[T.Belarus]</th>                          <td>   -0.0198</td> <td>    0.000</td> <td>  -61.699</td> <td> 0.000</td> <td>   -0.020</td> <td>   -0.019</td>
</tr>
<tr>
  <th>C(country_standard)[T.Belgium]</th>                          <td>   -0.0377</td> <td>    0.004</td> <td>  -10.749</td> <td> 0.000</td> <td>   -0.045</td> <td>   -0.031</td>
</tr>
<tr>
  <th>C(country_standard)[T.Belize]</th>                           <td>   -0.0360</td> <td>    0.003</td> <td>  -10.985</td> <td> 0.000</td> <td>   -0.042</td> <td>   -0.030</td>
</tr>
<tr>
  <th>C(country_standard)[T.Benin]</th>                            <td>   -0.0346</td> <td>    0.003</td> <td>  -11.541</td> <td> 0.000</td> <td>   -0.040</td> <td>   -0.029</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bermuda]</th>                          <td>   -0.0311</td> <td>    0.002</td> <td>  -14.576</td> <td> 0.000</td> <td>   -0.035</td> <td>   -0.027</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bhutan]</th>                           <td>   -0.0289</td> <td>    0.002</td> <td>  -16.575</td> <td> 0.000</td> <td>   -0.032</td> <td>   -0.026</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bolivia]</th>                          <td>   -0.0290</td> <td>    0.002</td> <td>  -16.421</td> <td> 0.000</td> <td>   -0.032</td> <td>   -0.026</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bosnia & Herzegovina]</th>             <td>   -0.0287</td> <td>    0.002</td> <td>  -17.221</td> <td> 0.000</td> <td>   -0.032</td> <td>   -0.025</td>
</tr>
<tr>
  <th>C(country_standard)[T.Botswana]</th>                         <td>   -0.0358</td> <td>    0.003</td> <td>  -10.890</td> <td> 0.000</td> <td>   -0.042</td> <td>   -0.029</td>
</tr>
<tr>
  <th>C(country_standard)[T.Brazil]</th>                           <td>    0.0278</td> <td>    0.003</td> <td>   10.303</td> <td> 0.000</td> <td>    0.023</td> <td>    0.033</td>
</tr>
<tr>
  <th>C(country_standard)[T.British Virgin Islands]</th>           <td>   -0.0303</td> <td>    0.002</td> <td>  -15.449</td> <td> 0.000</td> <td>   -0.034</td> <td>   -0.026</td>
</tr>
<tr>
  <th>C(country_standard)[T.Brunei]</th>                           <td>   -0.0217</td> <td>    0.000</td> <td> -109.011</td> <td> 0.000</td> <td>   -0.022</td> <td>   -0.021</td>
</tr>
<tr>
  <th>C(country_standard)[T.Bulgaria]</th>                         <td>    0.0062</td> <td>    0.003</td> <td>    1.957</td> <td> 0.050</td> <td>-8.85e-06</td> <td>    0.012</td>
</tr>
<tr>
  <th>C(country_standard)[T.Burkina Faso]</th>                     <td>   -0.0079</td> <td>    0.002</td> <td>   -4.684</td> <td> 0.000</td> <td>   -0.011</td> <td>   -0.005</td>
</tr>
<tr>
  <th>C(country_standard)[T.Burundi]</th>                          <td>   -0.0216</td> <td>    0.000</td> <td> -127.336</td> <td> 0.000</td> <td>   -0.022</td> <td>   -0.021</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cambodia]</th>                         <td>   -0.0208</td> <td> 9.57e-05</td> <td> -217.153</td> <td> 0.000</td> <td>   -0.021</td> <td>   -0.021</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cameroon]</th>                         <td>   -0.0210</td> <td>    0.000</td> <td> -205.692</td> <td> 0.000</td> <td>   -0.021</td> <td>   -0.021</td>
</tr>
<tr>
  <th>C(country_standard)[T.Canada]</th>                           <td>    0.0044</td> <td>    0.003</td> <td>    1.365</td> <td> 0.172</td> <td>   -0.002</td> <td>    0.011</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cape Verde]</th>                       <td>   -0.0371</td> <td>    0.004</td> <td>  -10.527</td> <td> 0.000</td> <td>   -0.044</td> <td>   -0.030</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cayman Islands]</th>                   <td>   -0.0326</td> <td>    0.003</td> <td>  -12.891</td> <td> 0.000</td> <td>   -0.038</td> <td>   -0.028</td>
</tr>
<tr>
  <th>C(country_standard)[T.Central African Republic]</th>         <td>   -0.0213</td> <td>    0.000</td> <td> -124.429</td> <td> 0.000</td> <td>   -0.022</td> <td>   -0.021</td>
</tr>
<tr>
  <th>C(country_standard)[T.Chad]</th>                             <td>   -0.0201</td> <td>    0.000</td> <td> -115.249</td> <td> 0.000</td> <td>   -0.020</td> <td>   -0.020</td>
</tr>
<tr>
  <th>C(country_standard)[T.Channel Islands]</th>                  <td>   -0.0297</td> <td>    0.002</td> <td>  -16.195</td> <td> 0.000</td> <td>   -0.033</td> <td>   -0.026</td>
</tr>
<tr>
  <th>C(country_standard)[T.Chile]</th>                            <td>    0.0045</td> <td>    0.003</td> <td>    1.293</td> <td> 0.196</td> <td>   -0.002</td> <td>    0.011</td>
</tr>
<tr>
  <th>C(country_standard)[T.China]</th>                            <td>   -0.0103</td> <td>    0.005</td> <td>   -2.085</td> <td> 0.037</td> <td>   -0.020</td> <td>   -0.001</td>
</tr>
<tr>
  <th>C(country_standard)[T.Colombia]</th>                         <td>    0.0125</td> <td>    0.002</td> <td>    6.997</td> <td> 0.000</td> <td>    0.009</td> <td>    0.016</td>
</tr>
<tr>
  <th>C(country_standard)[T.Comoros]</th>                          <td>   -0.0281</td> <td>    0.002</td> <td>  -17.712</td> <td> 0.000</td> <td>   -0.031</td> <td>   -0.025</td>
</tr>
<tr>
  <th>C(country_standard)[T.Congo - Brazzaville]</th>              <td>   -0.0208</td> <td> 9.13e-05</td> <td> -227.727</td> <td> 0.000</td> <td>   -0.021</td> <td>   -0.021</td>
</tr>
<tr>
  <th>C(country_standard)[T.Congo - Kinshasa]</th>                 <td>   -0.0200</td> <td>    0.000</td> <td>  -95.626</td> <td> 0.000</td> <td>   -0.020</td> <td>   -0.020</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cook Islands]</th>                     <td>   -0.0296</td> <td>    0.002</td> <td>  -16.295</td> <td> 0.000</td> <td>   -0.033</td> <td>   -0.026</td>
</tr>
<tr>
  <th>C(country_standard)[T.Costa Rica]</th>                       <td>    0.0041</td> <td>    0.004</td> <td>    1.136</td> <td> 0.256</td> <td>   -0.003</td> <td>    0.011</td>
</tr>
<tr>
  <th>C(country_standard)[T.Croatia]</th>                          <td>    0.0059</td> <td>    0.003</td> <td>    1.845</td> <td> 0.065</td> <td>   -0.000</td> <td>    0.012</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cuba]</th>                             <td>   -0.0197</td> <td>    0.000</td> <td>  -64.673</td> <td> 0.000</td> <td>   -0.020</td> <td>   -0.019</td>
</tr>
<tr>
  <th>C(country_standard)[T.Curaçao]</th>                          <td>   -0.0303</td> <td>    0.002</td> <td>  -15.445</td> <td> 0.000</td> <td>   -0.034</td> <td>   -0.026</td>
</tr>
<tr>
  <th>C(country_standard)[T.Cyprus]</th>                           <td>   -0.0372</td> <td>    0.003</td> <td>  -10.626</td> <td> 0.000</td> <td>   -0.044</td> <td>   -0.030</td>
</tr>
<tr>
  <th>C(country_standard)[T.Czechia]</th>                          <td>   -0.0160</td> <td>    0.003</td> <td>   -4.755</td> <td> 0.000</td> <td>   -0.023</td> <td>   -0.009</td>
</tr>
<tr>
  <th>C(country_standard)[T.Côte d’Ivoire]</th>                    <td>    0.0137</td> <td>    0.002</td> <td>    8.890</td> <td> 0.000</td> <td>    0.011</td> <td>    0.017</td>
</tr>
<tr>
  <th>C(country_standard)[T.Denmark]</th>                          <td>    0.0033</td> <td>    0.004</td> <td>    0.883</td> <td> 0.377</td> <td>   -0.004</td> <td>    0.011</td>
</tr>
<tr>
  <th>C(country_standard)[T.Djibouti]</th>                         <td>   -0.0200</td> <td>    0.000</td> <td> -100.491</td> <td> 0.000</td> <td>   -0.020</td> <td>   -0.020</td>
</tr>
<tr>
  <th>C(country_standard)[T.Dominica]</th>                         <td>   -0.0369</td> <td>    0.003</td> <td>  -10.625</td> <td> 0.000</td> <td>   -0.044</td> <td>   -0.030</td>
</tr>
<tr>
  <th>C(country_standard)[T.Dominican Republic]</th>               <td>    0.0101</td> <td>    0.002</td> <td>    4.360</td> <td> 0.000</td> <td>    0.006</td> <td>    0.015</td>
</tr>
<tr>
  <th>C(country_standard)[T.Ecuador]</th>                          <td>   -0.0079</td> <td>    0.002</td> <td>   -4.636</td> <td> 0.000</td> <td>   -0.011</td> <td>   -0.005</td>
</tr>
<tr>
  <th>C(country_standard)[T.Egypt]</th>                            <td>   -0.0211</td> <td>    0.000</td> <td> -205.793</td> <td> 0.000</td> <td>   -0.021</td> <td>   -0.021</td>
</tr>
<tr>
  <th>C(country_standard)[T.El Salvador]</th>                      <td>    0.0076</td> <td>    0.003</td> <td>    2.666</td> <td> 0.008</td> <td>    0.002</td> <td>    0.013</td>
</tr>
<tr>
  <th>C(country_standard)[T.Equatorial Guinea]</th>                <td>   -0.0186</td> <td>    0.001</td> <td>  -36.623</td> <td> 0.000</td> <td>   -0.020</td> <td>   -0.018</td>
</tr>
<tr>
  <th>C(country_standard)[T.Eritrea]</th>                          <td>   -0.0180</td> <td>    0.001</td> <td>  -28.143</td> <td> 0.000</td> <td>   -0.019</td> <td>   -0.017</td>
</tr>
<tr>
  <th>C(country_standard)[T.Estonia]</th>                          <td>    0.0042</td> <td>    0.004</td> <td>    1.165</td> <td> 0.244</td> <td>   -0.003</td> <td>    0.011</td>
</tr>
<tr>
  <th>C(country_standard)[T.Eswatini]</th>                         <td>   -0.0203</td> <td>    0.000</td> <td> -185.587</td> <td> 0.000</td> <td>   -0.021</td> <td>   -0.020</td>
</tr>
<tr>
  <th>C(country_standard)[T.Ethiopia]</th>                         <td>   -0.0203</td> <td>    0.000</td> <td> -106.350</td> <td> 0.000</td> <td>   -0.021</td> <td>   -0.020</td>
</tr>
<tr>
  <th>C(country_standard)[T.Faroe Islands]</th>                    <td>   -0.0297</td> <td>    0.002</td> <td>  -16.195</td> <td> 0.000</td> <td>   -0.033</td> <td>   -0.026</td>
</tr>
<tr>
  <th>C(country_standard)[T.Fiji]</th>                             <td>   -0.0286</td> <td>    0.002</td> <td>  -17.177</td> <td> 0.000</td> <td>   -0.032</td> <td>   -0.025</td>
</tr>
<tr>
  <th>C(country_standard)[T.Finland]</th>                          <td>    0.0033</td> <td>    0.004</td> <td>    0.882</td> <td> 0.378</td> <td>   -0.004</td> <td>    0.011</td>
</tr>
<tr>
  <th>C(country_standard)[T.France]</th>                           <td>    0.0265</td> <td>    0.003</td> <td>    9.758</td> <td> 0.000</td> <td>    0.021</td> <td>    0.032</td>
</tr>
<tr>
  <th>C(country_standard)[T.French Guiana]</th>                    <td>   -0.0374</td> <td>    0.003</td> <td>  -10.775</td> <td> 0.000</td> <td>   -0.044</td> <td>   -0.031</td>
</tr>
<tr>
  <th>C(country_standard)[T.French Polynesia]</th>                 <td>   -0.0303</td> <td>    0.002</td> <td>  -15.446</td> <td> 0.000</td> <td>   -0.034</td> <td>   -0.026</td>
</tr>
<tr>
  <th>C(country_standard)[T.Gabon]</th>                            <td>   -0.0212</td> <td>    0.000</td> <td> -184.198</td> <td> 0.000</td> <td>   -0.021</td> <td>   -0.021</td>
</tr>
<tr>
  <th>C(country_standard)[T.Gambia]</th>                           <td>   -0.0232</td> <td>    0.001</td> <td>  -46.181</td> <td> 0.000</td> <td>   -0.024</td> <td>   -0.022</td>
</tr>
<tr>
  <th>C(country_standard)[T.Georgia]</th>                          <td>    0.0329</td> <td>    0.002</td> <td>   17.390</td> <td> 0.000</td> <td>    0.029</td> <td>    0.037</td>
</tr>
<tr>
  <th>C(country_standard)[T.Germany]</th>                          <td>    0.0059</td> <td>    0.002</td> <td>    2.367</td> <td> 0.018</td> <td>    0.001</td> <td>    0.011</td>
</tr>
<tr>
  <th>C(country_standard)[T.Ghana]</th>                            <td>    0.0058</td> <td>    0.003</td> <td>    1.775</td> <td> 0.076</td> <td>   -0.001</td> <td>    0.012</td>
</tr>
<tr>
  <th>C(country_standard)[T.Gibraltar]</th>                        <td>   -0.0297</td> <td>    0.002</td> <td>  -16.193</td> <td> 0.000</td> <td>   -0.033</td> <td>   -0.026</td>
</tr>
<tr>
  <th>C(country_standard)[T.Greece]</th>                           <td>    0.0054</td> <td>    0.003</td> <td>    1.644</td> <td> 0.100</td> <td>   -0.001</td> <td>    0.012</td>
</tr>
<tr>
  <th>C(country_standard)[T.Greenland]</th>                        <td>   -0.0368</td> <td>    0.003</td> <td>  -10.933</td> <td> 0.000</td> <td>   -0.043</td> <td>   -0.030</td>
</tr>
<tr>
  <th>C(country_standard)[T.Grenada]</th>                          <td>   -0.0365</td> <td>    0.003</td> <td>  -10.797</td> <td> 0.000</td> <td>   -0.043</td> <td>   -0.030</td>
</tr>
<tr>
  <th>C(country_standard)[T.Guam]</th>                             <td>   -0.0314</td> <td>    0.002</td> <td>  -14.344</td> <td> 0.000</td> <td>   -0.036</td> <td>   -0.027</td>
</tr>
<tr>
  <th>C(country_standard)[T.Guatemala]</th>                        <td>    0.0134</td> <td>    0.002</td> <td>    8.216</td> <td> 0.000</td> <td>    0.010</td> <td>    0.017</td>
</tr>
<tr>
  <th>C(country_standard)[T.Guernsey]</th>                         <td>   -0.0299</td> <td>    0.002</td> <td>  -16.240</td> <td> 0.000</td> <td>   -0.033</td> <td>   -0.026</td>
</tr>
<tr>
  <th>C(country_standard)[T.Guinea]</th>                           <td>   -0.0268</td> <td>    0.001</td> <td>  -21.120</td> <td> 0.000</td> <td>   -0.029</td> <td>   -0.024</td>
</tr>
<tr>
  <th>C(country_standard)[T.Guinea-Bissau]</th>                    <td>   -0.0256</td> <td>    0.001</td> <td>  -24.048</td> <td> 0.000</td> <td>   -0.028</td> <td>   -0.023</td>
</tr>
<tr>
  <th>C(country_standard)[T.Guyana]</th>                           <td>   -0.0350</td> <td>    0.003</td> <td>  -11.469</td> <td> 0.000</td> <td>   -0.041</td> <td>   -0.029</td>
</tr>
<tr>
  <th>C(country_standard)[T.Haiti]</th>                            <td>   -0.0271</td> <td>    0.001</td> <td>  -19.934</td> <td> 0.000</td> <td>   -0.030</td> <td>   -0.024</td>
</tr>
<tr>
  <th>C(country_standard)[T.Honduras]</th>                         <td>    0.0137</td> <td>    0.002</td> <td>    8.631</td> <td> 0.000</td> <td>    0.011</td> <td>    0.017</td>
</tr>
<tr>
  <th>C(country_standard)[T.Hong Kong SAR China]</th>              <td>   -0.0302</td> <td>    0.002</td> <td>  -15.505</td> <td> 0.000</td> <td>   -0.034</td> <td>   -0.026</td>
</tr>
<tr>
  <th>C(country_standard)[T.Hungary]</th>                          <td>   -0.0343</td> <td>    0.003</td> <td>  -12.051</td> <td> 0.000</td> <td>   -0.040</td> <td>   -0.029</td>
</tr>
<tr>
  <th>C(country_standard)[T.Iceland]</th>                          <td>   -0.0380</td> <td>    0.004</td> <td>  -10.358</td> <td> 0.000</td> <td>   -0.045</td> <td>   -0.031</td>
</tr>
<tr>
  <th>C(country_standard)[T.India]</th>                            <td>   -0.0315</td> <td>    0.002</td> <td>  -17.868</td> <td> 0.000</td> <td>   -0.035</td> <td>   -0.028</td>
</tr>
<tr>
  <th>C(country_standard)[T.Indonesia]</th>                        <td>    0.0122</td> <td>    0.002</td> <td>    7.443</td> <td> 0.000</td> <td>    0.009</td> <td>    0.015</td>
</tr>
<tr>
  <th>C(country_standard)[T.Iran]</th>                             <td>   -0.0196</td> <td>    0.000</td> <td>  -54.069</td> <td> 0.000</td> <td>   -0.020</td> <td>   -0.019</td>
</tr>
<tr>
  <th>C(country_standard)[T.Iraq]</th>                             <td>   -0.0207</td> <td>    0.000</td> <td> -140.841</td> <td> 0.000</td> <td>   -0.021</td> <td>   -0.020</td>
</tr>
<tr>
  <th>C(country_standard)[T.Ireland]</th>                          <td>   -0.0377</td> <td>    0.004</td> <td>  -10.617</td> <td> 0.000</td> <td>   -0.045</td> <td>   -0.031</td>
</tr>
<tr>
  <th>C(country_standard)[T.Isle of Man]</th>                      <td>   -0.0297</td> <td>    0.002</td> <td>  -16.193</td> <td> 0.000</td> <td>   -0.033</td> <td>   -0.026</td>
</tr>
<tr>
  <th>C(country_standard)[T.Israel]</th>                           <td>    0.0056</td> <td>    0.003</td> <td>    1.718</td> <td> 0.086</td> <td>   -0.001</td> <td>    0.012</td>
</tr>
<tr>
  <th>C(country_standard)[T.Italy]</th>                            <td>    0.0056</td> <td>    0.003</td> <td>    1.960</td> <td> 0.050</td> <td> 1.44e-06</td> <td>    0.011</td>
</tr>
<tr>
  <th>C(country_standard)[T.Jamaica]</th>                          <td>   -0.0361</td> <td>    0.003</td> <td>  -10.969</td> <td> 0.000</td> <td>   -0.043</td> <td>   -0.030</td>
</tr>
<tr>
  <th>C(country_standard)[T.Japan]</th>                            <td>   -0.0340</td> <td>    0.002</td> <td>  -18.014</td> <td> 0.000</td> <td>   -0.038</td> <td>   -0.030</td>
</tr>
<tr>
  <th>C(country_standard)[T.Jersey]</th>                           <td>   -0.0368</td> <td>    0.003</td> <td>  -10.934</td> <td> 0.000</td> <td>   -0.043</td> <td>   -0.030</td>
</tr>
<tr>
  <th>C(country_standard)[T.Jordan]</th>                           <td>    0.0172</td> <td>    0.001</td> <td>   23.144</td> <td> 0.000</td> <td>    0.016</td> <td>    0.019</td>
</tr>
<tr>
  <th>C(country_standard)[T.Kazakhstan]</th>                       <td>   -0.0203</td> <td>    0.000</td> <td>  -95.139</td> <td> 0.000</td> <td>   -0.021</td> <td>   -0.020</td>
</tr>
<tr>
  <th>C(country_standard)[T.Kenya]</th>                            <td>    0.0131</td> <td>    0.002</td> <td>    7.984</td> <td> 0.000</td> <td>    0.010</td> <td>    0.016</td>
</tr>
<tr>
  <th>C(country_standard)[T.Kiribati]</th>                         <td>   -0.0380</td> <td>    0.004</td> <td>  -10.538</td> <td> 0.000</td> <td>   -0.045</td> <td>   -0.031</td>
</tr>
<tr>
  <th>C(country_standard)[T.Kosovo]</th>                           <td>   -0.0297</td> <td>    0.002</td> <td>  -16.194</td> <td> 0.000</td> <td>   -0.033</td> <td>   -0.026</td>
</tr>
<tr>
  <th>C(country_standard)[T.Kuwait]</th>                           <td>   -0.0274</td> <td>    0.001</td> <td>  -19.870</td> <td> 0.000</td> <td>   -0.030</td> <td>   -0.025</td>
</tr>
<tr>
  <th>C(country_standard)[T.Kyrgyzstan]</th>                       <td>   -0.0278</td> <td>    0.001</td> <td>  -19.066</td> <td> 0.000</td> <td>   -0.031</td> <td>   -0.025</td>
</tr>
<tr>
  <th>C(country_standard)[T.Laos]</th>                             <td>   -0.0191</td> <td>    0.000</td> <td>  -47.119</td> <td> 0.000</td> <td>   -0.020</td> <td>   -0.018</td>
</tr>
<tr>
  <th>C(country_standard)[T.Latvia]</th>                           <td>    0.0050</td> <td>    0.003</td> <td>    1.470</td> <td> 0.142</td> <td>   -0.002</td> <td>    0.012</td>
</tr>
<tr>
  <th>C(country_standard)[T.Lebanon]</th>                          <td>   -0.0278</td> <td>    0.001</td> <td>  -18.997</td> <td> 0.000</td> <td>   -0.031</td> <td>   -0.025</td>
</tr>
<tr>
  <th>C(country_standard)[T.Lesotho]</th>                          <td>   -0.0314</td> <td>    0.002</td> <td>  -13.673</td> <td> 0.000</td> <td>   -0.036</td> <td>   -0.027</td>
</tr>
<tr>
  <th>C(country_standard)[T.Liberia]</th>                          <td>    0.0131</td> <td>    0.002</td> <td>    7.841</td> <td> 0.000</td> <td>    0.010</td> <td>    0.016</td>
</tr>
<tr>
  <th>C(country_standard)[T.Libya]</th>                            <td>   -0.0217</td> <td>    0.000</td> <td> -119.281</td> <td> 0.000</td> <td>   -0.022</td> <td>   -0.021</td>
</tr>
<tr>
  <th>C(country_standard)[T.Liechtenstein]</th>                    <td>   -0.0381</td> <td>    0.004</td> <td>  -10.220</td> <td> 0.000</td> <td>   -0.045</td> <td>   -0.031</td>
</tr>
<tr>
  <th>C(country_standard)[T.Lithuania]</th>                        <td>    0.0046</td> <td>    0.003</td> <td>    1.331</td> <td> 0.183</td> <td>   -0.002</td> <td>    0.011</td>
</tr>
<tr>
  <th>C(country_standard)[T.Luxembourg]</th>                       <td>   -0.0386</td> <td>    0.004</td> <td>  -10.134</td> <td> 0.000</td> <td>   -0.046</td> <td>   -0.031</td>
</tr>
<tr>
  <th>C(country_standard)[T.Macao SAR China]</th>                  <td>   -0.0288</td> <td>    0.002</td> <td>  -16.806</td> <td> 0.000</td> <td>   -0.032</td> <td>   -0.025</td>
</tr>
<tr>
  <th>C(country_standard)[T.Madagascar]</th>                       <td>   -0.0279</td> <td>    0.002</td> <td>  -18.160</td> <td> 0.000</td> <td>   -0.031</td> <td>   -0.025</td>
</tr>
<tr>
  <th>C(country_standard)[T.Malawi]</th>                           <td>   -0.0081</td> <td>    0.002</td> <td>   -4.594</td> <td> 0.000</td> <td>   -0.012</td> <td>   -0.005</td>
</tr>
<tr>
  <th>C(country_standard)[T.Malaysia]</th>                         <td>   -0.0281</td> <td>    0.002</td> <td>  -18.600</td> <td> 0.000</td> <td>   -0.031</td> <td>   -0.025</td>
</tr>
<tr>
  <th>C(country_standard)[T.Maldives]</th>                         <td>   -0.0276</td> <td>    0.001</td> <td>  -19.321</td> <td> 0.000</td> <td>   -0.030</td> <td>   -0.025</td>
</tr>
<tr>
  <th>C(country_standard)[T.Mali]</th>                             <td>   -0.0276</td> <td>    0.001</td> <td>  -19.244</td> <td> 0.000</td> <td>   -0.030</td> <td>   -0.025</td>
</tr>
<tr>
  <th>C(country_standard)[T.Malta]</th>                            <td>   -0.0375</td> <td>    0.004</td> <td>  -10.525</td> <td> 0.000</td> <td>   -0.044</td> <td>   -0.030</td>
</tr>
<tr>
  <th>C(country_standard)[T.Marshall Islands]</th>                 <td>   -0.0376</td> <td>    0.004</td> <td>  -10.394</td> <td> 0.000</td> <td>   -0.045</td> <td>   -0.030</td>
</tr>
<tr>
  <th>C(country_standard)[T.Martinique]</th>                       <td>   -0.0311</td> <td>    0.002</td> <td>  -14.654</td> <td> 0.000</td> <td>   -0.035</td> <td>   -0.027</td>
</tr>
<tr>
  <th>C(country_standard)[T.Mauritania]</th>                       <td>   -0.0221</td> <td>    0.000</td> <td>  -87.205</td> <td> 0.000</td> <td>   -0.023</td> <td>   -0.022</td>
</tr>
<tr>
  <th>C(country_standard)[T.Mauritius]</th>                        <td>   -0.0367</td> <td>    0.003</td> <td>  -10.730</td> <td> 0.000</td> <td>   -0.043</td> <td>   -0.030</td>
</tr>
<tr>
  <th>C(country_standard)[T.Mexico]</th>                           <td>    0.0340</td> <td>    0.001</td> <td>   23.846</td> <td> 0.000</td> <td>    0.031</td> <td>    0.037</td>
</tr>
<tr>
  <th>C(country_standard)[T.Micronesia (Federated States of)]</th> <td>   -0.0357</td> <td>    0.003</td> <td>  -11.411</td> <td> 0.000</td> <td>   -0.042</td> <td>   -0.030</td>
</tr>
<tr>
  <th>C(country_standard)[T.Moldova]</th>                          <td>   -0.0289</td> <td>    0.002</td> <td>  -17.096</td> <td> 0.000</td> <td>   -0.032</td> <td>   -0.026</td>
</tr>
<tr>
  <th>C(country_standard)[T.Monaco]</th>                           <td>   -0.0367</td> <td>    0.003</td> <td>  -10.718</td> <td> 0.000</td> <td>   -0.043</td> <td>   -0.030</td>
</tr>
<tr>
  <th>C(country_standard)[T.Mongolia]</th>                         <td>    0.0066</td> <td>    0.003</td> <td>    2.161</td> <td> 0.031</td> <td>    0.001</td> <td>    0.013</td>
</tr>
<tr>
  <th>C(country_standard)[T.Montenegro]</th>                       <td>    0.0102</td> <td>    0.002</td> <td>    4.446</td> <td> 0.000</td> <td>    0.006</td> <td>    0.015</td>
</tr>
<tr>
  <th>C(country_standard)[T.Morocco]</th>                          <td>   -0.0273</td> <td>    0.001</td> <td>  -19.796</td> <td> 0.000</td> <td>   -0.030</td> <td>   -0.025</td>
</tr>
<tr>
  <th>C(country_standard)[T.Mozambique]</th>                       <td>   -0.0283</td> <td>    0.002</td> <td>  -16.991</td> <td> 0.000</td> <td>   -0.032</td> <td>   -0.025</td>
</tr>
<tr>
  <th>C(country_standard)[T.Myanmar (Burma)]</th>                  <td>   -0.0226</td> <td>    0.000</td> <td>  -64.692</td> <td> 0.000</td> <td>   -0.023</td> <td>   -0.022</td>
</tr>
<tr>
  <th>C(country_standard)[T.Namibia]</th>                          <td>   -0.0360</td> <td>    0.003</td> <td>  -10.746</td> <td> 0.000</td> <td>   -0.043</td> <td>   -0.029</td>
</tr>
<tr>
  <th>C(country_standard)[T.Nauru]</th>                            <td>   -0.0373</td> <td>    0.003</td> <td>  -10.815</td> <td> 0.000</td> <td>   -0.044</td> <td>   -0.031</td>
</tr>
<tr>
  <th>C(country_standard)[T.Nepal]</th>                            <td>   -0.0281</td> <td>    0.002</td> <td>  -18.199</td> <td> 0.000</td> <td>   -0.031</td> <td>   -0.025</td>
</tr>
<tr>
  <th>C(country_standard)[T.Netherlands]</th>                      <td>    0.0037</td> <td>    0.004</td> <td>    1.046</td> <td> 0.295</td> <td>   -0.003</td> <td>    0.011</td>
</tr>
<tr>
  <th>C(country_standard)[T.New Caledonia]</th>                    <td>   -0.0303</td> <td>    0.002</td> <td>  -15.453</td> <td> 0.000</td> <td>   -0.034</td> <td>   -0.026</td>
</tr>
<tr>
  <th>C(country_standard)[T.New Zealand]</th>                      <td>    0.0239</td> <td>    0.004</td> <td>    6.305</td> <td> 0.000</td> <td>    0.017</td> <td>    0.031</td>
</tr>
<tr>
  <th>C(country_standard)[T.Nicaragua]</th>                        <td>   -0.0261</td> <td>    0.001</td> <td>  -22.686</td> <td> 0.000</td> <td>   -0.028</td> <td>   -0.024</td>
</tr>
<tr>
  <th>C(country_standard)[T.Niger]</th>                            <td>   -0.0281</td> <td>    0.002</td> <td>  -18.192</td> <td> 0.000</td> <td>   -0.031</td> <td>   -0.025</td>
</tr>
<tr>
  <th>C(country_standard)[T.Nigeria]</th>                          <td>   -0.0275</td> <td>    0.001</td> <td>  -20.434</td> <td> 0.000</td> <td>   -0.030</td> <td>   -0.025</td>
</tr>
<tr>
  <th>C(country_standard)[T.Niue]</th>                             <td>   -0.0296</td> <td>    0.002</td> <td>  -16.281</td> <td> 0.000</td> <td>   -0.033</td> <td>   -0.026</td>
</tr>
<tr>
  <th>C(country_standard)[T.North Korea]</th>                      <td>   -0.0165</td> <td>    0.001</td> <td>  -15.723</td> <td> 0.000</td> <td>   -0.019</td> <td>   -0.014</td>
</tr>
<tr>
  <th>C(country_standard)[T.North Macedonia]</th>                  <td>    0.0129</td> <td>    0.002</td> <td>    7.589</td> <td> 0.000</td> <td>    0.010</td> <td>    0.016</td>
</tr>
<tr>
  <th>C(country_standard)[T.Northern Mariana Islands]</th>         <td>   -0.0303</td> <td>    0.002</td> <td>  -15.452</td> <td> 0.000</td> <td>   -0.034</td> <td>   -0.026</td>
</tr>
<tr>
  <th>C(country_standard)[T.Norway]</th>                           <td>    0.0238</td> <td>    0.004</td> <td>    6.364</td> <td> 0.000</td> <td>    0.017</td> <td>    0.031</td>
</tr>
<tr>
  <th>C(country_standard)[T.Oman]</th>                             <td>   -0.0209</td> <td> 9.25e-05</td> <td> -225.803</td> <td> 0.000</td> <td>   -0.021</td> <td>   -0.021</td>
</tr>
<tr>
  <th>C(country_standard)[T.Pakistan]</th>                         <td>    0.0055</td> <td>    0.000</td> <td>   21.969</td> <td> 0.000</td> <td>    0.005</td> <td>    0.006</td>
</tr>
<tr>
  <th>C(country_standard)[T.Palau]</th>                            <td>   -0.0386</td> <td>    0.004</td> <td>  -10.330</td> <td> 0.000</td> <td>   -0.046</td> <td>   -0.031</td>
</tr>
<tr>
  <th>C(country_standard)[T.Palestinian Territories]</th>          <td>   -0.0202</td> <td>    0.000</td> <td> -102.840</td> <td> 0.000</td> <td>   -0.021</td> <td>   -0.020</td>
</tr>
<tr>
  <th>C(country_standard)[T.Panama]</th>                           <td>    0.0057</td> <td>    0.003</td> <td>    1.739</td> <td> 0.082</td> <td>   -0.001</td> <td>    0.012</td>
</tr>
<tr>
  <th>C(country_standard)[T.Papua New Guinea]</th>                 <td>   -0.0293</td> <td>    0.002</td> <td>  -16.029</td> <td> 0.000</td> <td>   -0.033</td> <td>   -0.026</td>
</tr>
<tr>
  <th>C(country_standard)[T.Paraguay]</th>                         <td>    0.0126</td> <td>    0.002</td> <td>    7.035</td> <td> 0.000</td> <td>    0.009</td> <td>    0.016</td>
</tr>
<tr>
  <th>C(country_standard)[T.Peru]</th>                             <td>    0.0067</td> <td>    0.003</td> <td>    2.220</td> <td> 0.026</td> <td>    0.001</td> <td>    0.013</td>
</tr>
<tr>
  <th>C(country_standard)[T.Philippines]</th>                      <td>    0.0125</td> <td>    0.002</td> <td>    7.202</td> <td> 0.000</td> <td>    0.009</td> <td>    0.016</td>
</tr>
<tr>
  <th>C(country_standard)[T.Poland]</th>                           <td>   -0.0364</td> <td>    0.003</td> <td>  -11.278</td> <td> 0.000</td> <td>   -0.043</td> <td>   -0.030</td>
</tr>
<tr>
  <th>C(country_standard)[T.Portugal]</th>                         <td>   -0.0165</td> <td>    0.004</td> <td>   -4.701</td> <td> 0.000</td> <td>   -0.023</td> <td>   -0.010</td>
</tr>
<tr>
  <th>C(country_standard)[T.Puerto Rico]</th>                      <td>   -0.0349</td> <td>    0.003</td> <td>  -11.890</td> <td> 0.000</td> <td>   -0.041</td> <td>   -0.029</td>
</tr>
<tr>
  <th>C(country_standard)[T.Qatar]</th>                            <td>   -0.0207</td> <td>    0.000</td> <td> -166.201</td> <td> 0.000</td> <td>   -0.021</td> <td>   -0.020</td>
</tr>
<tr>
  <th>C(country_standard)[T.Romania]</th>                          <td>    0.0270</td> <td>    0.003</td> <td>    8.657</td> <td> 0.000</td> <td>    0.021</td> <td>    0.033</td>
</tr>
<tr>
  <th>C(country_standard)[T.Russia]</th>                           <td>   -0.0199</td> <td>    0.001</td> <td>  -36.359</td> <td> 0.000</td> <td>   -0.021</td> <td>   -0.019</td>
</tr>
<tr>
  <th>C(country_standard)[T.Rwanda]</th>                           <td>   -0.0208</td> <td> 8.99e-05</td> <td> -230.844</td> <td> 0.000</td> <td>   -0.021</td> <td>   -0.021</td>
</tr>
<tr>
  <th>C(country_standard)[T.Réunion]</th>                          <td>   -0.0321</td> <td>    0.002</td> <td>  -13.773</td> <td> 0.000</td> <td>   -0.037</td> <td>   -0.027</td>
</tr>
<tr>
  <th>C(country_standard)[T.Saint Martin (French part)]</th>       <td>   -0.0303</td> <td>    0.002</td> <td>  -15.436</td> <td> 0.000</td> <td>   -0.034</td> <td>   -0.026</td>
</tr>
<tr>
  <th>C(country_standard)[T.Samoa]</th>                            <td>   -0.0362</td> <td>    0.003</td> <td>  -10.895</td> <td> 0.000</td> <td>   -0.043</td> <td>   -0.030</td>
</tr>
<tr>
  <th>C(country_standard)[T.San Marino]</th>                       <td>   -0.0376</td> <td>    0.004</td> <td>  -10.374</td> <td> 0.000</td> <td>   -0.045</td> <td>   -0.031</td>
</tr>
<tr>
  <th>C(country_standard)[T.Saudi Arabia]</th>                     <td>   -0.0185</td> <td>    0.001</td> <td>  -28.014</td> <td> 0.000</td> <td>   -0.020</td> <td>   -0.017</td>
</tr>
<tr>
  <th>C(country_standard)[T.Senegal]</th>                          <td>   -0.0136</td> <td>    0.003</td> <td>   -4.633</td> <td> 0.000</td> <td>   -0.019</td> <td>   -0.008</td>
</tr>
<tr>
  <th>C(country_standard)[T.Serbia]</th>                           <td>    0.0082</td> <td>    0.003</td> <td>    3.019</td> <td> 0.003</td> <td>    0.003</td> <td>    0.013</td>
</tr>
<tr>
  <th>C(country_standard)[T.Seychelles]</th>                       <td>   -0.0295</td> <td>    0.002</td> <td>  -15.512</td> <td> 0.000</td> <td>   -0.033</td> <td>   -0.026</td>
</tr>
<tr>
  <th>C(country_standard)[T.Sierra Leone]</th>                     <td>    0.0126</td> <td>    0.002</td> <td>    7.098</td> <td> 0.000</td> <td>    0.009</td> <td>    0.016</td>
</tr>
<tr>
  <th>C(country_standard)[T.Singapore]</th>                        <td>   -0.0286</td> <td>    0.002</td> <td>  -17.774</td> <td> 0.000</td> <td>   -0.032</td> <td>   -0.025</td>
</tr>
<tr>
  <th>C(country_standard)[T.Sint Maarten]</th>                     <td>   -0.0303</td> <td>    0.002</td> <td>  -15.436</td> <td> 0.000</td> <td>   -0.034</td> <td>   -0.026</td>
</tr>
<tr>
  <th>C(country_standard)[T.Slovakia]</th>                         <td>    0.0049</td> <td>    0.003</td> <td>    1.441</td> <td> 0.150</td> <td>   -0.002</td> <td>    0.012</td>
</tr>
<tr>
  <th>C(country_standard)[T.Slovenia]</th>                         <td>   -0.0370</td> <td>    0.003</td> <td>  -10.793</td> <td> 0.000</td> <td>   -0.044</td> <td>   -0.030</td>
</tr>
<tr>
  <th>C(country_standard)[T.Solomon Islands]</th>                  <td>   -0.0327</td> <td>    0.003</td> <td>  -12.824</td> <td> 0.000</td> <td>   -0.038</td> <td>   -0.028</td>
</tr>
<tr>
  <th>C(country_standard)[T.Somalia]</th>                          <td>   -0.0182</td> <td>    0.001</td> <td>  -26.522</td> <td> 0.000</td> <td>   -0.020</td> <td>   -0.017</td>
</tr>
<tr>
  <th>C(country_standard)[T.South Africa]</th>                     <td>    0.0263</td> <td>    0.003</td> <td>    7.845</td> <td> 0.000</td> <td>    0.020</td> <td>    0.033</td>
</tr>
<tr>
  <th>C(country_standard)[T.South Korea]</th>                      <td>    0.0256</td> <td>    0.002</td> <td>   11.838</td> <td> 0.000</td> <td>    0.021</td> <td>    0.030</td>
</tr>
<tr>
  <th>C(country_standard)[T.South Sudan]</th>                      <td>   -0.0193</td> <td>    0.000</td> <td>  -59.394</td> <td> 0.000</td> <td>   -0.020</td> <td>   -0.019</td>
</tr>
<tr>
  <th>C(country_standard)[T.Spain]</th>                            <td>    0.0053</td> <td>    0.003</td> <td>    1.728</td> <td> 0.084</td> <td>   -0.001</td> <td>    0.011</td>
</tr>
<tr>
  <th>C(country_standard)[T.Sri Lanka]</th>                        <td>    0.0134</td> <td>    0.002</td> <td>    8.376</td> <td> 0.000</td> <td>    0.010</td> <td>    0.017</td>
</tr>
<tr>
  <th>C(country_standard)[T.St. Helena]</th>                       <td>   -0.0298</td> <td>    0.002</td> <td>  -16.245</td> <td> 0.000</td> <td>   -0.033</td> <td>   -0.026</td>
</tr>
<tr>
  <th>C(country_standard)[T.St. Kitts & Nevis]</th>                <td>   -0.0371</td> <td>    0.004</td> <td>  -10.554</td> <td> 0.000</td> <td>   -0.044</td> <td>   -0.030</td>
</tr>
<tr>
  <th>C(country_standard)[T.St. Lucia]</th>                        <td>   -0.0374</td> <td>    0.004</td> <td>  -10.345</td> <td> 0.000</td> <td>   -0.044</td> <td>   -0.030</td>
</tr>
<tr>
  <th>C(country_standard)[T.St. Vincent & Grenadines]</th>         <td>   -0.0372</td> <td>    0.004</td> <td>  -10.557</td> <td> 0.000</td> <td>   -0.044</td> <td>   -0.030</td>
</tr>
<tr>
  <th>C(country_standard)[T.Sudan]</th>                            <td>   -0.0190</td> <td>    0.000</td> <td>  -42.126</td> <td> 0.000</td> <td>   -0.020</td> <td>   -0.018</td>
</tr>
<tr>
  <th>C(country_standard)[T.Suriname]</th>                         <td>   -0.0355</td> <td>    0.003</td> <td>  -11.195</td> <td> 0.000</td> <td>   -0.042</td> <td>   -0.029</td>
</tr>
<tr>
  <th>C(country_standard)[T.Sweden]</th>                           <td>    0.0242</td> <td>    0.004</td> <td>    6.596</td> <td> 0.000</td> <td>    0.017</td> <td>    0.031</td>
</tr>
<tr>
  <th>C(country_standard)[T.Switzerland]</th>                      <td>   -0.0382</td> <td>    0.004</td> <td>  -10.609</td> <td> 0.000</td> <td>   -0.045</td> <td>   -0.031</td>
</tr>
<tr>
  <th>C(country_standard)[T.Syria]</th>                            <td>   -0.0183</td> <td>    0.001</td> <td>  -28.138</td> <td> 0.000</td> <td>   -0.020</td> <td>   -0.017</td>
</tr>
<tr>
  <th>C(country_standard)[T.São Tomé & Príncipe]</th>              <td>   -0.0353</td> <td>    0.003</td> <td>  -11.048</td> <td> 0.000</td> <td>   -0.042</td> <td>   -0.029</td>
</tr>
<tr>
  <th>C(country_standard)[T.Taiwan]</th>                           <td>   -0.0367</td> <td>    0.003</td> <td>  -11.001</td> <td> 0.000</td> <td>   -0.043</td> <td>   -0.030</td>
</tr>
<tr>
  <th>C(country_standard)[T.Tajikistan]</th>                       <td>   -0.0194</td> <td>    0.000</td> <td>  -54.549</td> <td> 0.000</td> <td>   -0.020</td> <td>   -0.019</td>
</tr>
<tr>
  <th>C(country_standard)[T.Tanzania]</th>                         <td>   -0.0284</td> <td>    0.002</td> <td>  -17.474</td> <td> 0.000</td> <td>   -0.032</td> <td>   -0.025</td>
</tr>
<tr>
  <th>C(country_standard)[T.Thailand]</th>                         <td>   -0.0233</td> <td>    0.000</td> <td>  -56.425</td> <td> 0.000</td> <td>   -0.024</td> <td>   -0.022</td>
</tr>
<tr>
  <th>C(country_standard)[T.Timor-Leste]</th>                      <td>   -0.0317</td> <td>    0.002</td> <td>  -13.772</td> <td> 0.000</td> <td>   -0.036</td> <td>   -0.027</td>
</tr>
<tr>
  <th>C(country_standard)[T.Togo]</th>                             <td>   -0.0272</td> <td>    0.001</td> <td>  -19.715</td> <td> 0.000</td> <td>   -0.030</td> <td>   -0.024</td>
</tr>
<tr>
  <th>C(country_standard)[T.Tonga]</th>                            <td>   -0.0368</td> <td>    0.003</td> <td>  -11.011</td> <td> 0.000</td> <td>   -0.043</td> <td>   -0.030</td>
</tr>
<tr>
  <th>C(country_standard)[T.Trinidad & Tobago]</th>                <td>   -0.0150</td> <td>    0.003</td> <td>   -4.627</td> <td> 0.000</td> <td>   -0.021</td> <td>   -0.009</td>
</tr>
<tr>
  <th>C(country_standard)[T.Tunisia]</th>                          <td>    0.0083</td> <td>    0.003</td> <td>    3.065</td> <td> 0.002</td> <td>    0.003</td> <td>    0.014</td>
</tr>
<tr>
  <th>C(country_standard)[T.Turkey]</th>                           <td>   -0.0253</td> <td>    0.001</td> <td>  -30.536</td> <td> 0.000</td> <td>   -0.027</td> <td>   -0.024</td>
</tr>
<tr>
  <th>C(country_standard)[T.Turkmenistan]</th>                     <td>   -0.0171</td> <td>    0.001</td> <td>  -20.348</td> <td> 0.000</td> <td>   -0.019</td> <td>   -0.015</td>
</tr>
<tr>
  <th>C(country_standard)[T.Turks & Caicos Islands]</th>           <td>   -0.0298</td> <td>    0.002</td> <td>  -16.201</td> <td> 0.000</td> <td>   -0.033</td> <td>   -0.026</td>
</tr>
<tr>
  <th>C(country_standard)[T.Tuvalu]</th>                           <td>   -0.0381</td> <td>    0.004</td> <td>  -10.524</td> <td> 0.000</td> <td>   -0.045</td> <td>   -0.031</td>
</tr>
<tr>
  <th>C(country_standard)[T.U.S. Virgin Islands]</th>              <td>   -0.0314</td> <td>    0.002</td> <td>  -14.325</td> <td> 0.000</td> <td>   -0.036</td> <td>   -0.027</td>
</tr>
<tr>
  <th>C(country_standard)[T.Uganda]</th>                           <td>   -0.0242</td> <td>    0.001</td> <td>  -33.088</td> <td> 0.000</td> <td>   -0.026</td> <td>   -0.023</td>
</tr>
<tr>
  <th>C(country_standard)[T.Ukraine]</th>                          <td>    0.0130</td> <td>    0.002</td> <td>    8.095</td> <td> 0.000</td> <td>    0.010</td> <td>    0.016</td>
</tr>
<tr>
  <th>C(country_standard)[T.United Arab Emirates]</th>             <td>   -0.0206</td> <td>    0.000</td> <td> -108.328</td> <td> 0.000</td> <td>   -0.021</td> <td>   -0.020</td>
</tr>
<tr>
  <th>C(country_standard)[T.United Kingdom]</th>                   <td>    0.0264</td> <td>    0.003</td> <td>    9.446</td> <td> 0.000</td> <td>    0.021</td> <td>    0.032</td>
</tr>
<tr>
  <th>C(country_standard)[T.United States]</th>                    <td>    0.0388</td> <td>    0.003</td> <td>   11.773</td> <td> 0.000</td> <td>    0.032</td> <td>    0.045</td>
</tr>
<tr>
  <th>C(country_standard)[T.Uruguay]</th>                          <td>    0.0041</td> <td>    0.004</td> <td>    1.146</td> <td> 0.252</td> <td>   -0.003</td> <td>    0.011</td>
</tr>
<tr>
  <th>C(country_standard)[T.Uzbekistan]</th>                       <td>   -0.0188</td> <td>    0.000</td> <td>  -39.615</td> <td> 0.000</td> <td>   -0.020</td> <td>   -0.018</td>
</tr>
<tr>
  <th>C(country_standard)[T.Vanuatu]</th>                          <td>   -0.0360</td> <td>    0.003</td> <td>  -10.995</td> <td> 0.000</td> <td>   -0.042</td> <td>   -0.030</td>
</tr>
<tr>
  <th>C(country_standard)[T.Vatican City]</th>                     <td>   -0.0299</td> <td>    0.002</td> <td>  -15.966</td> <td> 0.000</td> <td>   -0.034</td> <td>   -0.026</td>
</tr>
<tr>
  <th>C(country_standard)[T.Venezuela]</th>                        <td>   -0.0232</td> <td>    0.000</td> <td>  -52.706</td> <td> 0.000</td> <td>   -0.024</td> <td>   -0.022</td>
</tr>
<tr>
  <th>C(country_standard)[T.Vietnam]</th>                          <td>   -0.0199</td> <td>    0.000</td> <td>  -67.692</td> <td> 0.000</td> <td>   -0.020</td> <td>   -0.019</td>
</tr>
<tr>
  <th>C(country_standard)[T.Western Sahara]</th>                   <td>   -0.0048</td> <td>    0.004</td> <td>   -1.339</td> <td> 0.181</td> <td>   -0.012</td> <td>    0.002</td>
</tr>
<tr>
  <th>C(country_standard)[T.Yemen]</th>                            <td>   -0.0196</td> <td>    0.000</td> <td>  -63.932</td> <td> 0.000</td> <td>   -0.020</td> <td>   -0.019</td>
</tr>
<tr>
  <th>C(country_standard)[T.Zambia]</th>                           <td>   -0.0287</td> <td>    0.002</td> <td>  -16.287</td> <td> 0.000</td> <td>   -0.032</td> <td>   -0.025</td>
</tr>
<tr>
  <th>C(country_standard)[T.Zimbabwe]</th>                         <td>   -0.0232</td> <td>    0.001</td> <td>  -44.536</td> <td> 0.000</td> <td>   -0.024</td> <td>   -0.022</td>
</tr>
<tr>
  <th>transparency_index</th>                                      <td>    0.0431</td> <td>    0.009</td> <td>    4.618</td> <td> 0.000</td> <td>    0.025</td> <td>    0.061</td>
</tr>
<tr>
  <th>gdp</th>                                                     <td> -6.37e-16</td> <td> 3.22e-16</td> <td>   -1.977</td> <td> 0.048</td> <td>-1.27e-15</td> <td>-5.34e-18</td>
</tr>
<tr>
  <th>gini</th>                                                    <td> 1.024e-05</td> <td> 5.73e-06</td> <td>    1.788</td> <td> 0.074</td> <td>-9.82e-07</td> <td> 2.15e-05</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>4504.157</td> <th>  Durbin-Watson:     </th>  <td>   2.047</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>451409.936</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 7.006</td>  <th>  Prob(JB):          </th>  <td>    0.00</td> 
</tr>
<tr>
  <th>Kurtosis:</th>       <td>56.015</td>  <th>  Cond. No.          </th>  <td>9.45e+26</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors are robust to cluster correlation (cluster)<br/>[2] The smallest eigenvalue is 1.39e-26. This might indicate that there are<br/>strong multicollinearity problems or that the design matrix is singular.




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
      <td>0.580426</td>
      <td>0.0</td>
      <td>0.652288</td>
      <td>0.409391</td>
      <td>0.568511</td>
      <td>0.520235</td>
      <td>0.0</td>
      <td>0.214286</td>
      <td>0.012655</td>
      <td>1.929110e+10</td>
      <td>38.076323</td>
      <td>2.578007</td>
      <td>18.60000</td>
      <td>2.581000</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Albania</td>
      <td>2020</td>
      <td>0.580426</td>
      <td>0.0</td>
      <td>0.652288</td>
      <td>0.409391</td>
      <td>0.568511</td>
      <td>0.520235</td>
      <td>0.0</td>
      <td>0.357143</td>
      <td>0.525529</td>
      <td>1.527918e+10</td>
      <td>33.200000</td>
      <td>148.436569</td>
      <td>38.00000</td>
      <td>13.188000</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Algeria</td>
      <td>2020</td>
      <td>0.580426</td>
      <td>0.0</td>
      <td>0.652288</td>
      <td>0.409391</td>
      <td>0.568511</td>
      <td>0.520235</td>
      <td>0.0</td>
      <td>0.428571</td>
      <td>0.011143</td>
      <td>1.710913e+11</td>
      <td>27.600000</td>
      <td>168.449661</td>
      <td>29.10000</td>
      <td>6.211000</td>
    </tr>
    <tr>
      <th>76</th>
      <td>Andorra</td>
      <td>2020</td>
      <td>0.580426</td>
      <td>0.0</td>
      <td>0.652288</td>
      <td>0.409391</td>
      <td>0.568511</td>
      <td>0.520235</td>
      <td>0.0</td>
      <td>0.337613</td>
      <td>0.924444</td>
      <td>3.154058e+09</td>
      <td>38.076323</td>
      <td>1916.984497</td>
      <td>30.30687</td>
      <td>8.627504</td>
    </tr>
    <tr>
      <th>92</th>
      <td>Angola</td>
      <td>2020</td>
      <td>0.580426</td>
      <td>0.0</td>
      <td>0.652288</td>
      <td>0.409391</td>
      <td>0.568511</td>
      <td>0.520235</td>
      <td>0.0</td>
      <td>0.357143</td>
      <td>0.019782</td>
      <td>8.881570e+10</td>
      <td>51.300000</td>
      <td>36.737221</td>
      <td>16.80000</td>
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
      <td>...</td>
    </tr>
    <tr>
      <th>3529</th>
      <td>Venezuela</td>
      <td>2020</td>
      <td>0.580426</td>
      <td>0.0</td>
      <td>0.652288</td>
      <td>0.409391</td>
      <td>0.568511</td>
      <td>0.520235</td>
      <td>0.0</td>
      <td>0.928571</td>
      <td>0.010997</td>
      <td>4.823593e+11</td>
      <td>46.900000</td>
      <td>122.942413</td>
      <td>29.00000</td>
      <td>6.614000</td>
    </tr>
    <tr>
      <th>3545</th>
      <td>Vietnam</td>
      <td>2020</td>
      <td>0.580426</td>
      <td>0.0</td>
      <td>0.652288</td>
      <td>0.811616</td>
      <td>0.568511</td>
      <td>0.520235</td>
      <td>0.0</td>
      <td>0.214286</td>
      <td>0.293635</td>
      <td>2.619212e+11</td>
      <td>35.700000</td>
      <td>69.108612</td>
      <td>32.60000</td>
      <td>7.150000</td>
    </tr>
    <tr>
      <th>3569</th>
      <td>Yemen</td>
      <td>2020</td>
      <td>0.580426</td>
      <td>0.0</td>
      <td>0.652288</td>
      <td>0.409391</td>
      <td>0.568511</td>
      <td>0.520235</td>
      <td>0.0</td>
      <td>0.337613</td>
      <td>0.028759</td>
      <td>2.258108e+10</td>
      <td>36.700000</td>
      <td>7.451180</td>
      <td>20.30000</td>
      <td>2.922000</td>
    </tr>
    <tr>
      <th>3585</th>
      <td>Zambia</td>
      <td>2020</td>
      <td>0.580426</td>
      <td>0.0</td>
      <td>0.652288</td>
      <td>0.409391</td>
      <td>0.568511</td>
      <td>0.520235</td>
      <td>0.0</td>
      <td>0.428571</td>
      <td>0.399771</td>
      <td>2.330977e+10</td>
      <td>57.100000</td>
      <td>29.700403</td>
      <td>17.70000</td>
      <td>2.480000</td>
    </tr>
    <tr>
      <th>3601</th>
      <td>Zimbabwe</td>
      <td>2020</td>
      <td>0.580426</td>
      <td>0.0</td>
      <td>0.652288</td>
      <td>0.402851</td>
      <td>0.568511</td>
      <td>0.520235</td>
      <td>0.0</td>
      <td>0.500000</td>
      <td>0.271321</td>
      <td>2.144076e+10</td>
      <td>44.300000</td>
      <td>39.249222</td>
      <td>19.60000</td>
      <td>2.822000</td>
    </tr>
  </tbody>
</table>
<p>193 rows × 16 columns</p>
</div>




```python
#COVID outcomes model
reg_covid = smf.ols('covid_index ~ transparency_index + gdp + gini + percap_domestic_health_expenditure + median_age + aged_65_older', df_2020).fit()
```


```python
reg_covid.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>covid_index</td>   <th>  R-squared:         </th> <td>   0.372</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.356</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   22.19</td>
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 05 Mar 2021</td> <th>  Prob (F-statistic):</th> <td>1.98e-17</td>
</tr>
<tr>
  <th>Time:</th>                 <td>23:13:58</td>     <th>  Log-Likelihood:    </th> <td> -2.5796</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   193</td>      <th>  AIC:               </th> <td>   17.16</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   187</td>      <th>  BIC:               </th> <td>   36.74</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     5</td>      <th>                     </th>     <td> </td>   
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
  <th>Intercept</th>                          <td>   -0.1241</td> <td>    0.125</td> <td>   -0.996</td> <td> 0.320</td> <td>   -0.370</td> <td>    0.122</td>
</tr>
<tr>
  <th>transparency_index</th>                 <td>   -0.0720</td> <td>    0.072</td> <td>   -0.996</td> <td> 0.320</td> <td>   -0.215</td> <td>    0.071</td>
</tr>
<tr>
  <th>gdp</th>                                <td> 3.512e-15</td> <td> 9.91e-15</td> <td>    0.354</td> <td> 0.724</td> <td> -1.6e-14</td> <td> 2.31e-14</td>
</tr>
<tr>
  <th>gini</th>                               <td>   -0.0013</td> <td>    0.003</td> <td>   -0.433</td> <td> 0.665</td> <td>   -0.007</td> <td>    0.005</td>
</tr>
<tr>
  <th>percap_domestic_health_expenditure</th> <td> 5.507e-05</td> <td> 1.88e-05</td> <td>    2.936</td> <td> 0.004</td> <td> 1.81e-05</td> <td> 9.21e-05</td>
</tr>
<tr>
  <th>median_age</th>                         <td>    0.0193</td> <td>    0.005</td> <td>    3.955</td> <td> 0.000</td> <td>    0.010</td> <td>    0.029</td>
</tr>
<tr>
  <th>aged_65_older</th>                      <td>   -0.0068</td> <td>    0.008</td> <td>   -0.884</td> <td> 0.378</td> <td>   -0.022</td> <td>    0.008</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 3.923</td> <th>  Durbin-Watson:     </th> <td>   2.008</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.141</td> <th>  Jarque-Bera (JB):  </th> <td>   3.992</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.340</td> <th>  Prob(JB):          </th> <td>   0.136</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.818</td> <th>  Cond. No.          </th> <td>9.30e+18</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The smallest eigenvalue is 8.8e-12. This might indicate that there are<br/>strong multicollinearity problems or that the design matrix is singular.




```python
reg_pandemic_violations = smf.ols('pandemic_dem_violation_index ~ transparency_index + gdp + gini + percap_domestic_health_expenditure + median_age + aged_65_older', df_2020).fit()
```


```python
reg_pandemic_violations.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>    <td>pandemic_dem_violation_index</td> <th>  R-squared:         </th> <td>   0.220</td>
</tr>
<tr>
  <th>Model:</th>                         <td>OLS</td>             <th>  Adj. R-squared:    </th> <td>   0.199</td>
</tr>
<tr>
  <th>Method:</th>                   <td>Least Squares</td>        <th>  F-statistic:       </th> <td>   10.52</td>
</tr>
<tr>
  <th>Date:</th>                   <td>Fri, 05 Mar 2021</td>       <th>  Prob (F-statistic):</th> <td>6.49e-09</td>
</tr>
<tr>
  <th>Time:</th>                       <td>23:14:12</td>           <th>  Log-Likelihood:    </th> <td>  85.670</td>
</tr>
<tr>
  <th>No. Observations:</th>            <td>   193</td>            <th>  AIC:               </th> <td>  -159.3</td>
</tr>
<tr>
  <th>Df Residuals:</th>                <td>   187</td>            <th>  BIC:               </th> <td>  -139.8</td>
</tr>
<tr>
  <th>Df Model:</th>                    <td>     5</td>            <th>                     </th>     <td> </td>   
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
  <th>Intercept</th>                          <td>    0.3205</td> <td>    0.079</td> <td>    4.063</td> <td> 0.000</td> <td>    0.165</td> <td>    0.476</td>
</tr>
<tr>
  <th>transparency_index</th>                 <td>    0.1860</td> <td>    0.046</td> <td>    4.063</td> <td> 0.000</td> <td>    0.096</td> <td>    0.276</td>
</tr>
<tr>
  <th>gdp</th>                                <td> 1.543e-14</td> <td> 6.27e-15</td> <td>    2.460</td> <td> 0.015</td> <td> 3.06e-15</td> <td> 2.78e-14</td>
</tr>
<tr>
  <th>gini</th>                               <td>   -0.0014</td> <td>    0.002</td> <td>   -0.722</td> <td> 0.471</td> <td>   -0.005</td> <td>    0.002</td>
</tr>
<tr>
  <th>percap_domestic_health_expenditure</th> <td>-5.337e-05</td> <td> 1.19e-05</td> <td>   -4.494</td> <td> 0.000</td> <td>-7.68e-05</td> <td>-2.99e-05</td>
</tr>
<tr>
  <th>median_age</th>                         <td>    0.0020</td> <td>    0.003</td> <td>    0.645</td> <td> 0.519</td> <td>   -0.004</td> <td>    0.008</td>
</tr>
<tr>
  <th>aged_65_older</th>                      <td>   -0.0076</td> <td>    0.005</td> <td>   -1.555</td> <td> 0.122</td> <td>   -0.017</td> <td>    0.002</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>46.710</td> <th>  Durbin-Watson:     </th> <td>   2.050</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td> 107.507</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 1.081</td> <th>  Prob(JB):          </th> <td>4.52e-24</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 5.949</td> <th>  Cond. No.          </th> <td>9.30e+18</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The smallest eigenvalue is 8.8e-12. This might indicate that there are<br/>strong multicollinearity problems or that the design matrix is singular.


