
#import plots
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib



```python
#load up data using pandas
items = pd.read_csv('items.csv')

```


```python
#check data first and last
items.head()
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
      <th>item_nbr</th>
      <th>family</th>
      <th>class</th>
      <th>perishable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>96995</td>
      <td>GROCERY I</td>
      <td>1093</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>99197</td>
      <td>GROCERY I</td>
      <td>1067</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>103501</td>
      <td>CLEANING</td>
      <td>3008</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>103520</td>
      <td>GROCERY I</td>
      <td>1028</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>103665</td>
      <td>BREAD/BAKERY</td>
      <td>2712</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
items.tail()
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
      <th>item_nbr</th>
      <th>family</th>
      <th>class</th>
      <th>perishable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4095</th>
      <td>2132318</td>
      <td>GROCERY I</td>
      <td>1002</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4096</th>
      <td>2132945</td>
      <td>GROCERY I</td>
      <td>1026</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4097</th>
      <td>2132957</td>
      <td>GROCERY I</td>
      <td>1068</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4098</th>
      <td>2134058</td>
      <td>BEVERAGES</td>
      <td>1124</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4099</th>
      <td>2134244</td>
      <td>LIQUOR,WINE,BEER</td>
      <td>1364</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Describe the data
items.describe()
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
      <th>item_nbr</th>
      <th>class</th>
      <th>perishable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4.100000e+03</td>
      <td>4100.0000</td>
      <td>4100.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.251436e+06</td>
      <td>2169.6500</td>
      <td>0.240488</td>
    </tr>
    <tr>
      <th>std</th>
      <td>5.876872e+05</td>
      <td>1484.9109</td>
      <td>0.427432</td>
    </tr>
    <tr>
      <th>min</th>
      <td>9.699500e+04</td>
      <td>1002.0000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>8.181108e+05</td>
      <td>1068.0000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.306198e+06</td>
      <td>2004.0000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.904918e+06</td>
      <td>2990.5000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.134244e+06</td>
      <td>7780.0000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#check missing value null value
pd.isnull(items)
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
      <th>item_nbr</th>
      <th>family</th>
      <th>class</th>
      <th>perishable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4095</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4096</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4097</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4098</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4099</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>4100 rows × 4 columns</p>
</div>




```python
#Filling missing value use numpy
mean = np.mean(items)
#replace missing value with the mean
i = items.fillna(mean)
```


```python
#outlier

```


```python
# Distribution of various families of items

fig, x1 = plt.subplots()
fig.set_size_inches(12,8)
x1 = sns.countplot(y = "family", data = items)
```


![png](output_8_0.png)



```python
# Distribution of perishable goods by family

pf = pd.crosstab(items.family, items.perishable)
pf.plot.bar(figsize = (12, 7), stacked=True)
plt.legend(title='Perishable')
plt.show()
```


![png](output_9_0.png)



```python
# Distrbution of number of unique classes per family of items.

x20 = items.groupby(['family'])['class'].nunique()
x20
```




    family
    AUTOMOTIVE                     5
    BABY CARE                      1
    BEAUTY                         9
    BEVERAGES                     21
    BOOKS                          1
    BREAD/BAKERY                  15
    CELEBRATION                    4
    CLEANING                      26
    DAIRY                         22
    DELI                          13
    EGGS                           3
    FROZEN FOODS                  11
    GROCERY I                     67
    GROCERY II                     1
    HARDWARE                       1
    HOME AND KITCHEN I            20
    HOME AND KITCHEN II           12
    HOME APPLIANCES                1
    HOME CARE                      6
    LADIESWEAR                     1
    LAWN AND GARDEN                7
    LINGERIE                       3
    LIQUOR,WINE,BEER              19
    MAGAZINES                      4
    MEATS                          5
    PERSONAL CARE                 13
    PET SUPPLIES                   2
    PLAYERS AND ELECTRONICS        1
    POULTRY                        3
    PREPARED FOODS                10
    PRODUCE                       19
    SCHOOL AND OFFICE SUPPLIES     6
    SEAFOOD                        5
    Name: class, dtype: int64




```python
x21 = items.groupby(['family'])['class'].size()
x21
```




    family
    AUTOMOTIVE                      20
    BABY CARE                        1
    BEAUTY                          19
    BEVERAGES                      613
    BOOKS                            1
    BREAD/BAKERY                   134
    CELEBRATION                     31
    CLEANING                       446
    DAIRY                          242
    DELI                            91
    EGGS                            41
    FROZEN FOODS                    55
    GROCERY I                     1334
    GROCERY II                      14
    HARDWARE                         4
    HOME AND KITCHEN I              77
    HOME AND KITCHEN II             45
    HOME APPLIANCES                  1
    HOME CARE                      108
    LADIESWEAR                      21
    LAWN AND GARDEN                 26
    LINGERIE                        20
    LIQUOR,WINE,BEER                73
    MAGAZINES                        6
    MEATS                           84
    PERSONAL CARE                  153
    PET SUPPLIES                    14
    PLAYERS AND ELECTRONICS         17
    POULTRY                         54
    PREPARED FOODS                  26
    PRODUCE                        306
    SCHOOL AND OFFICE SUPPLIES      15
    SEAFOOD                          8
    Name: class, dtype: int64




```python
fig, ax = plt.subplots()
fig.set_size_inches(12, 6)
x20.plot.bar(color='skyblue')
plt.show()
```


![png](output_12_0.png)



```python

```
