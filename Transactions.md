

```python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

```


```python
!pip install py7zr
import py7zr
```

    Collecting py7zr
      Downloading py7zr-0.13.0-py3-none-any.whl (65 kB)
    Collecting pycryptodome
      Downloading pycryptodome-3.10.1-cp35-abi3-win_amd64.whl (1.6 MB)
    Collecting texttable
      Downloading texttable-1.6.3-py2.py3-none-any.whl (10 kB)
    Collecting ppmd-cffi<0.4.0,>=0.3.1
      Downloading ppmd_cffi-0.3.3-cp38-cp38-win_amd64.whl (44 kB)
    Collecting zstandard
      Downloading zstandard-0.15.1-cp38-cp38-win_amd64.whl (582 kB)
    Collecting multivolumefile<0.2.0,>=0.1.1
      Downloading multivolumefile-0.1.3-py3-none-any.whl (15 kB)
    Requirement already satisfied: cffi>=1.14.0 in c:\users\pc\anaconda3\lib\site-packages (from ppmd-cffi<0.4.0,>=0.3.1->py7zr) (1.14.0)
    Requirement already satisfied: pycparser in c:\users\pc\anaconda3\lib\site-packages (from cffi>=1.14.0->ppmd-cffi<0.4.0,>=0.3.1->py7zr) (2.20)
    Installing collected packages: pycryptodome, texttable, ppmd-cffi, zstandard, multivolumefile, py7zr
    Successfully installed multivolumefile-0.1.3 ppmd-cffi-0.3.3 py7zr-0.13.0 pycryptodome-3.10.1 texttable-1.6.3 zstandard-0.15.1
    


```python
df_trans = pd.read_csv('transactions.csv')
```


```python
df_trans.shape
```




    (83488, 3)




```python
df_trans.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 83488 entries, 0 to 83487
    Data columns (total 3 columns):
     #   Column        Non-Null Count  Dtype 
    ---  ------        --------------  ----- 
     0   date          83488 non-null  object
     1   store_nbr     83488 non-null  int64 
     2   transactions  83488 non-null  int64 
    dtypes: int64(2), object(1)
    memory usage: 1.9+ MB
    


```python
df_trans.head()
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
      <th>date</th>
      <th>store_nbr</th>
      <th>transactions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2013-01-01</td>
      <td>25</td>
      <td>770</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2013-01-02</td>
      <td>1</td>
      <td>2111</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2013-01-02</td>
      <td>2</td>
      <td>2358</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2013-01-02</td>
      <td>3</td>
      <td>3487</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2013-01-02</td>
      <td>4</td>
      <td>1922</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_trans.tail()
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
      <th>date</th>
      <th>store_nbr</th>
      <th>transactions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>83483</th>
      <td>2017-08-15</td>
      <td>50</td>
      <td>2804</td>
    </tr>
    <tr>
      <th>83484</th>
      <td>2017-08-15</td>
      <td>51</td>
      <td>1573</td>
    </tr>
    <tr>
      <th>83485</th>
      <td>2017-08-15</td>
      <td>52</td>
      <td>2255</td>
    </tr>
    <tr>
      <th>83486</th>
      <td>2017-08-15</td>
      <td>53</td>
      <td>932</td>
    </tr>
    <tr>
      <th>83487</th>
      <td>2017-08-15</td>
      <td>54</td>
      <td>802</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_trans['transactions'].describe()
```




    count    83488.000000
    mean      1694.602158
    std        963.286644
    min          5.000000
    25%       1046.000000
    50%       1393.000000
    75%       2079.000000
    max       8359.000000
    Name: transactions, dtype: float64




```python
trans_columns = df_trans.columns.tolist()

for i in range(0, len(trans_columns)):
    print("***",trans_columns[i],"***")
    print(df_trans[trans_columns[i]].nunique(),'개')
    print(df_trans[trans_columns[i]].value_counts(normalize=False))
```

    *** date ***
    1682 개
    2017-06-04    54
    2017-08-02    54
    2017-07-11    54
    2017-07-03    54
    2017-07-15    54
                  ..
    2016-01-04    14
    2014-01-01     2
    2015-01-01     1
    2013-01-01     1
    2017-01-01     1
    Name: date, Length: 1682, dtype: int64
    *** store_nbr ***
    54 개
    38    1678
    26    1678
    31    1678
    33    1678
    34    1678
    37    1678
    39    1678
    41    1677
    23    1677
    28    1677
    32    1677
    5     1677
    40    1677
    27    1677
    49    1677
    47    1677
    2     1677
    51    1677
    50    1677
    48    1677
    16    1677
    46    1677
    45    1677
    44    1677
    6     1676
    3     1676
    8     1676
    9     1676
    11    1676
    13    1676
    15    1676
    19    1676
    4     1676
    54    1676
    1     1676
    35    1676
    7     1675
    10    1675
    17    1674
    43    1672
    30    1655
    14    1638
    12    1616
    25    1615
    24    1577
    18    1566
    36    1551
    53    1167
    20     909
    29     874
    21     748
    42     720
    22     671
    52     118
    Name: store_nbr, dtype: int64
    *** transactions ***
    4993 개
    1207    90
    1200    86
    1304    81
    1296    80
    1282    80
            ..
    5380     1
    5722     1
    5556     1
    5316     1
    4005     1
    Name: transactions, Length: 4993, dtype: int64
    


```python
import seaborn as sns 
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
#Tổng quy mô khối lượng công việc theo cửa hàng
amount = (df_trans.groupby(['store_nbr']).sum())
fig, ax = plt.subplots()
fig.set_size_inches(12, 8)
ax = sns.barplot(x = amount.index, y= "transactions", data = amount)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 75, fontsize = 9)
```




    [Text(0, 0, '1'),
     Text(0, 0, '2'),
     Text(0, 0, '3'),
     Text(0, 0, '4'),
     Text(0, 0, '5'),
     Text(0, 0, '6'),
     Text(0, 0, '7'),
     Text(0, 0, '8'),
     Text(0, 0, '9'),
     Text(0, 0, '10'),
     Text(0, 0, '11'),
     Text(0, 0, '12'),
     Text(0, 0, '13'),
     Text(0, 0, '14'),
     Text(0, 0, '15'),
     Text(0, 0, '16'),
     Text(0, 0, '17'),
     Text(0, 0, '18'),
     Text(0, 0, '19'),
     Text(0, 0, '20'),
     Text(0, 0, '21'),
     Text(0, 0, '22'),
     Text(0, 0, '23'),
     Text(0, 0, '24'),
     Text(0, 0, '25'),
     Text(0, 0, '26'),
     Text(0, 0, '27'),
     Text(0, 0, '28'),
     Text(0, 0, '29'),
     Text(0, 0, '30'),
     Text(0, 0, '31'),
     Text(0, 0, '32'),
     Text(0, 0, '33'),
     Text(0, 0, '34'),
     Text(0, 0, '35'),
     Text(0, 0, '36'),
     Text(0, 0, '37'),
     Text(0, 0, '38'),
     Text(0, 0, '39'),
     Text(0, 0, '40'),
     Text(0, 0, '41'),
     Text(0, 0, '42'),
     Text(0, 0, '43'),
     Text(0, 0, '44'),
     Text(0, 0, '45'),
     Text(0, 0, '46'),
     Text(0, 0, '47'),
     Text(0, 0, '48'),
     Text(0, 0, '49'),
     Text(0, 0, '50'),
     Text(0, 0, '51'),
     Text(0, 0, '52'),
     Text(0, 0, '53'),
     Text(0, 0, '54')]




![png](output_10_1.png)



```python

```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-19-598f233fba87> in <module>
    ----> 1 amount_trans = pd.merge(amount, df_stores, left_on='store_nbr', right_on='store_nbr', how='left')
          2 amount_trans.sort_values(by=['transactions'], ascending=False)
    

    NameError: name 'df_stores' is not defined



```python

```