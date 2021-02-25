```python
import numpy as np
import scipy.stats
import pandas as pd
```


```python
import matplotlib
import matplotlib.pyplot as pp

from IPython import display
from ipywidgets import interact, widgets

%matplotlib inline
```


```python
import re
import mailbox
import csv
```


```python
import numpy as np # linear algebra
import pandas as pd # data processing
from statsmodels.tsa.stattools import adfuller
```


```python
df_oil=pd.read_csv('oil.csv')
```


```python
df_oil.shape
```




    (1218, 2)



Dữ liệu giá dầu gồm có 1218 dòng, 2 cột


```python
df_oil.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1218 entries, 0 to 1217
    Data columns (total 2 columns):
    date          1218 non-null object
    dcoilwtico    1175 non-null float64
    dtypes: float64(1), object(1)
    memory usage: 19.1+ KB



```python
df_oil.head()
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
      <th>dcoilwtico</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1/1/2013</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1/2/2013</td>
      <td>93.14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1/3/2013</td>
      <td>92.97</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1/4/2013</td>
      <td>93.12</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1/7/2013</td>
      <td>93.20</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_oil.tail()
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
      <th>dcoilwtico</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1213</th>
      <td>8/25/2017</td>
      <td>47.65</td>
    </tr>
    <tr>
      <th>1214</th>
      <td>8/28/2017</td>
      <td>46.40</td>
    </tr>
    <tr>
      <th>1215</th>
      <td>8/29/2017</td>
      <td>46.46</td>
    </tr>
    <tr>
      <th>1216</th>
      <td>8/30/2017</td>
      <td>45.96</td>
    </tr>
    <tr>
      <th>1217</th>
      <td>8/31/2017</td>
      <td>47.26</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_oil[df_oil['dcoilwtico'] == df_oil['dcoilwtico'].max()]
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
      <th>dcoilwtico</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>178</th>
      <td>9/6/2013</td>
      <td>110.62</td>
    </tr>
  </tbody>
</table>
</div>



Giá dầu cao nhất là 110.62 đô/thùng vào ngày 6 tháng 9 năm 2013


```python
df_oil[df_oil['dcoilwtico'] == df_oil['dcoilwtico'].min()]
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
      <th>dcoilwtico</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>812</th>
      <td>2/11/2016</td>
      <td>26.19</td>
    </tr>
  </tbody>
</table>
</div>



Giá dầu thấp nhất là 26.19 đô/thùng vào ngày 11 tháng 2 năm 2016


```python
df_oil.dcoilwtico.plot(kind='box')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x201a6dc10b8>




![png](output_14_1.png)



```python
df_oil['dcoilwtico'].describe()
```




    count    1175.000000
    mean       67.714366
    std        25.630476
    min        26.190000
    25%        46.405000
    50%        53.190000
    75%        95.660000
    max       110.620000
    Name: dcoilwtico, dtype: float64



Trung bình có giá là 67.714366 đô/thùng dầu. Khoảng số phân tử: Q3-Q1= 53.19-46.405=6.785. Khoảng cách giữa giá trị lớn nhất và nhỏ nhất là 110.62-26.19=84.43. Nhìn chung, dữ liệu có dạng lệch phải, không xuất hiện giá trị ngoại lai. Giá dầu có độ biến động tương đối lớn, điều này có thể thấy ở biểu đồ được minh họa là giá dầu có sự biến động mạnh cụ thể là giảm sút rõ rệt vaof nửa cuối năm 2014,


```python
df_oil.plot.line("date","dcoilwtico")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x201a7b0f978>




![png](output_17_1.png)


Chúng ta có thể thấy rằng giá dầu có những biến động mạnh và dài hạn với sự sụt giảm rõ rệt vào nửa cuối năm 2014. Trên thực tế, mặc dù có một số biến động, giá dầu vẫn ở mức như hồi đầu năm 2015. Do đó, chúng ta có thể thấy sự thay đổi đáng kể trong doanh thu của các cửa hàng vào khoảng cuối năm 2014. Nhìn vào số liệu bán hàng của đơn vị, điều này không rõ ràng: mặc dù doanh số bán hàng có vẻ giảm vào đầu năm 2015, nhưng vào cuối năm 2014, chúng đang tăng lên. Ngoài ra, việc giảm giá dầu dường như không có bất kỳ tác động nào đến doanh số bán hàng, vì nó đã được nhìn thấy từ lô bán hàng, không có mối liên hệ giữa việc giảm giá dầu với doanh số bán hàng, vì vậy chúng ta có thể nói rằng đặc điểm hoặc dữ liệu này là không quan trọng đối với chúng tôi và sẽ không được xem xét trong quá trình lập mô hình.


```python
#Điền giá trị missing value
mean=np.mean(df_oil)
```


```python
ii=df_oil.fillna(mean)
```


```python
ii
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
      <th>dcoilwtico</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1/1/2013</td>
      <td>67.714366</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1/2/2013</td>
      <td>93.140000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1/3/2013</td>
      <td>92.970000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1/4/2013</td>
      <td>93.120000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1/7/2013</td>
      <td>93.200000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1/8/2013</td>
      <td>93.210000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1/9/2013</td>
      <td>93.080000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1/10/2013</td>
      <td>93.810000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1/11/2013</td>
      <td>93.600000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1/14/2013</td>
      <td>94.270000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1/15/2013</td>
      <td>93.260000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1/16/2013</td>
      <td>94.280000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1/17/2013</td>
      <td>95.490000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1/18/2013</td>
      <td>95.610000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1/21/2013</td>
      <td>67.714366</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1/22/2013</td>
      <td>96.090000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1/23/2013</td>
      <td>95.060000</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1/24/2013</td>
      <td>95.350000</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1/25/2013</td>
      <td>95.150000</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1/28/2013</td>
      <td>95.950000</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1/29/2013</td>
      <td>97.620000</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1/30/2013</td>
      <td>97.980000</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1/31/2013</td>
      <td>97.650000</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2/1/2013</td>
      <td>97.460000</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2/4/2013</td>
      <td>96.210000</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2/5/2013</td>
      <td>96.680000</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2/6/2013</td>
      <td>96.440000</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2/7/2013</td>
      <td>95.840000</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2/8/2013</td>
      <td>95.710000</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2/11/2013</td>
      <td>97.010000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1188</th>
      <td>7/21/2017</td>
      <td>45.780000</td>
    </tr>
    <tr>
      <th>1189</th>
      <td>7/24/2017</td>
      <td>46.210000</td>
    </tr>
    <tr>
      <th>1190</th>
      <td>7/25/2017</td>
      <td>47.770000</td>
    </tr>
    <tr>
      <th>1191</th>
      <td>7/26/2017</td>
      <td>48.580000</td>
    </tr>
    <tr>
      <th>1192</th>
      <td>7/27/2017</td>
      <td>49.050000</td>
    </tr>
    <tr>
      <th>1193</th>
      <td>7/28/2017</td>
      <td>49.720000</td>
    </tr>
    <tr>
      <th>1194</th>
      <td>7/31/2017</td>
      <td>50.210000</td>
    </tr>
    <tr>
      <th>1195</th>
      <td>8/1/2017</td>
      <td>49.190000</td>
    </tr>
    <tr>
      <th>1196</th>
      <td>8/2/2017</td>
      <td>49.600000</td>
    </tr>
    <tr>
      <th>1197</th>
      <td>8/3/2017</td>
      <td>49.030000</td>
    </tr>
    <tr>
      <th>1198</th>
      <td>8/4/2017</td>
      <td>49.570000</td>
    </tr>
    <tr>
      <th>1199</th>
      <td>8/7/2017</td>
      <td>49.370000</td>
    </tr>
    <tr>
      <th>1200</th>
      <td>8/8/2017</td>
      <td>49.070000</td>
    </tr>
    <tr>
      <th>1201</th>
      <td>8/9/2017</td>
      <td>49.590000</td>
    </tr>
    <tr>
      <th>1202</th>
      <td>8/10/2017</td>
      <td>48.540000</td>
    </tr>
    <tr>
      <th>1203</th>
      <td>8/11/2017</td>
      <td>48.810000</td>
    </tr>
    <tr>
      <th>1204</th>
      <td>8/14/2017</td>
      <td>47.590000</td>
    </tr>
    <tr>
      <th>1205</th>
      <td>8/15/2017</td>
      <td>47.570000</td>
    </tr>
    <tr>
      <th>1206</th>
      <td>8/16/2017</td>
      <td>46.800000</td>
    </tr>
    <tr>
      <th>1207</th>
      <td>8/17/2017</td>
      <td>47.070000</td>
    </tr>
    <tr>
      <th>1208</th>
      <td>8/18/2017</td>
      <td>48.590000</td>
    </tr>
    <tr>
      <th>1209</th>
      <td>8/21/2017</td>
      <td>47.390000</td>
    </tr>
    <tr>
      <th>1210</th>
      <td>8/22/2017</td>
      <td>47.650000</td>
    </tr>
    <tr>
      <th>1211</th>
      <td>8/23/2017</td>
      <td>48.450000</td>
    </tr>
    <tr>
      <th>1212</th>
      <td>8/24/2017</td>
      <td>47.240000</td>
    </tr>
    <tr>
      <th>1213</th>
      <td>8/25/2017</td>
      <td>47.650000</td>
    </tr>
    <tr>
      <th>1214</th>
      <td>8/28/2017</td>
      <td>46.400000</td>
    </tr>
    <tr>
      <th>1215</th>
      <td>8/29/2017</td>
      <td>46.460000</td>
    </tr>
    <tr>
      <th>1216</th>
      <td>8/30/2017</td>
      <td>45.960000</td>
    </tr>
    <tr>
      <th>1217</th>
      <td>8/31/2017</td>
      <td>47.260000</td>
    </tr>
  </tbody>
</table>
<p>1218 rows × 2 columns</p>
</div>




```python
import numpy as np # linear algebra
import pandas as pd # data processing
from statsmodels.tsa.stattools import adfuller
```

# Kiểm định tính dừng của chuỗi dữ liệu


```python
## KIỂM ĐỊNH DICKEY FULLER:
result = adfuller(ii['dcoilwtico'])
print('ADF Statistic: {}'.format(result[0]))
print('p-value: {}'.format(result[1]))
print('Critical Values:')
for key, value in result[4].items():
    print('\t{}: {}'.format(key, value))
```

    ADF Statistic: -0.9803517410964506
    p-value: 0.7603629301235183
    Critical Values:
    	1%: -3.43577938005948
    	5%: -2.863937543790164
    	10%: -2.568046493171221



```python
def adfuller_test(ii):
    result=adfuller(ii)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )

if result[1] <= 0.05:
    print("Strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data is stationary (Chuỗi dừng)")
else:
    print("Weak evidence against null hypothesis,indicating it is non-stationary (Chuỗi không dừng) ")

adfuller_test(ii['dcoilwtico'])

```

    Weak evidence against null hypothesis,indicating it is non-stationary (Chuỗi không dừng) 
    ADF Test Statistic : -0.9803517410964506
    p-value : 0.7603629301235183
    #Lags Used : 10
    Number of Observations : 1207



```python
from statsmodels.tsa.stattools import kpss
```


```python
series=ii['dcoilwtico']
```


```python
series
```




    0       67.714366
    1       93.140000
    2       92.970000
    3       93.120000
    4       93.200000
    5       93.210000
    6       93.080000
    7       93.810000
    8       93.600000
    9       94.270000
    10      93.260000
    11      94.280000
    12      95.490000
    13      95.610000
    14      67.714366
    15      96.090000
    16      95.060000
    17      95.350000
    18      95.150000
    19      95.950000
    20      97.620000
    21      97.980000
    22      97.650000
    23      97.460000
    24      96.210000
    25      96.680000
    26      96.440000
    27      95.840000
    28      95.710000
    29      97.010000
              ...    
    1188    45.780000
    1189    46.210000
    1190    47.770000
    1191    48.580000
    1192    49.050000
    1193    49.720000
    1194    50.210000
    1195    49.190000
    1196    49.600000
    1197    49.030000
    1198    49.570000
    1199    49.370000
    1200    49.070000
    1201    49.590000
    1202    48.540000
    1203    48.810000
    1204    47.590000
    1205    47.570000
    1206    46.800000
    1207    47.070000
    1208    48.590000
    1209    47.390000
    1210    47.650000
    1211    48.450000
    1212    47.240000
    1213    47.650000
    1214    46.400000
    1215    46.460000
    1216    45.960000
    1217    47.260000
    Name: dcoilwtico, Length: 1218, dtype: float64




```python
#KIỂM ĐỊNH KPSS
from statsmodels.tsa.stattools import kpss
def kpss_test(series, **kw):    
    statistic, p_value, n_lags, critical_values = kpss(series, **kw)
    # Format Output
    print(f'KPSS Statistic: {statistic}')
    print(f'p-value: {p_value}')
    print(f'num lags: {n_lags}')
    print('Critial Values:')
    for key, value in critical_values.items():
        print(f'   {key} : {value}')
    print(f'Result: The series is {"not " if p_value < 0.05 else ""}stationary')

kpss_test(series)
```

    KPSS Statistic: 4.113911447239356
    p-value: 0.01
    num lags: 23
    Critial Values:
       10% : 0.347
       5% : 0.463
       2.5% : 0.574
       1% : 0.739
    Result: The series is not stationary


    C:\Users\DELL\Anaconda3\lib\site-packages\statsmodels\tsa\stattools.py:1276: InterpolationWarning: p-value is smaller than the indicated p-value
      warn("p-value is smaller than the indicated p-value", InterpolationWarning)



```python
#Lấy sai phân bậc 1 của chuỗi dữ liệu
series=ii['dcoilwtico'].diff()
```


```python
series
```




    0             NaN
    1       25.425634
    2       -0.170000
    3        0.150000
    4        0.080000
    5        0.010000
    6       -0.130000
    7        0.730000
    8       -0.210000
    9        0.670000
    10      -1.010000
    11       1.020000
    12       1.210000
    13       0.120000
    14     -27.895634
    15      28.375634
    16      -1.030000
    17       0.290000
    18      -0.200000
    19       0.800000
    20       1.670000
    21       0.360000
    22      -0.330000
    23      -0.190000
    24      -1.250000
    25       0.470000
    26      -0.240000
    27      -0.600000
    28      -0.130000
    29       1.300000
              ...    
    1188    -0.950000
    1189     0.430000
    1190     1.560000
    1191     0.810000
    1192     0.470000
    1193     0.670000
    1194     0.490000
    1195    -1.020000
    1196     0.410000
    1197    -0.570000
    1198     0.540000
    1199    -0.200000
    1200    -0.300000
    1201     0.520000
    1202    -1.050000
    1203     0.270000
    1204    -1.220000
    1205    -0.020000
    1206    -0.770000
    1207     0.270000
    1208     1.520000
    1209    -1.200000
    1210     0.260000
    1211     0.800000
    1212    -1.210000
    1213     0.410000
    1214    -1.250000
    1215     0.060000
    1216    -0.500000
    1217     1.300000
    Name: dcoilwtico, Length: 1218, dtype: float64




```python
from statsmodels.tsa.stattools import kpss
def kpss_test(series, **kw):    
    statistic, p_value, n_lags, critical_values = kpss(series, **kw)
    # Format Output
    print(f'KPSS Statistic: {statistic}')
    print(f'p-value: {p_value}')
    print(f'num lags: {n_lags}')
    print('Critial Values:')
    for key, value in critical_values.items():
        print(f'   {key} : {value}')
    print(f'Result: The series is {"not " if p_value < 0.05 else ""}stationary')

kpss_test(series)
```

    KPSS Statistic: nan
    p-value: nan
    num lags: 23
    Critial Values:
       10% : 0.347
       5% : 0.463
       2.5% : 0.574
       1% : 0.739
    Result: The series is stationary


Sau khi lấy sai phân bậc 1, chuỗi dữ liệu đã dừng.
