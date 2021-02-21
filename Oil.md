

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




    <matplotlib.axes._subplots.AxesSubplot at 0x21b4a024358>




![png](output_13_1.png)



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




    <matplotlib.axes._subplots.AxesSubplot at 0x21b4a0703c8>




![png](output_16_1.png)


Chúng ta có thể thấy rằng giá dầu có những biến động mạnh và dài hạn với sự sụt giảm rõ rệt vào nửa cuối năm 2014. Trên thực tế, mặc dù có một số biến động, giá dầu vẫn ở mức như hồi đầu năm 2015. Do đó, chúng ta có thể thấy sự thay đổi đáng kể trong doanh thu của các cửa hàng vào khoảng cuối năm 2014. Nhìn vào số liệu bán hàng của đơn vị, điều này không rõ ràng: mặc dù doanh số bán hàng có vẻ giảm vào đầu năm 2015, nhưng vào cuối năm 2014, chúng đang tăng lên. Ngoài ra, việc giảm giá dầu dường như không có bất kỳ tác động nào đến doanh số bán hàng, vì nó đã được nhìn thấy từ lô bán hàng, không có mối liên hệ giữa việc giảm giá dầu với doanh số bán hàng, vì vậy chúng ta có thể nói rằng đặc điểm hoặc dữ liệu này là không quan trọng đối với chúng tôi và sẽ không được xem xét trong quá trình lập mô hình.
