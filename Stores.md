

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor
sns.set_style('whitegrid')
import seaborn as sns
```


```python
%matplotlib inline
```


```python
stores_db = pd.read_csv('stores.csv')
```


```python
# Number of columns
len(stores_db.columns)
```




    5




```python
# Number of rows
len(stores_db)
```




    54




```python
# Check missing values
stores_db.isnull().sum()
```




    store_nbr    0
    city         0
    state        0
    type         0
    cluster      0
    dtype: int64




```python
# Thống kê mô tả
stores_db.describe()
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
      <th>store_nbr</th>
      <th>cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>54.000000</td>
      <td>54.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>27.500000</td>
      <td>8.481481</td>
    </tr>
    <tr>
      <th>std</th>
      <td>15.732133</td>
      <td>4.693395</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>14.250000</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>27.500000</td>
      <td>8.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>40.750000</td>
      <td>13.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>54.000000</td>
      <td>17.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Stores distribution across cities
stores_db['city'].value_counts()
```




    Quito            18
    Guayaquil         8
    Santo Domingo     3
    Cuenca            3
    Ambato            2
    Machala           2
    Manta             2
    Latacunga         2
    Ibarra            1
    Puyo              1
    El Carmen         1
    Cayambe           1
    Babahoyo          1
    Daule             1
    Riobamba          1
    Quevedo           1
    Esmeraldas        1
    Libertad          1
    Playas            1
    Guaranda          1
    Salinas           1
    Loja              1
    Name: city, dtype: int64




```python
fig, ax = plt.subplots()
fig.set_size_inches(8, 8)
ax = sns.countplot(y = stores_db['city'], data = stores_db) 
plt.title('Stores distribution across cities', fontsize = 18)
```




    Text(0.5, 1.0, 'Stores distribution across cities')




![png](output_8_1.png)


Trong 21 thành phố thì số lượng cửa hàng ở Quito là nhiều nhất với 18 cửa hàng và đứng thứ hai là Guayaquil với 8 cửa hàng. Các thành phố Libertad, Ibarra, Loja, Babahoyo , Daule, Esmeraldas, Playas, Guaranda , Cayambe, Riobamba, Puyo, Salinas, Quevedo, El Carmen mỗi thành phố đều có 1 cửa hàng.


```python
# Stores distribution across states
stores_db['state'].value_counts()
```




    Pichincha                         19
    Guayas                            11
    Manabi                             3
    Azuay                              3
    Santo Domingo de los Tsachilas     3
    Los Rios                           2
    Tungurahua                         2
    Cotopaxi                           2
    El Oro                             2
    Chimborazo                         1
    Santa Elena                        1
    Loja                               1
    Bolivar                            1
    Pastaza                            1
    Imbabura                           1
    Esmeraldas                         1
    Name: state, dtype: int64




```python
fig, ax = plt.subplots()
fig.set_size_inches(8, 8)
ax = sns.countplot(y = stores_db['state'], data = stores_db)
plt.title('Stores distribution across states', fontsize = 18)
```




    Text(0.5, 1.0, 'Stores distribution across states')




![png](output_11_1.png)


Trong 16 bang thì bang Pichincha có nhiều cửa hàng nhất với 19 cửa hàng, đứng thứ hai là bang Guayas với 11 cửa hàng. Các bang Loja, Esmeraldas, Santa Elena, Pastaza, Imbabura, Bolivar, Chimborazo mỗi bang đều có 1 cửa hàng.


```python
# Different types of stores
stores_db['type'].value_counts()
```




    D    18
    C    15
    A     9
    B     8
    E     4
    Name: type, dtype: int64




```python
fig, ax = plt.subplots()
fig.set_size_inches(10, 7)
ax = sns.countplot(x="type", data=stores_db, palette="Set2")
plt.title('Different types of stores', fontsize = 18)
```




    Text(0.5, 1.0, 'Different types of stores')




![png](output_14_1.png)


Cửa hàng loại D có số lượng nhiều nhất với 18 cửa hàng và ít nhất là cửa hàng loại E với 4 cửa hàng


```python
# Types of stores across cities
ct = pd.crosstab(stores_db.city, stores_db.type)
ct
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
      <th>type</th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
    </tr>
    <tr>
      <th>city</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ambato</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Babahoyo</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Cayambe</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Cuenca</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Daule</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>El Carmen</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Esmeraldas</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Guaranda</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Guayaquil</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Ibarra</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Latacunga</th>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Libertad</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Loja</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Machala</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Manta</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Playas</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Puyo</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Quevedo</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Quito</th>
      <td>6</td>
      <td>3</td>
      <td>2</td>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Riobamba</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Salinas</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Santo Domingo</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
ct.plot.bar(figsize = (12, 6), stacked=True)
plt.legend(title = 'type')
plt.title('Types of stores across cities', fontsize = 18)
plt.show()
```


![png](output_17_0.png)


Chỉ có hai thành phố Quito và Guayaquil là có đầy đủ bốn loại cửa hàng


```python
# Types of stores across states
ct = pd.crosstab(stores_db.state, stores_db.type)
ct.plot.bar(figsize = (12, 6), stacked=True)
plt.legend(title = 'type')
plt.title('Types of stores across states', fontsize = 18)
plt.show()
```


![png](output_19_0.png)



```python
Chỉ có hai bang Pichincha và Guayas là có đầy đủ bốn loại cửa hàng
```


```python
stores_db.sort_values(by=['state'])
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
      <th>store_nbr</th>
      <th>city</th>
      <th>state</th>
      <th>type</th>
      <th>cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>41</th>
      <td>42</td>
      <td>Cuenca</td>
      <td>Azuay</td>
      <td>D</td>
      <td>2</td>
    </tr>
    <tr>
      <th>38</th>
      <td>39</td>
      <td>Cuenca</td>
      <td>Azuay</td>
      <td>B</td>
      <td>6</td>
    </tr>
    <tr>
      <th>36</th>
      <td>37</td>
      <td>Cuenca</td>
      <td>Azuay</td>
      <td>D</td>
      <td>2</td>
    </tr>
    <tr>
      <th>18</th>
      <td>19</td>
      <td>Guaranda</td>
      <td>Bolivar</td>
      <td>C</td>
      <td>15</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>Riobamba</td>
      <td>Chimborazo</td>
      <td>C</td>
      <td>7</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>Latacunga</td>
      <td>Cotopaxi</td>
      <td>C</td>
      <td>15</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>Latacunga</td>
      <td>Cotopaxi</td>
      <td>C</td>
      <td>15</td>
    </tr>
    <tr>
      <th>40</th>
      <td>41</td>
      <td>Machala</td>
      <td>El Oro</td>
      <td>D</td>
      <td>4</td>
    </tr>
    <tr>
      <th>39</th>
      <td>40</td>
      <td>Machala</td>
      <td>El Oro</td>
      <td>C</td>
      <td>3</td>
    </tr>
    <tr>
      <th>42</th>
      <td>43</td>
      <td>Esmeraldas</td>
      <td>Esmeraldas</td>
      <td>E</td>
      <td>10</td>
    </tr>
    <tr>
      <th>23</th>
      <td>24</td>
      <td>Guayaquil</td>
      <td>Guayas</td>
      <td>D</td>
      <td>1</td>
    </tr>
    <tr>
      <th>28</th>
      <td>29</td>
      <td>Guayaquil</td>
      <td>Guayas</td>
      <td>E</td>
      <td>10</td>
    </tr>
    <tr>
      <th>29</th>
      <td>30</td>
      <td>Guayaquil</td>
      <td>Guayas</td>
      <td>C</td>
      <td>3</td>
    </tr>
    <tr>
      <th>31</th>
      <td>32</td>
      <td>Guayaquil</td>
      <td>Guayas</td>
      <td>C</td>
      <td>3</td>
    </tr>
    <tr>
      <th>33</th>
      <td>34</td>
      <td>Guayaquil</td>
      <td>Guayas</td>
      <td>B</td>
      <td>6</td>
    </tr>
    <tr>
      <th>26</th>
      <td>27</td>
      <td>Daule</td>
      <td>Guayas</td>
      <td>D</td>
      <td>1</td>
    </tr>
    <tr>
      <th>27</th>
      <td>28</td>
      <td>Guayaquil</td>
      <td>Guayas</td>
      <td>E</td>
      <td>10</td>
    </tr>
    <tr>
      <th>35</th>
      <td>36</td>
      <td>Libertad</td>
      <td>Guayas</td>
      <td>E</td>
      <td>10</td>
    </tr>
    <tr>
      <th>50</th>
      <td>51</td>
      <td>Guayaquil</td>
      <td>Guayas</td>
      <td>A</td>
      <td>17</td>
    </tr>
    <tr>
      <th>34</th>
      <td>35</td>
      <td>Playas</td>
      <td>Guayas</td>
      <td>C</td>
      <td>3</td>
    </tr>
    <tr>
      <th>25</th>
      <td>26</td>
      <td>Guayaquil</td>
      <td>Guayas</td>
      <td>D</td>
      <td>10</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>Ibarra</td>
      <td>Imbabura</td>
      <td>C</td>
      <td>15</td>
    </tr>
    <tr>
      <th>37</th>
      <td>38</td>
      <td>Loja</td>
      <td>Loja</td>
      <td>D</td>
      <td>4</td>
    </tr>
    <tr>
      <th>32</th>
      <td>33</td>
      <td>Quevedo</td>
      <td>Los Rios</td>
      <td>C</td>
      <td>3</td>
    </tr>
    <tr>
      <th>30</th>
      <td>31</td>
      <td>Babahoyo</td>
      <td>Los Rios</td>
      <td>B</td>
      <td>10</td>
    </tr>
    <tr>
      <th>51</th>
      <td>52</td>
      <td>Manta</td>
      <td>Manabi</td>
      <td>A</td>
      <td>11</td>
    </tr>
    <tr>
      <th>53</th>
      <td>54</td>
      <td>El Carmen</td>
      <td>Manabi</td>
      <td>C</td>
      <td>3</td>
    </tr>
    <tr>
      <th>52</th>
      <td>53</td>
      <td>Manta</td>
      <td>Manabi</td>
      <td>D</td>
      <td>13</td>
    </tr>
    <tr>
      <th>21</th>
      <td>22</td>
      <td>Puyo</td>
      <td>Pastaza</td>
      <td>C</td>
      <td>7</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>Quito</td>
      <td>Pichincha</td>
      <td>D</td>
      <td>13</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Quito</td>
      <td>Pichincha</td>
      <td>D</td>
      <td>13</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Quito</td>
      <td>Pichincha</td>
      <td>D</td>
      <td>8</td>
    </tr>
    <tr>
      <th>48</th>
      <td>49</td>
      <td>Quito</td>
      <td>Pichincha</td>
      <td>A</td>
      <td>11</td>
    </tr>
    <tr>
      <th>47</th>
      <td>48</td>
      <td>Quito</td>
      <td>Pichincha</td>
      <td>A</td>
      <td>14</td>
    </tr>
    <tr>
      <th>46</th>
      <td>47</td>
      <td>Quito</td>
      <td>Pichincha</td>
      <td>A</td>
      <td>14</td>
    </tr>
    <tr>
      <th>45</th>
      <td>46</td>
      <td>Quito</td>
      <td>Pichincha</td>
      <td>A</td>
      <td>14</td>
    </tr>
    <tr>
      <th>44</th>
      <td>45</td>
      <td>Quito</td>
      <td>Pichincha</td>
      <td>A</td>
      <td>11</td>
    </tr>
    <tr>
      <th>43</th>
      <td>44</td>
      <td>Quito</td>
      <td>Pichincha</td>
      <td>A</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Quito</td>
      <td>Pichincha</td>
      <td>D</td>
      <td>9</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Quito</td>
      <td>Pichincha</td>
      <td>D</td>
      <td>13</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>Quito</td>
      <td>Pichincha</td>
      <td>D</td>
      <td>8</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>Quito</td>
      <td>Pichincha</td>
      <td>B</td>
      <td>6</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>Quito</td>
      <td>Pichincha</td>
      <td>C</td>
      <td>15</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>Cayambe</td>
      <td>Pichincha</td>
      <td>B</td>
      <td>6</td>
    </tr>
    <tr>
      <th>16</th>
      <td>17</td>
      <td>Quito</td>
      <td>Pichincha</td>
      <td>C</td>
      <td>12</td>
    </tr>
    <tr>
      <th>17</th>
      <td>18</td>
      <td>Quito</td>
      <td>Pichincha</td>
      <td>B</td>
      <td>16</td>
    </tr>
    <tr>
      <th>19</th>
      <td>20</td>
      <td>Quito</td>
      <td>Pichincha</td>
      <td>B</td>
      <td>6</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>Quito</td>
      <td>Pichincha</td>
      <td>D</td>
      <td>8</td>
    </tr>
    <tr>
      <th>24</th>
      <td>25</td>
      <td>Salinas</td>
      <td>Santa Elena</td>
      <td>D</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Santo Domingo</td>
      <td>Santo Domingo de los Tsachilas</td>
      <td>D</td>
      <td>4</td>
    </tr>
    <tr>
      <th>15</th>
      <td>16</td>
      <td>Santo Domingo</td>
      <td>Santo Domingo de los Tsachilas</td>
      <td>C</td>
      <td>3</td>
    </tr>
    <tr>
      <th>20</th>
      <td>21</td>
      <td>Santo Domingo</td>
      <td>Santo Domingo de los Tsachilas</td>
      <td>B</td>
      <td>6</td>
    </tr>
    <tr>
      <th>49</th>
      <td>50</td>
      <td>Ambato</td>
      <td>Tungurahua</td>
      <td>A</td>
      <td>14</td>
    </tr>
    <tr>
      <th>22</th>
      <td>23</td>
      <td>Ambato</td>
      <td>Tungurahua</td>
      <td>D</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Distribution of different clusters
fig, ax = plt.subplots()
fig.set_size_inches(12, 7)
ax = sns.countplot(x="cluster", data=stores_db)
plt.title('Distribution of different clusters', fontsize = 18)
```




    Text(0.5, 1.0, 'Distribution of different clusters')




![png](output_22_1.png)



```python
type_cluster.unstack().plot(kind='bar',stacked=True, colormap= 'PuBu', figsize=(13,11),  grid=False)
plt.title('Stacked Barplot of Store types and their cluster distribution', fontsize=18)
plt.ylabel('Count of clusters in a particular store type', fontsize=16)
plt.xlabel('Store type', fontsize=16)
plt.show()
```


![png](output_23_0.png)



```python
# Types against clusters
plt.style.use('seaborn-white')
type_cluster = stores_db.groupby(['type','cluster']).size()
type_cluster
# diffrence between .size() vs .count()
# => size includes NaN values, count does not:
```




    type  cluster
    A     5          1
          11         3
          14         4
          17         1
    B     6          6
          10         1
          16         1
    C     3          7
          7          2
          12         1
          15         5
    D     1          3
          2          2
          4          3
          8          3
          9          2
          10         1
          13         4
    E     10         4
    dtype: int64




```python
# cluster of stores across the different cities
plt.style.use('seaborn-white')
city_cluster = stores_db.groupby(['city','cluster']).store_nbr.size()
city_cluster.unstack().plot(kind='bar',stacked=True, colormap= 'viridis', figsize=(13,11),  grid=False)
plt.title('Stacked Barplot of Store cluster opened for each city')
plt.ylabel('Count of stores for a particular city')
plt.show()
```


![png](output_25_0.png)



```python

```
