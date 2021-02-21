

```python
%matplotlib inline
```


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
holiday_db = pd.read_csv('holidays_events.csv', parse_dates=['date'])
```


```python
fig, ax = plt.subplots()
fig.set_size_inches(8, 6)
ax = sns.countplot( y="type", data=holiday_db, palette="RdBu")
```


![png](output_3_0.png)



```python
plt.style.use('seaborn-white')
holiday_local_type = holiday_db.groupby(['locale_name', 'type']).size()
holiday_local_type.unstack().plot(kind='bar',stacked=True, colormap= 'magma_r', figsize=(12,10),  grid=False)
plt.title('Stacked Barplot of locale name against event type')
plt.ylabel('Count of entries')
plt.show()
```


![png](output_4_0.png)



```python

```
