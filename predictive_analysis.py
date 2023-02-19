# -*- coding: utf-8 -*-
"""predictive analysis.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Jre0i1QcnzZ2ScduTk2EFGOyVDFBwFyy

Importing Library
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# %matplotlib inline
import seaborn as sns

"""Baca Data"""

dataset = 'anz.csv'
df = pd.read_csv("anz.csv")
df

#hapus
anz=df.drop(['merchant_code','bpay_biller_code', 'card_present_flag'],axis=1)

anz.describe()

"""Kemudian kita lihat info dari type data masing-masing variabel"""

anz.info()

"""Kasih Header"""

anz.head()

"""Menangani Outliers"""

sns.boxplot(x=anz['balance'])

sns.boxplot(x=anz['age'])

sns.boxplot(x=anz['amount'])

"""mengatasi Outliers"""

Q1 = anz.quantile(0.25)
Q3 = anz.quantile(0.75)
IQR=Q3-Q1
anz=anz[~((anz<(Q1-1.5*IQR))|(anz>(Q3+1.5*IQR))).any(axis=1)]
 
# Cek ukuran dataset setelah kita drop outliers
anz.shape

"""UNIVARIATE ANALYSIS"""

numerical_features = ['balance', 'age', 'amount']
categorical_features = ['status', 'account', 'currency', 'long_lat', 'txn_description', 'merchant_id',
                       'first_name', 'date', 'gender', 'merchant_suburb', 'merchant_state', 'merchant_state', 'extraction',
                        'transaction_id', 'country', 'customer_id', 'merchant_long_lat', 'movement' ]

"""Categorical Features"""

feature = categorical_features[0]
count = anz[feature].value_counts()
percent = 100*anz[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

feature = categorical_features[1]
count = anz[feature].value_counts()
percent = 100*anz[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

feature = categorical_features[2]
count = anz[feature].value_counts()
percent = 100*anz[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

"""Numerical Features"""

anz.hist(bins=50, figsize=(20,15))
plt.show()

"""MULTIVARIATE ANALYSIS

Categorical Analysis
"""

cat_features = anz.select_dtypes(include='object').columns.to_list()
 
for col in cat_features:
  sns.catplot(x=col, y="amount", kind="bar", dodge=False, height = 4, aspect = 3,  data=anz, palette="Set3")
  plt.title("Rata-rata 'amount' Relatif terhadap - {}".format(col))

"""Numerical Features"""

# Mengamati hubungan antar fitur numerik dengan fungsi pairplot()
sns.pairplot(anz, diag_kind = 'kde')

plt.figure(figsize=(10, 8))
correlation_matrix = anz.corr().round(2)
 
# Untuk menge-print nilai di dalam kotak, gunakan parameter anot=True
sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, )
plt.title("Correlation Matrix untuk Fitur Numerik ", size=20)

"""DATA PREPARATION

Encoding Fitur Kategori
"""

anz.info()

from sklearn.preprocessing import  OneHotEncoder
anz = pd.concat([anz, pd.get_dummies(anz['status'], prefix='status')],axis=1)
anz = pd.concat([anz, pd.get_dummies(anz['account'], prefix='account')],axis=1)
anz = pd.concat([anz, pd.get_dummies(anz['currency'], prefix='currency')],axis=1)
anz = pd.concat([anz, pd.get_dummies(anz['long_lat'], prefix='long_lat')],axis=1)
anz = pd.concat([anz, pd.get_dummies(anz['txn_description'], prefix='txn_description')],axis=1)
anz = pd.concat([anz, pd.get_dummies(anz['merchant_id'], prefix='merchant_id')],axis=1)
anz = pd.concat([anz, pd.get_dummies(anz['first_name'], prefix='first_name')],axis=1)
anz = pd.concat([anz, pd.get_dummies(anz['date'], prefix='date')],axis=1)
anz = pd.concat([anz, pd.get_dummies(anz['gender'], prefix='gender')],axis=1)
anz = pd.concat([anz, pd.get_dummies(anz['merchant_suburb'], prefix='merchant_suburb')],axis=1)
anz = pd.concat([anz, pd.get_dummies(anz['merchant_state'], prefix='merchant_state')],axis=1)
anz = pd.concat([anz, pd.get_dummies(anz['extraction'], prefix='extraction')],axis=1)
anz = pd.concat([anz, pd.get_dummies(anz['transaction_id'], prefix='transaction_id')],axis=1)
anz = pd.concat([anz, pd.get_dummies(anz['country'], prefix='country')],axis=1)
anz = pd.concat([anz, pd.get_dummies(anz['customer_id'], prefix='customer_id')],axis=1)
anz = pd.concat([anz, pd.get_dummies(anz['merchant_long_lat'], prefix='merchant_long_lat')],axis=1)
anz = pd.concat([anz, pd.get_dummies(anz['movement'], prefix='movement')],axis=1)



anz.drop(['status', 'account', 'currency', 'long_lat', 'txn_description', 'merchant_id',
                       'first_name', 'date', 'gender', 'merchant_suburb', 'merchant_state', 'merchant_state', 'extraction',
                        'transaction_id', 'country', 'customer_id', 'merchant_long_lat', 'movement'], axis=1, inplace=True)
anz.head()

"""Reduksi Dimensi dengan PCA

Train-Test-Split
"""

from sklearn.model_selection import train_test_split
 
X = anz.drop(["amount"],axis =1)
y = anz["amount"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 123)

print(f'Total # of sample in whole dataset: {len(X)}')
print(f'Total # of sample in train dataset: {len(X_train)}')
print(f'Total # of sample in test dataset: {len(X_test)}')

"""Pembagian Train-Test-Split dataseet 80:20

Standarisasi
"""

from sklearn.preprocessing import StandardScaler
 
numerical_features = ['balance', 'age']
scaler = StandardScaler()
scaler.fit(X_train[numerical_features])
X_train[numerical_features] = scaler.transform(X_train.loc[:, numerical_features])
X_train[numerical_features].head()

X_train[numerical_features].describe().round(4)

from sklearn.preprocessing import StandardScaler
 
numerical_features = ['balance', 'age']
scaler = StandardScaler()
scaler.fit(X_train[numerical_features])
X_train[numerical_features] = scaler.transform(X_train.loc[:, numerical_features])
X_train[numerical_features].head()

"""Berikut data frame untuk analisis ketiga model"""

# Siapkan dataframe untuk analisis model
models = pd.DataFrame(index=['train_mse', 'test_mse'], 
                      columns=['KNN', 'RandomForest', 'Boosting'])

"""Model Development dengan K-Nearest Neighbor"""

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
 
knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(X_train, y_train)
 
models.loc['train_mse','knn'] = mean_squared_error(y_pred = knn.predict(X_train), y_true=y_train)

"""Model Development dengan Random Forest"""

# Impor library yang dibutuhkan
from sklearn.ensemble import RandomForestRegressor
 
# buat model prediksi
RF = RandomForestRegressor(n_estimators=50, max_depth=16, random_state=55, n_jobs=-1)
RF.fit(X_train, y_train)
 
models.loc['train_mse','RandomForest'] = mean_squared_error(y_pred=RF.predict(X_train), y_true=y_train)

"""Model Development dengan Boosting Algorithm"""

from sklearn.ensemble import AdaBoostRegressor
 
boosting = AdaBoostRegressor(learning_rate=0.05, random_state=55)                             
boosting.fit(X_train, y_train)
models.loc['train_mse','Boosting'] = mean_squared_error(y_pred=boosting.predict(X_train), y_true=y_train)

"""EVALUASI MODEL"""

# Lakukan scaling terhadap fitur numerik pada X_test sehingga memiliki rata-rata=0 dan varians=1
X_test.loc[:, numerical_features] = scaler.transform(X_test[numerical_features])

mse = pd.DataFrame(columns=['train', 'test'], index=['KNN','RF','Boosting'])
 
# Buat dictionary untuk setiap algoritma yang digunakan
model_dict = {'KNN': knn, 'RF': RF, 'Boosting': boosting}
 
# Hitung Mean Squared Error masing-masing algoritma pada data train dan test
for name, model in model_dict.items():
    mse.loc[name, 'train'] = mean_squared_error(y_true=y_train, y_pred=model.predict(X_train))/1e3 
    mse.loc[name, 'test'] = mean_squared_error(y_true=y_test, y_pred=model.predict(X_test))/1e3
 
# Panggil mse
mse

fig, ax = plt.subplots()
mse.sort_values(by='test', ascending=False).plot(kind='barh', ax=ax, zorder=3)
ax.grid(zorder=0)

prediksi = X_test.iloc[:1].copy()
pred_dict = {'y_true':y_test[:1]}
for name, model in model_dict.items():
    pred_dict['prediksi_'+name] = model.predict(prediksi).round(1)
 
pd.DataFrame(pred_dict)