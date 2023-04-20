# EX-05-Feature-Generation


## AIM
To read the given data and perform Feature Generation process and save the data to a file. 

# Explanation
Feature Generation (also known as feature construction, feature extraction or feature engineering) is the process of transforming features into new features that better relate to the target.
 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature Generation techniques to all the feature of the data set
### STEP 4
Save the data to the file


# CODE

NAME:VAISHALI BALAMURUGAN
REG NO:212222230164

```
DATA.CSV
import pandas as pd
df=pd.read_csv("data.csv")
df

# feature generation
import category_encoders as ce
be=ce.BinaryEncoder()
ndf=be.fit_transform(df["bin_1"])
df["bin_1"] = be.fit_transform(df["bin_1"])
ndf

ndf2=be.fit_transform(df["bin_2"])
df["bin_2"] = be.fit_transform(df["bin_2"])
ndf2

df1=df.copy()
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,OneHotEncoder
import category_encoders as ce
be=ce.BinaryEncoder()
ohe=OneHotEncoder(sparse=False)
le=LabelEncoder()
oe=OrdinalEncoder()


df1["City"] = ohe.fit_transform(df1[["City"]])

temp=['Cold','Warm','Hot','Very Hot']
oe1=OrdinalEncoder(categories=[temp])
df1['Ord_1'] = oe1.fit_transform(df1[["Ord_1"]])

edu=['High School','Diploma','Bachelors','Masters','PhD']
oe2=OrdinalEncoder(categories=[edu])
df1['Ord_2']= oe2.fit_transform(df1[["Ord_2"]])
df1

# feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df2=pd.DataFrame(sc.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df2

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df3=pd.DataFrame(sc1.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df3

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df4=pd.DataFrame(sc2.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df4

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df5=pd.DataFrame(sc3.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df5
```
ENCODING.CSV
```
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df

# feature generation
import category_encoders as ce
be=ce.BinaryEncoder()
ndf=be.fit_transform(df["bin_1"])
df["bin_1"] = be.fit_transform(df["bin_1"])
ndf

ndf2=be.fit_transform(df["bin_2"])
df["bin_2"] = be.fit_transform(df["bin_2"])
ndf2

df1=df.copy()
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
le=LabelEncoder()
oe=OrdinalEncoder()

df1["nom_0"] = oe.fit_transform(df1[["nom_0"]])
temp=['Cold','Warm','Hot']
oe2=OrdinalEncoder(categories=[temp])
df1['ord_2'] = oe2.fit_transform(df1[['ord_2']])

df1

# feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df0=pd.DataFrame(sc.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df0

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df2=pd.DataFrame(sc1.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df2

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df3=pd.DataFrame(sc2.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df3

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df4=pd.DataFrame(sc3.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df4
```
TITANIC.CSV
```
import pandas as pd
df=pd.read_csv("titanic_dataset.csv")
df

#removing unwanted data
df.drop("Name",axis=1,inplace=True)
df.drop("Ticket",axis=1,inplace=True)
df.drop("Cabin",axis=1,inplace=True)

#data cleaning
df.isnull().sum()

df["Age"]=df["Age"].fillna(df["Age"].median())
df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])

df.isnull().sum()

df

#feature encoding
from category_encoders import BinaryEncoder
be=BinaryEncoder()
df["Sex"]=be.fit_transform(df[["Sex"]])
ndf=be.fit_transform(df["Sex"])
ndf

df1=df.copy()
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
embark=['S','C','Q']
e1=OrdinalEncoder(categories=[embark])
df1['Embarked'] = e1.fit_transform(df[['Embarked']])
df1

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df2=pd.DataFrame(sc.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df2

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df3=pd.DataFrame(sc1.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df3

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df4=pd.DataFrame(sc2.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df4

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df5=pd.DataFrame(sc3.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df5
```


# OUTPUT:
```
DATA CSV

INITIAL DATASET:
![image](https://user-images.githubusercontent.com/119390134/233389012-69c70093-9763-4f55-85d7-44a914c99038.png)

 BINARY ENCODING:
 ![image](https://user-images.githubusercontent.com/119390134/233386114-771292e4-1e05-446d-8d07-640c806cb5ac.png)

ENCODED DATASET:
![image](https://user-images.githubusercontent.com/119390134/233386219-5eef891e-e02e-4bc1-afbc-a8b1c82852e4.png)

DATA SCAALING USING MINMAXSCALER:
![image](https://user-images.githubusercontent.com/119390134/233386332-118b3209-30e5-43d1-8f79-1d28415ca877.png)

DATA SCALING USING MAXABSSCALER:
![image](https://user-images.githubusercontent.com/119390134/233386489-7dc4142d-44e1-4e23-86a8-f8a4e0dda89e.png)

DATA SCALING USING STANDARDSCALER:
![image](https://user-images.githubusercontent.com/119390134/233386634-1a3f474d-cda0-4465-90fb-45d3ab15d5a1.png)
```
```
ENCODING.CSV:

INTIAL DATASET:
![image](https://user-images.githubusercontent.com/119390134/233386802-5d3abd3f-0fe2-4432-9891-32be8c1024ac.png)

BINARY ENCODING:
![image](https://user-images.githubusercontent.com/119390134/233386915-f12da3b3-3fa3-4399-9bf5-b6e3cf6f323c.png)

ENCODED DATASET:
![image](https://user-images.githubusercontent.com/119390134/233387015-13caad64-d390-4002-9c4b-540f529cc7fb.png)

DATA SCALING USING MINMAXSCALER:
![image](https://user-images.githubusercontent.com/119390134/233387194-73c30bdc-2987-4b05-94d5-b654991441df.png)

DATA SCALING USING ROBUSTSCALER:
![image](https://user-images.githubusercontent.com/119390134/233387365-53247982-f93a-4737-9342-0ee4fc86daa2.png)

DATA SCALING USING MAXABSSCALER:
![image](https://user-images.githubusercontent.com/119390134/233387471-089ad9cd-90c2-4c39-99b9-9e2cc1040dbb.png)
```
```
TITANIC.CSV:

INITIAL DATASET:
![image](https://user-images.githubusercontent.com/119390134/233387710-8eb9f492-105b-4c36-9fb6-09843a5e8aad.png)

DATA CLEANING BEFORE ENCODING:
![image](https://user-images.githubusercontent.com/119390134/233387854-4aa4be15-b133-4232-a65c-2a42f8035b27.png)

CLEANED DATASET:
![image](https://user-images.githubusercontent.com/119390134/233387949-15e268d9-c08e-4aa6-8518-db3fc6dcbeb7.png)

BINARY ENCODING: ![image](https://user-images.githubusercontent.com/119390134/233388036-d0c796c2-da8d-4153-972b-e0ff8c55df42.png)

ENCODED DATASET:
![image](https://user-images.githubusercontent.com/119390134/233388225-6ca6e43a-b22e-42db-b824-5a08365da746.png)

DATA SCALING USING MINMAXSCALER:
![image](https://user-images.githubusercontent.com/119390134/233388314-ca29df8f-9106-4569-a2b6-bf691d52f005.png)

DATA SCAALING USING ROBUSTSCALER:
![image](https://user-images.githubusercontent.com/119390134/233388500-7269df07-1727-43d0-a7b8-ed6e312c4052.png)

DATA SCALING USING MAXABSSCALER:
![image](https://user-images.githubusercontent.com/119390134/233388640-92991b48-c786-468a-b398-70bb2bac91d6.png)
```
```
RESULT:

Feature Generation process and Feature Scaling process is applied to the given data frames sucessfully.
```










