
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import chi2_contingency
scale = MinMaxScaler()

df=pd.read_csv("/heart_failure_clinical_records_dataset.csv")
df.info()
df.isna().sum()
df['age'] = scale.fit_transform(df[['age']])
df['anaemia'] = scale.fit_transform(df[['anaemia']])
df['diabetes'] = scale.fit_transform(df[['diabetes']])
df['high_blood_pressure'] = scale.fit_transform(df[['high_blood_pressure']])
df['serum_creatinine'] = scale.fit_transform(df[['serum_creatinine']])
df['sex'] = scale.fit_transform(df[['sex']])
df['smoking'] = scale.fit_transform(df[['smoking']])
df['creatinine_phosphokinase'] = scale.fit_transform(df[['creatinine_phosphokinase']])
df['ejection_fraction'] = scale.fit_transform(df[['ejection_fraction']])
df['platelets'] = scale.fit_transform(df[['platelets']])
df['serum_sodium'] = scale.fit_transform(df[['serum_sodium']])
df['time'] = scale.fit_transform(df[['time']])

df.corr(method='pearson')
plt.figure(figsize=(16, 6))
sns.heatmap(df.corr(),vmin=-1, vmax=1, annot=True);

for name in df.columns[:]:
    CrosstabResult=pd.crosstab(index=df[name],columns=df['DEATH_EVENT'])
    # Performing Chi-sq test
    ChiSqResult = chi2_contingency(CrosstabResult)
    print('The P-Value for' ,name,'of the ChiSq Test is:', ChiSqResult[1])


#X_train,X_test,Y_train,Y_test=train_test_split(df.iloc[0:299,0:12],df.iloc[0:299,12],random_state=42,test_size=0.2)
X_train,X_test,Y_train,Y_test=train_test_split(df[['age','ejection_fraction','serum_creatinine','serum_sodium','time']],df.iloc[0:299,12],random_state=42,test_size=0.2)

model1=LinearRegression()
model1.fit(X_train,Y_train)
y_predict=model1.predict(X_test)
bool_y_predict=[1 if x>0.5 else 0 for x in y_predict]
from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test,bool_y_predict)
from sklearn.metrics import f1_score
print('f1 score is: ',f1_score(Y_test,bool_y_predict,average='weighted')*100)

