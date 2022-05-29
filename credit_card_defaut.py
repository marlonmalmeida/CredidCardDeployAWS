import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
#from google.colab import drive 
#drive.mount('/content/drive')
np.random.seed(42)
import pickle

#Importando do drive do Marlon
df = pd.read_csv(r"C://Users//marlo//deepprojetos//creditcard.csv")
#Importando do drive do Rodrigo
#os.chdir('/content/drive/MyDrive/Colab Notebooks/FGV MFEE 2021/Machine Learning/Trabalho Final')
#df=pd.read_csv('train.csv')

df.head()
default_prop = df["credit_card_default"].value_counts()/len(df)
default_prop
df["gender"].value_counts()
df.drop(df[df['gender']=='XNA'].index,inplace=True)
df["gender"].value_counts()
gender_prop = df["gender"].value_counts()/len(df)
gender_prop
gender_default_prop = df[df['credit_card_default']==1]['gender'].value_counts()/df["gender"].value_counts()
gender_default_prop
gender_resume = pd.concat([gender_prop, gender_default_prop], axis=1)*100
gender_resume.columns = ['% de Instâncias', '% de Default']
gender_resume
df2 = df.copy()
df['credit_score'].unique()
df.drop(df[df['credit_score'].isna()].index, inplace=True)
df['credit_score'].isna().sum()
df['cred_score_cat'] = pd.cut(df['credit_score'], bins=([0, 700, 850, np.inf]), labels=['<700', '<850', '850+'])
df.groupby('cred_score_cat')['customer_id'].count()/len(df)
df['cred_score_cat']
df.head()
df['gender_cred_score_cat'] = df['gender'].astype(str) + df['cred_score_cat'].astype(str)
df['gender_cred_score_cat']
gender_cred_score_prop = df.groupby('gender_cred_score_cat')['customer_id'].count()/len(df)
gender_cred_score_prop
df.drop('cred_score_cat', axis=1, inplace=True)
df.head()
df = df.reset_index()
df.head()
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
for train_index, test_index in split.split(df, df["gender_cred_score_cat"]):
    train_set = df.loc[train_index]
    test_set = df.loc[test_index]
gender_cred_score_prop_train = train_set.groupby('gender_cred_score_cat')['customer_id'].count()/len(train_set)
gender_cred_score_prop_test = test_set.groupby('gender_cred_score_cat')['customer_id'].count()/len(test_set)

compara_props_strat = pd.concat([gender_cred_score_prop, gender_cred_score_prop_test, gender_cred_score_prop_test],axis=1)
compara_props_strat.columns = ['Prop DF', 'Prop Train Set', 'Prop Test Set']
compara_props_strat
#Excluindo da base de treino e teste atraibuto Credit Score Cat
train_set.drop('gender_cred_score_cat', axis=1, inplace=True)
test_set.drop('gender_cred_score_cat', axis=1, inplace=True)
train_set.columns
test_set.columns
default = train_set.copy()
default.head()
todrop = ['credit_card_default', 'index']
default_X = train_set.drop(todrop, axis=1)
default_X.head()
default_Y = train_set["credit_card_default"].copy()
default_Y.head()

default_X.head()
cat = ['gender', 'owns_car', 'owns_house', 'occupation_type', 'migrant_worker']
cat
default_X_cat = default_X[cat]
default_X_cat
num = ['age', 'no_of_children', 'net_yearly_income', 'no_of_days_employed' ,
       'total_family_members', 'yearly_debt_payments', 'credit_limit',
       'credit_limit_used(%)', 'credit_score', 'prev_defaults', 'default_in_last_6months']
num
default_X_num = default_X[num]
default_X_num.head()







from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
                         ('imputer', SimpleImputer(strategy='median')),
                         ('std_scaler', StandardScaler())
])
default_X_num_prep = num_pipeline.fit_transform(default_X_num)
default_X_num_prep
default_X_num_prep.shape

pd.DataFrame(default_X_num_prep, columns=num).describe()
pd.DataFrame(default_X_num_prep, columns=num).hist(bins=50, figsize=(20,15))
plt.show
default_X_cat

from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()

default_X_cat_1hot = cat_encoder.fit_transform(default_X_cat)
default_X_cat_1hot = default_X_cat_1hot.toarray()
default_X_cat_1hot

from sklearn.compose import ColumnTransformer

num2 = ['age', 'net_yearly_income','credit_limit','credit_score','default_in_last_6months','credit_limit_used(%)']
cat2 = ['owns_house']

full_pipeline = ColumnTransformer([
                                   ("num", num_pipeline, num),
                                   ("cat",OneHotEncoder(), cat)])

full_pipeline2 = ColumnTransformer([("num", num_pipeline, num2)])
full_pipeline3 = ColumnTransformer([("num", num_pipeline, num2)])
                                 
default_X_new = default_X.copy()
todrop = ['customer_id','name', 'gender','owns_car','owns_house','no_of_children','no_of_days_employed','occupation_type','total_family_members','migrant_worker','yearly_debt_payments','prev_defaults']
default_X_new = default_X_new.drop(todrop, axis=1)
default_X_new.head()

default_X_prep = full_pipeline.fit_transform(default_X)
default_X_prep2 = full_pipeline2.fit_transform(default_X_new)

default_X_prep.shape
todrop = ['credit_card_default', 'index']
default_X_test = test_set.drop(todrop, axis=1)
default_X_test.head()

default_Y_test = test_set["credit_card_default"].copy()
default_Y_test.head()

default_X_test_prep = full_pipeline.transform(default_X_test)
default_X_test_prep


from imblearn.over_sampling import SMOTE
smt = SMOTE()

X1_train_smote = default_X_prep2 
Y1_train_smote = default_Y
X1_train_smote, Y1_train_smote = smt.fit_resample(X1_train_smote, Y1_train_smote)

#Função para retornar matriz de confusão, precision, recall e f1
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
def score_completo(Y, Y_PRED):
  conf_mx = confusion_matrix(Y, Y_PRED)
  conf_mx = pd.DataFrame(conf_mx, columns=['Negativo Previsto', 'Positivo Previsto'], index=['Negativo Real', 'Positivo Real'])
  accuracy = round(accuracy_score(Y, Y_PRED),4)
  precision = round(precision_score(Y, Y_PRED),4)
  recall = round(recall_score(Y, Y_PRED),4)
  f1 = round(f1_score(Y, Y_PRED),4)

  return print(f'\033[1mConfusion Matrix\033[0m \n{conf_mx}\n\n Accuracy: {accuracy} | Precision: {precision} | Recall: {recall} | F1: {f1}')

from sklearn.linear_model import LogisticRegression

lrsmote = LogisticRegression(max_iter=200)
lrsmotepronto = lrsmote.fit(X1_train_smote, Y1_train_smote)
modelpronto = lrsmote.fit(X1_train_smote, Y1_train_smote)
pred_train_lr_smote = lrsmotepronto.predict(X1_train_smote)
#Score sem validação cruzada
score_completo(Y1_train_smote, pred_train_lr_smote)



from sklearn.ensemble import GradientBoostingClassifier

gb_clf_2 = GradientBoostingClassifier(learning_rate=0.05, max_depth=19)
gb_pronto = gb_clf_2.fit(default_X_prep, default_Y)
gb_pronto2 = gb_clf_2.fit(default_X_prep2, default_Y)

#Predição sem validação cruzada
pred_train_gb_clf = gb_clf_2.predict(default_X_prep2)
#Score com predição sem validação cruzada
score_completo(default_Y, pred_train_gb_clf)


lr = LogisticRegression(max_iter=200)
lr_2 = lr.fit(default_X_prep2, default_Y)
pred_train_lr = lr.predict(default_X_prep2)
pred_train_lr

#Score sem validação cruzada
score_completo(default_Y, pred_train_lr)


from sklearn.ensemble import BaggingClassifier
bag_lr = BaggingClassifier(lr)
bag_lr.fit(default_X_prep2, default_Y)
pred_train_bag_lr = bag_lr.predict(default_X_prep2)
pred_train_bag_lr
score_completo(default_Y, pred_train_bag_lr)
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping





#Salvando modelo 
with open ('gb_pronto.pkl','wb') as f:
    pickle.dump(gb_pronto, f)

with open('lr_2.pkl','wb') as f:
    pickle.dump(lr_2,f)  
with open ('gb_pronto2.pkl','wb') as f:
    pickle.dump(gb_pronto2, f)
with open('full_pipeline2.pkl','wb') as f:
    pickle.dump(full_pipeline2,f)  
with open('full_pipeline3.pkl','wb') as f:
    pickle.dump(full_pipeline3,f) 
with open('lrsmotepronto.pkl','wb') as f:
    pickle.dump(lrsmotepronto ,f)
with open('modelpronto.pkl','wb') as f:
    pickle.dump(modelpronto,f)

