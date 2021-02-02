#
#
#
import pandas as pd
import numpy as np
import scripts.helpers as hlp
import missingno as msno
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, \
    classification_report
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

train_test = hlp.get_merge_df()


monthly_df = hlp.get_singular_monthly_expenditures()

merged_df = train_test.merge(monthly_df, how="left", on="musteri")
df = merged_df.copy()
droplist = ["musteri", "tarih"]

df.drop(droplist, axis=1, inplace=True)

type_list = ["yas", "kidem_suresi"]
for i in type_list:
    df[i] = df[i].astype(int)

num_cols = [col for col in df.columns if df[col].dtype != "O"]

df.info()
df.head()

egitim = list(df["egitim"].unique())
is_durumu = list(df["is_durumu"].unique())

egitim.pop(4)
is_durumu.pop(11)

for x in egitim:
    for y in is_durumu:
        try:
            df.loc[(df["egitim"] == x) & (df["is_durumu"] == y), "meslek_grubu"] = \
                df.loc[(df["egitim"] == x) & (df["is_durumu"] == y), "meslek_grubu"].fillna(
                    df.loc[(df["egitim"] == x) & (df["is_durumu"] == y), "meslek_grubu"].mode()[0])
        except:
            pass

fill_mod_col = ["egitim", "is_durumu", "meslek_grubu"]
for col in fill_mod_col:
    df[col] = df[col].fillna(df[col].mode()[0])


df.info()

df,new_cols = hlp.one_hot_encoder(df,["egitim","is_durumu","meslek_grubu"])
df.head()

df_train = df[df["target"].notnull()]
df_train.shape

df_test = df[df["target"].isnull()]
df_test.shape

df_train["target"] = df_train["target"].astype(int)
df_train["target"].dtype

X = df_train.drop("target", axis=1)
y = np.ravel(df_train[["target"]])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=123, stratify=y)

rf_params = {"max_depth": [3, 5, 8],
             "max_features": [8, 15, 25],
             "n_estimators": [200, 500, 1000],
             "min_samples_split": [2, 5, 10]}

rf = RandomForestClassifier(random_state=123)


gs_cv_rf = GridSearchCV(rf,
                        rf_params,
                        cv=10,
                        n_jobs=-1,
                        verbose=2).fit(X_train, y_train)

rf_tuned = RandomForestClassifier(**gs_cv_rf.best_params_, random_state=123).fit(X_train, y_train)

models = [("RF", rf_tuned)]

for name, model in models:
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    msg = "%s: (%f)" % (name, acc)
    print(msg)