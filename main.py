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
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, \
    classification_report
from sklearn.metrics import confusion_matrix, classification_report, f1_score, recall_score

from imblearn.under_sampling import RandomUnderSampler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

train_test = hlp.get_merge_df()

# Random Under Sampling Yapıyoruz
train_df = pd.read_csv("datasets/train.csv")
ranUnSample = RandomUnderSampler()
X_train = train_df.drop("target", axis=1)
y_train = np.ravel(train_df[["target"]])
X_ranUnSample, y_ranUnSample = ranUnSample.fit_resample(X_train, y_train, )
X_ranUnSample["target"] = y_ranUnSample

train_df = X_ranUnSample.copy()
test_df = pd.read_csv("datasets/test.csv")
merged = pd.concat([train_df, test_df], ignore_index=True)

monthly_df = hlp.get_singular_monthly_expenditures()
merged = merged.merge(monthly_df, how="left", on="musteri")
# merged_df = train_test.merge(monthly_df, how="left", on="musteri")

# df = merged_df.copy()

df = merged.copy()
len(df)
droplist = ["musteri", "tarih"]

df.drop(droplist, axis=1, inplace=True)

type_list = ["yas", "kidem_suresi"]
for i in type_list:
    df[i] = df[i].astype(int)

num_cols = [col for col in df.columns if df[col].dtype != "O"]

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

df, new_cols = hlp.one_hot_encoder(df, ["egitim", "is_durumu", "meslek_grubu"])

df_train = df[df["target"].notnull()]

df_test = df[df["target"].isnull()]

df_train["target"] = df_train["target"].astype(int)

X_train = df_train.drop("target", axis=1)
y_train = np.ravel(df_train[["target"]])

X_test = df_test.drop("target", axis=1)
# y_test = np.ravel(df_test[["target"]])


lgbm_tuned, best_params = hlp.lgbm_tuned_model(X_train, y_train)

# Submission File
y_preds = lgbm_tuned.predict(X_test)

hlp.do_submission(merged, y_preds, "file_name")

hlp.save_best_params("lgbm", best_params, 0.71037)
