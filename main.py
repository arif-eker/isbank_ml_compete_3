#
#
#
import pandas as pd
import numpy as np
import isbank_ml_compete_3.scripts.helpers as hlp

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.1f' % x)

train_test = hlp.get_merge_df()

# Random Under Sampling Yapıyoruz
train_df = pd.read_csv("datasets/train.csv")
ranUnSample = RandomUnderSampler(sampling_strategy= 0.3)
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


# *************************************************************

# yasın targettaki kırılımına göre aralık belirledik
merged.groupby("yas").agg({"target":"mean"})
bins = [18,25,36,42,50]
labels = ["18_25","26_36","37_42","43_50"]
merged["yas_aralik"] = pd.cut(merged["yas"],bins= bins,labels=labels)
merged.groupby("yas_aralik").agg({"target":["mean","count"]})


# kidem_suresi targettaki kırılımına göre aralık belirledik
merged["kidem_aralik"] = pd.qcut(merged["kidem_suresi"],5,labels=["q1","q2","q3","q4","q5"])
merged.head()
merged.groupby("kidem_aralik").agg({"target":"mean"})

# müşteri puanı
merged["kidem_yil"] =  round(merged["kidem_suresi"] / 12,0)
merged.head()


df = merged.copy()
len(df)
droplist = ["musteri", "tarih"]

df.drop(droplist, axis=1, inplace=True)

type_list = ["yas", "kidem_suresi"]
for i in type_list:
    df[i] = df[i].astype(int)

df["yas_aralik"] = df["yas_aralik"].astype("object")
df["kidem_aralik"] = df["kidem_aralik"].astype("object")

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

df, new_cols = hlp.one_hot_encoder(df, ["egitim", "is_durumu", "meslek_grubu","yas_aralik","kidem_aralik"])

X_train, y_train, X_test = hlp.get_train_test_data(df)

lgbm_tuned, best_params = hlp.lgbm_tuned_model(X_train, y_train)

# Submission File
y_preds = lgbm_tuned.predict(X_test)

hlp.do_submission(merged, y_preds, "lgbm_10_02_with_undersampling")

hlp.save_best_params("lgbm", best_params, 0.71037)
