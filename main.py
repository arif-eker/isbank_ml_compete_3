#
#
#
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, \
    classification_report
#
import pickle
import pandas as pd
import numpy as np
import scripts.helpers as hlp
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

# train_test = hlp.train_test_df()
train = pd.read_csv("datasets/train.csv")
test = pd.read_csv("datasets/test.csv")

hlp.fillna_with_mode(train)
hlp.fillna_with_mode(test)

mergee = pd.concat([train, test], ignore_index=True)
hlp.add_new_features(mergee)


X_train, y_train, X_test = hlp.train_test_split_data(mergee)
X_train, y_train = hlp.under_sampler(X_train, y_train)

train_test = X_train.copy()
train_test["target"] = y_train
train_test = pd.concat([train_test,X_test], ignore_index=True)

monthly_df = hlp.get_singular_monthly_expenditures()

merged = train_test.merge(monthly_df, how="left", on="musteri")

df = merged.copy()

droplist = ["musteri", "tarih"]

df.drop(droplist, axis=1, inplace=True)

# Kategorik olan eksik değerler kırılımlara göre dolduruluyor.
# hlp.fillna_with_mode(df)

# Yeni değişkenler türetiliyor.
# hlp.add_new_features(df)

# Kategorik değişkenler modele girmek üzere one-hot ediliyor.
df, new_cols = hlp.one_hot_encoder(df, ["egitim", "is_durumu", "meslek_grubu", "yas_aralik", "kidem_aralik",
                                        "Evlilik_Orani_Binde_2019"])

# Model için dataframe train ve test olmak üzere bölünüyor.
X_train, y_train, X_test = hlp.train_test_split_data(df)

# Under Sampling yapılmış eğitim seti
# X_train, y_train = hlp.under_sampler(X_train, y_train)

print(pd.DataFrame(y_train).value_counts())
# *********************************************************************** #

# SMOTE Over Sampling yapılmış eğitim seti
# X_train, y_train = hlp.over_sampler(X_train, y_train)


# LGBM modeli oluşturuluyor.
lgbm_tuned, best_params = hlp.lgbm_tuned_model(X_train, y_train)


pickle.dump(lgbm_tuned, open("datasets/" + "lgbm_tuned" + ".pkl", "wb"))
lgbm_from_pickle = pickle.load(open("datasets/lgbm_tuned.pkl", "rb"))
y_ppreedds = lgbm_from_pickle.predict(X_test)


# Eğitilen modelden tahminler yapılıyor.
y_preds = lgbm_tuned.predict(X_test)

# Yarışma için submission dosyası hazırlanıyor.
hlp.do_submission(merged, y_preds, "lgbm_13_02_sixth_under_with_evlilik")

# En iyi parametreler kaydediliyor.
hlp.save_best_params("lgbm", best_params, 0.71051)
