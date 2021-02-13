#
#
#
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, \
    classification_report
#

import pandas as pd
import numpy as np
import scripts.helpers as hlp
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

train_test = hlp.train_test_df()

monthly_df = hlp.get_singular_monthly_expenditures()

merged = train_test.merge(monthly_df, how="left", on="musteri")

df = merged.copy()

droplist = ["musteri", "tarih"]

df.drop(droplist, axis=1, inplace=True)

# Kategorik olan eksik değerler kırılımlara göre dolduruluyor.
hlp.fillna_with_mode(df)

# Yeni değişkenler türetiliyor.
hlp.add_new_features(df)

# Kategorik değişkenler modele girmek üzere one-hot ediliyor.
df, new_cols = hlp.one_hot_encoder(df, ["egitim", "is_durumu", "meslek_grubu", "yas_aralik", "kidem_aralik",
                                        "Evlilik_Orani_Binde_2019"])

# Model için dataframe train ve test olmak üzere bölünüyor.
X_train, y_train, X_test = hlp.train_test_split_data(df)

# Under Sampling yapılmış eğitim seti
# X_train, y_train = hlp.under_sampler(X_train, y_train)
#
# x_train, x_test, yy_train, yy_test = train_test_split(X_train, y_train, test_size=0.20)
#
# lgbm, new_best_params = hlp.lgbm_tuned_model(x_train, yy_train)
#
# y_pred = lgbm.predict(x_test)
# acc = accuracy_score(yy_test, y_pred)
# msg = "(%f)" % acc
# print(msg)
# print(classification_report(yy_test, y_pred))

# *********************************************************************** #

# SMOTE Over Sampling yapılmış eğitim seti
X_train, y_train = hlp.over_sampler(X_train, y_train)

print(pd.DataFrame(y_train).value_counts())

# LGBM modeli oluşturuluyor.
lgbm_tuned, best_params = hlp.lgbm_tuned_model(X_train, y_train)

# Eğitilen modelden tahminler yapılıyor.
y_preds = lgbm_tuned.predict(X_test)

# Yarışma için submission dosyası hazırlanıyor.
hlp.do_submission(merged, y_preds, "lgbm_13_02_overandunder_with_evlilik")

# En iyi parametreler kaydediliyor.
hlp.save_best_params("lgbm", best_params, 0.70945)
