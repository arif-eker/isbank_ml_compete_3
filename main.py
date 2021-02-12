#
#
#
import pandas as pd
import numpy as np
import scripts.helpers as hlp

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.1f' % x)

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
df, new_cols = hlp.one_hot_encoder(df, ["egitim", "is_durumu", "meslek_grubu", "yas_aralik", "kidem_aralik"])

# Model için dataframe train ve test olmak üzere bölünüyor.
X_train, y_train, X_test = hlp.train_test_split_data(df)

# Under Sampling yapılmış eğitim seti
X_train, y_train = hlp.under_sampler(X_train, y_train)

# SMOTE Over Sampling yapılmış eğitim seti
# X_train, y_train = hlp.over_sampler(X_train, y_train)

# LGBM modeli oluşturuluyor.
lgbm_tuned, best_params = hlp.lgbm_tuned_model(X_train, y_train)

# Eğitilen modelden tahminler yapılıyor.
y_preds = lgbm_tuned.predict(X_test)

# Yarışma için submission dosyası hazırlanıyor.
hlp.do_submission(merged, y_preds, "lgbm_11_02_with_secon_oversamplingandundersamping_afterall")

# En iyi parametreler kaydediliyor.
hlp.save_best_params("lgbm", best_params, 0.60124)
