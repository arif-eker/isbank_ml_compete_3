#
##
###
##
#

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV

pd.set_option('display.max_columns', None)


def one_hot_encoder(dataframe, categorical_columns, nan_as_category=False):
    """
    Drop_first doğrusal modellerde yapılması gerekli

    Ağaç modellerde gerekli değil ama yapılabilir.

    dummy_na eksik değerlerden değişken türettirir.

    :param dataframe: İşlem yapılacak dataframe
    :param categorical_columns: One-Hot Encode uygulanacak kategorik değişken adları
    :param nan_as_category: NaN değişken oluştursun mu? True/False
    :return: One-Hot Encode yapılmış dataframe ve bu işlem sonrası oluşan yeni değişken adlarını döndürür.
    """
    original_columns = list(dataframe.columns)

    dataframe = pd.get_dummies(dataframe, columns=categorical_columns,
                               dummy_na=nan_as_category, drop_first=False)

    new_columns = [col for col in dataframe.columns if col not in original_columns]

    return dataframe, new_columns


def find_correlation(dataframe, numeric_columns, target, corr_limit=0.50):
    """
    -> Sayısal değişkenlerin targetla olan korelasyonunu inceler.

    :param dataframe: İşlem yapılacak dataframe
    :param numeric_columns: Sayısal değişken adları
    :param target: Korelasyon ilişkisinde bakılacak hedef değişken
    :param corr_limit: Korelasyon sınırı. Sınırdan aşağısı düşük, yukarısı yüksek korelasyon
    :return: İlk değer düşük korelasyona sahip değişkenler, ikinci değer yüksek korelasyona sahip değişkenler
    """
    high_correlations = []

    low_correlations = []

    for col in numeric_columns:
        if col == target:
            pass

        else:
            correlation = dataframe[[col, target]].corr().loc[col, target]

            if abs(correlation) > corr_limit:
                high_correlations.append(col + " : " + str(correlation))

            else:
                low_correlations.append(col + " : " + str(correlation))

    return low_correlations, high_correlations


def get_merge_df():
    train_df = pd.read_csv("D:\my_projects\pycharms\isbank_ml_uc\datasets/train.csv")
    test_df = pd.read_csv("D:\my_projects\pycharms\isbank_ml_uc\datasets/test.csv")
    merge_df = pd.concat([train_df, test_df], ignore_index=True)
    return merge_df


def check_df(dataframe):
    print("Number of Features : {0} \n Dataframe Lenght : {1} ".format(dataframe.shape[1], dataframe.shape[0]))
    print("\n", dataframe.info())


def get_singular_monthly_expenditures():
    pure_df = pd.read_csv("D:\my_projects\pycharms\isbank_ml_uc\datasets/monthly_expenditures.csv")

    pure_df = pure_df.groupby(["musteri", "sektor"]).agg({"islem_adedi": "sum",
                                                          "aylik_toplam_tutar": "sum"})
    pure_df.reset_index(inplace=True)
    # Her bir müşterinin tekilleştirilmek üzere işlem bilgileri alındı.
    new_df = pure_df.groupby("musteri").agg({"islem_adedi": ["sum", "mean", "max", "min"],
                                             "aylik_toplam_tutar": ["sum", "mean", "max", "min"]})

    # Sütun isimlendirmesi düzeltildi
    new_df.columns = [col[0] + "_" + col[1] if col[1] else col[0] for col in new_df.columns]
    new_df.reset_index(inplace=True)

    pure_df, new_cols = one_hot_encoder(pure_df, ["sektor"])

    cols = [col for col in pure_df.columns if "sektor_" in col]

    for col in cols:
        pure_df[col] = pure_df["islem_adedi"] * pure_df[col]

    pure_processed = pure_df.groupby('musteri')[new_cols].mean().reset_index()

    agg_df = pure_processed.merge(new_df, how="left", on="musteri")

    return agg_df


def lgbm_tuned_model(x_train, y_train):
    """

    :param x_train: Train veri setinin değişkenleri
    :param y_train: Train veri setinin hedef değişkeni
    :return: İlk değer tune edilmiş model nesnesi, ikinci değer bu modelin en iyi parametreleri
    """
    lgbm_params = {"learning_rate": [0.001, 0.01, 0.1],
                   "n_estimators": [200, 500, 750, 1000],
                   "max_depth": [3, 5, 8, 10],
                   "colsample_bytree": [1, 0.8, 0.5],
                   "num_leaves": [32, 64, 128]}

    lgbm = LGBMClassifier(random_state=123)

    gs_cv_lgbm = GridSearchCV(lgbm,
                              lgbm_params,
                              cv=10,
                              n_jobs=-1,
                              verbose=2).fit(x_train, y_train)

    lgbm_tuned = LGBMClassifier(**gs_cv_lgbm.best_params_, random_state=123).fit(x_train, y_train)

    return lgbm_tuned, gs_cv_lgbm.best_params_


def rf_tuned_model(x_train, y_train):
    """

    :param x_train: Train veri setinin değişkenleri
    :param y_train: Train veri setinin hedef değişkeni
    :return: İlk değer tune edilmiş model nesnesi, ikinci değer bu modelin en iyi parametreleri
    """
    rf_params = {"max_depth": [3, 5, 8],
                 "max_features": [8, 15, 20],
                 "n_estimators": [200, 500, 750, 1000],
                 "min_samples_split": [2, 5, 8, 10]}

    rf = RandomForestClassifier(random_state=123)

    gs_cv_rf = GridSearchCV(rf,
                            rf_params,
                            cv=10,
                            n_jobs=-1,
                            verbose=2).fit(x_train, y_train)

    rf_tuned = RandomForestClassifier(**gs_cv_rf.best_params_, random_state=123).fit(x_train, y_train)

    return rf_tuned, gs_cv_rf.best_params_
