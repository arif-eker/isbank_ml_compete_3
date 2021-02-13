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

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE

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


def train_test_df():
    """

    Train ve test verilerini birleştirip, döndürür.

    :return: birleştirilmiş dataframe
    """
    train_df = pd.read_csv("datasets/train.csv")
    test_df = pd.read_csv("datasets/test.csv")
    merge_df = pd.concat([train_df, test_df], ignore_index=True)
    return merge_df


def check_df(dataframe):
    print("Number of Features : {0} \n Dataframe Lenght : {1} ".format(dataframe.shape[1], dataframe.shape[0]))
    print("\n", dataframe.info())


def get_singular_monthly_expenditures():
    """

    Müşterilerin aylık harcama verilerinin, karşılaştırma yapabilmek için tekilleştirilmiş halde alınmasını sağlar.
    Her bir müşteri için tek bir gözlem birimi vardır.

    :return: teklleştirilmiş verilerden oluşan bir dataframe
    """
    pure_df = pd.read_csv("datasets/monthly_expenditures.csv")

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


def fillna_with_mode(dataframe):
    """

    Bu fonksiyon verilen dataframe'deki  kategorik eksik değerleri kırılımlara göre doldurur.

    :param dataframe: Eksik değeri olan dataframe
    :return:
    """
    # Sınıflar alınıyor.
    egitim = list(dataframe["egitim"].unique())
    is_durumu = list(dataframe["is_durumu"].unique())

    # Her ikisinde de "nan" olan sınıf düşürülüyor.
    egitim.pop(4)
    is_durumu.pop(11)

    for x in egitim:
        for y in is_durumu:
            try:
                dataframe.loc[(dataframe["egitim"] == x) & (dataframe["is_durumu"] == y), "meslek_grubu"] = \
                    dataframe.loc[(dataframe["egitim"] == x) & (dataframe["is_durumu"] == y), "meslek_grubu"].fillna(
                        dataframe.loc[
                            (dataframe["egitim"] == x) & (dataframe["is_durumu"] == y), "meslek_grubu"].mode()[0])
            except:
                pass

    fill_mod_col = ["egitim", "is_durumu", "meslek_grubu"]
    for col in fill_mod_col:
        dataframe[col] = dataframe[col].fillna(dataframe[col].mode()[0])


def add_new_features(dataframe):
    """

    Bu fonksiyon verilen dataframe için değişken türetir.

    :param dataframe: İçerisine değişken türetilecek dataframe
    :return:
    """
    # Yaş aralığı belirleniyor.
    bins = [18, 24, 29, 34, 39, 44, 50]
    labels = ["18_24", "25_29", "30_34", "35_39", "40_44", "45_50"]

    dataframe["yas_aralik"] = pd.cut(dataframe["yas"], bins=bins, labels=labels)
    dataframe["yas_aralik"] = dataframe["yas_aralik"].astype("object")

    # Kıdem süresi aralığı belirleniyor.
    dataframe["kidem_aralik"] = pd.qcut(dataframe["kidem_suresi"], 5, labels=["q1", "q2", "q3", "q4", "q5"])
    dataframe["kidem_aralik"] = dataframe["kidem_aralik"].astype("object")

    # Kaç yıllık müşteri belirleniyor. virgülden sonraki rakam 0.5 ve üstüyse bir üst basamağa yuvarlanır. 2.6 = 3 olur.
    dataframe["kidem_yil"] = round(dataframe["kidem_suresi"] / 12, 0)
    dataframe.drop("kidem_suresi", axis=1, inplace=True)

    seg_map_binde_2019 = {
        "18_24": 4.4530,
        "25_29": 4.5453,
        "30_34": 1.8451,
        "35_39": 0.8403,
        "40_44": 0.4422,
        "45_50": 0.2646,
    }
    dataframe["Evlilik_Orani_Binde_2019"] = dataframe["yas_aralik"].replace(seg_map_binde_2019, regex=True)

    # Aylara Göre Dolar Kuru
    dataframe["01_2019_Kur"] = 5.3759
    dataframe["02_2019_Kur"] = 5.2769
    dataframe["03_2019_Kur"] = 5.4666
    dataframe["04_2019_Kur"] = 5.7617
    dataframe["05_2019_Kur"] = 6.0560
    dataframe["06_2019_Kur"] = 5.8127
    dataframe["07_2019_Kur"] = 5.6718
    dataframe["08_2019_Kur"] = 5.6511
    dataframe["09_2019_Kur"] = 5.7150
    dataframe["10_2019_Kur"] = 5.7993
    dataframe["11_2019_Kur"] = 5.7431
    dataframe["12_2019_Kur"] = 5.8572

    # Aylara Göre Gram Altın Fiyatları
    dataframe["01_2019_GramAltin"] = 222.65
    dataframe["02_2019_GramAltin"] = 223.75
    dataframe["03_2019_GramAltin"] = 228.43
    dataframe["04_2019_GramAltin"] = 238.45
    dataframe["05_2019_GramAltin"] = 249.52
    dataframe["06_2019_GramAltin"] = 254.30
    dataframe["07_2019_GramAltin"] = 257.09
    dataframe["08_2019_GramAltin"] = 272.43
    dataframe["09_2019_GramAltin"] = 276.43
    dataframe["10_2019_GramAltin"] = 278.30
    dataframe["11_2019_GramAltin"] = 270.52
    dataframe["12_2019_GramAltin"] = 278.11

    # Aylara Göre Çeyrek Altın Fiyatları
    dataframe["01_2019_CeyrekAltin"] = 357.73
    dataframe["02_2019_CeyrekAltin"] = 359.49
    dataframe["03_2019_CeyrekAltin"] = 367.01
    dataframe["04_2019_CeyrekAltin"] = 383.11
    dataframe["05_2019_CeyrekAltin"] = 400.90
    dataframe["06_2019_CeyrekAltin"] = 408.57
    dataframe["07_2019_CeyrekAltin"] = 413.05
    dataframe["08_2019_CeyrekAltin"] = 437.70
    dataframe["09_2019_CeyrekAltin"] = 444.13
    dataframe["10_2019_CeyrekAltin"] = 447.14
    dataframe["11_2019_CeyrekAltin"] = 434.64
    dataframe["12_2019_CeyrekAltin"] = 446.82

    # Aylara Göre Enflasyon Oranı
    dataframe["01_2019_Enflasyon"] = 20.35
    dataframe["02_2019_Enflasyon"] = 19.67
    dataframe["03_2019_Enflasyon"] = 19.71
    dataframe["04_2019_Enflasyon"] = 19.50
    dataframe["05_2019_Enflasyon"] = 18.71
    dataframe["06_2019_Enflasyon"] = 15.72
    dataframe["07_2019_Enflasyon"] = 16.65
    dataframe["08_2019_Enflasyon"] = 15.01
    dataframe["09_2019_Enflasyon"] = 9.26
    dataframe["10_2019_Enflasyon"] = 8.55
    dataframe["11_2019_Enflasyon"] = 10.56
    dataframe["12_2019_Enflasyon"] = 11.84

    # Aylara Göre Tüketici Güven Ekdeksi
    dataframe["01_2019_GuvenEndeksi"] = 80.50
    dataframe["02_2019_GuvenEndeksi"] = 79.21
    dataframe["03_2019_GuvenEndeksi"] = 81.30
    dataframe["04_2019_GuvenEndeksi"] = 83.61
    dataframe["05_2019_GuvenEndeksi"] = 76.89
    dataframe["06_2019_GuvenEndeksi"] = 79.75
    dataframe["07_2019_GuvenEndeksi"] = 78.25
    dataframe["08_2019_GuvenEndeksi"] = 79.14
    dataframe["09_2019_GuvenEndeksi"] = 77.65
    dataframe["10_2019_GuvenEndeksi"] = 78.46
    dataframe["11_2019_GuvenEndeksi"] = 81.28
    dataframe["12_2019_GuvenEndeksi"] = 80.72


def lgbm_tuned_model(x_train, y_train):
    """

    LGBM modeli kurar, modeli tune eder ve geriye tune edilmiş model ile bu model için gereken en iyi parametreleri döndürür.

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

    Random Forest için model kurar, bu modeli tune eder ve geriye tune edilmiş model ile bu model için gereken en iyi parametreleri döndürür.

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


def do_submission(dataframe, y_predictions, file_name):
    """
    Bu fonksiyon tahmin edilen değişkenlerden ve verilen dosya adı ile bir submission dosyası oluşturur.
    Dosyayı datasets klasörü altında kaydeder.

    :param dataframe: Dataframe
    :param y_predictions: Tahmin edilen y değişenleri
    :param file_name: Submisson dosyasının adı
    :return:
    """
    sub = pd.DataFrame()
    # sub["musteri"] = merged_df[merged_df["target"].isnull()]["musteri"]
    sub["musteri"] = dataframe[dataframe["target"].isnull()]["musteri"]
    sub["target"] = y_predictions
    file_path = "datasets/" + file_name + ".csv"
    sub.to_csv(file_path, index=False)


def save_best_params(model_name, best_parameters, point):
    """
    Modeller için en iyi parametreleri kaydeder.

    :param model_name: Hangi model kullanıldı ise o modelin ismi
    :param best_parameters: Modelin best parametreleri
    :param point: Bu modelin yarışmadaki puanı
    :return:
    """
    f_add = open("best_params/best_params.txt", "a")

    if model_name == "rf":
        f_add.writelines(
            "\nmax_depth : {0} -- max_features : {1} -- min_samples_split : {2} -- n_estimators : {3}".format(
                best_parameters["max_depth"],
                best_parameters["max_features"],
                best_parameters["min_samples_split"],
                best_parameters["n_estimators"]))

        f_add.writelines("\nYukarıdaki parametreler RF için: {0}  submission puanına sahip.\n".format(point))
        f_add.close()
    elif model_name == "lgbm":
        f_add.writelines(
            "\nlearning_rate : {0} -- n_estimators : {1} -- max_depth : {2} -- colsample_bytree : {3} -- num_leaves : {4}".format(
                best_parameters["learning_rate"],
                best_parameters["n_estimators"],
                best_parameters["max_depth"],
                best_parameters["colsample_bytree"],
                best_parameters["num_leaves"]))

        f_add.writelines("\nYukarıdaki parametreler LGBM için: {0}  submission puanına sahip.\n".format(point))
        f_add.close()
    else:
        print("Geçerli bir model ismi giriniz!!!")


def train_test_split_data(dataframe):
    """

    Verilen dataframe'den X_train, y_train, X_test verilerini elde ederek döndürür.

    :param dataframe: Train ve Test için bölümlere ayrılacak ana dataframe
    :return: X_train ; train edilecek veri. y_train ; train için hedef değişken. X_test ; test edilecek veri.
    """
    df_train = dataframe[dataframe["target"].notnull()]

    df_test = dataframe[dataframe["target"].isnull()]

    x_train = df_train.drop("target", axis=1)
    y_train = np.ravel(df_train[["target"]])

    x_test = df_test.drop("target", axis=1)
    # y_test = np.ravel(df_test[["target"]])

    return x_train, y_train, x_test


def under_sampler(x_train, y_train):
    ranUnSample = RandomUnderSampler()

    x_ranUnSample, y_ranUnSample = ranUnSample.fit_resample(x_train, y_train)

    return x_ranUnSample, y_ranUnSample


def over_sampler(x_train, y_train):
    oversample = SMOTE(sampling_strategy=0.06)

    x_smote, y_smote = oversample.fit_resample(x_train, y_train)

    x_under, y_under = under_sampler(x_smote, y_smote)

    return x_under, y_under
