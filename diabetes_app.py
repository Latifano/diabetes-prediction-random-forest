import itertools
import pandas as pd
import numpy

# matplotlib
import matplotlib.pyplot as plt

# skicit learn
from sklearn.preprocessing import Normalizer, MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from scipy.stats import randint

from collections import Counter

# imbalance learn undersample
from imblearn.under_sampling import RandomUnderSampler

# Streamlit
import streamlit as st
import time
import pickle
from PIL import Image
img = Image.open('diabetes.png')

# Read Model
diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))
# Import Dataset
diabetes_dataset = pd.read_csv('diabetes.csv')

# ========= Hitung Jumlah Fitur =========
# daftar fitur / kolom dataset
features = list(diabetes_dataset.columns)

# konstanta jumlah fitur / kolom dataset
num_of_feature = len(features)


# ========= Cleaning Data =========
columnScore = {}
for i in range(0, num_of_feature - 1):
    columnScore[features[i]] = len(
        diabetes_dataset[diabetes_dataset[features[i]] == 0].index) / diabetes_dataset[features[i]].shape[0]

target = {}
for i in range(0, num_of_feature - 1):
    target[features[i]] = diabetes_dataset[features[num_of_feature - 1]
                                           ].corr(diabetes_dataset[features[i]])

avg_target = sum(target.values()) / len(target)

for i in range(0, num_of_feature - 1):
    if (columnScore[features[i]] > 0.2 and target[features[i]]):
        diabetes_dataset.drop(features[i], inplace=True, axis=1)

features = list(diabetes_dataset.columns)
num_of_feature = len(features)

listIndex = []

for i in range(0, num_of_feature - 1):
    listIndex.append(
        list(diabetes_dataset[diabetes_dataset[features[i]] == 0].index))

rowsNewWithMV = set(listIndex[0])

data_frame_new = diabetes_dataset.drop(rowsNewWithMV)


# ========= Transformasi Data =========
def groupPregnancies(pgn):
    if pgn > 0 and pgn < 6:
        return 0
    else:
        return 1


data_frame_new['Pregnancies'] = data_frame_new['Pregnancies'].apply(
    lambda x: groupPregnancies(x))


def groupGlucose(glc):
    if glc > 0 and glc < 140:
        return 0
    elif glc >= 140 and glc <= 199:
        return 1
    else:
        return 2


data_frame_new['Glucose'] = data_frame_new['Glucose'].apply(
    lambda x: groupGlucose(x))


def groupBloodPressure(bp):
    if bp > 0 and bp < 80:
        return 0
    elif bp >= 80 and bp <= 89:
        return 1
    else:
        return 2


data_frame_new['BloodPressure'] = data_frame_new['BloodPressure'].apply(
    lambda x: groupBloodPressure(x))


def groupBMI(bmi):
    if bmi > 0 and bmi < 18.5:
        return 0
    elif bmi >= 18.5 and bmi <= 22.9:
        return 1
    elif bmi >= 23 and bmi <= 29.0:
        return 2
    else:
        return 3


data_frame_new['BMI'] = data_frame_new['BMI'].apply(lambda x: groupBMI(x))


def groupDiabetesPedigreeFunction(dpf):
    if dpf > 0 and dpf < 0.376:
        return 0
    else:
        return 1


data_frame_new['DiabetesPedigreeFunction'] = data_frame_new['DiabetesPedigreeFunction'].apply(
    lambda x: groupDiabetesPedigreeFunction(x))


def groupAge(age):
    if age > 0 and age < 25:
        return 0
    if age >= 26 and age <= 45:
        return 1
    if age >= 46 and age <= 65:
        return 2
    else:
        return 3


data_frame_new['Age'] = data_frame_new['Age'].apply(lambda x: groupAge(x))


# ========= Data Balancing =========
x = data_frame_new.drop(columns='Outcome', axis=1)
y = data_frame_new['Outcome']

rus = RandomUnderSampler(random_state=0)
x_resampled, y_resampled = rus.fit_resample(x, y)


# ========= Pembagian Data =========
# Memisahkan data training dan data testing
x_train, x_test, y_train, y_test = train_test_split(
    x_resampled, y_resampled, test_size=0.1, stratify=y_resampled, random_state=2)


# ========= Normalisasi Data =========
scale_x = MinMaxScaler()
x_train = scale_x.fit_transform(x_train)
x_test = scale_x.transform(x_test)


# ========= Data train with Random Forest =========
rf = RandomForestClassifier()
rf.fit(x_train, y_train)

# ========= Modelling =========
x_train_prediction = rf.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)
x_test_prediction = rf.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)

# ========= Hyperparameter & Confusion Matrix =========
# param_dist = {'n_estimators': randint(50, 100), 'max_depth': randint(1, 20)}
# # Menggunakan random search untuk mencari hyperparameters terbaik
# rand_search = RandomizedSearchCV(
#     rf,  param_distributions=param_dist, n_iter=5, cv=5)
# # Fit random search dengan data
# rand_search.fit(x_train, y_train)
# # Buat variabel untuk model terbaik
# best_rf = rand_search.best_estimator_
# # Print hyperparameters terbaik
# print('Best hyperparameters:',  rand_search.best_params_)

# y_pred = best_rf.predict(x_test)
# # Buat the confusion matrix
# cm = confusion_matrix(y_test, y_pred)

# y_pred = best_rf.predict(x_test)

# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)

test_data_accuracy = round((test_data_accuracy * 100), 2)

df_final = x_resampled
df_final['target'] = y_resampled

# ========================== Streamllit ==========================

# untuk Judul website
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon=img
)

st.title("Diabetes Classification")
st.write("with Random Forest Algorithm")
st.write(f"**_Model's Accuracy_** : :green[**{test_data_accuracy}**]%")
st.write("")

tab1, tab2 = st.tabs(["Hasil Klasifikasi", "Transformasi Data"])


# Tab 1 (Sidebar user input)
with tab1:
    st.sidebar.header("**User Input** Sidebar")
    st.sidebar.write(
        "Lihat pada menu Transformasi Data untuk mengetahui kategori / skor tiap atribut")

    # Pregnancies = st.sidebar.text_input('Input Jumlah Kehamilan')
    # Glucose = st.sidebar.text_input('Input Glukosa')
    # BloodPressure = st.sidebar.text_input('Input Tekanan Darah')
    # BMI = st.sidebar.text_input('Input Indeks Massa Tubuh')
    # DiabetesPedigreeFunction = st.sidebar.text_input(
    #     'Input Presentase Keturunan Diabetes')
    # Age = st.sidebar.text_input('Input Umur')

    # ===== Kehamilan =====
    # values = ["Hamil", "Tidak Hamil"]
    # Pregnancies = st.sidebar.selectbox(
    #     "Input Kehamilan", (
    #         values
    #     ),
    #     placeholder="Input Pernah Hamil atau Tidak"
    # )
    # if "Hamil":
    #     return 0
    # elif "Tidak Hamil":
    #     return 1

    # st.sidebar.write("")

    Pregnancies = st.sidebar.number_input(
        label=":blue[**Input Jumlah Kehamilan**]", min_value=df_final['Pregnancies'].min())
    st.sidebar.write(
        f":orange[Min] value: :orange[**{df_final['Pregnancies'].min()}**], :red[Max] value: :red[**{df_final['Pregnancies'].max()}**]")
    st.sidebar.write("")

    # ===== Glukosa =====
    Glucose = st.sidebar.number_input(
        label=":blue[**Input Glukosa**]", min_value=df_final['Glucose'].min())
    st.sidebar.write(
        f":orange[Min] value: :orange[**{df_final['Glucose'].min()}**], :red[Max] value: :red[**{df_final['Glucose'].max()}**]")
    st.sidebar.write("")

    # ===== Tekanan Darah =====
    BloodPressure = st.sidebar.number_input(
        label=":blue[**Input Tekanan Darah**]", min_value=df_final['BloodPressure'].min())
    st.sidebar.write(
        f":orange[Min] value: :orange[**{df_final['BloodPressure'].min()}**], :red[Max] value: :red[**{df_final['BloodPressure'].max()}**]")
    st.sidebar.write("")

    # ===== Indeks Massa Tubuh =====
    BMI = st.sidebar.number_input(
        label=":blue[**Input Indeks Massa Tubuh**]", min_value=df_final['BMI'].min())
    st.sidebar.write(
        f":orange[Min] value: :orange[**{df_final['BMI'].min()}**], :red[Max] value: :red[**{df_final['BMI'].max()}**]")
    st.sidebar.write("")

    # ===== DiabetesPedigreeFunction =====
    DiabetesPedigreeFunction = st.sidebar.number_input(
        label=":blue[**Input Presentase Keturunan Diabetes**]", min_value=df_final['DiabetesPedigreeFunction'].min())
    st.sidebar.write(
        f":orange[Min] value: :orange[**{df_final['DiabetesPedigreeFunction'].min()}**], :red[Max] value: :red[**{df_final['DiabetesPedigreeFunction'].max()}**]")
    st.sidebar.write("")

    # ===== Umur =====
    Age = st.sidebar.number_input(
        label=":blue[**Input Umur**]", min_value=df_final['Age'].min())
    st.sidebar.write(
        f":orange[Min] value: :orange[**{df_final['Age'].min()}**], :red[Max] value: :red[**{df_final['Age'].max()}**]")
    st.sidebar.write("")

    # Preview Tab 1 (Prediction Result)
    data = {
        'Umur': Age,
        'Kehamilan': Pregnancies,
        'Glukosa': Glucose,
        'Tekanan Darah': BloodPressure,
        'Indeks Massa Tubuh': BMI,
        'Presentase Keturunan Diabetes': DiabetesPedigreeFunction
    }

    preview_df = pd.DataFrame(data, index=['input'])

    st.header("Data Hasil Input Pengguna")
    st.write("")
    st.dataframe(preview_df.iloc[:, :6])
    st.write("")

    result = ":violet[-]"

    predict_button = st.button("**Predict**", type="primary")

    st.write("")

    if predict_button:
        inputs = [[Age, Pregnancies, Glucose,
                   BloodPressure, BMI, DiabetesPedigreeFunction]]
        prediction = diabetes_model.predict(inputs)[0]

        bar = st.progress(0)

        status_text = st.empty()

        for i in range(1, 101):
            status_text.text(f"{i}% complete")
            bar.progress(i)
            time.sleep(0.01)
            if i == 100:
                time.sleep(1)
                status_text.empty()
                bar.empty()
        if prediction == 0:
            result = ":green[**Pasien Tidak Terkena Diabetes**]"
        else:
            result = ":red[**Pasien Terkena Diabetes**]"

    st.write("")
    st.write("")
    st.subheader("Hasil :")
    st.subheader(result)


# Tab 2 (Informasi Data transform)
with tab2:
    # Kehamilan
    st.write("Jumlah Kehamilan")
    dt1 = pd.DataFrame(
        {

            "pregnancies": ["< 6x", "> 6x"],
            "category": ["Rendah", "Tinggi"],
            "score": ["0", "1"]
        }
    )
    st.dataframe(
        dt1,
        column_config={
            "pregnancies": "Kehamilan",
            "category": "Kategori",
            "score": "skor"
        }
    )

    # Glucose
    st.write("Glukosa")
    dt2 = pd.DataFrame(
        {

            "glucose": ["< 140 mg/dL", "140 - 199 mg/dL", "> 200mg/dL"],
            "category": ["Normal", "Sedang", "Tinggi"],
            "score": ["0", "1", "2"]
        }
    )
    st.dataframe(
        dt2,
        column_config={
            "glucose": "Glukosa",
            "category": "Kategori",
            "score": "skor"
        }
    )

    # BloodPressure
    st.write("Tekanan Darah")
    dt3 = pd.DataFrame(
        {

            "bloodp": ["< 80 mmHg", "80 - 89 mmHg", "> 89 mmHg"],
            "category": ["Normal", "Sedang", "Tinggi"],
            "score": ["0", "1", "2"]
        }
    )
    st.dataframe(
        dt3,
        column_config={
            "bloodp": "Tekanan Darah",
            "category": "Kategori",
            "score": "skor"
        }
    )

    # BMI
    st.write("Indeks Massa Tubuh")
    dt4 = pd.DataFrame(
        {

            "bmi": ["< 18.5", "18.5 - 22.9", "23 - 29.9", "> 30"],
            "category": ["Kurang", "Normal", "Berlebih", "Obesitas"],
            "score": ["0", "1", "2", "3"]
        }
    )
    st.dataframe(
        dt4,
        column_config={
            "bmi": "BMI",
            "category": "Kategori",
            "score": "skor"
        }
    )

    # Diabetes Pedigree Function
    st.write("Presentase Keturunan Diabetes")
    dt5 = pd.DataFrame(
        {

            "dpf": ["< 0.376", ">= 0.376"],
            "category": ["Normal", "Tinggi"],
            "score": ["0", "1"]
        }
    )
    st.dataframe(
        dt5,
        column_config={
            "dpf": "Jumlah",
            "category": "Kategori",
            "score": "skor"
        }
    )

    # Umur
    st.write("Umur")
    dt6 = pd.DataFrame(
        {

            "age": ["< 25 Tahun", "26 - 45 Tahun", "46 - 65 Tahun", "> 65 Tahun"],
            "category": ["Remaja", "Dewasa", "Lansia", "Tua"],
            "score": ["0", "1", "2", "3"]
        }
    )
    st.dataframe(
        dt6,
        column_config={
            "age": "Umur",
            "category": "Kategori",
            "score": "skor"
        }
    )
