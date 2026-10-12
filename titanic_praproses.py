import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)

def judul(teks):
    print("\n" + "=" * 78 + f"\n{teks}\n" + "=" * 78)

# -----------------------------------------------------------------------------
# 1. MEMUAT DATA & INFO / TIPE KOLOM
# -----------------------------------------------------------------------------
df = pd.read_csv("train.csv")
judul("1. INFO DATA & TIPE KOLOM")
print(f"Jumlah baris dan kolom awal : {df.shape}")
print(df.dtypes)

# Koreksi tipe: Sex & Embarked adalah kategori (bukan sekadar teks bebas)
# Pclass bertipe int tetapi bermakna ordinal (1 > 2 > 3), dipertahankan sebagai angka berurutan
# -----------------------------------------------------------------------------
# 2. KATEGORI: CEK EJAAN / KAPITALISASI
# -----------------------------------------------------------------------------
judul("2. CEK EJAAN / KAPITALISASI KATEGORI")
for kol in ["Sex", "Embarked"]:
    print(f"Nilai unik {kol} (sebelum dirapikan): {sorted(df[kol].dropna().unique())}")
df["Sex"] = df["Sex"].str.strip().str.lower()
df["Embarked"] = df["Embarked"].str.strip().str.upper()
for kol in ["Sex", "Embarked"]:
    print(f"Nilai unik {kol} (sesudah dirapikan): {sorted(df[kol].dropna().unique())}")
df["Sex"] = df["Sex"].astype("category")
df["Embarked"] = df["Embarked"].astype("category")

# -----------------------------------------------------------------------------
# 3. MISSING VALUES (DIDETEKSI) & DUPLIKASI (DICEK PADA DATA ASLI)
# -----------------------------------------------------------------------------
judul("3. MISSING VALUES & DUPLIKASI (DATA ASLI)")
mv = df.isna().sum()
print(pd.DataFrame({"jumlah_missing": mv, "persen": (mv / len(df) * 100).round(2)})[mv > 0])
print(f"\nBaris duplikat (seluruh kolom)        : {df.duplicated().sum()}")
print(f"Baris duplikat (tanpa PassengerId)    : {df.drop(columns='PassengerId').duplicated().sum()}")
df = df.drop_duplicates()   # tidak ada yang terbuang, tetapi tetap dijalankan sebagai prosedur
print(f"Jumlah baris setelah penanganan duplikat: {len(df)}")

# -----------------------------------------------------------------------------
# 4. SELEKSI & PEMBANGUNAN FITUR
# -----------------------------------------------------------------------------
judul("4. SELEKSI & PEMBANGUNAN FITUR")
# Fitur baru (feature engineering)
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

def ambil_title(nama):
    t = re.search(r",\s*([^\.]+)\.", nama).group(1).strip()
    peta = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
    t = peta.get(t, t)
    return t if t in ["Mr", "Mrs", "Miss", "Master"] else "Rare"
df["Title"] = df["Name"].apply(ambil_title)
print("Distribusi Title:", df["Title"].value_counts().to_dict())

print("\nCabin dibuang (77% kosong). PassengerId, Name, Ticket dibuang (ID/identitas/kardinalitas tinggi).")

fitur = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare",
         "Embarked", "FamilySize", "IsAlone", "Title"]          # 10 fitur
target = "Survived"
data = df[fitur + [target]].copy()
print(f"\nJumlah fitur yang dipakai: {len(fitur)} -> {fitur}")

# -----------------------------------------------------------------------------
# 5. PEMBAGIAN DATASET 80:20 (DILAKUKAN SEBELUM IMPUTASI/SCALING)
# -----------------------------------------------------------------------------
judul("5. PEMBAGIAN DATASET (80% LATIH : 20% UJI)")
X_train, X_test, y_train, y_test = train_test_split(
    data[fitur], data[target], test_size=0.2, random_state=42)
print(f"Data latih: {X_train.shape[0]} baris | Data uji: {X_test.shape[0]} baris")

# -----------------------------------------------------------------------------
# 6. MISSING VALUES: IMPUTASI (statistik dihitung dari DATA LATIH saja)
# -----------------------------------------------------------------------------
judul("6. IMPUTASI MISSING VALUES")
median_age = X_train["Age"].median()
modus_emb = X_train["Embarked"].mode()[0]
print(f"Median Age (data latih) = {median_age} | Modus Embarked (data latih) = {modus_emb}")
for d in (X_train, X_test):
    d["Age"] = d["Age"].fillna(median_age)
    d["Embarked"] = d["Embarked"].fillna(modus_emb)
print("Sisa missing latih/uji:", int(X_train.isna().sum().sum()), "/", int(X_test.isna().sum().sum()))

# -----------------------------------------------------------------------------
# 7. TRANSFORMASI KATEGORIKAL -> NUMERIK
# -----------------------------------------------------------------------------
judul("7. ENCODING DATA KATEGORIKAL")
X_train["Sex"] = X_train["Sex"].map({"male": 0, "female": 1}).astype(int)   # biner
X_test["Sex"] = X_test["Sex"].map({"male": 0, "female": 1}).astype(int)
# Pclass = ordinal -> tetap 1 < 2 < 3 (sudah numerik berurutan)
# Embarked & Title = nominal -> One-Hot Encoding (tidak ada urutan)
X_train = pd.get_dummies(X_train, columns=["Embarked", "Title"], dtype=int)
X_test = pd.get_dummies(X_test, columns=["Embarked", "Title"], dtype=int)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
print("Kolom setelah encoding:", list(X_train.columns))

# -----------------------------------------------------------------------------
# 8. NORMALISASI (Min-Max) - parameter min/max dari DATA LATIH
# -----------------------------------------------------------------------------
judul("8. NORMALISASI MIN-MAX")
kolom_num = ["Age", "Fare", "SibSp", "Parch", "FamilySize", "Pclass"]
mn, mx = X_train[kolom_num].min(), X_train[kolom_num].max()
X_train[kolom_num] = (X_train[kolom_num] - mn) / (mx - mn)
X_test[kolom_num] = (X_test[kolom_num] - mn) / (mx - mn)
print(f"Kolom dinormalisasi ke rentang [0,1]: {kolom_num}")

# -----------------------------------------------------------------------------
# 9. 5 DATA TERATAS & 5 DATA TERBAWAH HASIL PEMROSESAN
#     (gabungan data latih + uji, diurutkan sesuai urutan baris dataset asli)
# -----------------------------------------------------------------------------
hasil = pd.concat([X_train.assign(Survived=y_train), X_test.assign(Survived=y_test)]).sort_index()
# nama kolom dipersingkat hanya untuk tampilan agar muat di layar/laporan
singkat = {"FamilySize": "FamSize", "Embarked_C": "Emb_C", "Embarked_Q": "Emb_Q", "Embarked_S": "Emb_S",
           "Title_Master": "T_Master", "Title_Miss": "T_Miss", "Title_Mr": "T_Mr",
           "Title_Mrs": "T_Mrs", "Title_Rare": "T_Rare"}
judul("9. 5 DATA TERATAS HASIL PEMROSESAN")
print(hasil.head(5).round(3).rename(columns=singkat).to_string())
judul("9. 5 DATA TERBAWAH HASIL PEMROSESAN")
print(hasil.tail(5).round(3).rename(columns=singkat).to_string())
print(f"\nBentuk data hasil pemrosesan: {hasil.shape} (termasuk kolom target Survived)")

# -----------------------------------------------------------------------------
# 10. PEMODELAN: GAUSSIAN NAIVE BAYES (NIM GANJIL)
# -----------------------------------------------------------------------------
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

judul("10. HASIL EVALUASI MODEL GAUSSIAN NAIVE BAYES")
acc_train = accuracy_score(y_train, model.predict(X_train))
acc_test = accuracy_score(y_test, y_pred)
print(f"Akurasi data latih : {acc_train * 100:.2f}%")
print(f"Akurasi data uji   : {acc_test * 100:.2f}%")
print(f"Selisih latih-uji  : {abs(acc_train - acc_test) * 100:.2f} poin (kecil -> tidak ada indikasi overfitting berat)")
print("\nConfusion Matrix (baris = aktual, kolom = prediksi):")
print(pd.DataFrame(confusion_matrix(y_test, y_pred),
                   index=["Aktual 0", "Aktual 1"], columns=["Pred 0", "Pred 1"]))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Tidak selamat (0)", "Selamat (1)"]))