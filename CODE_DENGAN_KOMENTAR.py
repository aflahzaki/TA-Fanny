# ==========================================
# TAHAP 2: EXPLORATORY DATA ANALYSIS (EDA)
# Tujuan: Memahami karakteristik dataset sebelum pemodelan
# ==========================================

# Menampilkan informasi dasar dataset: jumlah baris dan kolom
print("=== INFORMASI DATASET ===")
print(f"Total Baris dan Kolom: {df.shape}")  # df.shape mengembalikan tuple (jumlah_baris, jumlah_kolom)

# Menampilkan tipe data setiap kolom/fitur (int, float, object, dll)
print("\n=== TIPE DATA FITUR ===")
print(df.dtypes)  # Penting untuk mengetahui fitur mana yang numerik vs kategorikal

# Pengecekan kualitas data: missing value dan duplikat
print("\n=== PENGECEKAN KUALITAS DATA ===")
print(f"Jumlah Missing Value:\n{df.isnull().sum()}")  # Menghitung jumlah data kosong per kolom
print(f"\nJumlah Duplikat: {df.duplicated().sum()} baris")  # Menghitung baris yang identik

# Jika ditemukan data duplikat, hapus agar tidak mengganggu pelatihan model
if df.duplicated().sum() > 0:
    df = df.drop_duplicates()  # Menghapus baris duplikat, menyisakan satu saja
    print("Data duplikat telah dihapus.")

print("\n=== DISTRIBUSI KELAS TARGET ===")
# Menampilkan 5 baris data pertama untuk melihat struktur data secara visual
display(df.head())

# Visualisasi distribusi jumlah data per kelas (jenis tanaman)
plt.figure(figsize=(12, 6))  # Mengatur ukuran kanvas grafik
sns.countplot(data=df, x='label', palette='viridis', order=df['label'].value_counts().index)
# countplot = bar chart yang menghitung frekuensi kemunculan setiap kategori
# palette='viridis' = skema warna hijau-biru-kuning
# order = mengurutkan bar berdasarkan jumlah data terbanyak
plt.title('Distribusi Jumlah Data per Jenis Tanaman', fontsize=14, fontweight='bold')
plt.xlabel('Jenis Tanaman', fontsize=12)
plt.ylabel('Jumlah', fontsize=12)
plt.xticks(rotation=45, ha='right')  # Memutar label sumbu X agar tidak saling tumpang tindih
plt.tight_layout()  # Mengatur layout agar tidak terpotong
plt.show()

# Interpretasi hasil EDA
print("""
Interpretasi EDA:
Dataset ini 'perfectly balanced' karena setiap kelas memiliki jumlah data yang sama persis.
Ini sangat penting agar model tidak bias terhadap kelas mayoritas.
""")


# ==========================================
# TAHAP 3: PREPROCESSING & TRAIN-TEST SPLIT
# Tujuan: Menyiapkan data agar siap diproses oleh model ML
# ==========================================

# Import library yang dibutuhkan
from sklearn.preprocessing import LabelEncoder      # Untuk mengubah label teks menjadi angka
from sklearn.model_selection import train_test_split  # Untuk membagi data menjadi train dan test

# 1. Memisahkan Fitur (X) dan Target (y)
# X = semua kolom KECUALI 'label' (fitur input: N, P, K, suhu, kelembapan, dll)
# y = kolom 'label' saja (target output: jenis tanaman yang direkomendasikan)
X = df.drop('label', axis=1)  # axis=1 berarti menghapus kolom (bukan baris)
y = df['label']

# 2. Label Encoding pada target
# Model ML tidak bisa memproses teks seperti "rice", "wheat"
# LabelEncoder mengubahnya menjadi angka: rice=0, wheat=1, dst.
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # fit = belajar mapping, transform = terapkan mapping

# 3. Train-Test Split (80% untuk Training, 20% untuk Testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,       # 20% data disisihkan sebagai data ujian akhir
    stratify=y_encoded,  # WAJIB: memastikan proporsi setiap jenis tanaman sama di train & test
    random_state=42      # Agar hasil pembagian selalu konsisten (reproducible)
)

print("=== PEMBAGIAN DATA SELESAI ===")
print(f"Data Training (Untuk belajar & CV) : {X_train.shape[0]} baris")
print(f"Data Testing (Dikunci untuk Ujian Akhir) : {X_test.shape[0]} baris")

# Catatan penting tentang pencegahan Data Leakage
print("""
Catatan Anti Data Leakage:
Mulai titik ini, X_test dan y_test disembunyikan dari proses pelatihan.
Semua validasi, pencarian pola, dan fitting model hanya akan menggunakan X_train.
""")


# ==========================================
# TAHAP 4: CROSS-VALIDATION PADA DATA TRAINING
# Tujuan: Mengukur performa model secara fair menggunakan K-Fold CV
# ==========================================

# Import semua library yang diperlukan
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate  # Untuk K-Fold CV
from xgboost import XGBClassifier          # Base Learner 1: XGBoost
from lightgbm import LGBMClassifier        # Base Learner 2: LightGBM
from catboost import CatBoostClassifier    # Base Learner 3: CatBoost
from sklearn.ensemble import StackingClassifier    # Arsitektur Stacking
from sklearn.linear_model import LogisticRegression  # Meta-Learner
from IPython.display import display

# Inisialisasi 3 Base Learners (model dasar)
xgb_model = XGBClassifier(random_state=42, eval_metric='mlogloss')
# XGBoost: algoritma boosting dengan regularisasi anti-overfitting
# eval_metric='mlogloss' = metrik evaluasi untuk multi-class classification

lgbm_model = LGBMClassifier(random_state=42, verbose=-1)
# LightGBM: boosting yang lebih efisien (leaf-wise growth)
# verbose=-1 = menyembunyikan pesan log agar output tidak berisik

cat_model = CatBoostClassifier(random_state=42, verbose=0)
# CatBoost: boosting yang unggul menangani fitur kategorikal
# verbose=0 = menyembunyikan progress bar pelatihan

# Inisialisasi Stacking Ensemble
# Stacking = menggabungkan prediksi dari beberapa base learner
# lalu dipelajari oleh meta-learner untuk keputusan akhir
estimators = [
    ('xgb', xgb_model),    # Base learner 1
    ('lgbm', lgbm_model),  # Base learner 2
    ('cat', cat_model)     # Base learner 3
]
stacking_model = StackingClassifier(
    estimators=estimators,                          # Daftar base learners
    final_estimator=LogisticRegression(max_iter=2000),  # Meta-learner (penggabung)
    cv=5  # Internal CV untuk menghasilkan probabilitas Out-of-Fold bagi meta-learner
)

# Kumpulkan semua model dalam dictionary untuk diiterasi
models = {
    'XGBoost': xgb_model,
    'LightGBM': lgbm_model,
    'CatBoost': cat_model,
    'Stacking Ensemble': stacking_model
}

# Setup Stratified K-Fold Cross Validation
# Stratified = memastikan proporsi setiap kelas sama di setiap fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# n_splits=5 = data dibagi menjadi 5 bagian, model diuji 5x
# shuffle=True = data diacak sebelum dibagi

# Metrik evaluasi yang akan dihitung per fold
scoring_metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
# weighted = memperhitungkan jumlah sampel per kelas

cv_results_summary = []  # Wadah untuk menyimpan hasil CV semua model

print("Memulai proses Cross-Validation 5-Fold pada Data Training... (Mohon tunggu)\n")

# Fungsi bantuan: mengubah format angka dari titik ke koma (format Indonesia)
def format_indo(mean_val, std_val):
    mean_str = f"{mean_val:.4f}".replace('.', ',')  # 4 desimal, titik jadi koma
    std_str = f"{std_val:.4f}".replace('.', ',')
    return f"{mean_str} ± {std_str}"  # Format: mean ± std

# Loop untuk menjalankan CV pada setiap model
for name, model in models.items():
    # cross_validate = menjalankan K-Fold CV dan mengembalikan skor per fold
    # n_jobs=1 = menggunakan 1 core CPU agar RAM stabil (tidak crash di Colab)
    cv_scores = cross_validate(model, X_train, y_train, cv=skf, scoring=scoring_metrics, n_jobs=1)

    # Menyimpan hasil rata-rata ± standar deviasi ke dalam list
    cv_results_summary.append({
        'Model': name,
        'CV Accuracy': format_indo(cv_scores['test_accuracy'].mean(), cv_scores['test_accuracy'].std()),
        'CV Precision': format_indo(cv_scores['test_precision_weighted'].mean(), cv_scores['test_precision_weighted'].std()),
        'CV Recall': format_indo(cv_scores['test_recall_weighted'].mean(), cv_scores['test_recall_weighted'].std()),
        'CV F1-Score': format_indo(cv_scores['test_f1_weighted'].mean(), cv_scores['test_f1_weighted'].std())
    })

# Konversi hasil ke DataFrame untuk ditampilkan sebagai tabel
df_cv = pd.DataFrame(cv_results_summary)

# Fungsi styling: membuat baris Stacking Ensemble dicetak tebal (bold)
def highlight_stacking(baris):
    if baris['Model'] == 'Stacking Ensemble':
        return ['font-weight: bold'] * len(baris)  # Semua kolom di-bold
    else:
        return [''] * len(baris)  # Tidak ada styling

# Menerapkan style dan menyembunyikan index angka
tabel_cantik = df_cv.style.apply(highlight_stacking, axis=1).hide(axis="index")

print("Tabel 2. Hasil 5-Fold Cross Validation pada Data Pelatihan")
display(tabel_cantik)  # Menampilkan tabel cantik di Jupyter/Colab


# ==========================================
# TAHAP 5 & 6: PELATIHAN FINAL, EVALUASI, & ANALISIS OVERFITTING
# Tujuan: Melatih model pada seluruh data training, lalu uji pada data testing
# yang belum pernah dilihat model sebelumnya
# ==========================================

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from IPython.display import display

print("Melatih model final pada seluruh data X_train...\n")

test_eval_results = []  # Wadah hasil evaluasi pada test set

for name, model in models.items():
    # PELATIHAN FINAL: model belajar dari SELURUH data training
    model.fit(X_train, y_train)

    # Prediksi pada data TRAINING (untuk menghitung gap overfitting)
    train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)  # Akurasi pada data latih

    # Prediksi pada data TESTING (ujian akhir - data yang belum pernah dilihat)
    test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)  # Akurasi pada data uji

    # Menghitung GAP = selisih akurasi train vs test
    # Jika gap besar = model menghafal data (overfitting)
    gap = train_acc - test_acc

    # Klasifikasi status overfitting berdasarkan nilai gap
    if gap < 0.02:
        status = "Tidak overfitting"      # Gap < 2% = aman
    elif gap < 0.05:
        status = "Perlu diwaspadai"       # Gap 2-5% = hati-hati
    else:
        status = "Indikasi overfitting"   # Gap > 5% = bermasalah

    # Menyimpan semua metrik evaluasi (format Indonesia: koma sebagai desimal)
    test_eval_results.append({
        'Model': name,
        'Train Acc': str(round(train_acc, 4)).replace('.', ','),
        'Test Acc': str(round(test_acc, 4)).replace('.', ','),
        'Precision': str(round(precision_score(y_test, test_pred, average='weighted'), 4)).replace('.', ','),
        'Recall': str(round(recall_score(y_test, test_pred, average='weighted'), 4)).replace('.', ','),
        'F1-Score': str(round(f1_score(y_test, test_pred, average='weighted'), 4)).replace('.', ','),
        'Gap': str(round(gap, 4)).replace('.', ','),
        'Status': status
    })

df_eval = pd.DataFrame(test_eval_results)

# Menggabungkan hasil CV (dari Tahap 4) dengan hasil Test (Tahap 5)
# Tujuannya: membandingkan performa CV vs Test dalam satu tabel
df_final_comparison = pd.merge(df_cv[['Model', 'CV Accuracy']], df_eval, on='Model')
df_final_comparison.rename(columns={'CV Accuracy': 'CV Mean ± Std'}, inplace=True)

# Mengurutkan kolom agar mudah dibaca
kolom_urut = ['Model', 'Train Acc', 'CV Mean ± Std', 'Test Acc', 'Precision', 'Recall', 'F1-Score', 'Gap', 'Status']
df_final_comparison = df_final_comparison[kolom_urut]

# Menerapkan styling bold untuk Stacking Ensemble
tabel_akhir_cantik = df_final_comparison.style.apply(highlight_stacking, axis=1).hide(axis="index")

print("=== TABEL EVALUASI FINAL & ANALISIS OVERFITTING ===")
display(tabel_akhir_cantik)


# ==========================================
# TAHAP 7: CONFUSION MATRIX & ANALISIS KESALAHAN
# Tujuan: Melihat secara detail di mana model salah menebak
# ==========================================

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# Mengambil prediksi dari model Stacking (model utama yang dianalisis)
stack_final_pred = models['Stacking Ensemble'].predict(X_test)

# Menghitung Confusion Matrix
# Baris = label asli, Kolom = prediksi model
cm = confusion_matrix(y_test, stack_final_pred)

# Visualisasi Confusion Matrix sebagai heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_,   # Label sumbu X = nama tanaman
            yticklabels=le.classes_)   # Label sumbu Y = nama tanaman
# annot=True = menampilkan angka di setiap sel
# fmt='d' = format angka integer (bukan desimal)
# cmap='Blues' = skema warna biru
plt.title('Confusion Matrix - Stacking Ensemble (Test Set)', fontsize=14, fontweight='bold')
plt.xlabel('Prediksi Sistem')       # Apa yang ditebak model
plt.ylabel('Kondisi Riil (Label Asli)')  # Apa yang sebenarnya
plt.tight_layout()
plt.show()

# Menampilkan Classification Report (Precision, Recall, F1 per kelas)
print("\n=== CLASSIFICATION REPORT STACKING ENSEMBLE ===")
print(classification_report(y_test, stack_final_pred, target_names=le.classes_))

# Analisis Kesalahan: mencari data mana yang salah diprediksi
print("\n=== MISCLASSIFICATION ANALYSIS (STACKING ENSEMBLE) ===")
errors = np.where(y_test != stack_final_pred)[0]  # Indeks data yang salah tebak

print(f"Total data salah tebak : {len(errors)} dari {len(y_test)} data uji.")

if len(errors) > 0:
    print("Rincian data yang tertukar:")
    for idx in errors:
        real_label = le.classes_[y_test[idx]]          # Label asli
        pred_label = le.classes_[stack_final_pred[idx]]  # Label prediksi
        print(f"- Lahan index ke-{idx}: Asli [{real_label.upper()}] namun diprediksi sebagai [{pred_label.upper()}]")

    print("\n*Insight Tambahan untuk Pembahasan:*")
    print("Perhatikan kelas tanaman yang tertukar. Seringkali tanaman yang salah diprediksi memiliki ")
    print("kebutuhan nutrisi tanah (N, P, K) atau iklim (suhu, curah hujan) yang sangat mirip secara agronomis.")
else:
    print("Model Stacking berhasil menebak seluruh data uji dengan sempurna!")


# =====================================================================
# TAHAP 8: SHAP GLOBAL INTERPRETABILITY (VERSI AKURAT & KOMPREHENSIF)
# Tujuan: Menjelaskan fitur mana yang paling berpengaruh terhadap
# seluruh prediksi model secara GLOBAL (level populasi)
# =====================================================================

import shap
import matplotlib.pyplot as plt

print("Menghitung SHAP Global secara komprehensif (Bisa memakan waktu 5-15 menit)...")

# 1. MEMBUAT BACKGROUND DATA MENGGUNAKAN K-MEANS
# Background data = representasi ringkas dari data training
# K-Means merangkum 1.760 data latih menjadi 22 profil (centroid)
# Angka 22 dipilih karena ada 22 jenis tanaman — setiap centroid mewakili 1 profil tanaman
X_train_summary = shap.kmeans(X_train, 22)

# 2. Inisialisasi KernelExplainer
# KernelExplainer = metode SHAP yang model-agnostic (bisa untuk semua jenis model)
# Parameter 1: fungsi predict_proba dari model Stacking (output = probabilitas per kelas)
# Parameter 2: background data (ringkasan X_train)
explainer_global = shap.KernelExplainer(stacking_model.predict_proba, X_train_summary)

# 3. MENGHITUNG SHAP VALUES UNTUK SELURUH DATA UJI (440 baris)
# Ini menghitung kontribusi setiap fitur terhadap prediksi di setiap data point
# Output: matrix berukuran (440 x 7 x 22) — 440 data, 7 fitur, 22 kelas
shap_values_global = explainer_global.shap_values(X_test)

# 4. Mengatur ukuran kanvas visualisasi
plt.figure(figsize=(16, 8))

# 5. Membuat SHAP Summary Plot (Bar Plot)
# Menampilkan rata-rata |SHAP value| per fitur — fitur teratas = paling berpengaruh
shap.summary_plot(
    shap_values_global,       # Nilai SHAP yang sudah dihitung
    X_test,                   # Data yang sesuai (untuk nama fitur)
    plot_type="bar",          # Tipe grafik: bar chart
    class_names=le.classes_,  # Nama-nama kelas (jenis tanaman)
    show=False,               # Jangan langsung tampilkan (kita atur dulu layout-nya)
    color_bar=False           # Sembunyikan color bar default
)

# 6. Mengatur posisi grafik agar tidak terpotong
plt.subplots_adjust(left=0.2, right=0.75, top=0.85, bottom=0.15)

# 7. Menambahkan legenda (daftar nama tanaman)
plt.legend(
    loc='center left',           # Posisi legenda: kiri tengah
    bbox_to_anchor=(1.05, 0.5),  # Geser ke luar area grafik
    ncol=2,                      # 2 kolom agar tidak terlalu panjang ke bawah
    fontsize='medium',
    title="Komoditas Tanaman",
    title_fontsize='large',
    frameon=True,                # Tampilkan border kotak legenda
    borderpad=1
)

# 8. Menambahkan judul dan label sumbu
plt.title("Analisis Kepentingan Fitur Global (Seluruh 440 Data Uji)",
          fontsize=18, pad=30, fontweight='bold')
plt.xlabel("Mean |SHAP Value| (Rata-rata pengaruh terhadap seluruh prediksi)",
           fontsize=14, labelpad=15)
plt.ylabel("Parameter Tanah dan Iklim", fontsize=14, labelpad=20)

plt.show()


# =====================================================================
# TAHAP 9 & 10: SHAP LOCAL INTERPRETABILITY (WATERFALL PLOT)
# Tujuan: Menjelaskan MENGAPA model memberikan rekomendasi SPESIFIK
# untuk SATU data point tertentu (level individu)
# =====================================================================

import shap
import matplotlib.pyplot as plt
import numpy as np

# Membuat ulang prediksi Stacking pada data test
# (agar variabel 'stack_pred' selalu tersedia meskipun cell sebelumnya tidak dijalankan)
stack_pred = stacking_model.predict(X_test)

# ===== SKENARIO 1: TEBAKAN BENAR (Ideal Case) =====
print("Menyiapkan SHAP Lokal (Skenario 1 - Tebakan Sempurna)...")

# Pilih data point pertama (index 0) sebagai contoh analisis
idx_sampel = 0
sample_df = X_test.iloc[[idx_sampel]]  # Ambil 1 baris data sebagai DataFrame

# Ambil kelas yang diprediksi oleh model untuk data ini
predicted_class_idx = stack_pred[idx_sampel]
nama_tanaman = le.classes_[predicted_class_idx]  # Konversi angka kembali ke nama tanaman

# Menggunakan TreeExplainer pada CatBoost (salah satu base learner)
# TreeExplainer lebih cepat dan presisi dibanding KernelExplainer untuk model tree-based
explainer_lokal = shap.TreeExplainer(stacking_model.named_estimators_['cat'])
# named_estimators_['cat'] = mengakses model CatBoost dari dalam arsitektur Stacking

# Menghitung SHAP values untuk 1 data point ini
shap_values_all = explainer_lokal.shap_values(sample_df)

plt.close('all')  # Menutup semua grafik sebelumnya agar tidak bertumpuk
plt.figure(figsize=(10, 6))

# Mengekstrak base value dan kontribusi SHAP untuk kelas yang diprediksi
# (handling perbedaan format output antar versi SHAP)
try:
    base_val = explainer_lokal.expected_value[predicted_class_idx]
    # expected_value = nilai dasar prediksi (sebelum fitur apapun dipertimbangkan)
    shap_contrib = shap_values_all[0][:, predicted_class_idx]
    # shap_values_all[0] = kontribusi fitur untuk data point pertama
    # [:, predicted_class_idx] = ambil kolom untuk kelas yang diprediksi
except Exception:
    base_val = explainer_lokal.expected_value[predicted_class_idx]
    shap_contrib = shap_values_all[predicted_class_idx][0]

# Membuat Waterfall Plot: visualisasi kontribusi setiap fitur
# Merah = mendorong prediksi KE ARAH kelas ini (positif)
# Biru = mendorong prediksi MENJAUHI kelas ini (negatif)
shap.plots._waterfall.waterfall_legacy(
    base_val,                              # Titik awal (base value)
    shap_contrib,                          # Kontribusi setiap fitur
    feature_names=X_test.columns.tolist(), # Nama-nama fitur
    show=False
)

plt.title(f"Analisis Lokal: Mengapa Rekomendasinya {nama_tanaman.upper()}?",
          pad=20, fontweight='bold', fontsize=14)
plt.tight_layout()
plt.show()


# ===== SKENARIO 2: TEBAKAN SALAH (Error Case) =====
# Tujuan: Menjelaskan MENGAPA model salah menebak — fitur mana yang menyesatkan
print("Menyiapkan SHAP Lokal (Skenario 2 - Salah Tebak)...")

# Mencari data point yang salah diprediksi
salah_tebak = np.where(y_test != stack_pred)[0]  # Array berisi indeks data yang salah

# Jika ada data yang salah tebak, gunakan yang pertama
# Jika tidak ada (sempurna), gunakan index cadangan
if len(salah_tebak) > 0:
    idx_sampel_2 = salah_tebak[0]  # Ambil data salah pertama
else:
    idx_sampel_2 = 5  # Fallback: gunakan index 5 sebagai contoh

# Identifikasi label prediksi vs label asli
predicted_class_idx_2 = stack_pred[idx_sampel_2]
nama_tanaman_prediksi = le.classes_[predicted_class_idx_2]  # Apa yang ditebak model
nama_tanaman_asli = le.classes_[y_test[idx_sampel_2]]       # Apa yang sebenarnya

# Hitung SHAP values untuk data point yang salah ini
shap_values_all_2 = explainer_lokal.shap_values(X_test.iloc[[idx_sampel_2]])

plt.close('all')
plt.clf()  # Membersihkan figure sebelumnya
plt.figure(figsize=(10, 6))

# Ekstrak base value dan kontribusi SHAP (sama seperti Skenario 1)
try:
    base_val_2 = explainer_lokal.expected_value[predicted_class_idx_2]
    shap_contrib_2 = shap_values_all_2[0][:, predicted_class_idx_2]
except Exception:
    base_val_2 = explainer_lokal.expected_value[predicted_class_idx_2]
    shap_contrib_2 = shap_values_all_2[predicted_class_idx_2][0]

# Waterfall Plot untuk kasus salah tebak
# Grafik ini menunjukkan fitur mana yang "menyesatkan" model
shap.plots._waterfall.waterfall_legacy(
    base_val_2,
    shap_contrib_2,
    feature_names=X_test.columns.tolist(),
    show=False
)

plt.title(f"Skenario 2: Lahan {nama_tanaman_asli.upper()} yang disalahartikan sebagai {nama_tanaman_prediksi.upper()}",
          fontsize=14, fontweight='bold', pad=30)
plt.tight_layout()
plt.show()
