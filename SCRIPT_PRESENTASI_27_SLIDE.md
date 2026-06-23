# SCRIPT PRESENTASI SEMINAR PROPOSAL (27 SLIDE)
## "Optimasi dan Interpretasi Prediksi Risiko Stroke Menggunakan Stacking Ensemble Learning dan SHAP pada Data Kesehatan Tidak Seimbang"

**Presenter:** Stiefanny Dwi Chandra (103062300102)
**Durasi Total:** ~15 menit
**Sumber:** Presentasi Seminar Proposal Fanny.pdf

### Catatan Format:
- *[italic dalam kurung]* = instruksi presenter, TIDAK dibacakan
- **[X]** = nomor referensi dari daftar pustaka proposal
- Referensi yang digunakan adalah daftar pustaka PROPOSAL [1]-[17]

---

## SLIDE 1 — COVER (~30 detik)

*[Slide: Judul, Nama, NIM, Dosen Pembimbing]*

Bismillahirrahmanirrahim.

Assalamualaikum warahmatullahi wabarakatuh.

Yang terhormat Bapak/Ibu dosen penguji dan pembimbing, serta hadirin yang saya hormati.

Perkenalkan, nama saya Stiefanny Dwi Chandra dengan NIM 103062300102 dari Program Studi Sarjana Teknologi Informasi, Fakultas Informatika, Universitas Telkom Jakarta. Dengan dosen pembimbing pertama Ibu Nurul Ilmi, S.Kom., M.T. dan dosen pembimbing kedua Bapak Achmad Udin Zailani, S.Kom., M.Kom.

Pada kesempatan ini, saya akan mempresentasikan proposal tugas akhir saya yang berjudul **"Optimasi dan Interpretasi Prediksi Risiko Stroke Menggunakan Stacking Ensemble Learning dan SHAP pada Data Kesehatan Tidak Seimbang."**

---

## SLIDE 2 — DAFTAR ISI (~15 detik)

*[Slide: Bab 1, Bab 2, Bab 3, Daftar Pustaka]*

Pemaparan saya terbagi menjadi tiga bagian utama: **Bab 1 Pendahuluan**, **Bab 2 Kajian Pustaka**, dan **Bab 3 Perancangan Sistem**, dilengkapi dengan daftar pustaka. Saya akan mulai dari Bab 1.

---

## SLIDE 3 — HEADER BAB 1: PENDAHULUAN (~5 detik)

*[Slide: Transisi Bab 1]*

Baik, Bapak/Ibu, saya mulai dari Bab 1 yaitu Pendahuluan.

---

## SLIDE 4 — LATAR BELAKANG (1/2) (~2 menit)

*[Slide: Stroke, Solusi, Permasalahan, Data dan ML, Class Imbalance, Pemilihan Model, Performa Prediksi, Interpretasi]*

Bapak/Ibu, penelitian ini dilatarbelakangi oleh beberapa aspek yang saling berkaitan:

**Stroke** merupakan kondisi medis serius yang menyebabkan gangguan fungsi otak akibat penyumbatan atau pecahnya pembuluh darah. Identifikasi dini faktor risiko stroke sangat krusial karena faktor-faktor tersebut berasal dari kombinasi berbagai karakteristik pasien yang kompleks **[1]**.

**Data dan Machine Learning.** Perkembangan machine learning memberikan pendekatan yang dapat mempelajari pola dari data kesehatan dan menghasilkan prediksi berdasarkan hubungan antar variabel klinis **[1]**. Pendekatan ensemble learning melalui metode stacking dapat menggabungkan beberapa model sehingga menghasilkan representasi prediksi yang lebih beragam **[5]**.

**Class Imbalance.** Dalam penerapan machine learning pada bidang kesehatan, data yang digunakan sering memiliki distribusi kelas yang tidak seimbang. Kondisi ini menyebabkan model lebih dominan mempelajari kelas mayoritas sehingga kemampuan mendeteksi kasus minoritas menjadi kurang optimal **[2]**.

**Pemilihan Model.** Penggunaan satu algoritma terkadang memiliki keterbatasan dalam menangkap pola kompleks. Pada data tabular, model berbasis decision tree ensemble mampu menangani hubungan non-linear antar fitur tanpa memerlukan asumsi hubungan linear **[7]**.

**Performa Prediksi.** Threshold default 0,5 sering kali tidak optimal pada data tidak seimbang **[11]**, sehingga diperlukan optimasi batas keputusan.

**Interpretasi.** Model prediksi perlu dianalisis tidak hanya berdasarkan evaluasi numerik, tetapi juga melalui pemahaman mengenai alasan model menghasilkan suatu keputusan. Aspek kepercayaan terhadap model menjadi perhatian penting dalam penerapan AI di bidang kesehatan **[8]**.

---

## SLIDE 5 — LATAR BELAKANG (2/2): KENAPA? (~2 menit) **[SLIDE KUNCI]**

*[Slide: Kenapa Stacking Ensemble? Kenapa F2-Score? Kenapa SHAP?]*

*[Bicarakan bagian ini lebih pelan dan tegas — ini inti dari kebaruan penelitian]*

Bapak/Ibu, sekarang saya jelaskan **mengapa** ketiga metode ini dipilih:

**Kenapa Stacking Ensemble?** Agar mampu memprediksi risiko stroke dari data rekam medis yang sangat kompleks dengan lebih presisi dibandingkan menggunakan satu model tunggal. Faria et al. tahun 2024 membuktikan bahwa stacking mampu mengungguli classifier tunggal **[1]**. Mashi et al. tahun 2024 juga menunjukkan bahwa menggabungkan beberapa model mampu memitigasi overfitting dan meningkatkan akurasi prediksi pada data medis **[5]**. Saya menggunakan XGBoost **[14]**, LightGBM **[15]**, dan CatBoost **[16]** sebagai base learners karena Grinsztajn et al. tahun 2022 membuktikan tree-based models masih mengungguli deep learning pada data tabular berukuran sedang **[7]**. Logistic Regression dipilih sebagai meta-learner karena Zian et al. tahun 2021 menunjukkan pengklasifikasi linear yang diregularisasi adalah pilihan paling stabil untuk mencegah overfitting pada level meta **[4]**.

**Kenapa Optimasi F2-Score?** Karena metrik ini mengutamakan keberhasilan mendeteksi pasien stroke (Recall), namun tetap menyertakan batasan presisi agar model tidak menebak secara sembarangan. Zhang et al. membuktikan bahwa threshold default 0,5 menghasilkan prediksi yang bias terhadap kelas mayoritas pada data imbalanced **[11]**. F2-Score memberikan bobot empat kali lebih besar pada Recall dibanding Precision. Dalam konteks medis, gagal mendeteksi pasien stroke (False Negative) jauh lebih berbahaya dibanding salah mendeteksi orang sehat **[12]**.

**Kenapa Metode SHAP?** Untuk menjelaskan perhitungan matematis algoritma secara transparan, sehingga setiap hasil prediksi dapat divalidasi dan dipercaya oleh tenaga medis berdasarkan gejala klinis riil pasien. Agafonov et al. tahun 2024 menegaskan bahwa trustworthy AI adalah syarat mutlak agar AI diterima di bidang kesehatan **[8]**. Moulaei et al. tahun 2024 menerapkan SHAP pada prediksi stroke namun hanya pada model tunggal **[10]**. Lu dan Qiu tahun 2023 menegaskan bahwa model black-box tanpa penjelasan akan mengurangi penerimaannya oleh komunitas medis **[17]**. Saya mengembangkan pendekatan **SHAP Dua Level** — level ensemble untuk validasi algoritmik, dan level klinis untuk validasi diagnosis.

---

## SLIDE 6 — RUMUSAN MASALAH (~1 menit)

*[Slide: 3 Rumusan Masalah]*

Berdasarkan latar belakang tersebut, saya merumuskan tiga rumusan masalah:

**Pertama:** Bagaimana membangun model prediksi stroke menggunakan metode Stacking Ensemble yang terjamin zero-leakage dan membandingkan performanya dengan Random Forest sebagai baseline? *[Gap dari [1] yang rentan leakage, diperkuat [13] tentang bahaya leakage]*

**Kedua:** Bagaimana pengaruh integrasi SMOTE dan threshold tuning dalam mengatasi class imbalance dan meningkatkan deteksi kelas minoritas? *[Gap dari [6] yang tidak menangani imbalance, diperkuat [11] tentang threshold]*

**Ketiga:** Bagaimana menguraikan kontribusi fitur klinis menggunakan metode SHAP? *[Gap dari [10] yang hanya pada model tunggal, bukan stacking]*

---

## SLIDE 7 — TUJUAN PENELITIAN (~1 menit)

*[Slide: 3 Tujuan]*

Sejalan dengan rumusan masalah, tujuan penelitian ini:

**Tujuan pertama:** Membangun dan mengevaluasi arsitektur Stacking Ensemble yang bebas kebocoran data (zero-leakage) serta membandingkan signifikansi performanya terhadap model baseline Random Forest **[1][4][9]**.

**Tujuan kedua:** Menganalisis efektivitas penerapan Pipeline SMOTE dan Threshold Optimization berbasis F2-Score dalam meminimalkan kegagalan deteksi (False Negative) pada data yang sangat tidak seimbang, serta mengevaluasi trade-off antara Recall dan Precision **[2][3][11][12]**.

**Tujuan ketiga:** Menjelaskan faktor-faktor yang memengaruhi hasil prediksi secara hierarkis menggunakan metode SHAP dua level, baik dari segi kontribusi algoritma pembentuk maupun faktor klinis pasien **[8][10][17]**.

Intinya, Bapak/Ibu, penelitian ini bertujuan menghasilkan model yang **akurat, sensitif, dan transparan**.

---

## SLIDE 8 — KONTRIBUSI PENELITIAN (~45 detik)

*[Slide: 4 Kontribusi]*

Kontribusi utama penelitian ini:

**01.** Penerapan Stacking Ensemble Learning untuk prediksi stroke — menggabungkan XGBoost **[14]**, LightGBM **[15]**, dan CatBoost **[16]** yang terbukti mengungguli deep learning pada data tabular **[7]**.

**02.** Penanganan class imbalance menggunakan SMOTE **[3]** yang diisolasi di dalam Pipeline untuk menjamin zero data leakage **[13]**.

**03.** Optimasi model dengan fokus pada F2-Score — threshold dicari secara matematis, bukan default 0,5 **[11][12]**.

**04.** Interpretasi model menggunakan SHAP dua level — memberikan insight faktor risiko stroke secara transparan **[10][17]**.

---

## SLIDE 9 — JADWAL PENELITIAN (~20 detik)

*[Slide: Tabel 8 Kegiatan x 5 Bulan]*

Penelitian ini dirancang dalam delapan fase selama lima bulan — mulai dari studi literatur, EDA, data preparation, model development, threshold optimization, pengujian, SHAP, hingga penyusunan laporan.

Selanjutnya, Bab 2 Kajian Pustaka.

---

## SLIDE 10 — HEADER BAB 2: KAJIAN PUSTAKA (~5 detik)

*[Slide: Transisi Bab 2]*

Baik, saya lanjutkan ke Bab 2 yaitu Kajian Pustaka.

---

## SLIDE 11 — PENELITIAN TERDAHULU (Paper [1] dan [6]) (~1.5 menit)

*[Slide: Matriks Paper [1] Faria et al. dan [6] Li & Mintah]*

**Penelitian pertama:** Faria et al. tahun 2024 **[1]** — "Stroke Detection Through Ensemble Learning: A Stacking Approach." Membuktikan stacking mampu mengungguli classifier tunggal. **Gap:** Belum menggunakan Explainable AI, belum menerapkan threshold tuning berbasis F2-Score, dan rentan data leakage karena SMOTE diterapkan sebelum data splitting.

**Penelitian kedua:** Li dan Mintah tahun 2025 **[6]** — "Stroke Prediction Using a Stacked Ensemble Classifier." Menunjukkan pentingnya preprocessing dalam meningkatkan performa. **Gap:** Fokus pada akurasi umum, belum menangani class imbalance secara khusus, belum menggunakan threshold tuning maupun SHAP.

Kedua celah inilah yang menjadi dasar pengembangan arsitektur saya.

---

## SLIDE 12 — PENELITIAN TERDAHULU (Paper [9] dan [10]) (~1.5 menit)

*[Slide: Matriks Paper [9] Hwangbo et al. dan [10] Moulaei et al.]*

**Penelitian ketiga:** Hwangbo et al. tahun 2022 **[9]** — "Stacking Ensemble Learning Model to Predict 6-Month Mortality in Ischemic Stroke Patients." Membuktikan efektivitas stacking pada data medis nyata. **Gap:** Fokus pada mortalitas pasca-stroke, tidak membahas class imbalance, threshold tuning, maupun interpretabilitas model.

**Penelitian keempat:** Moulaei et al. tahun 2024 **[10]** — "Explainable AI for Stroke Prediction through Comparison of DL and ML Models." Menekankan pentingnya interpretabilitas dalam kesehatan. **Gap:** Tidak menggunakan stacking ensemble dan belum menerapkan threshold optimization pada data imbalanced. SHAP hanya pada model tunggal.

Penelitian saya mengintegrasikan keandalan stacking dari **[9]** dengan kebutuhan interpretabilitas dari **[10]**, sekaligus menambahkan zero-leakage pipeline dan threshold optimization.

---

## SLIDE 13 — PENELITIAN TERDAHULU (Paper [11]) (~45 detik)

*[Slide: Matriks Paper [11] Zhang et al.]*

**Penelitian kelima:** Zhang et al. **[11]** — "Threshold Moving Approaches for Addressing the Class Imbalance Problem." Membuktikan secara matematis bahwa threshold default 0,5 tidak selalu optimal dan sering menghasilkan prediksi bias terhadap kelas mayoritas.

**Gap:** Kajian bersifat umum pada klasifikasi multi-label, belum diterapkan pada kasus stroke dan tidak menggunakan stacking.

Penelitian saya mengadaptasi prinsip threshold moving dari **[11]** secara spesifik pada hasil cross-validation model Stacking, dengan fungsi objektif F2-Score untuk meminimalkan False Negative pada diagnosis stroke.

---

## SLIDE 14 — HEADER BAB 3: PERANCANGAN SISTEM (~5 detik)

*[Slide: Transisi Bab 3]*

Baik, Bapak/Ibu, selanjutnya Bab 3 yaitu Perancangan Sistem.

---

## SLIDE 15 — DATASET DAN KETIMPANGAN KELAS (~45 detik)

*[Slide: 5.110 baris, 95:5 ratio, Kaggle Fedesoriano]*

Penelitian ini menggunakan pendekatan kuantitatif dengan metode eksperimen.

Dataset yang digunakan: **Stroke Prediction Dataset** dari Kaggle (Fedesoriano).
- Total: **5.110 baris** observasi rekam medis anonim
- Kelas Mayoritas: **4.861 pasien sehat** (95,13%)
- Kelas Minoritas: **249 pasien stroke** (4,87%)

Dataset ini telah divalidasi penggunaannya oleh Faria et al. **[1]** dan Li & Mintah **[6]**. Ketimpangan kelas yang ekstrem — 95 berbanding 5 — menjadikan strategi penanganan class imbalance sebagai komponen kritikal dalam penelitian ini.

---

## SLIDE 16 — DESAIN SISTEM (~30 detik)

*[Slide: Diagram Desain Sistem]*

Ini adalah gambaran umum desain sistem saya. Arsitektur terdiri dari enam fase yang saling berhubungan — mulai dari eksplorasi data hingga interpretasi SHAP. Setiap fase dirancang untuk menghasilkan model yang akurat, sensitif terhadap kasus stroke, dan mampu menjelaskan faktor-faktor yang memengaruhi keputusan prediksi.

---

## SLIDE 17 — SKENARIO & ALUR SISTEM PREDIKSI (~20 detik)

*[Slide: Daftar 6 Fase]*

Alur sistem prediksi terdiri dari enam fase:
1. Eksplorasi dan Pemahaman Dataset (EDA)
2. Persiapan Data (Data Preparation)
3. Pengembangan Model Prediksi (Model Development)
4. Threshold Optimization dan Evaluasi Model
5. Pengujian pada Hold-Out Test Set
6. Explainable Artificial Intelligence (SHAP)

Saya jelaskan satu per satu.

---

## SLIDE 18 — FASE 1: EDA (~30 detik)

*[Slide: Identifikasi Struktur, Analisis Ketimpangan, Cek Missing Value, Visualisasi EDA]*

Fase pertama: Exploratory Data Analysis.

**Identifikasi Struktur Data** — pemilahan fitur numerik (age, avg_glucose_level, bmi) dan kategorikal (gender, smoking_status, work_type).

**Analisis Ketimpangan** — validasi empiris rasio 95:5 yang menjustifikasi mengapa akurasi konvensional tidak bisa diandalkan **[2][12]**.

**Cek Missing Value** — fitur BMI memiliki nilai kosong. Berbeda dengan Faria et al. **[1]** yang menghapus baris kosong, saya menggunakan **imputasi median** karena distribusi fisiologis rentan outlier.

**Visualisasi EDA** — memahami pola distribusi data sebelum pemodelan.

---

## SLIDE 19 — FASE 2: PERSIAPAN DATA (~45 detik)

*[Slide: Training/Testing Split, SMOTE Inside Pipeline, Pipeline Preprocessing]*

Fase kedua: lini pertahanan utama untuk integritas eksperimen.

**Stratified Split (80:20)** — mempertahankan proporsi kelas. Data testing **diisolasi sepenuhnya** sejak awal.

**Pipeline Preprocessing** — imputasi median + One-Hot Encoding terintegrasi dalam satu jalur.

**SMOTE Inside Pipeline** — dan ini yang paling krusial, Bapak/Ibu. SMOTE **hanya** dieksekusi pada data training di setiap fold. Data testing **tidak pernah** tersentuh SMOTE. Salmi et al. **[2]** menegaskan: *"Proposing a new sampling method and using it before the train-test split will incorrectly tell us of its effectiveness."* Van de Mortel dan van Wingen **[13]** membuktikan bahwa sirkularitas kebocoran data menggelembungkan akurasi semu dan membatasi generalisasi.

---

## SLIDE 20 — FASE 3: PENGEMBANGAN MODEL (~1 menit)

*[Slide: Baseline Random Forest, Arsitektur Stacking, Base Learners, Meta Learner]*

Fase ketiga: pembangunan arsitektur model.

**Baseline: Random Forest** — tolok ukur awal. Model tree-based ensemble yang stabil pada data tabular **[7]**.

**Base Learners:** Tiga algoritma Gradient Boosting:
- **XGBoost** **[14]** — pencarian pola kompleks dengan regularisasi anti-overfitting
- **LightGBM** **[15]** — efisiensi komputasi melalui leaf-wise growth
- **CatBoost** **[16]** — penanganan fitur kategorikal secara native via ordered boosting

Hyperparameter tuning menggunakan RandomizedSearchCV dengan **F2-Score sebagai objective function** — memaksa model memprioritaskan deteksi sejak awal.

**Arsitektur Stacking:** Base learners menghasilkan probabilitas **Out-of-Fold (OOF)** melalui Stratified K-Fold CV. Probabilitas ini menjadi fitur baru untuk **Logistic Regression sebagai Meta-Learner** **[4]** — dengan class weighting untuk penalti lebih besar pada kegagalan deteksi stroke.

---

## SLIDE 21 — FASE 4: THRESHOLD OPTIMIZATION & EVALUASI (~1 menit)

*[Slide: Probabilitas OOF, Threshold Optimal, Evaluasi Repeated SKFCV, Laporan Metrik]*

Fase keempat: menentukan ambang batas keputusan optimal.

**Probabilitas OOF** dihasilkan menggunakan Stratified K-Fold CV sebagai acuan perhitungan threshold.

**Threshold Optimization:** Zhang et al. **[11]** membuktikan: *"When the distribution is highly unbalanced, predictions based on lambda=0.5 may be biased with respect to the major class."* Oleh karena itu, saya mengevaluasi probabilitas secara dinamis dan mencari titik threshold berdasarkan **F2-Score tertinggi** — dengan tetap menjaga Precision pada level logis (Precision Guard).

**Evaluasi Stabilitas:** Repeated Stratified K-Fold CV (5-fold x 3 repetisi = 15 skenario) untuk membuktikan konsistensi model.

**Metrik Komprehensif:** Accuracy, Precision, Recall, Metric Gap, F1-Score, F2-Score, ROC-AUC, PR-AUC — dilaporkan dalam format Mean ± Std **[12]**.

---

## SLIDE 22 — FASE 5: PENGUJIAN HOLD-OUT TEST (~30 detik)

*[Slide: Pelatihan Final, Penerapan Threshold, Evaluasi Performa, Visualisasi]*

Fase kelima: pembuktian generalisasi model.

**Pelatihan Final** — arsitektur Stacking dilatih ulang dengan 100% data training.

**Penerapan Threshold Klinis** — probabilitas dikonversi menggunakan threshold optimal F2-Score, **bukan** default 0,5.

**Evaluasi Performa** — verifikasi apakah performa stabil pada data baru atau drop (indikasi overfitting).

**Visualisasi** — Confusion Matrix dan kurva ROC untuk membedah distribusi TP/FP/TN/FN.

---

## SLIDE 23 — FASE 6: SHAP (~1 menit) **[SLIDE KUNCI]**

*[Slide: Local Explanation (Waterfall Plot), Global Interpretation (Summary Plot), Meta Learner Analysis]*

Fase terakhir dan **kebaruan utama** penelitian ini.

Lu dan Qiu **[17]** menegaskan: *"Most existing studies treated these models as black boxes and rarely provided explanations, which might reduce their acceptance by the medical community."* Agafonov et al. **[8]** juga menyerukan bahwa transparansi adalah syarat mutlak untuk trustworthy AI di healthcare.

Oleh karena itu, saya mengusulkan **Dual-Layer SHAP Framework**:

**Meta Learner Analysis (Layer 1 — Algorithmic Validation):** LinearExplainer pada Meta-Learner untuk membedah kontribusi probabilitas setiap base learner terhadap keputusan akhir. Memvalidasi arsitektur stacking bekerja harmonis.

**Global Interpretation — Summary Plot (Layer 2a):** Memvisualisasikan pola hubungan fitur klinis vs risiko stroke pada level populasi. Memvalidasi arah pengaruh fitur konsisten dengan kaidah medis.

**Local Explanation — Waterfall Plot (Layer 2b):** Dekonstruksi risiko per individu. Memberikan justifikasi medis mengapa seorang pasien diklasifikasikan berisiko tinggi.

Pendekatan dua level ini memberikan **akuntabilitas ganda** — transparansi algoritmik sekaligus transparansi klinis **[8][10][17]**.

---

## SLIDE 24, 25, 26 — DAFTAR PUSTAKA (~15 detik)

*[Slide: Referensi [1]-[15] sesuai slide presentasi]*

Bapak/Ibu, seluruh argumen yang saya paparkan didukung oleh 17 referensi ilmiah dari jurnal dan konferensi bereputasi. Daftar lengkapnya dapat dilihat pada slide ini.

---

## SLIDE 27 — PENUTUP (~30 detik)

*[Slide: Sekian Terima Kasih]*

Baik, Bapak/Ibu, demikian pemaparan proposal tugas akhir saya.

Sebagai ringkasan, penelitian ini mengusulkan pendekatan end-to-end yang mengintegrasikan:
1. Arsitektur **Stacking Ensemble** yang terjamin bebas kebocoran data **[1][4][13]**
2. **Pipeline SMOTE + Threshold Optimization berbasis F2-Score** untuk memaksimalkan deteksi kelas minoritas **[2][3][11][12]**
3. **SHAP Dua Level** untuk transparansi keputusan secara komputasional dan medis **[8][10][17]**

Demikian presentasi dari saya. Mohon maaf apabila ada kekurangan. Apakah ada pertanyaan dari Bapak/Ibu dosen?

Terima kasih.
Wassalamualaikum warahmatullahi wabarakatuh.

---
---

# PETA SITASI PER SLIDE

| Slide | Argumen | Referensi |
|-------|---------|-----------|
| 4 | ML untuk prediksi stroke | [1] Faria et al., 2024 |
| 4 | Ensemble mengurangi bias | [5] Mashi et al., 2024 |
| 4 | Class imbalance pada data medis | [2] Salmi et al., 2024 |
| 4 | Tree-based > DL pada tabular | [7] Grinsztajn et al., 2022 |
| 4 | Threshold 0,5 tidak optimal | [11] Zhang et al. |
| 4 | Trustworthy AI | [8] Agafonov et al., 2024 |
| 5 | Stacking > model tunggal | [1] Faria; [5] Mashi |
| 5 | Meta-learner LR stabil | [4] Zian et al., 2021 |
| 5 | XGBoost/LightGBM/CatBoost | [14][15][16] |
| 5 | F2-Score memprioritaskan Recall | [11] Zhang; [12] Araf |
| 5 | SHAP untuk transparansi medis | [8] Agafonov; [10] Moulaei; [17] Lu & Qiu |
| 11 | Gap: Faria leakage | [1]; [13] van de Mortel |
| 11 | Gap: Li belum imbalance | [6] Li & Mintah |
| 12 | Gap: Hwangbo belum XAI | [9] Hwangbo |
| 12 | Gap: Moulaei model tunggal | [10] Moulaei |
| 13 | Threshold moving | [11] Zhang et al. |
| 19 | SMOTE inside pipeline | [2] Salmi; [13] van de Mortel |
| 20 | Base learners | [14] XGBoost; [15] LightGBM; [16] CatBoost |
| 20 | Meta-learner | [4] Zian et al. |
| 21 | Threshold F2-Score | [11] Zhang; [12] Araf |
| 23 | SHAP Layer 1 (algoritmik) | [4] Zian; [17] Lu & Qiu |
| 23 | SHAP Layer 2 (klinis) | [10] Moulaei; [8] Agafonov |

---

# PANDUAN JAWAB PERTANYAAN DOSEN

## Q1: "Dataset Kaggle valid untuk penelitian?"
"Dataset ini divalidasi oleh Faria et al. [1] di IEEE iCACCESS 2024, Li & Mintah [6] di ICCIP 2024, dan Moulaei et al. [10] di Scientific Reports 2024. Karakteristiknya sesuai: tabular medis, fitur numerik+kategorikal, class imbalance ekstrem. Dataset publik memungkinkan reproducibility."

## Q2: "Kenapa SMOTE, bukan metode lain?"
"Abdelhamid & Desai [3] menguji berbagai metode dan SMOTE stabil pada data ekstrem. Undersampling tidak cocok: minoritas hanya 249. Pembeda utama bukan metode-nya tapi penerapan di dalam Pipeline — zero-leakage sesuai van de Mortel [13]."

## Q3: "Kenapa F2-Score, bukan F1?"
"Zhang et al. [11]: threshold 0,5 bias pada data imbalanced. F2 bobot 4x pada Recall. False Negative stroke = kehilangan golden period. F1 terlalu 'adil' untuk biaya kesalahan asimetris. Didukung Araf et al. [12] tentang cost-sensitive learning."

## Q4: "Kenapa Stacking, bukan single model?"
"Faria [1] membuktikan stacking unggul. Tiga base learner [14][15][16] memberikan diversity. Grinsztajn [7]: tree-based state-of-the-art pada ~10K data. Zian [4]: LR meta-learner paling stabil."

## Q5: "Jelaskan teknis zero data leakage."
"Van de Mortel [13]: oversampling sebelum splitting = sirkularitas = akurasi semu. Mekanisme: SMOTE dikunci di Pipeline imblearn, hanya pada training fold. Fold validasi tanpa SMOTE. Evaluasi jujur dan valid."

## Q6: "SHAP Dua Level kebaruannya?"
"Moulaei [10]: SHAP pada model tunggal. Lu & Qiu [17]: SHAP pada stacking tapi satu level. Saya: Layer 1 (LinearExplainer pada meta-learner) + Layer 2 (KernelExplainer global+lokal). Akuntabilitas ganda sesuai Agafonov [8]."

## Q7: "Jika Random Forest lebih baik?"
"Hipotesis dapat ditolak = temuan valid. Dasar kuat: Faria [1], Zian [4], Hwangbo [9]. Kontribusi zero-leakage, threshold, SHAP independen dari hasil perbandingan."

## Q8: "Meta-learner kenapa Logistic Regression?"
"Zian [4]: LR meta-learner paling stabil. Alasan: (1) Input hanya 3 probabilitas — model kompleks overfitting. (2) Tugas layer 2 hanya sintesis berbobot. (3) Koefisien LR bisa langsung diinterpretasi SHAP LinearExplainer."

---

# DAFTAR PUSTAKA PROPOSAL

[1] Faria et al., 2024 — Stroke Detection Through Ensemble Learning: A Stacking Approach. iCACCESS 2024.
[2] Salmi et al., 2024 — Handling imbalanced medical datasets. Artif Intell Rev.
[3] Abdelhamid & Desai, 2024 — Balancing the Scales: Class Imbalance in Binary Classification.
[4] Zian et al., 2021 — Stacked Ensembles with Different Meta-Learners. IEEE Access.
[5] Mashi et al., 2024 — An Ensemble Approach for Stroke Prediction. IEEE MCSoC.
[6] Li & Mintah, 2025 — Stroke Prediction Using a Stacked Ensemble Classifier. ICCIP.
[7] Grinsztajn et al., 2022 — Why tree-based models outperform deep learning on tabular data.
[8] Agafonov et al., 2024 — Trustworthy AI for healthcare. Front Digit Health.
[9] Hwangbo et al., 2022 — Stacking ensemble to predict stroke mortality. Sci Rep.
[10] Moulaei et al., 2024 — Explainable AI for stroke prediction. Sci Rep.
[11] Zhang et al. — Threshold Moving for Class Imbalance. ACM.
[12] Araf et al., 2024 — Cost-sensitive learning for imbalanced medical data. Artif Intell Rev.
[13] van de Mortel & van Wingen, 2025 — Data leakage in ML studies. Mol Psychiatry.
[14] Chen & Guestrin, 2016 — XGBoost. ACM SIGKDD.
[15] Ke et al., 2017 — LightGBM. NeurIPS.
[16] Prokhorenkova et al., 2017 — CatBoost. NeurIPS.
[17] Lu & Qiu, 2023 — Explainable prediction using stacked ensemble. BMC Med Inform.
