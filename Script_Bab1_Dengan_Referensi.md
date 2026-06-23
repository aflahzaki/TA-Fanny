# SCRIPT PRESENTASI SEMINAR PROPOSAL - BAB 1 (DENGAN ACUAN REFERENSI)
## "Optimasi dan Interpretasi Prediksi Risiko Stroke Menggunakan Stacking Ensemble Learning dan SHAP pada Data Kesehatan Tidak Seimbang"

**Presenter:** Stiefanny Dwi Chandra (103062300102)  
**Durasi Target:** ~15 menit (Bab 1)  
**Catatan:** 
- Teks dalam *[kurung miring]* = instruksi presenter
- Teks dalam **[Ref X]** = referensi yang mendukung argumen (untuk ditampilkan di slide atau diingat saat ditanya dosen)

---

## PEMBUKAAN (~1 menit)

*[Slide: Cover / Judul]*

Bismillahirrahmanirrahim.

Assalamualaikum warahmatullahi wabarakatuh.

Yang terhormat Bapak/Ibu dosen penguji dan pembimbing, serta hadirin yang saya hormati.

Perkenalkan, nama saya Stiefanny Dwi Chandra dengan NIM 103062300102 dari Program Studi Sarjana Teknologi Informasi, Fakultas Informatika, Universitas Telkom Jakarta.

Pada kesempatan kali ini, saya akan mempresentasikan proposal tugas akhir saya yang berjudul **"Optimasi dan Interpretasi Prediksi Risiko Stroke Menggunakan Stacking Ensemble Learning dan SHAP pada Data Kesehatan Tidak Seimbang."**

Baik, saya akan memulai pemaparan dari Bab 1 yaitu Pendahuluan.

---

## 1.1 LATAR BELAKANG (~5 menit)

### Paragraf 1: Konteks Masalah Stroke & Machine Learning

*[Slide: Latar Belakang - Urgensi Prediksi Stroke]*

Bapak/Ibu, penelitian ini dilatarbelakangi oleh permasalahan serius di bidang kesehatan, yaitu penyakit stroke.

Stroke merupakan kondisi medis yang menyebabkan gangguan fungsi otak akibat penyumbatan atau pecahnya pembuluh darah. Proses identifikasi risiko stroke menjadi penting karena faktor yang berhubungan dengan kejadian stroke berasal dari kombinasi berbagai karakteristik pasien yang kompleks.

Berdasarkan penelitian Faria et al. pada tahun 2024, perkembangan machine learning memberikan pendekatan yang dapat mempelajari pola dari data kesehatan dan menghasilkan prediksi berdasarkan hubungan antar variabel klinis pasien **[Ref 1: Faria et al., 2024 - iCACCESS]**.

### Paragraf 2: Masalah Class Imbalance

*[Slide: Permasalahan - Class Imbalance pada Data Medis]*

Namun, Bapak/Ibu, ada permasalahan mendasar yang menjadi tantangan dalam penerapan machine learning pada bidang kesehatan, yaitu **class imbalance** atau ketidakseimbangan kelas.

Menurut Salmi et al. dalam review komprehensif tahun 2024 yang dimuat di Artificial Intelligence Review, ketidakseimbangan kelas terjadi ketika jumlah sampel pada kelas tertentu jauh lebih sedikit dibandingkan kelas lainnya. Kondisi ini menyebabkan model lebih dominan mempelajari kelas mayoritas sehingga kemampuan mendeteksi kasus minoritas seperti pasien stroke menjadi kurang optimal **[Ref 2: Salmi et al., 2024 - Artificial Intelligence Review]**.

Pada dataset yang digunakan dalam penelitian ini, kelas stroke hanya mencakup 4,87% dari total data, sedangkan kelas non-stroke mencapai 95,13%. Ketimpangan ini sangat berbahaya karena model bisa menghasilkan akurasi tinggi yang bersifat ilusi, padahal sebenarnya gagal mendeteksi pasien yang berisiko stroke.

### Paragraf 3: Solusi SMOTE dalam Pipeline (PENEKANAN)

*[Slide: Solusi 1 - SMOTE dalam Pipeline (Zero-Leakage)]*

**Solusi pertama** yang saya usulkan untuk menangani permasalahan class imbalance ini adalah penerapan **Synthetic Minority Oversampling Technique atau SMOTE**.

Berdasarkan kajian Abdelhamid dan Desai tahun 2024, SMOTE bekerja dengan membuat data sintetis baru pada kelas minoritas berdasarkan pola kedekatan antar data sehingga distribusi kelas menjadi lebih seimbang **[Ref 3: Abdelhamid & Desai, 2024]**.

Yang menjadi **pembeda utama** dari penelitian ini, Bapak/Ibu, adalah SMOTE tidak diterapkan secara langsung pada keseluruhan data, melainkan **diisolasi secara ketat di dalam Pipeline pelatihan**. Hal ini merujuk pada temuan van de Mortel dan van Wingen tahun 2025 yang membuktikan bahwa kebocoran data atau data leakage dalam machine learning dapat merusak validitas estimasi performa model secara fundamental **[Ref 13: van de Mortel & van Wingen, 2025 - Molecular Psychiatry]**.

Dengan pendekatan pipeline ini, proses SMOTE hanya dilakukan pada data training setiap fold, sehingga data pengujian tetap merepresentasikan kondisi asli dan evaluasi model menjadi valid.

Selain SMOTE, pembobotan kelas atau class weighting juga diterapkan pada meta-learner sebagai strategi tambahan, merujuk pada kajian Araf et al. tahun 2024 tentang cost-sensitive learning pada data medis tidak seimbang **[Ref 12: Araf et al., 2024 - Artificial Intelligence Review]**.

### Paragraf 4: Solusi Stacking Ensemble (PENEKANAN)

*[Slide: Solusi 2 - Stacking Ensemble Learning]*

**Solusi kedua** adalah pendekatan **Stacking Ensemble Learning**.

Bapak/Ibu, penggunaan satu algoritma terkadang memiliki keterbatasan dalam menangkap pola kompleks pada data kesehatan. Penelitian Mashi et al. tahun 2024 menunjukkan bahwa pendekatan ensemble dapat diterapkan pada prediksi stroke dengan mengombinasikan beberapa algoritma klasifikasi untuk memperoleh informasi prediksi yang lebih beragam **[Ref 5: Mashi et al., 2024 - IEEE MCSoC]**.

Secara lebih spesifik, Zian et al. pada tahun 2021 dalam IEEE Access menunjukkan bahwa metode stacking ensemble dengan evaluasi empiris terhadap berbagai meta-learner terbukti efektif dalam menangani klasifikasi data tidak seimbang **[Ref 4: Zian et al., 2021 - IEEE Access]**.

Dalam penelitian ini, saya menggunakan **XGBoost** yang dikembangkan oleh Chen dan Guestrin tahun 2016 **[Ref 14]**, **LightGBM** oleh Ke et al. tahun 2017 **[Ref 15]**, dan **CatBoost** oleh Prokhorenkova et al. tahun 2017 **[Ref 16]** sebagai base learner. Pemilihan ketiga algoritma berbasis gradient boosting ini merujuk pada temuan Grinsztajn et al. tahun 2022 yang menunjukkan bahwa model berbasis decision tree ensemble masih mengungguli deep learning pada data tabular **[Ref 7: Grinsztajn et al., 2022]**.

Ketiga model ini kemudian digabungkan melalui Logistic Regression sebagai meta-learner yang menghasilkan keputusan akhir.

### Paragraf 5: Solusi Threshold Optimization + SHAP (PENEKANAN)

*[Slide: Solusi 3 - Threshold Optimization & SHAP Dua Level]*

**Solusi ketiga** mencakup dua aspek sekaligus, yaitu **Threshold Optimization** dan **Explainability melalui SHAP**.

Untuk threshold optimization, Zhang et al. telah membuktikan secara matematis bahwa penggunaan batas keputusan default sebesar 0,5 sering kali berakibat fatal dan tidak menghasilkan performa terbaik pada data yang timpang **[Ref 11: Zhang et al., 2021 - ACM]**. Oleh karena itu, dalam penelitian ini, threshold dicari secara matematis menggunakan fungsi objektif **F2-Score** yang memberikan bobot lebih besar pada Recall, agar model lebih sensitif mendeteksi pasien stroke.

Untuk aspek transparansi, Agafonov et al. tahun 2024 menekankan bahwa trustworthy AI adalah syarat mutlak dalam healthcare agar sistem pendukung keputusan dapat dipercaya **[Ref 8: Agafonov et al., 2024 - Frontiers in Digital Health]**. Penelitian Moulaei et al. tahun 2024 juga menegaskan pentingnya interpretabilitas SHAP dalam prediksi stroke, namun penerapannya masih terbatas pada model tunggal **[Ref 10: Moulaei et al., 2024 - Scientific Reports]**.

Untuk melengkapi keterbatasan tersebut, saya mengusulkan **SHAP Dua Level** yang mengadaptasi pendekatan Lu dan Qiu tahun 2023 dalam menerapkan explainability pada stacking ensemble di bidang serebrovaskular **[Ref 17: Lu & Qiu, 2023 - BMC Medical Informatics]**:
- **Level 1 (Ensemble):** Analisis kontribusi setiap base learner terhadap keputusan meta-learner
- **Level 2 (Klinis):** Analisis faktor klinis pasien secara global dan lokal

*[Jeda sebentar]*

Jadi secara ringkas, Bapak/Ibu, penelitian ini mengintegrasikan tiga solusi — Stacking Ensemble, Pipeline SMOTE dengan Threshold Optimization, dan SHAP Dua Level — untuk menyelesaikan celah-celah metodologis yang masih terbuka pada penelitian-penelitian sebelumnya.

---

## 1.2 RUMUSAN MASALAH (~2 menit)

*[Slide: Rumusan Masalah]*

Baik, berdasarkan latar belakang dan celah penelitian tersebut, saya merumuskan tiga rumusan masalah:

**Rumusan masalah pertama:** Bagaimana membangun arsitektur Stacking Ensemble yang terdiri dari XGBoost, LightGBM, dan CatBoost yang terisolasi dari kebocoran data — atau zero-leakage — dan bagaimana performanya dibandingkan dengan model baseline Random Forest?

> *Acuan: Rumusan ini muncul dari gap yang ditemukan pada penelitian Faria et al. [1] yang belum menjamin zero-leakage, dan dikuatkan oleh van de Mortel & van Wingen [13] tentang bahaya data leakage.*

**Rumusan masalah kedua:** Bagaimana pengaruh kombinasi Pipeline SMOTE dan Threshold Optimization berbasis F2-Score terhadap peningkatan kemampuan model dalam mendeteksi kelas minoritas, yaitu pasien yang berisiko stroke?

> *Acuan: Rumusan ini muncul dari gap Li & Mintah [6] yang tidak menangani class imbalance, dan landasan matematis Zhang et al. [11] tentang bahaya threshold default.*

**Rumusan masalah ketiga:** Bagaimana menguraikan proses pengambilan keputusan model secara transparan menggunakan metode SHAP pada level ensemble dan level klinis?

> *Acuan: Rumusan ini muncul dari keterbatasan Moulaei et al. [10] yang hanya menerapkan SHAP pada model tunggal, bukan arsitektur stacking.*

Ketiga rumusan masalah ini saling berkaitan — yang pertama fokus pada arsitektur, yang kedua pada optimasi deteksi, dan yang ketiga pada interpretabilitas model.

---

## 1.3 TUJUAN PENELITIAN (~2 menit)

*[Slide: Tujuan Penelitian]*

Sejalan dengan rumusan masalah tersebut, tujuan penelitian ini adalah:

**Tujuan pertama:** Membangun dan mengevaluasi arsitektur Stacking Ensemble yang bebas kebocoran data serta membandingkan signifikansi performanya terhadap model baseline Random Forest.

> *Ini merujuk pada pendekatan Hwangbo et al. [9] yang membuktikan keandalan stacking pada skenario medis klinis, namun belum menyentuh aspek zero-leakage.*

**Tujuan kedua:** Menganalisis efektivitas penerapan Pipeline SMOTE dan Threshold Optimization berbasis F2-Score dalam meminimalkan kegagalan deteksi atau False Negative pada data yang sangat tidak seimbang, serta mengevaluasi trade-off antara Recall dan Precision.

> *Pendekatan cost-sensitive ini didukung oleh Araf et al. [12] dan pembuktian matematis Zhang et al. [11].*

**Tujuan ketiga:** Menjelaskan faktor-faktor yang memengaruhi hasil prediksi secara hierarkis menggunakan metode SHAP dua level, baik dari segi kontribusi algoritma pembentuk maupun faktor klinis pasien.

> *Mengembangkan konsep Lu & Qiu [17] yang menerapkan explainability pada stacking di bidang serebrovaskular.*

Jadi pada intinya, Bapak/Ibu, penelitian ini bertujuan menghasilkan model prediksi stroke yang **akurat, sensitif, dan transparan** — tiga aspek yang dalam penelitian-penelitian sebelumnya belum pernah diintegrasikan secara utuh dalam satu framework.

---

## 1.4 BATASAN MASALAH (~1.5 menit)

*[Slide: Batasan Masalah]*

Untuk menjaga fokus penelitian, saya menetapkan delapan batasan masalah:

**Pertama**, dataset yang digunakan adalah Stroke Prediction Dataset dari Fedesoriano melalui platform Kaggle — merupakan data rekam medis anonim yang terdiri dari 5.110 observasi dengan 12 atribut.

**Kedua**, permasalahan difokuskan pada **klasifikasi biner**: berisiko stroke versus tidak berisiko stroke.

**Ketiga**, Random Forest digunakan sebagai baseline pembanding, karena merujuk pada Grinsztajn et al. [7], Random Forest adalah model ensemble stabil yang sering dijadikan acuan awal pada data tabular. Sedangkan arsitektur utama menggunakan XGBoost [14], LightGBM [15], dan CatBoost [16] sebagai base learners.

**Keempat**, SMOTE dilakukan secara ketat di dalam Pipeline menggunakan framework Imbalanced-Learn untuk menjamin zero data leakage [13].

**Kelima**, threshold dicari secara matematis melalui optimasi F2-Score, bukan menggunakan nilai default 0,5 [11].

**Keenam**, evaluasi menggunakan metrik komprehensif: Accuracy, Precision, Recall, F1-Score, F2-Score, ROC-AUC, dan PR-AUC [2][12].

**Ketujuh**, interpretasi menggunakan SHAP dua level [10][17].

**Kedelapan**, penelitian dibatasi pada tahap pemodelan analitik dan tidak mencakup implementasi aplikasi klinis.

---

## 1.5 HIPOTESIS (~2 menit)

*[Slide: Hipotesis Penelitian]*

Selanjutnya, Bapak/Ibu, saya mengajukan tiga hipotesis:

**Hipotesis pertama (H1):** Stacking Ensemble mampu menghasilkan performa prediksi stroke yang lebih baik dibandingkan model tunggal Random Forest berdasarkan metrik Recall, F1-Score, F2-Score, ROC-AUC, dan PR-AUC.

> *Landasan: Faria et al. [1] dan Zian et al. [4] membuktikan superioritas stacking terhadap model tunggal. Hwangbo et al. [9] mengkonfirmasi keandalan stacking pada domain medis serebrovaskular.*

**Hipotesis kedua (H2):** Penerapan SMOTE di dalam Pipeline yang dikombinasikan dengan Threshold Optimization mampu meningkatkan kemampuan model mengenali kelas minoritas pada prediksi stroke.

> *Landasan: Zhang et al. [11] membuktikan bahwa threshold default 0,5 berakibat fatal pada data imbalanced. Abdelhamid & Desai [3] menunjukkan efektivitas SMOTE dalam menyeimbangkan distribusi kelas.*

**Hipotesis ketiga (H3):** Penerapan SHAP pada arsitektur Stacking Ensemble mampu menguraikan keputusan prediksi secara transparan melalui analisis dua level.

> *Landasan: Moulaei et al. [10] menegaskan SHAP krusial untuk transparansi di healthcare. Lu & Qiu [17] menunjukkan SHAP dapat diterapkan pada stacking untuk penyakit serebrovaskular.*

Ketiga hipotesis ini nantinya akan divalidasi melalui serangkaian eksperimen terukur dengan metrik yang komprehensif.

---

## 1.6 & 1.7 RENCANA DAN JADWAL KEGIATAN (~1 menit)

*[Slide: Rencana Kegiatan]*

Bapak/Ibu, pelaksanaan penelitian ini dirancang dalam **delapan fase** yang sistematis:

1. **Studi Literatur** — Kajian referensi ilmiah terkait
2. **EDA** — Memahami karakteristik dataset
3. **Data Preparation** — Splitting, preprocessing, dan SMOTE dalam pipeline [3][13]
4. **Model Development** — Baseline (Random Forest [7]) dan Stacking Ensemble (XGBoost [14], LightGBM [15], CatBoost [16]) dengan Logistic Regression sebagai meta-learner [4]
5. **Threshold Optimization & Evaluasi** — Stratified K-Fold CV, optimasi F2-Score [11], evaluasi Repeated CV
6. **Pengujian Hold-Out Test Set** — Validasi generalisasi model
7. **SHAP Dua Level** — LinearExplainer pada meta-learner, KernelExplainer untuk analisis klinis [10][17]
8. **Analisis Hasil & Penyusunan Laporan**

*[Slide: Jadwal Kegiatan]*

Keseluruhan tahapan direncanakan selesai dalam **lima bulan** sebagaimana tertera pada tabel jadwal kegiatan.

---

## PENUTUP BAB 1 (~0.5 menit)

*[Slide: Ringkasan Bab 1]*

Demikian pemaparan Bab 1 Pendahuluan dari proposal tugas akhir saya.

Sebagai ringkasan, penelitian ini mengusulkan pendekatan **end-to-end** yang mengintegrasikan:

1. **Arsitektur Stacking Ensemble** yang terjamin bebas kebocoran data [1][4][13]
2. **Pipeline SMOTE + Threshold Optimization berbasis F2-Score** untuk memaksimalkan deteksi kelas minoritas [3][11][12]
3. **SHAP Dua Level** untuk transparansi keputusan yang dapat dipertanggungjawabkan secara komputasional dan medis [10][17]

Selanjutnya, saya akan melanjutkan pemaparan ke Bab 2 yaitu Kajian Pustaka.

*[Atau: "Demikian pemaparan dari saya. Apakah ada pertanyaan dari Bapak/Ibu dosen?"]*

---

## PETA REFERENSI PER ARGUMEN (CHEAT SHEET)

| Argumen/Klaim | Referensi Pendukung |
|---|---|
| Machine learning untuk prediksi stroke | [1] Faria et al., 2024; [5] Mashi et al., 2024 |
| Class imbalance pada data medis | [2] Salmi et al., 2024; [12] Araf et al., 2024 |
| SMOTE untuk penanganan imbalance | [3] Abdelhamid & Desai, 2024 |
| Stacking Ensemble efektif pada data imbalanced | [4] Zian et al., 2021 |
| Stacking pada prediksi stroke | [1] Faria et al., 2024; [6] Li & Mintah, 2025 |
| Stacking pada kasus medis serebrovaskular | [9] Hwangbo et al., 2022 |
| Tree-based models unggul pada data tabular | [7] Grinsztajn et al., 2022 |
| Trustworthy AI di healthcare | [8] Agafonov et al., 2024 |
| SHAP pada prediksi stroke | [10] Moulaei et al., 2024 |
| Threshold default 0,5 fatal pada data imbalanced | [11] Zhang et al., 2021 |
| Cost-sensitive learning pada data medis | [12] Araf et al., 2024 |
| Data leakage merusak validitas model | [13] van de Mortel & van Wingen, 2025 |
| XGBoost | [14] Chen & Guestrin, 2016 |
| LightGBM | [15] Ke et al., 2017 |
| CatBoost | [16] Prokhorenkova et al., 2017 |
| SHAP pada stacking ensemble (serebrovaskular) | [17] Lu & Qiu, 2023 |

---

## PANDUAN JAWAB PERTANYAAN DOSEN

### Q1: "Kenapa F2-Score, bukan F1-Score?"
**A:** Karena F2-Score memberikan bobot lebih besar pada Recall (β=2). Dalam konteks medis, lebih berbahaya jika pasien stroke tidak terdeteksi (False Negative) dibandingkan salah mendeteksi orang sehat sebagai berisiko (False Positive). Zhang et al. [11] membuktikan secara matematis bahwa optimasi threshold harus disesuaikan dengan konteks keparahan kesalahan. F1-Score memperlakukan Precision dan Recall setara, sedangkan F2-Score memprioritaskan deteksi.

### Q2: "Kenapa harus zero-leakage? Apa dampaknya kalau ada leakage?"
**A:** Merujuk van de Mortel dan van Wingen [13], data leakage menyebabkan model melihat informasi dari data testing selama pelatihan, sehingga evaluasi menghasilkan performa yang terlalu optimis. Pada konteks SMOTE, jika oversampling dilakukan sebelum splitting, data sintetis bisa mengandung informasi dari data testing, sehingga model "curang" dan performanya tidak merepresentasikan kemampuan asli pada data baru.

### Q3: "Kenapa meta-learner-nya Logistic Regression?"
**A:** Merujuk Zian et al. [4] yang melakukan evaluasi empiris terhadap berbagai meta-learner, Logistic Regression dipilih karena: (1) bersifat linear dan sederhana sehingga tidak overfitting pada fitur meta yang sedikit (hanya 3 probabilitas dari base learner), (2) mendukung class weighting secara native, dan (3) koefisiennya langsung dapat diinterpretasi menggunakan SHAP LinearExplainer untuk analisis level ensemble.

### Q4: "Apa bedanya penelitian ini dengan penelitian Faria et al. [1]?"
**A:** Faria et al. [1] membuktikan efektivitas stacking untuk deteksi stroke, namun memiliki tiga keterbatasan: (1) belum menjamin zero-leakage, (2) belum menerapkan threshold optimization, dan (3) belum mengintegrasikan Explainable AI. Penelitian ini menutup ketiga gap tersebut secara bersamaan dalam satu framework utuh.

### Q5: "SHAP dua level bedanya apa dengan SHAP biasa?"
**A:** SHAP biasa pada penelitian sebelumnya [10] hanya diterapkan pada satu level, yaitu langsung melihat kontribusi fitur pasien terhadap model tunggal. Pada penelitian ini, Level 1 menganalisis kontribusi probabilitas setiap base learner (XGBoost, LightGBM, CatBoost) terhadap keputusan meta-learner — ini memvalidasi apakah arsitektur stacking bekerja dengan harmonis. Level 2 baru masuk ke analisis fitur klinis pasien pada model dominan. Pendekatan ini mengadaptasi konsep Lu & Qiu [17] pada stacking di bidang serebrovaskular.

### Q6: "Kenapa dataset Fedesoriano, bukan dataset lain?"
**A:** Dataset ini dipilih karena: (1) memiliki kombinasi fitur numerik dan kategorikal yang representatif, (2) memiliki class imbalance yang ekstrem (4,87% stroke vs 95,13% non-stroke) sehingga sesuai dengan permasalahan penelitian, (3) bersifat tabular medis yang cocok untuk tree-based models [7], dan (4) banyak digunakan sebagai benchmark dalam penelitian prediksi stroke [1][5][6][10] sehingga memungkinkan komparasi.

---

*Script ini disusun untuk durasi ~15 menit dengan penekanan pada solusi dan setiap argumen didukung oleh referensi akademis.*
