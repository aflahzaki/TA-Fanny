# SCRIPT PRESENTASI PER SLIDE - SEMINAR PROPOSAL
## "Optimasi dan Interpretasi Prediksi Risiko Stroke Menggunakan Stacking Ensemble Learning dan SHAP pada Data Kesehatan Tidak Seimbang"

**Presenter:** Stiefanny Dwi Chandra (103062300102)
**Durasi Total:** ~15 menit
**Catatan:**
- *[italic]* = instruksi presenter, TIDAK dibacakan
- **[Ref X]** = sitasi yang mendukung argumen
- Referensi mengacu pada daftar pustaka di slide presentasi (bukan proposal)

---


## SLIDE 1 - COVER (~30 detik)

*[Slide: Judul, Nama, NIM, Dosen Pembimbing]*

Bismillahirrahmanirrahim.

Assalamualaikum warahmatullahi wabarakatuh.

Yang terhormat Bapak/Ibu dosen penguji dan pembimbing, serta hadirin yang saya hormati.

Perkenalkan, nama saya Stiefanny Dwi Chandra dengan NIM 103062300102 dari Program Studi Sarjana Teknologi Informasi, Fakultas Informatika, Universitas Telkom Jakarta. Dengan dosen pembimbing pertama Ibu Nurul Ilmi, S.Kom., M.T. dan dosen pembimbing kedua Bapak Achmad Udin Zailani, S.Kom., M.Kom.

Pada kesempatan ini, saya akan mempresentasikan proposal tugas akhir saya yang berjudul **"Optimasi dan Interpretasi Prediksi Risiko Stroke Menggunakan Stacking Ensemble Learning dan SHAP pada Data Kesehatan Tidak Seimbang."**

---


## SLIDE 2 - DAFTAR ISI (~15 detik)

*[Slide: Daftar Isi - Bab 1, 2, 3, Daftar Pustaka]*

Baik, pemaparan saya akan terbagi menjadi tiga bagian utama. **Bab 1 Pendahuluan** yang mencakup latar belakang, rumusan masalah, tujuan, dan kontribusi penelitian. **Bab 2 Kajian Pustaka** yang membahas penelitian terdahulu dan landasan teori. Dan **Bab 3 Perancangan Sistem** yang menjelaskan desain dan alur sistem prediksi. Saya akan mulai dari Bab 1.

---


## SLIDE 3 - HEADER BAB 1: PENDAHULUAN (~10 detik)

*[Slide: Transisi ke Bab 1]*

Baik, Bapak/Ibu, saya akan memulai pemaparan dari Bab 1 yaitu Pendahuluan.

---


## SLIDE 4 - LATAR BELAKANG (~2.5 menit) **[SLIDE KUNCI - PENEKANAN SOLUSI]**

*[Slide: Latar Belakang - Stroke, Data & ML, Permasalahan, Solusi]*

Bapak/Ibu, penelitian ini dilatarbelakangi oleh empat aspek utama yang saling berkaitan.

**Pertama, mengenai Stroke.** Berdasarkan data dari World Stroke Organization tahun 2022, stroke merupakan penyebab kematian kedua dan penyebab disabilitas ketiga secara global **[Ref 1: Feigin et al., 2022 - International Journal of Stroke]**. Kondisi ini menyebabkan gangguan fungsi otak akibat penyumbatan atau pecahnya pembuluh darah, sehingga identifikasi dini terhadap faktor risiko stroke menjadi sangat krusial.

**Kedua, mengenai Data dan Machine Learning.** Perkembangan machine learning memberikan pendekatan yang menjanjikan untuk mempelajari pola dari data kesehatan. Mridha et al. tahun 2023 membuktikan bahwa automated stroke prediction menggunakan machine learning mampu memberikan early intervention yang bermakna **[Ref 2: Mridha et al., 2023 - IEEE Access]**. Byna et al. tahun 2024 dalam critical review mereka juga mengkonfirmasi bahwa machine learning sudah banyak diaplikasikan pada prediksi stroke dengan berbagai pendekatan **[Ref 5: Byna et al., 2024]**.

**Ketiga, mengenai Permasalahan.** Namun, ada tantangan utama yang sering dihadapi, yaitu **class imbalance**. Data rekam medis stroke umumnya didominasi oleh kelas mayoritas atau pasien sehat, sehingga pemodelan konvensional rentan bias dan gagal mendeteksi pasien berisiko tinggi. Pada dataset yang saya gunakan, kelas stroke hanya mencakup 4,87% dari total 5.110 data. Hal ini menyebabkan model yang tampak akurat tinggi justru gagal mendeteksi pasien stroke yang sebenarnya.

*[Bicarakan lebih pelan dan tegas di bagian ini]*

**Keempat, dan yang paling penting, mengenai Solusi.** Untuk mengatasi permasalahan tersebut, saya mengusulkan **tiga solusi terintegrasi**:

1. **Stacking Ensemble Learning** — menggabungkan kekuatan XGBoost, LightGBM, dan CatBoost sebagai base learners dengan Logistic Regression sebagai meta-learner, karena pendekatan ensemble terbukti mampu mengurangi bias dan meningkatkan akurasi prediksi pada data medis **[Ref 8: James et al., 2024; Ref 9: Abdelsamie, 2026]**.

2. **Pipeline SMOTE + Threshold Optimization berbasis F2-Score** — SMOTE diisolasi di dalam pipeline untuk mencegah data leakage, dan batas keputusan dicari secara matematis menggunakan F2-Score agar model lebih sensitif mendeteksi kelas minoritas **[Ref 14: Chawla et al., 2011; Ref 10: Barman et al., 2025]**.

3. **Explainable AI menggunakan SHAP Dua Level** — untuk membongkar sifat black-box model dan memberikan transparansi diagnosis bagi tenaga medis **[Ref 12: Kaya et al., 2026; Ref 15: Agrawal & Sharma, 2024]**.

---


## SLIDE 5 - RUMUSAN MASALAH (~1 menit)

*[Slide: 3 Rumusan Masalah]*

Berdasarkan latar belakang tersebut, saya merumuskan tiga rumusan masalah:

**Rumusan masalah pertama:** Bagaimana membangun model prediksi stroke menggunakan metode Stacking Ensemble? Rumusan ini muncul karena penelitian sebelumnya seperti James et al. [8] telah menunjukkan efektivitas ensemble pada kasus stroke, namun belum menjamin arsitektur yang bebas dari kebocoran data.

**Rumusan masalah kedua:** Bagaimana pengaruh integrasi SMOTE dan threshold tuning dalam mengatasi class imbalance? Ini didasari oleh temuan Barman et al. tahun 2025 yang menunjukkan bahwa kombinasi class imbalance handling dan threshold tuning dapat meningkatkan kemampuan deteksi pada prediksi stroke **[Ref 10: Barman et al., 2025 - IEEE BECITHCON]**.

**Rumusan masalah ketiga:** Bagaimana menguraikan kontribusi fitur klinis menggunakan metode SHAP? Rumusan ini didasari oleh kebutuhan transparansi dalam AI kesehatan, sebagaimana ditegaskan oleh Kaya et al. tahun 2026 bahwa XAI merupakan keharusan untuk membangun kepercayaan pengguna **[Ref 12: Kaya et al., 2026 - IEEE Access]**.

---


## SLIDE 6 - TUJUAN PENELITIAN (~1 menit)

*[Slide: 3 Tujuan Penelitian]*

Sejalan dengan rumusan masalah, tujuan penelitian ini adalah:

**Tujuan pertama:** Membangun dan mengevaluasi arsitektur Stacking Ensemble yang bebas kebocoran data atau zero-leakage, serta membandingkan signifikansi performanya terhadap model baseline Random Forest. Random Forest dipilih sebagai baseline karena merupakan model tree-based ensemble yang stabil dan sering menjadi acuan awal **[Ref 7: Amin & Ilyas, 2026]**.

**Tujuan kedua:** Menganalisis efektivitas penerapan Pipeline SMOTE dan Threshold Optimization berbasis F2-Score dalam meminimalkan kegagalan deteksi atau False Negative, serta mengevaluasi trade-off antara Recall dan Precision. F2-Score dipilih karena memberikan bobot lebih besar pada Recall — dalam konteks medis, lebih berbahaya gagal mendeteksi pasien stroke dibanding salah mendeteksi orang sehat.

**Tujuan ketiga:** Menjelaskan faktor-faktor yang memengaruhi hasil prediksi secara hierarkis menggunakan SHAP dua level — baik kontribusi algoritmik maupun faktor klinis pasien.

Jadi intinya, Bapak/Ibu, penelitian ini bertujuan menghasilkan model yang **akurat, sensitif, dan transparan**.

---


## SLIDE 7 - KONTRIBUSI PENELITIAN (~1 menit)

*[Slide: 4 Kontribusi Penelitian]*

Bapak/Ibu, kontribusi utama dari penelitian ini ada empat:

**Pertama**, penerapan Stacking Ensemble Learning untuk prediksi stroke — menggabungkan kekuatan XGBoost, LightGBM, dan CatBoost yang terbukti secara empiris masih mengungguli deep learning pada data tabular klinis **[Ref 7: Amin & Ilyas, 2026; didukung oleh Grinsztajn et al., 2022 dari proposal]**.

**Kedua**, penanganan class imbalance menggunakan SMOTE yang diisolasi secara ketat di dalam pipeline pelatihan. Ini memastikan zero data leakage sehingga evaluasi model benar-benar valid dan tidak menghasilkan ilusi performa.

**Ketiga**, optimasi model dengan fokus pada F2-Score. Tidak seperti penelitian sebelumnya yang menggunakan threshold default 0,5, sistem saya mencari ambang batas optimal secara matematis untuk meminimalkan False Negative — atau pasien stroke yang gagal terdeteksi.

**Keempat**, interpretasi model menggunakan SHAP yang memberikan insight tentang faktor-faktor apa saja yang paling memengaruhi risiko stroke. Ini menjadikan model tidak hanya akurat, tetapi juga dapat dipahami dan dipercaya oleh tenaga medis **[Ref 12: Kaya et al., 2026; Ref 15: Agrawal & Sharma, 2024]**.

---


## SLIDE 8 - JADWAL PENELITIAN (~30 detik)

*[Slide: Tabel Jadwal 8 Kegiatan x 5 Bulan]*

Untuk jadwal pelaksanaan, penelitian ini dirancang dalam delapan fase selama lima bulan. Dimulai dari studi literatur di bulan pertama, dilanjutkan dengan EDA, data preparation, model development, threshold optimization, pengujian, implementasi SHAP, hingga analisis hasil dan penyusunan laporan di bulan kelima. Setiap fase saling berkaitan dan didesain secara sistematis untuk memastikan penelitian berjalan terarah.

Selanjutnya, saya akan membahas Bab 2 yaitu Kajian Pustaka.

---


## SLIDE 9 - HEADER BAB 2: KAJIAN PUSTAKA (~10 detik)

*[Slide: Transisi ke Bab 2]*

Baik, Bapak/Ibu, saya lanjutkan ke Bab 2 yaitu Kajian Pustaka yang terdiri dari penelitian terdahulu dan landasan teori.

---


## SLIDE 10 - PENELITIAN TERDAHULU (Tabel 1: Ref [1] dan [6]) (~1.5 menit)

*[Slide: Matriks Tinjauan Pustaka - Paper [1] Faria et al. dan [6] Ozkara et al.]*

Bapak/Ibu, saya akan memaparkan penelitian-penelitian terdahulu yang menjadi landasan penelitian ini.

**Penelitian pertama** dari Faria et al. tahun 2024 yang berjudul "Stroke Detection Through Ensemble Learning: A Stacking Approach." Penelitian ini membuktikan bahwa teknik stacking ensemble mampu mengungguli classifier tunggal dalam mendeteksi risiko stroke. Sebagaimana kutipan langsung dari paper tersebut: *"Stacked ensemble techniques have demonstrated superior performance compared to individual models."*

Namun, Bapak/Ibu, penelitian Faria et al. memiliki **tiga celah metodologis**: belum menerapkan Explainable AI, belum melakukan threshold optimization, dan rentan terhadap data leakage karena tidak menggunakan pipeline khusus untuk penanganan imbalance data. Selain itu, di halaman 2 paper mereka, Faria et al. menghapus baris data kosong (dropping null values) yang mereka akui sendiri dapat menyebabkan hilangnya informasi berharga. Penelitian saya memperbaiki hal ini dengan imputasi median.

**Penelitian kedua** dari Ozkara et al. tahun 2023 **[Ref 6]** tentang prediksi fungsional stroke menggunakan machine learning. Penelitian ini fokus pada preprocessing dan feature selection, namun belum menangani class imbalance secara khusus dan belum menggunakan threshold tuning maupun SHAP.

Kedua celah inilah yang menjadi dasar pengembangan penelitian saya.

---


## SLIDE 11 - PENELITIAN TERDAHULU (Tabel 2: Ref [9] dan [10]) (~1.5 menit)

*[Slide: Matriks Tinjauan Pustaka - Paper [9] Hwangbo et al. dan [10] Barman et al.]*

**Penelitian ketiga** dari Hwangbo et al. tahun 2022 yang berjudul "Stacking Ensemble Learning Model to Predict 6-Month Mortality in Ischemic Stroke Patients." Penelitian ini memvalidasi keandalan metode stacking pada skenario medis klinis nyata di rumah sakit. Kutipan dari paper mereka menyatakan: *"Stacking ensemble learning is an algorithm structure consisting of more than one level of ML algorithms that is known to produce a more reliable model."*

Namun, Hwangbo et al. fokus pada **mortalitas pasca-stroke**, bukan deteksi dini. Mereka juga belum membahas class imbalance, threshold optimization, maupun interpretabilitas model. Arsitektur saya mengadaptasi keandalan komputasi stacking mereka, tetapi untuk skenario yang berbeda yaitu **screening preventif** pada populasi umum, dan dengan meta-learner Logistic Regression yang lebih stabil dibanding ANN yang mereka gunakan.

**Penelitian keempat** dari Barman et al. tahun 2025 **[Ref 10]** tentang "Robust Brain Stroke Prediction via Ensemble Machine Learning with Class Imbalance Handling and Threshold Tuning." Penelitian ini sangat relevan karena sudah mengkombinasikan ensemble learning dengan penanganan imbalance dan threshold tuning. Namun, penerapan SHAP pada arsitektur stacking belum dieksplorasi. Penelitian saya menutup celah interpretabilitas tersebut.

---


## SLIDE 12 - PENELITIAN TERDAHULU (Tabel 3: Ref [11]) (~45 detik)

*[Slide: Matriks Tinjauan Pustaka - Paper [11] Tahyudin et al.]*

**Penelitian kelima** dari Tahyudin et al. tahun 2024 **[Ref 11]** yang membahas optimasi prediksi mortalitas stroke melalui analisis faktor risiko dan teknik hyperparameter tuning. Penelitian ini membuktikan bahwa penyesuaian parameter model sangat signifikan dalam meningkatkan performa prediksi pada kasus stroke.

Namun, Tahyudin et al. tidak menggunakan arsitektur stacking dan belum menerapkan threshold optimization berbasis F2-Score yang secara spesifik menargetkan minimalisasi False Negative. Penelitian saya menggunakan landasan pentingnya tuning dari paper ini, tetapi mengembangkannya ke arah optimasi batas keputusan yang lebih fokus pada sensitivitas deteksi kelas minoritas.

---


## SLIDE 13 - POSISI PENELITIAN / RESEARCH GAP (~1 menit)

*[Slide: Research Gap - Zero Leakage SMOTE, F2-Score Threshold Optimization, Two-Level SHAP]*

Bapak/Ibu, berdasarkan analisis terhadap lima penelitian terdahulu tersebut, saya mengidentifikasi **tiga celah penelitian utama** yang menjadi posisi penelitian saya:

**Celah pertama: Zero Leakage SMOTE.** Sebagian besar penelitian terdahulu menerapkan oversampling sebelum pembagian data validasi. Berdasarkan kajian ilmiah, jika SMOTE dilakukan sebelum pemisahan data latih-uji, maka akan menghasilkan hasil yang tidak realistis dan menyesatkan — karena data sintetis memiliki korelasi sirkuler dengan data testing. Oleh karena itu, saya mengisolasi SMOTE di dalam Pipeline.

**Celah kedua: F2-Score Threshold Optimization.** Penelitian terdahulu menggunakan ambang batas default 0,5 yang terbukti berakibat fatal pada data imbalanced. Saya mengoptimasi threshold secara matematis menggunakan F2-Score yang memprioritaskan Recall — karena dalam konteks stroke, gagal mendeteksi pasien berisiko jauh lebih berbahaya.

**Celah ketiga: Two-Level SHAP.** Penerapan SHAP pada penelitian sebelumnya hanya pada model tunggal atau satu level. Saya mengusulkan **Dual-Layer SHAP Framework**: Level 1 untuk validasi algoritmik (kontribusi base learner), dan Level 2 untuk validasi klinis (faktor risiko pasien).

Tiga kontribusi inilah yang membedakan penelitian saya dari penelitian-penelitian sebelumnya.

---


## SLIDE 14 - HEADER LANDASAN TEORI (~10 detik)

*[Slide: Transisi ke Landasan Teori]*

Selanjutnya, Bapak/Ibu, saya akan memaparkan landasan teori yang mendasari penelitian ini.

---


## SLIDE 15 - LANDASAN TEORI (Bagian 1) (~1.5 menit)

*[Slide: Imbalanced Data, SMOTE & Zero-Leakage Pipeline, Pipeline Preprocessing, Random Forest]*

Bapak/Ibu, landasan teori pertama berkaitan dengan permasalahan yang ingin diselesaikan:

**Imbalanced Data & Stroke.** Berdasarkan Salmi et al. tahun 2024, data rekam medis stroke umumnya didominasi oleh kelas mayoritas atau pasien sehat. Pengklasifikasi konvensional cenderung memprioritaskan akurasi keseluruhan yang tinggi, yang berpotensi mengarah pada kesalahan klasifikasi pasien berisiko sebagai individu sehat. Ini adalah masalah fundamental yang harus diatasi.

**SMOTE & Zero-Leakage Pipeline.** Untuk menangani class imbalance, SMOTE digunakan untuk menyintesis data minoritas **[Ref 14: Chawla et al., 2011]**. Namun, penerapannya harus ketat. Van de Mortel dan van Wingen tahun 2025 membuktikan bahwa menggunakan metode sampling baru sebelum pemisahan latih-uji akan secara tidak benar menginformasikan efektivitasnya. Oleh karena itu, SMOTE diisolasi di dalam Pipeline.

**Pipeline Preprocessing.** Integrasi seluruh tahapan prapemrosesan — imputasi, scaling, encoding — ke dalam satu jalur berurutan diterapkan secara ketat pada setiap lipatan validasi untuk memastikan nol risiko kebocoran data.

**Random Forest sebagai Baseline.** Digunakan sebagai model pembanding utama. Mashi et al. tahun 2024 **[Ref 8 presentasi / Ref 5 proposal]** membuktikan bahwa metode ensemble mampu memitigasi overfitting dan meningkatkan akurasi prediksi. Random Forest menjadi tolok ukur evaluasi yang adil terhadap peningkatan performa arsitektur stacking.

---


## SLIDE 16 - LANDASAN TEORI (Bagian 2) (~1.5 menit)

*[Slide: Gradient Boosting Models, Stacking Ensemble, Optimasi Threshold & F2-Score, Explainable AI (SHAP)]*

**Gradient Boosting Models.** Pemanfaatan tree-based ensemble tingkat lanjut yaitu XGBoost, LightGBM, dan CatBoost terbukti secara empiris masih mengungguli deep learning dalam memproses data tabular klinis yang kompleks. Ini berdasarkan penelitian Grinsztajn et al. tahun 2022, dan menjadi alasan utama pemilihan ketiga algoritma ini sebagai base learner.

Secara spesifik: XGBoost memiliki regularisasi yang mencegah overfitting, LightGBM menggunakan pendekatan leaf-wise yang lebih efisien secara komputasi dengan teknik histogram dan GOSS untuk memfokuskan pembelajaran pada sampel yang sulit, dan CatBoost memiliki keunggulan dalam menangani fitur kategorikal secara native melalui ordered boosting.

**Stacking Ensemble Learning.** Berdasarkan Faria et al. tahun 2024, arsitektur stacking mengagregasi prediksi probabilitas dari beberapa algoritma dasar menggunakan meta-learner untuk meningkatkan stabilitas dan akurasi prediksi stroke secara signifikan.

**Optimasi Threshold & F2-Score.** Penerapan cost-sensitive learning yang menggeser ambang batas probabilitas model untuk memprioritaskan sensitivitas deteksi, berdasarkan Araf et al. tahun 2024, guna meminimalkan False Negative klinis.

**Explainable AI (SHAP).** Implementasi SHAP pada model stacking untuk membongkar sifat black-box dan memberikan transparansi diagnosis ganda — global dan spesifik per pasien — bagi tenaga medis, berdasarkan Moulaei et al. tahun 2024 dan Abousaber tahun 2025 **[Ref 13: Abousaber, 2025]**.

---


## SLIDE 17 - HEADER BAB 3: PERANCANGAN SISTEM (~10 detik)

*[Slide: Transisi ke Bab 3]*

Baik, Bapak/Ibu, selanjutnya saya akan memaparkan Bab 3 yaitu Perancangan Sistem.

---


## SLIDE 18 - DATASET DAN KETIMPANGAN KELAS (~45 detik)

*[Slide: Dataset, Metode Penelitian]*

Bapak/Ibu, penelitian ini menggunakan pendekatan kuantitatif dengan metode eksperimen.

Mengenai dataset, saya menggunakan **Stroke Prediction Dataset** dari platform Kaggle yang dipublikasikan oleh Fedesoriano. Dataset ini berisi **5.110 baris observasi** rekam medis anonim, dengan:
- Kelas mayoritas: **4.861 pasien sehat** (95,13%)
- Kelas minoritas: **249 pasien stroke** (4,87%)

Dataset ini dipilih karena telah divalidasi penggunaannya oleh berbagai penelitian terdahulu **[Ref 8: James et al., 2024]** dan memiliki karakteristik yang sesuai dengan permasalahan penelitian ini: kombinasi fitur numerik dan kategorikal, distribusi kelas yang sangat tidak seimbang, serta bersifat data tabular medis.

Ketimpangan kelas yang ekstrem inilah — 95 berbanding 5 — yang menjadikan strategi penanganan class imbalance menjadi komponen kritikal dalam penelitian ini.

---


## SLIDE 19 - DESAIN SISTEM (~30 detik)

*[Slide: Diagram Desain Sistem]*

Bapak/Ibu, ini adalah gambaran umum desain sistem yang saya rancang. Arsitektur sistem terdiri dari enam fase utama yang saling berhubungan, mulai dari eksplorasi data, persiapan data, pengembangan model, optimasi threshold, pengujian, hingga interpretasi menggunakan SHAP. Setiap fase dirancang untuk menghasilkan model prediksi yang akurat, sensitif terhadap kasus stroke, dan mampu menjelaskan faktor-faktor yang memengaruhi keputusan prediksi.

Izinkan saya menjelaskan masing-masing fase secara detail.

---


## SLIDE 20 - SKENARIO & ALUR SISTEM PREDIKSI (~30 detik)

*[Slide: Daftar 6 Fase]*

Keseluruhan alur sistem prediksi terdiri dari enam fase:
- Fase 1: Eksplorasi dan Pemahaman Dataset
- Fase 2: Persiapan Data
- Fase 3: Pengembangan Model Prediksi
- Fase 4: Threshold Optimization dan Evaluasi Model
- Fase 5: Pengujian pada Hold-Out Test Set
- Fase 6: Explainable Artificial Intelligence menggunakan SHAP

Seluruh alur dibangun dengan fokus utama pada pencegahan kebocoran data dan optimasi sensitivitas model. Saya akan jelaskan satu per satu.

---


## SLIDE 21 - FASE 1: EKSPLORASI DAN PEMAHAMAN DATASET (~30 detik)

*[Slide: EDA - Identifikasi Struktur, Analisis Ketimpangan, Cek Missing Value, Visualisasi]*

Fase pertama adalah Exploratory Data Analysis. Di fase ini, saya melakukan empat aktivitas utama:

**Pertama**, identifikasi struktur data dan pemilahan fitur menjadi fitur numerik (seperti age, avg_glucose_level, bmi) dan fitur kategorikal (seperti gender, smoking_status).

**Kedua**, analisis ketimpangan kelas untuk memvalidasi secara empiris bahwa rasio stroke vs non-stroke memang sangat timpang — ini menjadi justifikasi utama mengapa akurasi konvensional tidak dapat diandalkan.

**Ketiga**, pengecekan missing value. Secara khusus, fitur BMI memiliki sejumlah nilai kosong. Berbeda dengan Faria et al. yang menghapus baris kosong, saya menggunakan **imputasi median** karena distribusi data fisiologis manusia rentan terhadap outlier.

**Keempat**, visualisasi EDA untuk memahami pola distribusi data.

---


## SLIDE 22 - FASE 2: PERSIAPAN DATA (~45 detik)

*[Slide: Data Preparation - Training/Testing Split, SMOTE Inside Pipeline, Pipeline Preprocessing]*

Fase kedua adalah **lini pertahanan utama** untuk menjaga integritas eksperimen.

**Pertama**, pemisahan data menggunakan **Stratified Split** — 80% untuk training dan 20% untuk testing. Teknik stratified memastikan proporsi pasien stroke tetap sama di kedua bagian. Data testing diisolasi sepenuhnya dan **tidak pernah tersentuh** sampai pengujian akhir.

**Kedua**, Pipeline Preprocessing yang mengintegrasikan imputasi median untuk fitur numerik, StandardScaler untuk standarisasi, dan One-Hot Encoding untuk fitur kategorikal.

**Ketiga**, dan yang paling krusial: **SMOTE ditempatkan di dalam Pipeline**. Ini adalah pembeda utama penelitian saya. Berdasarkan Salmi et al. tahun 2024: *"Menggunakan metode sampling sebelum pemisahan latih-uji akan secara tidak benar menginformasikan efektivitasnya."* Dengan pendekatan pipeline, SMOTE hanya dilakukan secara internal pada data latih di setiap fold, sehingga data testing tetap steril dan merepresentasikan kondisi asli **[Ref 14: Chawla et al., 2011 — metode SMOTE]**.

---


## SLIDE 23 - FASE 3: PENGEMBANGAN MODEL PREDIKSI (~1 menit)

*[Slide: Model Development - Baseline, Base Learners, Arsitektur Stacking, Meta Learner]*

Fase ketiga adalah pembangunan arsitektur model.

**Baseline: Random Forest.** Digunakan sebagai tolok ukur awal untuk memvalidasi apakah arsitektur stacking yang lebih kompleks benar-benar memberikan peningkatan performa yang signifikan.

**Base Learners:** Tiga algoritma Gradient Boosting yang andal pada data tabular:
- **XGBoost** — pencarian pola kompleks dengan regularisasi untuk mencegah overfitting
- **LightGBM** — efisiensi komputasi melalui histogram-based learning dan leaf-wise growth
- **CatBoost** — penanganan fitur kategorikal secara native melalui ordered boosting

Ketiga model ini dioptimasi menggunakan RandomizedSearchCV dengan **F2-Score sebagai fungsi objektif**, sehingga sejak awal pelatihan model sudah dipaksa memprioritaskan sensitivitas deteksi.

**Arsitektur Stacking Ensemble:** Ketiga base learner menghasilkan probabilitas **Out-of-Fold (OOF)** melalui cross-validation. Probabilitas OOF ini menjadi fitur baru yang dipelajari oleh **Logistic Regression sebagai Meta-Learner**. Meta-learner juga menggunakan **class weighting** untuk memberikan penalti lebih besar jika model gagal mendeteksi pasien stroke.

---


## SLIDE 24 - FASE 4: THRESHOLD OPTIMIZATION DAN EVALUASI MODEL (~1 menit)

*[Slide: Probabilitas OOF, Threshold Optimal, Evaluasi Repeated SKFCV, Laporan Metrik]*

Fase keempat bertujuan menentukan ambang batas keputusan optimal.

**Pertama**, menghasilkan **prediksi OOF** menggunakan Stratified K-Fold Cross Validation. Prediksi OOF ini digunakan sebagai input meta-learner dan sebagai acuan perhitungan threshold.

**Kedua**, **optimasi threshold.** Bapak/Ibu, model klasifikasi secara bawaan menggunakan threshold 0,5. Pada data medis yang tidak seimbang, ini berisiko tinggi menghasilkan banyak False Negative. Berdasarkan landasan teoritis dari Zhang et al. yang membuktikan bahwa threshold default sering berakibat fatal, saya mengevaluasi probabilitas prediksi secara dinamis dan menentukan titik threshold optimal berdasarkan pencapaian **F2-Score tertinggi** — dengan tetap menjaga Precision pada level yang logis.

**Ketiga**, evaluasi stabilitas menggunakan **Repeated Stratified K-Fold Cross Validation** (5-fold, 3 repetisi) — artinya model diuji melewati 15 skenario pembagian data yang berbeda.

**Keempat**, pelaporan metrik yang komprehensif: Accuracy, Precision, Recall, Metric Gap, F1-Score, F2-Score, ROC-AUC, dan PR-AUC — dilaporkan dalam format **Mean ± Standard Deviation** untuk membuktikan kestabilan prediksi.

---


## SLIDE 25 - FASE 5: PENGUJIAN HOLD-OUT TEST (~45 detik)

*[Slide: Pelatihan Final, Penerapan Threshold, Evaluasi Performa, Visualisasi]*

Fase kelima adalah pembuktian empiris kemampuan generalisasi model.

**Pertama**, pelatihan final — seluruh arsitektur stacking dilatih ulang menggunakan 100% data training.

**Kedua**, data testing yang telah diisolasi sejak Fase 2 dan **belum pernah terpapar** proses pelatihan maupun SMOTE, dimasukkan ke model untuk menghasilkan probabilitas prediksi.

**Ketiga**, penerapan threshold klinis — probabilitas tidak dikonversi menggunakan default 0,5, melainkan menggunakan **threshold optimal berbasis F2-Score** yang telah dihitung di Fase 4.

**Keempat**, evaluasi performa pada data baru untuk memverifikasi apakah model tetap stabil atau mengalami penurunan yang mengindikasikan overfitting.

Hasil dari fase ini juga divisualisasikan melalui **Confusion Matrix** dan **kurva ROC** untuk memudahkan analisis kinerja model.

---


## SLIDE 26 - FASE 6: EXPLAINABLE AI (SHAP) (~1 menit) **[SLIDE KUNCI]**

*[Slide: SHAP - Global Interpretation, Local Explanation, Feature Ranking]*

Fase terakhir dan menjadi **kebaruan utama** penelitian ini: implementasi SHAP melalui **Dual-Layer Interpretability Framework**.

Bapak/Ibu, sebagian besar studi yang ada memperlakukan model machine learning sebagai "kotak hitam" dan jarang memberikan penjelasan — yang mana dapat mengurangi penerimaannya oleh komunitas medis. Lu & Qiu tahun 2023 menegaskan bahwa meningkatkan transparansi model ML di ranah medis adalah keharusan.

Oleh karena itu, saya mengusulkan interpretasi dua level:

**Layer 1: Algorithmic Validation** — menggunakan LinearExplainer pada Meta-Learner (Logistic Regression) untuk membedah kontribusi probabilitas setiap base learner. Ini memvalidasi secara matematis algoritma mana yang paling berkontribusi terhadap keputusan akhir.

**Layer 2: Clinical Diagnostics** — menggunakan KernelExplainer pada model secara keseluruhan untuk menghasilkan:
- **Global Interpretation (Summary Plot):** Memvisualisasikan pola hubungan fitur klinis dengan risiko stroke pada level populasi — memvalidasi apakah arah pengaruh fitur konsisten dengan kaidah medis.
- **Local Explanation (Waterfall Plot):** Dekonstruksi risiko pada level individual — memberikan justifikasi medis mengapa seorang pasien tertentu diklasifikasikan berisiko tinggi.
- **Feature Ranking (Bar Plot):** Mengidentifikasi fitur-fitur yang paling berpengaruh.

Pendekatan dua level ini memberikan **akuntabilitas ganda**: transparansi algoritmik untuk validasi sistem, dan transparansi klinis untuk validasi diagnosis **[Ref 12: Kaya et al., 2026; Ref 13: Abousaber, 2025; Ref 15: Agrawal & Sharma, 2024]**.

---


## SLIDE 27, 28, 29 - DAFTAR PUSTAKA (~15 detik)

*[Slide: Daftar Pustaka]*

Bapak/Ibu, seluruh argumen dan landasan teori yang saya paparkan didukung oleh 15 referensi ilmiah yang telah terverifikasi, mulai dari jurnal bereputasi seperti IEEE Access, Scientific Reports, BMC Medical Informatics, hingga konferensi internasional seperti IEEE BECITHCON dan ICCCNT. Daftar lengkapnya dapat dilihat pada slide ini.

---


## SLIDE 30 - PENUTUP (~30 detik)

*[Slide: Sekian Terima Kasih]*

Baik, Bapak/Ibu, demikian pemaparan proposal tugas akhir saya.

Sebagai ringkasan, penelitian ini mengusulkan pendekatan end-to-end yang mengintegrasikan:
1. Arsitektur **Stacking Ensemble** yang terjamin bebas kebocoran data
2. **Pipeline SMOTE + Threshold Optimization berbasis F2-Score** untuk memaksimalkan deteksi kelas minoritas
3. **SHAP Dua Level** untuk transparansi keputusan yang dapat dipertanggungjawabkan secara komputasional maupun medis

Demikian presentasi dari saya. Mohon maaf apabila ada kekurangan dalam penyampaian. Apakah ada pertanyaan dari Bapak/Ibu dosen?

Terima kasih.
Wassalamualaikum warahmatullahi wabarakatuh.

---


---
---

# LAMPIRAN: PETA SITASI PER SLIDE

| Slide | Argumen Utama | Referensi Pendukung (Daftar Pustaka Presentasi) |
|-------|--------------|------------------------------------------------|
| 4 | Stroke sebagai masalah global | [1] Feigin et al., 2022 - WSO Global Stroke Fact Sheet |
| 4 | ML untuk prediksi stroke | [2] Mridha et al., 2023; [5] Byna et al., 2024 |
| 4 | Ensemble efektif pada stroke | [8] James et al., 2024; [9] Abdelsamie, 2026 |
| 4 | SMOTE untuk class imbalance | [14] Chawla et al., 2011 |
| 4 | Threshold tuning pada stroke | [10] Barman et al., 2025 |
| 4 | XAI untuk transparansi | [12] Kaya et al., 2026; [15] Agrawal & Sharma, 2024 |
| 5 | Stacking untuk stroke | [8] James et al., 2024 |
| 5 | SMOTE + threshold tuning | [10] Barman et al., 2025; [14] Chawla et al., 2011 |
| 5 | SHAP untuk interpretasi | [12] Kaya et al., 2026 |
| 6 | Baseline Random Forest | [7] Amin & Ilyas, 2026 |
| 7 | Stacking > single model | [8] James et al., 2024; [9] Abdelsamie, 2026 |
| 7 | Zero-leakage pipeline | Salmi et al., 2024 (dari proposal [2]) |
| 7 | F2-Score optimization | [10] Barman et al., 2025 |
| 7 | SHAP interpretability | [12] Kaya et al., 2026; [15] Agrawal & Sharma, 2024 |
| 10 | Stacking ungguli single model | Faria et al., 2024 (dari proposal [1]) |
| 11 | Stacking pada medis nyata | Hwangbo et al., 2022 (dari proposal [9]) |
| 11 | Ensemble + imbalance + tuning | [10] Barman et al., 2025 |
| 12 | Threshold default fatal | Tahyudin et al., 2024; Zhang et al. (dari proposal [11]) |
| 15 | Imbalanced data masalah | Salmi et al., 2024 (dari proposal [2]) |
| 15 | SMOTE zero-leakage | [14] Chawla et al., 2011; van de Mortel, 2025 (proposal [13]) |
| 15 | Random Forest baseline | [8] James et al., 2024 |
| 16 | Tree-based > deep learning pada tabular | Grinsztajn et al., 2022 (dari proposal [7]) |
| 16 | Stacking ensemble | Faria et al., 2024 (dari proposal [1]) |
| 16 | Cost-sensitive / threshold | Araf et al., 2024 (dari proposal [12]) |
| 16 | SHAP pada stacking | [13] Abousaber, 2025; Moulaei et al., 2024 (proposal [10]) |
| 22 | SMOTE inside pipeline | [14] Chawla et al., 2011; Salmi et al., 2024 |
| 23 | XGBoost, LightGBM, CatBoost | Chen & Guestrin 2016; Ke et al. 2017; Prokhorenkova et al. 2017 |
| 24 | Threshold optimization F2 | Zhang et al. (proposal [11]); [10] Barman et al., 2025 |
| 26 | SHAP Dua Level | Lu & Qiu, 2023 (proposal [17]); [12] Kaya et al., 2026 |

---


# LAMPIRAN: PANDUAN JAWAB PERTANYAAN DOSEN

## Q1: "Penelitian Faria et al. (2024) sudah mencapai akurasi 99,63%. Mengapa Anda meneliti ulang hal yang sama?"

**Jawaban:**
"Bapak/Ibu Penguji yang saya hormati, Faria et al. (2024) memang telah membuktikan keandalan arsitektur Stacking Ensemble pada kasus stroke. Namun, penelitian mereka memiliki tiga celah metodologi yang secara logis disempurnakan dalam proposal saya:

Pertama, Penanganan Data Kosong: Di halaman 2 paper mereka, Faria et al. membuang nilai kosong (dropping null values) yang mereka akui sendiri menyebabkan hilangnya informasi berharga. Pada desain saya, hilangnya data dicegah dengan menerapkan imputasi median untuk fitur numerik.

Kedua, Pencegahan Data Leakage: Faria et al. melakukan oversampling sebelum pembagian data validasi, yang menghasilkan akurasi 99% yang semu akibat data leakage. Saya mengoreksi hal ini dengan mengurung SMOTE ketat di dalam Pipeline pada setiap putaran cross-validation.

Ketiga, Optimasi dan Interpretasi: Faria et al. menggunakan ambang batas default 0,5 dan tidak memiliki transparansi model. Sistem saya menggeser ambang batas tersebut melalui Threshold Optimization berbasis F2-Score dan membuka logika keputusannya menggunakan SHAP.

Dengan demikian, penelitian saya tidak sekadar meniru arsitektur mereka, melainkan membangun sistem yang secara metodologi jauh lebih ketat, valid, dan dapat dijelaskan secara medis."

---

## Q2: "Mengapa Anda menggunakan banyak metrik dan tidak berfokus pada Akurasi saja?"

**Jawaban:**
"Bapak/Ibu Penguji, penolakan saya terhadap metrik Akurasi tunggal didasarkan pada kajian literatur dari Araf et al. (2024) dan Salmi et al. (2024). Pada himpunan data medis dengan rasio ketimpangan 95 berbanding 5, sebuah model yang memprediksi seluruh pasien sebagai 'tidak stroke' akan secara otomatis memperoleh Akurasi 95%. Namun, secara klinis model tersebut gagal total karena tidak dapat mendeteksi satu pun pasien stroke.

Mengandalkan Akurasi pada data miring menghasilkan ilusi performa yang menyesatkan. Oleh karena itu, saya mengimplementasikan analisis Metric Gap serta memprioritaskan metrik Recall dan F2-Score. Secara logika medis, model yang kehilangan sedikit Akurasi namun berhasil mendeteksi lebih banyak pasien berisiko stroke jauh lebih bernilai dan aman."

---

## Q3: "Kenapa F2-Score, bukan F1-Score?"

**Jawaban:**
"F2-Score memberikan bobot lebih besar pada Recall dibanding Precision, dengan beta = 2. Dalam konteks prediksi stroke, biaya kesalahan tidak simetris — gagal mendeteksi pasien stroke (False Negative) jauh lebih berbahaya dibanding salah mendeteksi orang sehat sebagai berisiko (False Positive). Pasien stroke yang tidak terdeteksi bisa kehilangan golden period untuk penanganan medis.

F1-Score memperlakukan Precision dan Recall setara, yang tidak sesuai dengan urgensi klinis kasus ini. Oleh karena itu, F2-Score dipilih sebagai fungsi objektif karena secara matematis mendorong model untuk meminimalkan False Negative."

---

## Q4: "Kenapa meta-learner-nya Logistic Regression, bukan Neural Network?"

**Jawaban:**
"Pemilihan Logistic Regression sebagai meta-learner didasari tiga alasan:

Pertama, Logistic Regression bersifat linear dan sederhana sehingga tidak overfitting pada fitur meta yang jumlahnya sedikit — hanya 3 probabilitas dari base learner. Berbeda dengan Hwangbo et al. (2022) yang menggunakan ANN sebagai meta-learner dan rentan overfitting.

Kedua, Logistic Regression mendukung class weighting secara native, sehingga kita bisa memberikan penalti lebih besar pada kegagalan deteksi kelas stroke.

Ketiga, koefisien Logistic Regression dapat langsung diinterpretasi menggunakan SHAP LinearExplainer untuk analisis Layer 1. Jika menggunakan ANN, proses interpretasi algoritmik menjadi jauh lebih kompleks dan kurang transparan."

---

## Q5: "SHAP dua level bedanya apa dengan SHAP biasa?"

**Jawaban:**
"SHAP biasa pada penelitian sebelumnya seperti Moulaei et al. (2024) hanya diterapkan pada satu level — langsung melihat kontribusi fitur pasien terhadap model tunggal.

Pada penelitian ini, saya mengusulkan Dual-Layer Framework:
- Layer 1 menganalisis kontribusi probabilitas setiap base learner (XGBoost, LightGBM, CatBoost) terhadap keputusan meta-learner — ini memvalidasi apakah arsitektur stacking bekerja harmonis.
- Layer 2 baru masuk ke analisis fitur klinis pasien pada model secara keseluruhan — memberikan interpretasi global dan lokal.

Lu & Qiu (2023) menerapkan SHAP langsung pada hasil akhir ensemble tanpa membedah hierarkinya. Pendekatan dua level saya memberikan jaminan akuntabilitas ganda: akuntabilitas komputasi sekaligus akuntabilitas medis."

---

## Q6: "Kenapa dataset Fedesoriano dari Kaggle? Apakah valid untuk penelitian?"

**Jawaban:**
"Dataset ini dipilih karena empat alasan:

Pertama, memiliki kombinasi fitur numerik dan kategorikal yang representatif terhadap faktor risiko stroke.

Kedua, memiliki class imbalance yang ekstrem (4,87% stroke) sehingga sesuai dengan permasalahan penelitian.

Ketiga, bersifat data tabular medis yang cocok untuk tree-based models berdasarkan Grinsztajn et al. (2022).

Keempat, dataset ini telah banyak digunakan sebagai benchmark dalam penelitian prediksi stroke oleh James et al. [8], Barman et al. [10], dan berbagai penelitian lain, sehingga memungkinkan komparasi yang adil dengan hasil penelitian terdahulu."

---

## Q7: "Bagaimana Anda menjamin LightGBM tidak membuang data pasien stroke saat training?"

**Jawaban:**
"Jaminan tersebut dipegang oleh algoritma GOSS (Gradient-based One-Side Sampling) milik LightGBM. Secara prinsip matematis, berdasarkan Ke et al. (2017), baris data dengan gradien kecil adalah pasien-pasien berlabel normal yang polanya sudah dikuasai model. GOSS membuang data normal tersebut, dan mengunci 100% retensi pada baris data bergradien besar — yaitu pasien-pasien dengan gejala stroke yang rumit dan belum dikuasai model. Algoritma ini justru memaksa sistem memusatkan seluruh daya komputasinya pada pasien berisiko tinggi."

---

## Q8: "Apa dampak nyata jika ada data leakage?"

**Jawaban:**
"Jika SMOTE dilakukan sebelum pembagian data, data sintetis yang dibuat dari sampel minoritas bisa sangat mirip atau identik dengan data yang masuk ke fold testing. Akibatnya, model 'curang' — ia mengenali data testing bukan karena benar-benar belajar pola, tapi karena sudah pernah melihat versi sintetis dari data tersebut. Ini menghasilkan skor evaluasi yang terlalu optimis. Berdasarkan van de Mortel dan van Wingen (2025), kebocoran data semacam ini dapat merusak validitas estimasi performa model secara fundamental. Itulah mengapa saya mengisolasi SMOTE ketat di dalam Pipeline."
