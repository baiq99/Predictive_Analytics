## Project Overview
Kinerja akademik siswa merupakan faktor krusial yang memengaruhi kesuksesan mereka dalam dunia pendidikan dan masa depan. Di era digital saat ini, institusi pendidikan memiliki akses terhadap berbagai data siswa, mulai dari nilai ujian, kehadiran, partisipasi kegiatan luar sekolah, hingga informasi demografis. Sayangnya, potensi data ini sering belum dimanfaatkan secara maksimal untuk menghasilkan wawasan strategis.

Metode evaluasi konvensional umumnya bersifat reaktif, baru memberikan perhatian setelah munculnya masalah seperti penurunan nilai atau absensi tinggi. Oleh karena itu, diperlukan pendekatan prediktif yang dapat mengenali risiko sejak dini dan memungkinkan intervensi proaktif oleh pihak sekolah.

Analitik prediktif memungkinkan identifikasi pola kinerja siswa melalui data historis, sehingga membantu pengambilan keputusan berbasis data dalam menyusun intervensi yang tepat sasaran. Dalam proyek ini, digunakan empat model machine learning, yaitu Random Forest, Naive Bayes, Support Vector Machine (SVM), dan Extreme Gradient Boosting (XGBoost). Tujuannya adalah membandingkan kinerja keempat algoritma tersebut untuk menentukan model terbaik dalam memprediksi performa akademik berdasarkan data yang tersedia dari Kaggle.

## Business Understanding

### Problem Statements
Berdasarkan latar belakang tersebut, maka rincian permasalahan yang dapat dibahas pada proyek ini yakni:
1. Berapa banyak waktu belajar mingguan (*Study Time Weekly*) yang optimal untuk meningkatkan GPA siswa?
2. Apakah absensi siswa (*Absences*) berkorelasi negatif dengan GPA mereka?
3. Apakah tutoring (bimbingan belajar) memengaruhi GPA siswa?
4. Apakah terdapat perbedaan performa akademik antara siswa laki-laki dan perempuan (*Gender*) dalam hal GPA?
5. Bagaimana partisipasi siswa dalam kegiatan ekstrakurikuler (*Extracurricular, Sports, Music, Volunteering*) memengaruhi GPA mereka?
6. Apakah dukungan orang tua (*Parental Support*) berhubungan langsung dengan GPA siswa?
7. Faktor mana yang paling berpengaruh terhadap prediksi GPA siswa ketika mempertimbangkan semua atribut (*Age, Gender, Parental Education, Study Time Weekly, Absences, Tutoring, Parental Support, Extracurricular, Sports, Music, Volunteering*)
8. Apa model terbaik yang dapat digunakan untuk memprediksi kinerja siswa?

### Goals
Berdasarkan problem statements, berikut tujuan yang ingin dicapai pada proyek ini.
1. Menampilkan durasi belajar yang lebih efektif.
2. Mendeteksi pola absensi yang berdampak pada penurunan performa akademik.
3. Menilai pengaruh bimbingan belajar untuk meningkatkan performa siswa.
4. Mengidentifikasi apakah ada kesenjangan gender dalam pencapaian akademik.
5. Menilai dampak kegiatan non-akademik terhadap kinerja akademik.
6. Mengukur pentingnya keterlibatan orang tua dalam keberhasilan belajar siswa.
7. Mengukur Faktor yang paling berpengaruh terhadap prediksi GPA Siswa
8. Menemukan model terbaik berdasarkan akurasi tertinggi untuk memprediksi kinerja siswa.

### Solution Statement
1. Melakukan proses *Exploratory Data Analysis* (EDA) untuk menampilkan durasi belajar yang lebih efektif, mendeteksi pola absensi yang berdampak pada penurunan performa akademik, menilai efektivitas bimbingan belajar untuk meningkatkan performa siswa, mengidentifikasi apakah ada kesenjangan gender dalam pencapaian akademik, menilai dampak kegiatan non-akademik terhadap kinerja akademik, mengukur pentingnya keterlibatan orang tua dalam keberhasilan belajar siswa
2. Menggunakan 4 model *machine learning* yaitu *Extreme Gradient Boosting* (XGBoost), *Support Vector Machine* (SVM), *Naive Bayes*, dan *Random Forest* untuk memprediksi kinerja siswa
3. Menggunakan confusion matrix dan f1 score pada masing-masing model *machine learning* untuk menemukan model terbaik berdasarkan akurasi tertinggi.

## Data Understanding
Dataset yang digunakan dalam proyek prediksi kinerja siswa diperoleh dari platform Kaggle melalui tautan (https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset), yang dipublikasikan oleh Rabie El Kharoua pada 13 Juni 2024. Dataset ini mencakup informasi mendetail mengenai 2.392 siswa sekolah menengah, termasuk aspek demografis, kebiasaan belajar, peran orang tua, aktivitas ekstrakurikuler, serta hasil akademik mereka. Variabel target yang digunakan, yaitu GradeClass, mengelompokkan nilai siswa ke dalam beberapa kategori, sehingga menjadikan dataset ini relevan untuk penelitian di bidang pendidikan, analisis statistik, dan pengembangan model prediktif. Dataset tersebut tersedia dalam satu file berformat CSV.

### Deskripsi Variabel
Dataset ini memiliki 15 variabel dengan keterangan sebagai berikut.

| Variabel           | Keterangan |
|--------------------|------------|
| StudentID          | Pengidentifikasi unik yang diberikan kepada setiap siswa (1001 hingga 3392) |
| Age                | Usia siswa berkisar antara 15 hingga 18 tahun |
| Gender             | Jenis kelamin siswa, di mana 0 mewakili Laki-laki dan 1 mewakili Perempuan |
| Ethnicity          | Etnis siswa, dikodekan sebagai berikut: 0(Kaukasia), 1(Afrika Amerika), 2(Asia), 3(Lainnya) |
| ParentalEducation  | Tingkat pendidikan orang tua, dikodekan sebagai berikut: 0(Tidak Ada), 1(Sekolah Menengah Atas), 2(Beberapa Perguruan Tinggi), 3(Sarjana), 4(Lebih Tinggi) |
| StudyTimeWeekly    | Waktu belajar mingguan dalam jam, berkisar antara 0 hingga 20 |
| Absences           | Jumlah ketidakhadiran selama tahun ajaran, berkisar antara 0 hingga 30 |
| Tutoring           | Status bimbingan belajar, di mana 0 menunjukkan Tidak dan 1 menunjukkan Ya |
| ParentalSupport    | Tingkat dukungan orang tua, dikodekan sebagai berikut: 0(Tidak Ada), 1(Rendah), 2(Sedang), 3(Tinggi), 4(Sangat Tinggi) |
| Extracurricular    | Partisipasi dalam kegiatan ekstrakurikuler, di mana 0 menunjukkan Tidak dan 1 menunjukkan Ya |
| Sports             | Partisipasi dalam olahraga, di mana 0 menunjukkan Tidak dan 1 menunjukkan Ya |
| Music              | Partisipasi dalam kegiatan musik, di mana 0 menunjukkan Tidak dan 1 menunjukkan Ya |
| Volunteering       | Partisipasi dalam kesukarelaan, di mana 0 menunjukkan Tidak dan 1 menunjukkan Ya |
| GPA                | Nilai Rata-rata Poin pada skala 2,0 hingga 4,0 |
| GradeClass         | Klasifikasi nilai siswa berdasarkan IPK (0: 'A' (IPK >= 3,5)), (1: 'B' (3,0 <= IPK < 3,5)), (2: 'C' (2,5 <= IPK < 3,0)), (3: 'D' (2,0 <= IPK < 2,5)), (4: 'F' (IPK < 2,0)) |

### Penanganan Missing Value dan Data Duplikat  
Pada tahap ini dilakukan pengecekan terhadap data yang tidak valid di dalam dataset. Setelah diperiksa, tidak ditemukan kolom dengan nilai null. Selain itu, tidak ditemukan data ganda atau duplikat. Dengan demikian, dataset siap digunakan untuk proses analisis selanjutnya.

### Konversi Nilai Kategorikal Numerik Menjadi Objek (String)  
Karena fitur kategorikal pada dataset sudah berbentuk numerik, maka perlu dilakukan konversi nilai numerik tersebut menjadi bentuk string. Langkah ini bertujuan agar label pada fitur bisa ditampilkan secara lebih informatif dalam proses visualisasi saat eksplorasi data univariat dan multivariat.

### Analisis Univariat (Univariate EDA)  
Tahap pertama dalam analisis ini adalah memisahkan variabel menjadi dua jenis, yaitu numerik dan kategorikal.  
- Variabel numerik meliputi: `Age`, `StudyTimeWeekly`, `Absences`, `GPA`.  
- Variabel kategorikal meliputi: `Gender`, `Ethnicity`, `ParentalEducation`, `Tutoring`, `ParentalSupport`, `Extracurricular`, `Sports`, `Music`, `Volunteering`, `GradeClass`.

Selanjutnya, dilakukan pengamatan terhadap nilai unik dalam masing-masing kolom kategorikal. Hasilnya menunjukkan:
1. Kolom Gender memiliki 2 nilai;
2. Kolom Ethnicity memiliki 4 nilai;
3. Kolom ParentalEducation memiliki 5 nilai;
4. Kolom Tutoring memiliki 2 nilai;
5. Kolom ParentalSupport memiliki 5 nilai;
6. Kolom Extracurricular, Sports, Music, dan Volunteering masing-masing memiliki 2 nilai;
7. Kolom GradeClass sebagai variabel target memiliki 5 nilai berbeda.

Pada tahap selanjutnya, dibuat visualisasi data kategorikal dalam bentuk grafik batang menggunakan pustaka `matplotlib`.  
Interpretasi grafik:  
1. Jumlah siswa laki-laki dan perempuan seimbang.  
2. Sebagian besar siswa berasal dari etnis Kaukasia.  
3. Pendidikan orang tua didominasi oleh lulusan perguruan tinggi.  
4. Sebagian besar siswa tidak mengikuti bimbingan belajar.  
5. Dukungan orang tua cenderung berada pada level sedang dan tinggi.  
6. Partisipasi siswa dalam kegiatan ekstrakurikuler, olahraga, musik, dan sukarelawan terbilang rendah.

Langkah berikutnya adalah visualisasi data numerikal.  
Interpretasi grafik numerikal:  
1. Usia siswa mayoritas berada di rentang 15–17 tahun, tanpa outlier.  
2. Waktu belajar per minggu sebagian besar antara 5–14 jam.  
3. Jumlah ketidakhadiran siswa rata-rata antara 6–23 hari.  
4. Nilai prestasi (GPA) siswa berkisar antara 1,2–2,7, dan tidak terdapat outlier.

Selanjutnya dilakukan analisis distribusi jumlah pada variabel target `GradeClass` untuk mengetahui persebaran siswa.  
Interpretasi:  
Kategori prestasi terbanyak adalah Grade F (terendah) dengan jumlah 1.211 siswa, sedangkan Grade A (tertinggi) memiliki jumlah paling sedikit, yaitu 107 siswa.

Visualisasi persebaran GradeClass juga dilakukan.  
Interpretasi:  
1. Grade F menempati porsi terbesar (50,6%).  
2. Grade A menempati porsi terkecil (4,5%).  
3. Sisanya tersebar di Grade B (11,2%), C (16,3%), dan D (17,3%).

Langkah terakhir dalam univariate EDA adalah membuat histogram dari variabel numerikal.  
Interpretasi:  
Distribusi usia, waktu belajar, ketidakhadiran, dan GPA cenderung menyerupai distribusi normal.

### Analisis Multivariat (Multivariate EDA)  
Analisis ini dilakukan untuk melihat hubungan antara dua variabel (bivariate).  

#### 1. Hubungan antara StudyTimeWeekly dan GPA  
Interpretasi:  
Siswa dengan waktu belajar lebih banyak cenderung memiliki GPA yang lebih tinggi.

#### 2. Hubungan antara Absences dan GPA  
Interpretasi:  
Tingkat ketidakhadiran siswa berkorelasi negatif terhadap prestasi, semakin banyak absen, semakin rendah GPA.

#### 3. Hubungan antara Tutoring dan GradeClass  
Interpretasi:  
Sebagian besar siswa yang tidak mengikuti bimbingan belajar mendapatkan nilai yang rendah (Grade F).

#### 4. Hubungan antara Gender dan GradeClass  
Interpretasi:  
Siswa laki-laki cenderung memiliki prestasi lebih tinggi dibandingkan perempuan.

#### 5. Hubungan antara kegiatan non-akademik (Extracurricular, Sports, Music, Volunteering) dan GPA  
Interpretasi:  
Minimnya partisipasi siswa dalam kegiatan non-akademik berpengaruh pada menurunnya nilai GPA.

#### 6. Hubungan antara ParentalSupport dan GradeClass  
Interpretasi:  
Dukungan orang tua yang tinggi sangat berkontribusi terhadap peningkatan prestasi siswa.

#### 7. Korelasi Antar Variabel (Heatmap)  
Interpretasi:  
- GPA memiliki korelasi negatif yang cukup kuat terhadap ketidakhadiran.  
- GPA memiliki korelasi positif lemah terhadap waktu belajar per minggu.

#### 8. Plot Scatter Korelasi Positif dan Negatif  
Interpretasi:  
- Terdapat korelasi negatif yang kuat antara GPA dan ketidakhadiran.  
- Terdapat korelasi positif lemah antara GPA dan waktu belajar.

## Persiapan Data (Data Preparation)  
Tahapan ini bertujuan mengubah data ke dalam format yang sesuai untuk proses pemodelan. Langkah-langkah yang dilakukan meliputi:

1. Menghapus kolom yang tidak diperlukan.  
2. Melakukan encoding pada fitur kategorikal.  
3. Membagi dataset menggunakan fungsi `train_test_split` dari pustaka `sklearn`.

### Menghapus Kolom yang Tidak Digunakan  
Beberapa kolom yang tidak digunakan dalam analisis, seperti `StudentID`, `Ethnicity`, dan `ParentalEducation`, dihapus menggunakan fungsi `drop()`. Setelah penghapusan, dataset terdiri dari 2 kolom bertipe float64, 2 kolom bertipe int64, dan 8 kolom bertipe string (objek).

### Encoding Fitur Kategori
Pada bagian ini, karena fitur kategori dalam dataset sudah diubah menjadi objek (string) pada tahap eksplorasi data, kita akan mengubahnya menjadi format numerik agar dapat diproses oleh algoritma machine learning. Encoding fitur kategori dilakukan dalam tiga cara, yaitu:

1. *Label Encoding*: Mengonversi nilai kategori menjadi angka integer (`0` dan `1`).
2. *One Hot Encoding*: Mengubah setiap kategori menjadi kolom biner terpisah untuk data yang tidak terurut.
3. *Ordinal Encoding*: Memberikan nilai integer berdasarkan urutan kategori.

Hasil setelah preprocessing data dapat dilihat dalam gambar berikut.

### Train-Test-Split
Langkah pertama adalah mengonversi data objek ke data numerik dengan memanggil fungsi konversi objek ke numerik. Selanjutnya, karena variabel target kita adalah GradeClass, kolom tersebut akan dihapus dari data dan dipisahkan menjadi variabel baru. Pembagian data dilakukan dengan rasio 80% untuk training dan 20% untuk testing menggunakan `train_test_split` dari pustaka sklearn.

## Model Development

Pada bagian ini, akan dibangun empat model machine learning untuk menguji seberapa baik akurasi model dalam memprediksi prestasi siswa. Keempat model ini masing-masing menggunakan algoritma yang berbeda, dengan tujuan untuk memperoleh hasil yang optimal bagi prediksi prestasi akademik siswa.

### 1. Model Development dengan Random Forest

**Random Forest** adalah algoritma pembelajaran ensemble yang sangat populer dalam tugas klasifikasi dan regresi. Algoritma ini bekerja dengan membuat sejumlah pohon keputusan selama pelatihan, yang kemudian digabungkan untuk menghasilkan prediksi yang lebih akurat. Proses penggabungan ini bisa dilakukan melalui voting (untuk klasifikasi) atau rata-rata (untuk regresi), yang dapat membantu mengurangi overfitting.

Pada pemodelan ini, *Random Forest* diimplementasikan menggunakan `RandomForestClassifier` dari library `sklearn.ensemble`. Data training yang digunakan adalah `X_train` dan `y_train`, sedangkan `X_test` dan `y_test` digunakan untuk menguji model dengan data testing yang tidak ada pada data training. Beberapa parameter yang digunakan pada model ini adalah:

- **n_estimators**: Jumlah pohon keputusan (tree) yang akan dibangun dalam hutan.
- **criterion**: Fungsi untuk menentukan kualitas pembagian data, yaitu menggunakan `entropy`.
- **max_depth**: Kedalaman maksimum dari setiap pohon keputusan.
- **random_state**: Menentukan nilai acak yang digunakan untuk setiap iterasi guna memastikan reprodusibilitas model.

Pada proyek ini, parameter yang dipilih adalah:
- `n_estimators = 200`
- `criterion = "entropy"`
- `max_depth = 10`
- `random_state = 50`

Dengan konfigurasi ini, Random Forest bertujuan untuk meminimalkan kesalahan prediksi dengan memanfaatkan banyak pohon keputusan, meningkatkan stabilitas dan akurasi.

### 2. Model Development dengan Extreme Gradient Boosting (XGBoost)

**Extreme Gradient Boosting (XGBoost)** merupakan salah satu algoritma boosting yang sangat kuat untuk tugas klasifikasi dan regresi. XGBoost dirancang untuk efisiensi, fleksibilitas, dan performa tinggi, serta sering digunakan dalam berbagai kompetisi machine learning karena kemampuan optimisasi dan generalisasi yang sangat baik.

Pada pemodelan ini, XGBoost diimplementasikan menggunakan `XGBClassifier` dari library `xgboost`, dengan menggunakan `X_train` dan `y_train` untuk melatih model dan `X_test` serta `y_test` untuk menguji model dengan data yang tidak digunakan dalam pelatihan. Parameter-parameter utama yang digunakan dalam model XGBoost meliputi:

- **max_depth**: Kedalaman maksimum dari pohon keputusan.
- **n_estimators**: Jumlah pohon keputusan yang akan dibangun selama proses boosting.
- **random_state**: Menentukan nilai acak untuk setiap iterasi guna memastikan hasil yang dapat direproduksi.
- **learning_rate**: Mengatur langkah-langkah yang diambil dalam proses pembelajaran untuk meminimalkan fungsi kerugian.
- **n_jobs**: Menentukan jumlah CPU thread yang digunakan untuk menjalankan algoritma XGBoost.

Pada proyek ini, parameter yang digunakan adalah:
- `max_depth = 6`
- `n_estimators = 125`
- `random_state = 30`
- `learning_rate = 0.01`
- `n_jobs = -1`

Dengan konfigurasi ini, XGBoost akan berusaha untuk memaksimalkan performa model dengan memperkecil kesalahan prediksi melalui teknik boosting yang efisien.

### 3. Model Development dengan Support Vector Machine (SVM)

**Support Vector Machine (SVM)** adalah algoritma yang sangat efektif untuk klasifikasi dan regresi, terutama dalam kasus data non-linear. SVM bekerja dengan mencari hyperplane optimal yang memisahkan data dalam ruang fitur yang lebih tinggi. Algoritma ini mendukung berbagai jenis kernel untuk menangani data non-linear yang tidak dapat dipisahkan dengan garis lurus.

Pada pemodelan ini, SVM diimplementasikan menggunakan `SVC` dari library `sklearn.svm`, dengan menggunakan `X_train` dan `y_train` untuk melatih model, serta `X_test` dan `y_test` untuk menguji model. Parameter utama yang digunakan pada model ini adalah:

- **kernel**: Tipe kernel yang digunakan untuk mentransformasikan input data ke ruang fitur yang lebih tinggi, dengan tujuan mencari hyperplane terbaik.
- **gamma**: Menentukan pengaruh setiap contoh training dalam model.
- **random_state**: Menetapkan nilai acak untuk setiap iterasi, yang berguna untuk memastikan bahwa hasil eksperimen dapat direproduksi.

Pada proyek ini, parameter yang digunakan adalah:
- `kernel = 'rbf'`
- `gamma = 'auto'`
- `random_state = 50`

Model SVM bertujuan untuk menemukan batas keputusan yang optimal dengan menggunakan kernel yang cocok untuk data non-linear, dengan harapan dapat menghasilkan prediksi yang lebih akurat dan stabil.

### 4. Model Development dengan Naive Bayes

**Naive Bayes** adalah algoritma klasifikasi probabilistik yang didasarkan pada Teorema Bayes. Algoritma ini bekerja dengan asumsi bahwa semua fitur dalam data adalah independen satu sama lain, meskipun dalam kenyataannya asumsi ini sering kali tidak sepenuhnya valid. Meskipun begitu, Naive Bayes sering kali memberikan hasil yang baik dalam kasus tertentu, terutama pada data yang sangat besar.

Pada pemodelan ini, Naive Bayes diimplementasikan menggunakan `GaussianNB` dari library `sklearn.naive_bayes`. Algoritma ini digunakan dengan data numerik, dan `X_train` serta `y_train` digunakan untuk melatih model, sementara `X_test` dan `y_test` digunakan untuk menguji model dengan data yang tidak digunakan dalam pelatihan. Salah satu parameter yang penting dalam model ini adalah:

- **var_smoothing**: Nilai kecil yang ditambahkan pada varians setiap fitur untuk mencegah masalah numerik ketika varians menjadi terlalu kecil. Nilai `1e-9` (atau `0.000000001`) digunakan untuk tujuan ini, yang dapat membantu mengatasi masalah yang mungkin timbul akibat pembagian dengan angka yang sangat kecil.

Pada proyek ini, Naive Bayes diatur untuk menggunakan `var_smoothing = 1e-9`.

Dengan model ini, diharapkan dapat menghasilkan prediksi yang efisien dan cukup akurat meskipun menggunakan asumsi independensi antar fitur yang disederhanakan.

## Evaluation

Pada proyek ini, penilaian model dilakukan menggunakan matriks confusion, akurasi, dan skor F1 sebagai metrik evaluasi untuk masing-masing model. Evaluasi ini bertujuan untuk mengetahui seberapa baik performa setiap model dalam memprediksi hasil berdasarkan data uji yang telah disiapkan. Sebelum membahas lebih lanjut tentang hasil evaluasi, berikut penjelasan mengenai cara mendapatkan akurasi, skor F1, dan bagaimana matriks confusion digunakan.

### Matriks Confusion, Akurasi, dan Skor F1

1. **Matriks Confusion**  
   Matriks confusion adalah alat yang digunakan untuk menilai performa model klasifikasi dengan menunjukkan jumlah prediksi yang benar dan salah untuk setiap kelas. Matriks ini membantu untuk lebih memahami bagaimana model melakukan kesalahan dalam klasifikasinya. Format dasar matriks confusion adalah sebagai berikut:

   \[
   \text{Confusion Matrix} =
   \begin{bmatrix}
   \text{TP} & \text{FP} \\
   \text{FN} & \text{TN}
   \end{bmatrix}
   \]

   - **True Positive (TP)**: Jumlah data yang diprediksi positif dengan benar.
   - **True Negative (TN)**: Jumlah data yang diprediksi negatif dengan benar.
   - **False Positive (FP)**: Jumlah data yang diprediksi positif namun seharusnya negatif.
   - **False Negative (FN)**: Jumlah data yang diprediksi negatif namun seharusnya positif.

2. **Akurasi**  
   Akurasi adalah persentase dari jumlah prediksi yang benar dibandingkan dengan total data yang diuji. Akurasi dihitung dengan rumus:

   \[
   \text{Akurasi} = \frac{TP + TN}{TP + TN + FP + FN}
   \]

3. **Skor F1**  
   Skor F1 adalah rata-rata harmonik dari precision dan recall. Skor F1 memberikan gambaran yang lebih baik tentang keseimbangan antara keduanya. Semakin tinggi skor F1, semakin baik performa model dalam memprediksi data.

   Rumusnya adalah:

   \[
   F1\ \text{Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
   \]

4. **Precision dan Recall**
   - **Precision**: Proporsi prediksi positif yang benar-benar benar. 
     \[
     \text{Precision} = \frac{TP}{TP + FP}
     \]
   - **Recall (Sensitivity)**: Proporsi data positif yang terdeteksi dengan benar oleh model.
     \[
     \text{Recall} = \frac{TP}{TP + FN}
     \]

### Penerapan Matriks Confusion, Akurasi, dan Skor F1

#### 1. Model Development dengan Random Forest

Pada model Random Forest, berikut adalah hasil matriks confusion, akurasi, dan skor F1:

![Random Forest Confusion Matrix](https://github.com/user-attachments/assets/2fefd179-4ab8-451b-bb4f-6ad86f5ee29d)

Dari hasil tersebut, diperoleh akurasi sebesar 92.69% dengan skor F1 mencapai 0.93. Dalam hal kesalahan klasifikasi, terdapat 8 data yang salah prediksi pada Grade A dan 14 data yang salah pada Grade F.

#### 2. Model Development dengan XGBoost

Berikut adalah hasil evaluasi untuk model XGBoost:

![XGBoost Confusion Matrix](https://github.com/user-attachments/assets/6a28e4c0-af31-4ff7-b177-cdbdf09e3ba7)

Dari gambar di atas, diperoleh akurasi 93.32% dan skor F1 sebesar 0.93. Terdapat 5 data yang salah prediksi pada Grade A dan 15 data pada Grade F.

#### 3. Model Development dengan SVM

Berikut adalah hasil evaluasi untuk model Support Vector Machine (SVM):

![SVM Confusion Matrix](https://github.com/user-attachments/assets/9958e5a6-ea7f-4618-9e03-0e5404e45e22)

Pada model SVM, akurasi yang diperoleh adalah 78.08% dengan skor F1 sebesar 0.77. Terlihat terdapat 16 data yang salah prediksi pada Grade A dan 28 data pada Grade F.

#### 4. Model Development dengan Naive Bayes

Berikut adalah hasil evaluasi untuk model Naive Bayes:

![Naive Bayes Confusion Matrix](https://github.com/user-attachments/assets/bd837e95-d081-46cd-a221-17f51ce7af33)

Model Naive Bayes memperoleh akurasi sebesar 79.33% dengan skor F1 sebesar 0.79. Kesalahan klasifikasi terjadi pada 19 data Grade A dan 13 data Grade F.

### Hasil Evaluasi

Untuk memberikan gambaran yang lebih jelas tentang perbandingan performa masing-masing model, berikut ini adalah bar plot yang menunjukkan akurasi setiap model yang diuji.

![Akurasi Per Model](https://github.com/user-attachments/assets/c0584047-c778-4d72-a451-b48e44764df5)

Berdasarkan hasil evaluasi, model **XGBoost** terbukti menjadi model terbaik dengan akurasi dan skor F1 tertinggi, serta jumlah kesalahan klasifikasi paling sedikit, terutama pada Grade A. Meskipun model **Random Forest** memberikan hasil yang hampir setara, XGBoost masih lebih unggul dalam hal keakuratan. **SVM** menunjukkan performa yang lebih rendah dibandingkan kedua model tersebut, dan **Naive Bayes** memiliki akurasi terendah meskipun lebih cepat dalam proses pelatihan.

## Kesimpulan

Berdasarkan hasil analisis dan evaluasi model, dapat disimpulkan bahwa:
1. Terdapat hubungan positif antara durasi belajar mingguan yang lebih tinggi dengan performa akademik, meskipun peningkatannya tidak terlalu signifikan. Oleh karena itu, disarankan agar waktu belajar siswa berada di atas 20 jam per minggu.
2. Absensi yang tinggi berhubungan dengan penurunan prestasi akademik, yang menunjukkan pentingnya intervensi bagi siswa dengan tingkat absensi tinggi.
3. Siswa yang mengikuti bimbingan belajar cenderung memiliki GPA yang lebih tinggi dibandingkan mereka yang tidak mengikuti bimbingan belajar.
4. Perbedaan jenis kelamin tidak menunjukkan perbedaan signifikan dalam prestasi akademik.
5. Aktivitas ekstrakurikuler, olahraga, dan musik memiliki pengaruh terhadap nilai GPA siswa, sementara kegiatan sukarela tidak menunjukkan dampak signifikan.
6. Dukungan orang tua terbukti berperan penting dalam peningkatan prestasi akademik siswa.
7. Hasil **Exploratory Data Analysis (EDA)** menunjukkan bahwa faktor internal seperti durasi belajar dan absensi, serta faktor eksternal seperti dukungan orang tua dan partisipasi dalam kegiatan non-akademik, mempengaruhi performa akademik siswa.
8. Dari keempat model yang diuji (**XGBoost**, **SVM**, **Naive Bayes**, dan **Random Forest**), **XGBoost** terbukti menjadi model yang paling efektif untuk memprediksi performa siswa pada dataset ini, dengan akurasi dan skor F1 tertinggi.

## Referensi

1. Abdul Rahman. "Klasifikasi Performa Akademik Siswa Menggunakan Metode Decision Tree dan Naive Bayes", Vol. 13 No.1 (2023) 22-31, ISSN 2503-3247. SINTA Peringkat 4, diakses pada 28 November 2024.
2. Dicoding. Diakses pada 6 Juli 2024 dari https://www.dicoding.com/academies/319-machine-learning-terapan
3. Arif Fahrudin, Harco Leslie Hendric Spits Warnars. "Prediksi Performa Siswa Dengan Metode SAW", vol. 9, no. 1, 2020, P-ISSN 2089-1245, E-ISSN 2655-4925. KILAT, diakses pada 29 November 2024.

# Predictive_Analytics
