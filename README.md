# ml_ANZ-Customer-Transaction-Prediction
Predict the customer's annual revenue from ANZ's read transaction data
## Prediksi Transaksi Customer ANZ
Mengulas status Internet Banking di Australia saat ini. tentang Perbankan Internet dan memeriksa status saat ini sehubungan dengan bank di internet, layanan yang mereka sediakan, kesulitan yang dihadapi pelanggan, dan tindakan perbaikan yang diperlukan oleh bank, diakhiri dengan pengamatan bahwa bank-bank Australia tertinggal dari rekan-rekan mereka di AS, Eropa dan Jepang dalam menyediakan layanan perbankan di Internet dan meminta upaya serius dari bank, spesialis komputer, akademisi, dan lainnya untuk mempopulerkan area yang akan datang ini jika perbankan tidak untuk melihat abad pertengahan di Australia jika dibandingkan dengan standar Dunia[1]. Deregulasi progresif sistem keuangan Australia, yang diumumkan oleh Komite Penyelidikan Campbell (1981), telah menciptakan tekanan besar untuk penyesuaian struktural dalam industri perbankan Australia. Pergerakan ANZ baru-baru ini untuk mengambil alih Grindlays, bank internasional yang berbasis di Inggris, adalah upaya signifikan pertama dalam internasionalisasi menyeluruh oleh bank Australia. Pengambilalihan tersebut dianalisis dari model internasionalisasi bank yang dikembangkan oleh Fujita dan Ishigaki. Yang sangat menarik bagi ANZ adalah perwakilan bank asing yang dimiliki Grindlays. Hal ini tidak hanya melengkapi perwakilan internasional ANZ yang ada, tetapi juga memungkinkan pembeli Australia untuk mengatasi hambatan peraturan timbal balik yang mencegah bank Australia beroperasi di banyak lokasi lepas pantai. Dengan mengakuisisi Grindlays Bank, ANZ memperoleh keunggulan internasionalisasi atas saingannya di Australia. Potensi penuhnya hanya akan terlihat setelah paket aktivitas baru yang dibawa oleh Grindlays Bank telah terintegrasi dan dirasionalisasi sepenuhnya. ANZ telah memperoleh keunggulan internasionalisasi atas saingannya di Australia. Potensi penuhnya hanya akan terlihat setelah paket aktivitas baru yang dibawa oleh Grindlays Bank telah terintegrasi dan dirasionalisasi sepenuhnya[2]. Dataseet tugas ini didasarkan pada kumpulan data transaksi yang disintesis, berisi transaksi selama 3 bulan untuk 100 pelanggan hipotetis, berisi pembelian, transaksi berulang & transaksi gaji. Kumpulan data dirancang untuk mensimulasikan perilaku transaksi realistis yang diamati dalam data transaksi baca ANZ. Target utama dalam tugas ini adalah:  Menghasilkan variabel target untuk masalah & membuat model yang dapat memprediksi pendapatan tahunan pelanggan.


## Business Understanding
### Problem Statements
* Bagaimana memprediksi pendapatan tahunan pelanggan dari data transaksi baca ANZ?
* Model evaluasi manakah yang paling akurat untuk memprediksi data transaksi dari ANZ?

### Goals
Menjelaskan tujuan dari pernyataan masalah:
* Menghasilkan variabel target untuk masalah & membuat model yang dapat memprediksi pendapatan tahunan pelanggan yang komprehensif dengan teknik EDA.
* Membuat beberapa rekomendasi model algoritma, kemudian mencoba menemukan fitur yang akan meningkatkan model menggunakan metodologi yang paling akurat


### Solution statements
* Menggunakan Exploratory data analysis atau sering disingkat EDA untuk proses investigasi awal pada data untuk menganalisis karakteristik, menemukan pola, anomali, dan memeriksa asumsi pada data. 
* Mengembangkan model machine learning dengan tiga algoritma. Kemudian, mengevaluasi performa masing-masing algoritma dan menentukan algoritma mana yang memberikan hasil prediksi terbaik. Ketiga algoritma yang akan kita gunakan, antara lain: *K-Nearest Neighbor*, *Random Forest*, *Boosting Algorithm*

## Data Understanding
Kumpulan data transaksi yang disintesis yang berisi transaksi selama 3 bulan untuk 100 pelanggan hipotetis. Kumpulan data ini dirancang untuk mensimulasikan perilaku transaksi realistis yang diamati dalam data transaksi nyata ANZ berisi pembelian, transaksi berulang, dan transaksi gaji.
sumber data ANZ https://www.kaggle.com/datasets/ashraf1997/anz-synthesised-transaction-dataset

Variabel-variabel pada ANZ dataset adalah sebagai berikut:
* merchant_code: identifikasi kode merchant
* merchant_id: pengidentifikasi pedagang (106e1272-44ab-4dcb-a438-dd98e0071e51 paling umum)
* merchant_latitude: lintang pedagang (paling umum -37,82)
* merchant_longitude: bujur merchant (paling umum 151,21)
* merchant_state: di mana ia berada (NSW adalah yang paling umum)
* merchant_suburb: pinggiran kota tertentu di mana ia berada (Melbourne paling umum)
* amount: jumlah transaksi
* movement: jenis transaksi, kredit atau debit (debit terkait dengan pengeluaran, kredit terkait dengan pembayaran gaji)
* status: status transaksi, mungkin beberapa transaksi belum disetujui (diposting)
* transaction_id: pengidentifikasi unik untuk setiap transaksi yang dilakukan
* txn_description: kategori pembayaran
* card_present_flag: kemungkinan menunjukkan apakah pembayaran dilakukan secara virtual atau fisik
* date: data kapan transaksi dibuat, tersibuk adalah 9/28/2018
* extraction: tanggal lain, mungkin waktu yang tepat disertakan (tetapi perlu diekstraksi), bukan hanya tanggalnya
* bpay_biller_code: bpay biasanya memiliki nilai unik
* long_lat: lokasi yang terkait dengan transaksi
* account: nomor rekening, dataset berisi total 100 rekening bank unik & transaksinya
* customer_id: 100 pengidentifikasi pelanggan unik
* first_name: Nama depan paling umum yang terkait dengan transaksi tersebut adalah Michael
* age: sebagian besar transaksi dilakukan oleh usia 26 tahun
* balance: saldo sebelum transaksi terjadi, atau mungkin setelahnya, yang pertama lebih masuk akal
* gender: jenis kelamin yang melakukan transaksi

Melakukan beberapa tahapan yang diperlukan untuk memahami data, EDA Menangani Missing Value,Univariate Analysis, Exploratory Data Analysis - Multivariate Analysis

* EDA Menangani Missing Value
![pi](https://user-images.githubusercontent.com/123156703/215402738-11befd36-54df-49d0-a0c8-2cb2ee2b4859.png)
Gambar 1. Value NAN pada dataseet

Dari hasil pd.read atau hasil fungsi describe() terdapat value nan yang pada dataseeet merchant_kode, bpay_biller_code, dan card_present_flag perlu di hapus dengan fungsi drop. kemudian menggunakan fungsi Outliers.

![outliers](https://user-images.githubusercontent.com/123156703/215409314-e032a12c-f095-4492-bbf4-7db3a85b1078.png)
Gambar 2. Boxplot dengan Value Balance

![outliers2](https://user-images.githubusercontent.com/123156703/215409317-178a5204-ef31-444c-a4f4-0fbbb31aca34.png)
Gambar 3. Boxplot dengan Value Age

![outliers3](https://user-images.githubusercontent.com/123156703/215409318-fa6a56c4-f51a-43cd-9b7b-08b053350b24.png)
Gambar 4. Boxplot dengan Value Amount

Outliers adalah sampel yang nilainya sangat jauh dari cakupan umum data utama adalah hasil pengamatan yang kemunculannya sangat jarang dan berbeda dari data hasil pengamatan lainnya. pada percobaan tersebut ternyata terdapat outliers pada data ANZ, kemudian, tindakan untuk mengatasi outliers dengan fungsi persamaan :

   |Batas atas = Q3 + 1.5 * IQR |
    | ------ |

   | Batas bawah = Q1 - 1.5 * IQR |
    | ------ |

* EDA Univariate Analysis
Selanjutnya, proses analisis data dengan teknik Univariate EDA, bagi fitur pada dataset menjadi dua bagian, yaitu numerical features dan categorical features. Lakukan analisis terhadap fitur kategori terlebih dahulu. 

![categoricalfeature_status](https://user-images.githubusercontent.com/123156703/215409307-0c087496-0aca-443b-8895-a2df3f44bd8d.png)
Gambar 5. Categorical Features

Berdasarkan kesimpulan deskripsi variabel, Terdapat 2 kategori pada fitur jumlah autorized lebih banyak dari posted. 
![univariatenumericalfeatures](https://user-images.githubusercontent.com/123156703/215409320-0e960c8a-7340-4809-89c4-edc9bd5d5941.png)
Gambar 6. Numerical Features

Sedangkan di kategori numrical features Peningkatan amount sebanding dengan penurunan jumlah sampel. Hal ini dapat kita lihat jelas dari histogram "amount" yang grafiknya mengalami penurunan seiring dengan semakin banyaknya jumlah sampel (sumbu x).



* EDA Multivariate Analysis
Multivariate EDA menunjukkan hubungan antara dua atau lebih variabel pada data. Multivariate EDA yang menunjukkan hubungan antara dua variabel biasa disebut sebagai bivariate EDA. Selanjutnya, melakukan analisis data pada fitur kategori dan numerik. 
Pada fitur 'status', memiliki jumlah perbedaan yang lumayan jauh yaitu memiliki selisih 17 amount
pada fitur 'amount' terbanyak berkisar antara 60-70 transaksi
pada fitur 'account' nilai tertinggi mencapai 70 transaksi
pada fitur 'currency' hanya terdapat satu variabel AUD yang memilki amount di total 29
pada fitur 'long_lat' jumlah total amount mencapai angka 70
pada fitur 'txn_description' jumlah total transaksi terbanyak pada PHONE BANK
pada fitur 'merchant_id' terjadi keseimbangan antara setiap transaksi dengan jumlah merchant
pada fitur 'first_nama' untuk jumlah transaksi stabil
pada fitur 'date' jumlah transaksi terbnyak yaitu di atas 40
Pada fitur 'gender' selisih jumlah transaksi gender F dan M tidak jauh beda
pada fitur 'merchant_suburb' memiliki jumlah amount yang stabil dengan yang naik dan turunnya
pada fitur 'merchant_state' jumlah transaksi penggunaan hampir sama
pada fitur 'transaction_id' memiliki jumlah variabel x, y yang stabil ketika ada keniakan dan ada penurunan
pada fitur 'transaction_id' jumlah amount tertinggi diatas 100
pada fitur 'country' hanya memilki satu variabel yaitu variabel dengan nama daerah Australia yang memeiliki jumlah total transaksi 28-29.
pada fitur 'customer_id' nilai transaksi tertinggi mencapai 70
pada fitur 'merchant_long_lat' memiliki jumlah transaksi yang stabil
pada fitur 'movement' hanya terdapat variabel debit yang hampir mencapai 30 amount

![corelasi](https://user-images.githubusercontent.com/123156703/215409311-1a11bad3-cb07-4c34-827d-1fad51d5e4bd.png)
Gambar 7. Korelasi Multivariate Analysis

Penjelasan mengenai hubungan korelasi antar fitur.  Koefisien korelasi berkisar antara -1 dan +1. Mengukur kekuatan hubungan antara dua variabel serta arahnya (positif atau negatif). Mengenai kekuatan hubungan antar variabel, semakin dekat nilainya ke 1 atau -1, korelasinya semakin kuat. Sedangkan, semakin dekat nilainya ke 0, korelasinya semakin lemah. Pada grafik korelasi, Jika kita amati, fitur ‘age’, ‘balance, ‘amount’ memiliki skor korelasi yang lumayan mendekati. Sedangkan korelasi yang sangat rendah hanya mencapai angka 0.06.

## Data Preparation
Data preparation merupakan tahapan penting dalam proses pengembangan model machine learning. Ini adalah tahap di mana kita melakukan proses transformasi pada data sehingga menjadi bentuk yang cocok untuk proses pemodelan. pada bagian ini terdapata 4 tahap persiapan data akan tetapi kita akan menggunakan 3 saja sesaui dengan kebutuhan yaitu:Encoding fitur kategori, Pembagian dataset dengan fungsi train_test_split dari library sklearn, Standarisasi.

* Encoding fitur kategori.
Untuk melakukan proses encoding fitur kategori, salah satu teknik yang umum dilakukan adalah teknik one-hot-encoding. Library scikit-learn menyediakan fungsi ini untuk mendapatkan fitur baru yang sesuai sehingga dapat mewakili variabel kategori.Proses ini berfungsi untuk merubah variabel kategori menjadi variabel numerik.
* Pembagian dataset dengan fungsi train_test_split dari library sklearn.
Membagi dataset menjadi data latih (train) dan data uji (test) merupakan hal yang harus kita lakukan sebelum membuat model. Kita perlu mempertahankan sebagian data yang ada untuk menguji seberapa baik generalisasi model terhadap data baru. Ketahuilah bahwa setiap transformasi yang kita lakukan pada data juga merupakan bagian dari model. Karena data uji (test set) berperan sebagai data baru, kita perlu melakukan semua proses transformasi dalam data latih. Inilah alasan mengapa langkah awal adalah membagi dataset sebelum melakukan transformasi apa pun. Tujuannya adalah agar kita tidak mengotori data uji dengan informasi yang kita dapat dari data latih. Proporsi pembagian data latih dan uji biasanya adalah 90:10.
 Hasilnya:
Tabel 1. Proprosi pembagian datalatih dan data uji
    |Total # of sample in whole dataset: 9054 |
    | ------ |
    |Total # of sample in train dataset: 8148 |
    | Total # of sample in test dataset: 906 |
    
    
* Standarisasi.
Standardisasi adalah teknik transformasi yang paling umum digunakan dalam tahap persiapan pemodelan. Untuk fitur numerik, kita tidak akan melakukan transformasi dengan one-hot-encoding seperti pada fitur kategori. Kita akan menggunakan teknik StandarScaler dari library Scikitlearn, 
StandardScaler melakukan proses standarisasi fitur dengan mengurangkan mean (nilai rata-rata) kemudian membaginya dengan standar deviasi untuk menggeser distribusi.  StandardScaler menghasilkan distribusi dengan standar deviasi sama dengan 1 dan mean sama dengan 0. Sekitar 68% dari nilai akan berada di antara -1 dan 1.

## Modeling
Pada tahap ini, kita akan mengembangkan model machine learning dengan tiga algoritma. Kemudian, kita akan mengevaluasi performa masing-masing algoritma dan menentukan algoritma mana yang memberikan hasil prediksi terbaik. Ketiga algoritma yang akan kita gunakan, antara lain:
* *K-Nearest Neighbor*
* *Random Forest*
* *Boosting Algorithm*

* K-Nearest Neighbor
KNN adalah algoritma yang relatif sederhana dibandingkan dengan algoritma lain. Algoritma KNN menggunakan ‘kesamaan fitur’ untuk memprediksi nilai dari setiap data yang baru. Dengan kata lain, setiap data baru diberi nilai berdasarkan seberapa mirip titik tersebut dalam set pelatihan. KNN bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k tetangga terdekat (dengan k adalah sebuah angka positif). Nah, itulah mengapa algoritma ini dinamakan *K-nearest neighbor* (sejumlah k tetangga terdekat). KNN bisa digunakan untuk kasus klasifikasi dan regresi. Meskipun algoritma KNN mudah dipahami dan digunakan, ia memiliki kekurangan jika dihadapkan pada jumlah fitur atau dimensi yang besar. Permasalahan ini sering disebut sebagai *curse of dimensionality* (kutukan dimensi). Pada dasarnya, permasalahan ini muncul ketika jumlah sampel meningkat secara eksponensial seiring dengan jumlah dimensi (fitur) pada data. Jadi, jika  menggunakan model KNN, pastikan data yang digunakan memiliki fitur yang relatif sedikit. Kita menggunakan k = 10 tetangga dan metric Euclidean untuk mengukur jarak antara titik. Pada tahap ini kita hanya melatih data training dan menyimpan data testing untuk tahap evaluasi yang akan dibahas di Modul Evaluasi Model.

* Random Forest
Random forest merupakan salah satu model machine learning yang termasuk ke dalam kategori ensemble (group) learning. Apa itu model ensemble? Sederhananya, ia merupakan model prediksi yang terdiri dari beberapa model dan bekerja secara bersama-sama. Ide dibalik model ensemble adalah sekelompok model yang bekerja bersama menyelesaikan masalah. Sehingga, tingkat keberhasilan akan lebih tinggi dibanding model yang bekerja sendirian. Pada model ensemble, setiap model harus membuat prediksi secara independen. Kemudian, prediksi dari setiap model ensemble ini digabungkan untuk membuat prediksi akhir. 
Berikut adalah parameter-parameter yang digunakan:
    a. n_estimator: jumlah trees (pohon) di forest. Di sini kita set n_estimator=50.
    b. max_depth: kedalaman atau panjang pohon. Ia merupakan ukuran seberapa banyak pohon dapat membelah (splitting) untuk membagi setiap node ke dalam jumlah pengamatan yang diinginkan  max_depth=16.
    c. random_state: digunakan untuk mengontrol random number generator yang digunakan random_state=55. 
    d. n_jobs: jumlah job (pekerjaan) yang digunakan secara paralel. Ia merupakan komponen untuk mengontrol thread atau proses yang berjalan secara paralel. n_jobs=-1 artinya semua proses berjalan secara paralel.

* Boosting Algorithm
Seperti namanya, boosting, algoritma ini bertujuan untuk meningkatkan performa atau akurasi prediksi. Caranya adalah dengan menggabungkan beberapa model sederhana dan dianggap lemah (*weak learners*) sehingga membentuk suatu model yang kuat (strong ensemble learner). Algoritma boosting muncul dari gagasan mengenai apakah algoritma yang sederhana seperti *linear regression* dan *decision tree* dapat dimodifikasi untuk dapat meningkatkan performa.
Berikut merupakan parameter-parameter yang digunakan pada potongan kode:
a. learning_rate: bobot yang diterapkan pada setiap regressor di masing-masing proses iterasi boosting, learning_rate=0.05.
b. random_state: digunakan untuk mengontrol random number generator yang digunakan random_state=55.

## Evaluation
Mengevaluasi model regresi sebenarnya relatif sederhana. Secara umum, hampir semua metrik adalah sama. Jika prediksi mendekati nilai sebenarnya, performanya baik. Sedangkan jika tidak, performanya buruk. Secara teknis, selisih antara nilai sebenarnya dan nilai prediksi disebut eror. Maka, semua metrik mengukur seberapa kecil nilai eror tersebut. Metrik yang akan kita gunakan pada prediksi ini adalah MSE atau *Mean Squared Error* yang menghitung jumlah selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi.
mari kita lihat hasil prediksi menggunakan beberapa harga dari data test.


Metrik yang akan kita gunakan pada prediksi ini adalah MSE atau Mean Squared Error yang menghitung jumlah selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi. MSE didefinisikan dalam persamaan berikut

![formula](https://user-images.githubusercontent.com/123156703/215319673-c374edc2-1c5f-486f-93fc-3ff127a2d6f9.jpeg)
Gambar 8. Formula untuk menghitung MSE

Keterangan:
N = jumlah dataset
yi = nilai sebenarnya
y_pred = nilai prediksi

Saat menghitung nilai Mean Squared Error pada data train dan test, kita membaginya dengan nilai 1e3. Hal ini bertujuan agar nilai mse berada dalam skala yang tidak terlalu besar.
Hasil evaluasi pada data latih dan data test adalah sebagai berikut:
Tabel 2. Hasil evaluasi data latih dan data test
 |  | train	 | test |   
| ------ | ------ | ------ |
|KNN | 0.260764 | 0.476869
| RF | 0.239543 | 0.352439
|Boosting | 0.354629 | 0.369491

Untuk mengujinya, buat prediksi menggunakan beberapa harga dari data test.
Tabel 3. hasil uji prediksi harga dari dataseet
  | y_true | 9.93	 |
| ------ | ------ |
|prediksi_KNN | 22.0 |
| prediksi_RF | 26.8 |
|prediksi_Boosting | 29.2 |

## Kesimpulan
![evaluasimodel](https://user-images.githubusercontent.com/123156703/215409312-7d4e8582-4dc8-4b08-8418-a75edd84c608.png)
Gambar 9. Prediksi pada masing-masing model

Terlihat bahwa prediksi dengan K-Nearest Neighbor (KNN) memberikan hasil yang paling mendekati.
Untuk melakukan peningkatan performa, lakukanlah hal yang sama (pengaturan parameter) pada semua algoritma yang digunakan. Selain itu, melakukan optimasi parameter dengan menerapkan teknik Grid Search.

## Refrensi
[1]	M. Sathye, “Internet Banking in Australia,” SSRN Electron. J., pp. 1996–1998, 2005, doi: 10.2139/ssrn.38222.
[2]	J. Hirst and M. J. Taylor, “The internationalisation of Australian banking: further moves by the ANZ,” Aust. Geogr., vol. 16, no. 4, pp. 291–295, 1985, doi: 10.1080/00049188508702886.




