# Customer Personality Analysis for Optimize Marketing Campaign
**Dataset** : Provided by Rakamin Academy <br>
**Programming Language** : Python <br>
**Libraries** : Pandas, NumPy, Matplotlib, Seaborn

## Overview
Sebuah perusahaan dapat berkembang dengan pesat saat mengetahui perilaku customer personality nya, sehingga dapat memberikan layanan serta manfaat lebih baik kepada customers yang berpotensi menjadi loyal customers. <br>
Dengan mengolah data historical marketing campaign dapat menaikkan performa dan menyasar customers yang tepat agar dapat bertransaksi di platform perusahaan. <br>
Dari insight data tersebut fokus kita adalah membuat sebuah model prediksi kluster sehingga memudahkan perusahaan dalam membuat keputusan. <br>

## Data Cleaning
Dataset berisi data pembelian dari semua toko, data iklan yang diterima, dan data pelanggan toko. <br>
Dataset memiliki 2240 baris dan 30 fitur. <br>
Beberapa tahapan yang dilakukan pada data cleaning:
1. Mengatasi missing values pada kolom dengan cara mengisi missing values dengan nilai median.
2. Memfilter outlier pada data.

## Feature Engineering
Pada tahap feature engineering, dilakukan pembuatan feature baru berdasarkan feature yang sudah ada dengan tujuan untuk membuat analisis menjadi lebih insightful. Feature baru ini dapat mengungkap informasi tambahan atau menggabungkan beberapa fitur yang saling berhubungan untuk membentuk fitur yang lebih kuat. <br>

Tabel 1 — Feature Engineering
 **New Feature** | **Source** |
-----------------|--------------|
Membership Duration | 2023 - Dt_Customer
Age | 2023 - Year_Birth
Age_Categories | Age
Total_Children | Kidhome + Teenhome
Total_Transaction | NumDealsPurchases + NumWebPurchases + NumCatalogPurchases + NumStorePurchases
Total_Spending | MntCoke + MntFruits + MntMeatProducts + MntFishProducts + MntSweet
Total_Accepted_Campaign | AcceptedCmp1 + AcceptedCmp2 + AcceptedCmp3 + AcceptedCmp4 + AcceptedCmp5
CVR | Total_Transaction x NumWebVisitsMonth/100

## Exploratory Data Analysis
### Descriptive Analysis
<p align="center">
    <kbd> <img src="https://github.com/arllarima/Customer_Personality_Analysis_to_Optimize_Marketing_Campaign/assets/130117653/a7e6f098-a331-4f79-b0ea-7ef2fc80f072" width=900px> </kbd> <br>
    Gambar 1 — Descriptive Analysis
    </p>
Dari data descriptive, dapat kita analisa bahwa: <br>
- Rata-rata pelanggan telah menjadi member selama 10 tahun <br>
- Rata-rata pelanggan hanya memiliki satu anak <br>
- Rata-rata pelanggan melakukan 13-14 transaksi <br>
- Rata-rata pelanggan berbelanja mengeluarkan biaya Rp.405.127 <br>
- Mayoritas pelanggan tidak menerima campaign <br>
- Konversi kunjungan web dengan pembelian mayoritas pelanggan hanya sebesar 3.29% <br>

### Conversion Rate by Income and Spending
Pada tahap ini akan dilakukan analisis mengenai hubungan Conversion Rate dengan income, total spending, dan usia pelanggan. <br>
<p align="center">
    <kbd> <img src="https://github.com/arllarima/Customer_Personality_Analysis_to_Optimize_Marketing_Campaign/assets/130117653/c8c401c6-fc5c-453f-a9fe-8d58c59328c7" width=700px> </kbd> <br>
    Gambar 2 - Plot Korelasi Conversion Rate (CVR) dengan Income dan Total Spending
</p>
Dari visualisasi data diatas, dapat kita analisa bahwa: <br>
- Terlihat adanya korelasi positif antara Conversion Rate khususnya dengan Income dan Total Spending. <br>
- Semakin tinggi Pendapatan dan Total Spending pelanggan, maka Conversion Rate juga semakin tinggi. <br>
- Income dan Total Spending menunjukkan kapasitas keuangan pelanggan, pelanggan dengan kapasitas keuangan yang lebih tinggi memiliki Conversion Rate yang lebih tinggi. <br>

### Conversion Rate by Age
<p align="center">
    <kbd> <img src="https://github.com/arllarima/Customer_Personality_Analysis_to_Optimize_Marketing_Campaign/assets/130117653/080aaac6-0477-48ee-b5f8-d165e0732e86" width=400px> </kbd> <br>
     Gambar 3 - Plot Korelasi Conversion Rate (CVR) dengan Age
</p>
Dari visualisasi data diatas, dapat kita analisa bahwa: <br>
- Usia tidak menunjukkan korelasi yang tinggi dengan Conversion Rate. <br>
- Conversion Rate terdistribusi dengan baik pada setiap umur, hal ini menunjukkan bahwa umur tidak berpengaruh signifikan terhadap Conversion Rate pelanggan. <br>

## Data Modeling with K-Means Clustering
### Data Pre-Processing
Sebelum melakukan data modeling, terdapat beberapa tahap pre-processing data yang perlu dilakukan yaitu: <br>
1. Melakukan encoding pada fitur kategorikal agar dapat diolah oleh algoritma machine learning.
2. Melakukan standardisasi fitur untuk memastikan skala data seragam dan menghindari bias dalam model. <br>

### Data Modeling
Setelah pre-processing data selesai, kita menggunakan Principal Component Analysis (PCA) untuk mengurangi dimensi data dengan mempertahankan informasi yang signifikan.  Ini membantu meningkatkan kinerja model dan mengatasi masalah multicollinearity antara fitur. <br>
Selanjutnya, langkah penting dalam proses ini adalah menentukan jumlah cluster terbaik. Dalam analisis ini, Elbow Method digunakan untuk memilih jumlah cluster yang optimal. Berdasarkan hasil analisis, jumlah cluster terbaik yang ditemukan adalah 4. Karena ketika n_cluster = 4, skor inersianya tidak berubah secara signifikan. <br>
<p align="center">
    <kbd> <img src="https://github.com/arllarima/Customer_Personality_Analysis_to_Optimize_Marketing_Campaign/assets/130117653/cec8ca73-069e-4b1f-b5ff-2fc93856577f" width=600px> </kbd> <br>
    Gambar 4 - Elbow Method
</p>

### Clustering
Setelah menentukan jumlah cluster yang optimal, kita menggunakan algoritma K-means untuk mengelompokkan data berdasarkan fitur yang serupa. Ini membantu kita mengidentifikasi pola dalam data dan memahami setiap kelompok. <br>
<p align="center">
    <kbd> <img src="https://github.com/arllarima/Customer_Personality_Analysis_to_Optimize_Marketing_Campaign/assets/130117653/641114e5-24c3-4d8c-bff3-1d2776be6211" width=600px> </kbd> <br>
    Gambar 5 - Hasil Clustering dengan Algoritma K-Means
</p>

Hasil dari pemodelan ini menunjukkan bahwa cluster-cluster terbentuk dengan baik dan memisahkan data dengan jelas berdasarkan karakteristiknya. Ini menunjukkan bahwa algoritma clustering berfungsi dengan baik. <br>

## Customer Personality Analysis
Customer Personality Analysis bertujuan untuk memahami perbedaan, kesamaan dan menemukan karakteristik unik masing-masing cluster. Dengan pemahaman ini, perusahaan bisa mengambil tindakan yang lebih cocok dan strategi yang lebih spesifik untuk masing-masing kelompok.<br>

### Cluster Analytical Statistic
<p align="center">
    <kbd> <img src="https://github.com/arllarima/Customer_Personality_Analysis_to_Optimize_Marketing_Campaign/assets/130117653/9363acf4-264d-474d-bba6-b5b05fe028cf" width=600px> </kbd> <br>
    Gambar 6 - Hasil Rata-Rata Tiap Cluster
</p>
<br>
<p align="center">
    <kbd> <img src="https://github.com/arllarima/Customer_Personality_Analysis_to_Optimize_Marketing_Campaign/assets/130117653/2e83583f-cb1b-46c8-a2f9-a53466478e55" width=600px> </kbd> <br>
    Gambar 7 - Hasil Distribusi Nilai Tiap Cluster
</p>
Berdasarkan hasil analisis clustering, ditemukan karakteristik masing-masing segmen cluster sebagai berikut:<br>
<br>

**Cluster 0: Low Spender Customer**<br>
1. Rata-rata melakukan 18 transaksi dengan pengeluaran bulanan sekitar Rp 425.867.
2. Pendapatan rata-rata pada segmen ini tergolong cukup tinggi, yakni sekitar Rp 52.857.438 per tahun.
3. Tingkat konversi pada segmen ini adalah sedang, sekitar 3%.
4. Mayoritas berada di umur 50 th - 60 th an.

**Cluster 1: Very Low Spender Customer**<br>
1. Rata-rata hanya melakukan 7 transaksi dengan pengeluaran bulanan sekitar Rp 65.308.
2. Pendapatan rata-rata pada segmen ini tergolong terendah, yakni sekitar Rp 33.063.385 per tahun.
3. Tingkat konversi pada segmen ini adalah yang terendah, hanya sekitar 1%.
4. Mayoritas berada di umur 40 th - 50 th an.

**Cluster 2: Mid Spender Customer**<br>
1. Rata-rata melakukan sekitar 20 transaksi dengan pengeluaran bulanan mencapai Rp 1.001.405.
2. Pendapatan rata-rata pada segmen ini adalah yang tertinggi, yakni sekitar Rp 70.331.247 per tahun.
3. Tingkat konversi pada segmen ini adalah yang tertinggi, sekitar 9%.
4. Mayotitas berada di umur 40 th - 60 th an.

**Cluster 3: High Spender Customer**<br>
1. Rata-rata melakukan sekitar 24 transaksi dengan pengeluaran bulanan mencapai Rp 1.112.685.
2. Pendapatan rata-rata pada segmen ini tergolong cukup tinggi, yakni sekitar Rp 66.503.343 per tahun.
3. Tingkat konversi pada segmen ini adalah cukup sedang, sekitar 5%.
4. Mayoritas berada di umur 50 th - 60 th an.

### Percentage of Each Cluster
<p align="center">
    <kbd> <img src="https://github.com/arllarima/Customer_Personality_Analysis_to_Optimize_Marketing_Campaign/assets/130117653/5b40cfa0-575d-47f5-abaa-4b8a50f40603" width=600px> </kbd> <br>
    Gambar 8 - Persentase Populasi dari Tiap Cluster
</p>

### Customer Analysis on Web Visit, Purchasing History, and Accepted Campaign
<p align="center">
    <kbd> <img src="https://github.com/arllarima/Customer_Personality_Analysis_to_Optimize_Marketing_Campaign/assets/130117653/da7f54ce-40b2-43b0-947c-a724b87509ef" width=600px> </kbd> <br>
    Gambar 9 - Distribusi Cluster
</p>
<br>
<p align="center">
    <kbd> <img src="https://github.com/arllarima/Customer_Personality_Analysis_to_Optimize_Marketing_Campaign/assets/130117653/60d65266-83ae-42f9-a3e8-46cfaf63f821" width=600px> </kbd> <br>
    Gambar 10 - Persentase Accepted Campaign Tiap Cluster
</p>

**Insights:** <br>

**Cluster Very Low Spender:** <br>
1. Tingkat kunjungan ke situs web perusahaan paling tinggi dibandingkan dengan cluster lain.
2. Respons terhadap kampanye yang disajikan masih rendah.
3. Memiliki jumlah populasi terbanyak.
4. Diperlukan strategi untuk meningkatkan keterlibatan dan respons terhadap kampanye.

**Cluster Low Spender:** <br>
1. Tingkat kunjungan web cukup tinggi.
2. Sering berbelanja di platform website dan toko fisik.
3. Aktif menggunakan kupon penawaran dan promo.
4. Respons kampanye masih dapat ditingkatkan. <br>

**Rekomendasi:** Customisasi kampanye sesuai preferensi kebutuhan dan keinginan cluster.

**Cluster Mid Spender:** <br>
1. Mayoritas pelanggan jarang mengunjungi situs web perusahaan.
2. Tetapi sering berbelanja di semua platform.
3. Respons kampanye tidak terlalu tinggi.
4. Jarang menggunakan promo. <br>

**Rekomendasi:** Optimalkan saluran komunikasi lain seperti email, media sosial, atau platform online lainnya untuk efektif menjangkau kelompok ini.

**Cluster High Spender:** <br>
1. Paling responsif terhadap kampanye.
2. Sering berbelanja di semua platform.
3. Tingkat kunjungan web cukup tinggi.
4. Cukup aktif menggunakan promo. <br>

**Rekomendasi:** Tingkatkan interaksi dengan kampanye yang lebih menarik dan relevan.



