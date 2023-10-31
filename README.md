# Predict Customer Personality to Boost Marketing Campaign by Using Machine Learning
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

Dari data descriptive di atas, dapat di analisa bahwa:
- Rata-rata pelanggan telah menjadi member selama 10 tahun
- Rata-rata pelanggan hanya memiliki satu anak
- Rata-rata pelanggan melakukan 13-14 transaksi
- Rata-rata pelanggan berbelanja mengeluarkan biaya Rp 405.127
- Mayoritas pelanggan tidak menerima campaign
- Konversi kunjungan web dengan pembelian mayoritas pelanggan hanya sebesar 3.29%

### Conversion Rate by Income and Spending
Pada tahap ini akan dilakukan analisis mengenai hubungan Conversion Rate dengan income, total spending, dan usia pelanggan. <br>
<p align="center">
    <kbd> <img src="https://github.com/arllarima/Customer_Personality_Analysis_to_Optimize_Marketing_Campaign/assets/130117653/c8c401c6-fc5c-453f-a9fe-8d58c59328c7" width=700px> </kbd> <br>
    Gambar 2 - Plot Korelasi Conversion Rate (CVR) dengan Income dan Total Spending
</p>

Dari visualisasi data di atas, dapat kita analisa bahwa:
- Terlihat adanya korelasi positif antara Conversion Rate khususnya dengan Income dan Total Spending.
- Semakin tinggi Pendapatan dan Total Spending pelanggan, maka Conversion Rate juga semakin tinggi.
- Income dan Total Spending menunjukkan kapasitas keuangan pelanggan, pelanggan dengan kapasitas keuangan yang lebih tinggi memiliki Conversion Rate yang lebih tinggi.

### Conversion Rate by Age
<p align="center">
    <kbd> <img src="https://github.com/arllarima/Customer_Personality_Analysis_to_Optimize_Marketing_Campaign/assets/130117653/080aaac6-0477-48ee-b5f8-d165e0732e86" width=400px> </kbd> <br>
     Gambar 3 - Plot Korelasi Conversion Rate (CVR) dengan Age
</p>

Dari visualisasi data di atas, dapat kita analisa bahwa:
- Usia tidak menunjukkan korelasi yang tinggi dengan Conversion Rate.
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
- Rata-rata melakukan 18 transaksi dengan pengeluaran bulanan sekitar Rp 425.867.
- Pendapatan rata-rata pada segmen ini tergolong cukup tinggi, yakni sekitar Rp 52.857.438 per tahun.
- Tingkat konversi pada segmen ini adalah sedang, sekitar 3%.
- Mayoritas berada di umur 50 th - 60 th an.

**Cluster 1: Very Low Spender Customer**<br>
- Rata-rata hanya melakukan 7 transaksi dengan pengeluaran bulanan sekitar Rp 65.308.
- Pendapatan rata-rata pada segmen ini tergolong terendah, yakni sekitar Rp 33.063.385 per tahun.
- Tingkat konversi pada segmen ini adalah yang terendah, hanya sekitar 1%.
- Mayoritas berada di umur 40 th - 50 th an.

**Cluster 2: Mid Spender Customer**<br>
- Rata-rata melakukan sekitar 20 transaksi dengan pengeluaran bulanan mencapai Rp 1.001.405.
- Pendapatan rata-rata pada segmen ini adalah yang tertinggi, yakni sekitar Rp 70.331.247 per tahun.
- Tingkat konversi pada segmen ini adalah yang tertinggi, sekitar 9%.
- Mayotitas berada di umur 40 th - 60 th an.

**Cluster 3: High Spender Customer**<br>
- Rata-rata melakukan sekitar 24 transaksi dengan pengeluaran bulanan mencapai Rp 1.112.685.
- Pendapatan rata-rata pada segmen ini tergolong cukup tinggi, yakni sekitar Rp 66.503.343 per tahun.
- Tingkat konversi pada segmen ini adalah cukup sedang, sekitar 5%.
- Mayoritas berada di umur 50 th - 60 th an.

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
- Tingkat kunjungan ke situs web perusahaan paling tinggi dibandingkan dengan cluster lain.
- Respons terhadap campaign yang disajikan masih rendah.
- Memiliki jumlah populasi terbanyak.
- Diperlukan strategi untuk meningkatkan keterlibatan dan respons terhadap campaign.

**Cluster Low Spender:** <br>
- Tingkat kunjungan web cukup tinggi.
- Sering berbelanja di platform website dan toko fisik.
- Aktif menggunakan kupon penawaran dan promo.
- Respons campaign masih dapat ditingkatkan. <br>

**Rekomendasi:**
- Customisasi campaign sesuai preferensi kebutuhan dan keinginan cluster. <br>

**Cluster Mid Spender:** <br>
- Mayoritas pelanggan jarang mengunjungi situs web perusahaan.
- Tetapi sering berbelanja di semua platform.
- Respons campaign tidak terlalu tinggi.
- Jarang menggunakan promo. <br>

**Rekomendasi:**
- Optimalkan saluran komunikasi lain seperti email, media sosial, atau platform online lainnya untuk efektif menjangkau kelompok ini. <br>

**Cluster High Spender:** <br>
- Paling responsif terhadap campaign.
- Sering berbelanja di semua platform.
- Tingkat kunjungan web cukup tinggi.
- Cukup aktif menggunakan promo. <br>

**Rekomendasi:**
- Tingkatkan interaksi dengan campaign yang lebih menarik dan relevan.

### Customer Analysis Based on Total Spending per Product
<p align="center">
    <kbd> <img src="https://github.com/arllarima/Customer_Personality_Analysis_to_Optimize_Marketing_Campaign/assets/130117653/a1ca0db9-ceb7-46a5-8718-d6d0cfe65620" width=600px> </kbd> <br>
    Gambar 11 - Distribusi Produk Tiap Cluster
</p>
<br>
<p align="center">
    <kbd> <img src="https://github.com/arllarima/Customer_Personality_Analysis_to_Optimize_Marketing_Campaign/assets/130117653/927054c1-6b19-49c2-8a53-cc25f85acd15" width=600px> </kbd> <br>
    Gambar 12 - Total Spending Product by Cluster
</p>

**Insights:** <br>
Pelanggan di semua cluster mengeluarkan uang paling banyak untuk produk Coke dan Meat, sedangkan produk Sweet dan Fruits paling sedikit. <br>
Berikut urutan produk favorit tiap cluster: <br>
- High Spender = coke, meat, gold, fish, fruits, sweet
- Mid Spender = coke, meat, fish, gold, fruits, sweet
- Low Spender = coke, meat, gold, fish, fruits, sweet
- Very Low Spender = coke, meat, gold, fish, fruits, sweet

## Business Recommendation
**Very Low Spender:**
- Populasi terbanyak, yaitu 50.24 %
- Memiliki total spending dan income terendah, namun tingkat kunjungan website tertinggi.
- Respons terhadap campaign yang disajikan masih rendah.
- Conversion rate terendah, hanya sekitar 1%.
- Mayoritas berada di umur 40 th - 50 th an.
- Urutan produk favorit: coke, meat, gold, fish, fruits, sweet

**Rekomendasi:**
- Pelanggan dari kelompok ini sering mengunjungi website, jadi perusahaan bisa menyesuaikan konten dan tawaran diskon khusus sesuai dengan minat mereka.
- Perusahaan dapat mengingatkan pelanggan di kelompok ini tentang produk atau layanan yang mereka lihat di website dengan iklan yang disesuaikan, sehingga mereka lebih mungkin untuk melanjutkan pembelian.
- Karena konversi rendah dan respons kurang baik, perusahaan perlu fokus pada konten yang memberikan informasi dan solusi yang bermanfaat kepada pelanggan, sehingga meningkatkan keterlibatan dan kepercayaan mereka.

**Low Spender:** <br>
- Populasi 24.08 %
- Tingkat kunjungan web cukup tinggi.
- Sering berbelanja di platform website dan toko fisik.
- Aktif menggunakan kupon penawaran dan promo.
- Respons terhadap campaign tidak terlalu tinggi.
- Conversion rate sedang, sekitar 3%.
- Mayoritas berada di umur 50 th - 60 th an.
- Urutan produk favorit: coke, meat, gold, fish, fruits, sweet

**Rekomendasi:**
- Customisasi campaign sesuai preferensi kebutuhan dan keinginan cluster.
- Karena cluster ini aktif menggunakan promo, kita dapat menawarkan cashback atau voucher diskon untuk transaksi berikutnya jika mereka mencapai batas pembelian tertentu.
- Bundling Produk: Buat bundel produk dengan harga khusus. Ini dapat mendorong mereka untuk membeli lebih banyak item dalam satu transaksi.

**Mid Spender:**
- Populasi 14.08 %
- Memiliki income tertinggi
- Tingkat kunjungan website terendah, namun sering berbelanja di semua platform (web, catalog, store)
- Jarang menggunakan promo.
- Respons terhadap campaign tidak terlalu tinggi.
- Conversion rate tertinggi, sekitar 9%.
- Mayotitas berada di umur 40 th - 60 th an
- Urutan produk favorit: coke, meat, fish, gold, fruits, sweet

**Rekomendasi:**
- Karena kelompok ini kurang aktif di website, perusahaan dapat menggunakan email, pesan teks, atau media sosial sebagai cara komunikasi alternatif untuk campaign. Ini membantu meningkatkan interaksi dan kesadaran pelanggan.
- Pastikan pengalaman pengguna yang baik di website dan saat berinteraksi dengan produk atau layanan perusahaan.
- Tawarkan mereka untuk bergabung dengan program loyalitas yang memberikan poin atau hadiah spesial untuk setiap transaksi.

**High Spender:**
- Populasi terendah, yaitu 11.58 %.
- Memiliki total spending tertinggi dan sering berbelanja di semua platform (web, catalog, store)
- Tingkat kunjungan web cukup tinggi
- Paling responsif terhadap campaign
- Cukup aktif menggunakan promo
- Conversion rate sedang, sekitar 5%.
- Mayoritas berada di umur 50 th - 60 th an.
- Urutan produk favorit: coke, meat, gold, fish, fruits, sweet

**Rekomendasi:**
- Tingkatkan interaksi dengan campaign yang lebih menarik dan relevan.
- Menawarkan diskon dan program loyalitas yang memberikan poin atau hadiah spesial untuk setiap transaksi, hal ini dapat menjaga minat pelanggan agar terus berbelanja.
- Tawarkan layanan pelanggan premium seperti pengiriman ekspres atau akses awal ke penawaran eksklusif.

