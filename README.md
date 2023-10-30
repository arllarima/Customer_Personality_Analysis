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
<br>

### Conversion Rate vs Income, Spending, and Age
Pada tahap ini akan dilakukan analisis mengenai hubungan Conversion Rate dengan income, total spending, dan usia pelanggan. <br>
<p align="center">
    <kbd> <img src="https://github.com/arllarima/Customer_Personality_Analysis_to_Optimize_Marketing_Campaign/assets/130117653/c8c401c6-fc5c-453f-a9fe-8d58c59328c7" width=700px> </kbd> <br>
    Gambar 2 - Plot Korelasi Conversion Rate (CVR) dengan Income dan Total Spending
</p>
Dari visualisasi data diatas, dapat kita analisa bahwa: <br>
- Terlihat adanya korelasi positif antara Conversion Rate khususnya dengan Income dan Total Spending. <br>
- Semakin tinggi Pendapatan dan Total Spending pelanggan, maka Conversion Rate juga semakin tinggi. <br>
- Income dan Total Spending menunjukkan kapasitas keuangan pelanggan, pelanggan dengan kapasitas keuangan yang lebih tinggi memiliki Conversion Rate yang lebih tinggi. <br>
<br>
<p align="center">
    <kbd> <img src="https://github.com/arllarima/Customer_Personality_Analysis_to_Optimize_Marketing_Campaign/assets/130117653/080aaac6-0477-48ee-b5f8-d165e0732e86" width=400px> </kbd> <br>
     Gambar 3 - Plot Korelasi Conversion Rate (CVR) dengan Age
</p>
Dari visualisasi data diatas, dapat kita analisa bahwa: <br>
Usia tidak menunjukkan korelasi yang tinggi denganConversion Rate. Conversion Rate terdistribusi dengan baik pada setiap umur, hal ini menunjukkan bahwa umur tidak berpengaruh signifikan terhadap Conversion Rate pelanggan. <br>



