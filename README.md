# 🏠 UTS Ngakan — Prediksi Harga Properti dengan Polynomial Regression

Proyek ini dibuat untuk memenuhi Ujian Tengah Semester (UTS) mata kuliah **Machine Learning**.
Model yang digunakan adalah **Polynomial Regression** dengan regularisasi **Ridge dan Lasso**, untuk memprediksi **harga properti** berdasarkan fitur-fitur seperti luas tanah, luas bangunan, jumlah kamar, umur bangunan, dan jarak ke pusat kota.

---

## ⚙️ 1. Cara Install Dependencies

Pastikan kamu sudah menginstal **Python versi 3.8+** di komputer.

### 🪟 **Untuk Windows**

Buka terminal (CMD atau PowerShell), lalu jalankan:

```bash
py -m ensurepip --upgrade
py -m pip install --upgrade pip
py -m pip install numpy pandas matplotlib seaborn scikit-learn joblib
```

## ▶️ 2. Cara Menjalankan Kode

### **Langkah 1 – Buka Project di VS Code**

1. Jalankan **Visual Studio Code (VS Code)**.
2. Klik menu **File → Open Folder...** dan pilih folder tempat kamu menyimpan file ini.
3. Pastikan file utama `uts_ngakan_prediksi_polynomial.py` terlihat di panel sebelah kiri (Explorer).

### **Langkah 2 – Jalankan Program**

Ada dua cara untuk menjalankan:

#### 🅐 Melalui Tombol “Run”

1. Buka file `uts_ngakan_prediksi_polynomial.py`.
2. Klik tombol **▶ Run Python File** di pojok kanan atas.
3. Tunggu proses berjalan hingga selesai.

#### 🅑 Melalui Terminal

1. Tekan `Ctrl + ~` untuk membuka terminal di VS Code.
2. Jalankan perintah:

   ```bash
   py uts_ngakan_prediksi_polynomial.py
   ```

   atau (tergantung versi Python)

   ```bash
   python uts_ngakan_prediksi_polynomial.py
   ```

### **Langkah 3 – Melihat Hasil**

- Hasil statistik dan metrik model akan muncul di terminal.
- Beberapa visualisasi (histogram, scatter, heatmap, grafik evaluasi) akan muncul dalam jendela terpisah.
- File model terbaik akan tersimpan otomatis:

  ```
  scaler_price.pkl
  best_model_bundle.pkl
  ```

---

## 3. Struktur Project

```
📂 UTS_Prediksi_Harga_Properti/
│
├── uts_ngakan_prediksi_polynomial.py     # File utama berisi seluruh pipeline proyek
├── scaler_price.pkl                      # Hasil penyimpanan StandardScaler
├── best_model_bundle.pkl                 # Hasil penyimpanan model terbaik (Ridge/Lasso)
├── README.md                             # Dokumentasi proyek (file ini)
│
├── (Output)
│   ├── histogram_distribusi.png          # Visualisasi histogram fitur
│   ├── correlation_matrix.png            # Heatmap korelasi antar fitur
│   ├── train_test_r2_plot.png            # Grafik perbandingan R2 Train vs Test
│   └── ridge_lasso_analysis.png          # Grafik pengaruh alpha pada Ridge & Lasso
│
└── (Opsional jika disertakan)
    └── dataset_price.csv                 # Dataset sintetis hasil generate
```

---

## 4. Ringkasan Alur Program

1. **Data Preparation**

   - Membuat data sintetis >200 sampel dengan hubungan non-linear.
   - Menyimpan dataset dan scaler.

2. **Exploratory Data Analysis (EDA)**

   - Menampilkan statistik deskriptif, histogram, scatter plot, heatmap, dan outlier.

3. **Data Preprocessing**

   - Split 70:30 (train:test) dan scaling fitur dengan `StandardScaler`.

4. **Model Implementation**

   - Polynomial Regression untuk degree 1–5.
   - Evaluasi Linear, Ridge, dan Lasso Regression.

5. **Model Evaluation**

   - Menghitung R², RMSE, dan MAPE untuk train/test.
   - Visualisasi performa tiap derajat polinomial.

6. **Regularization Analysis**

   - Membandingkan pengaruh `alpha` pada Ridge dan Lasso menggunakan grafik log-scale.

7. **Model Selection & Prediction**

   - Cross-validation (5-fold) untuk memilih model terbaik.
   - Menyimpan model dan membuat fungsi prediksi properti baru dengan interval kepercayaan 95%.

8. **Kesimpulan**

   - Model terbaik: **Polynomial Regression (degree 2–3) dengan Ridge (alpha=1)**.

---

## 5. Penulis

**Ngakan Made Dwi Pramana Putra**
Program Studi: Informatika
Proyek: _UTS Machine Learning — Prediksi Harga Properti dengan Polynomial Regression_
