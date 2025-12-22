<div align="center">

# ☕ SMART ROASTERY: SISTEM KONTROL KUALITAS

### _Kerangka Perbandingan: Custom CNN vs. MobileNetV2 vs. ResNet50V2_

![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge&logo=github)
![Framework](https://img.shields.io/badge/Framework-TensorFlow_%7C_Keras-FF6F00?style=for-the-badge&logo=tensorflow)
![Interface](https://img.shields.io/badge/Interface-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)
![Manager](https://img.shields.io/badge/Dependency-PDM-purple?style=for-the-badge&logo=pdm)
![Dataset](<https://img.shields.io/badge/Dataset-CoffeeBean_USK_(Kaggle)-20BEFF?style=for-the-badge&logo=kaggle>)

**[Metodologi](#-metodologi)** • **[Benchmark](#-hasil-perbandingan)** • **[Penilaian](#-penilaian-laboratorium-umm)** • **[Instalasi](#-instalasi--setup-pdm)**

</div>

## 📖 Deskripsi Proyek

**Smart Roastery** adalah kerangka kerja visi komputer yang dirancang untuk mengotomatisasi proses kontrol kualitas pada produksi kopi. Dalam ranah pertanian presisi dan teknologi pangan, kemampuan untuk mengklasifikasikan kualitas biji dengan cepat sangat penting untuk menjaga standar produk dan meningkatkan nilai ekspor.

Proyek ini menerapkan pendekatan multi-arsitektur untuk membandingkan performa **Custom Convolutional Neural Network (CNN)** yang dibangun dari nol dengan arsitektur Transfer Learning mutakhir yaitu **MobileNetV2** dan **ResNet50V2**. Studi ini berfokus pada pencarian kompromi terbaik antara akurasi klasifikasi, stabilitas model, dan latensi inferensi untuk penerapan pada dashboard roastery.

---

## 📘 Dataset: Coffee Bean USK (4 Kelas)

Proyek ini menggunakan **Coffee Bean Dataset (USK)** yang bersumber dari Kaggle. Dataset ini berisi foto makro berkualitas tinggi dari biji kopi yang dikategorikan berdasarkan cacat fisik dan mutasi genetik.

- **Sumber Dataset:** [Kaggle - Coffee Bean Dataset by Bergling Murphy](https://www.kaggle.com/datasets/berglingmurphy/coffeebean-usk)
- **Jumlah Kelas:** 4 Kategori

| **Kategori**  | **Karakteristik Visual**                                  | **Tingkat Kualitas** |
| :------------ | :-------------------------------------------------------- | :------------------- |
| **Premium**   | Biji utuh, bentuk sempurna, warna seragam, tanpa retakan. | ⭐⭐⭐⭐⭐ (Tinggi)  |
| **Peaberry**  | Biji tunggal, bentuk bulat atau oval, padat.              | ⭐⭐⭐⭐ (Spesialti) |
| **Longberry** | Bentuk memanjang, tubuh ramping, faktor bentuk khas.      | ⭐⭐⭐⭐ (Spesialti) |
| **Defect**    | Biji pecah, terkelupas, hitam, jamur, atau berongga.      | ⭐ (Reject)          |

### 📊 Alur Pra-pemrosesan

- **Pengubahan Ukuran:** $128 \times 128$ piksel (RGB).
- **Normalisasi:** Skala piksel `1./255`.
- **Augmentasi:** Rotasi, shear, zoom, dan flip horizontal. Diterapkan pada data latih.

---

## 🛠 Metodologi

Proyek ini mengevaluasi tiga pendekatan arsitektur yang berbeda.

### 1. Custom CNN (Baseline)

Model _Convolutional Neural Network_ sederhana yang dibangun dari nol.

- **Struktur:** 3 blok konvolusi dan max pooling, diikuti flatten dan dense layer.
- **Tujuan:** Sebagai tolok ukur performa model sederhana.

### 2. MobileNetV2 (Transfer Learning)

Model pra-latih yang ringan dan efisien.

- **Karakteristik:** Menggunakan _depthwise separable convolutions_.
- **Penggunaan:** _Feature extractor_ dengan bobot ImageNet dibekukan dan classifier kustom.
- **Keunggulan:** Inferensi cepat. Cocok untuk perangkat edge.

### 3. ResNet50V2 (Transfer Learning)

Model pra-latih yang dalam dengan _residual connections_.

- **Karakteristik:** Mengatasi masalah _vanishing gradient_ pada jaringan dalam.
- **Penggunaan:** _Feature extractor_ dan classifier kustom.
- **Keunggulan:** Mampu menangkap fitur visual yang sangat detail dan kompleks.

1. **Ingesti Data:** Pengumpulan citra biji kopi dari Dataset USK.
2. **Perancangan Arsitektur:**

   - **Custom CNN:** Jaringan konvolusi 3 blok sebagai baseline.
   - **MobileNetV2:** Dioptimalkan untuk kecepatan dan latensi rendah. Siap untuk edge.
   - **ResNet50V2:** Dirancang untuk kedalaman ekstraksi fitur dengan residual connection.

3. **Deployment:** Integrasi ke dashboard bertema “Coffee Shop” menggunakan Streamlit dengan **overlay HUD OpenCV**.

---

## 📊 Hasil Perbandingan

Data berikut diambil langsung dari log riwayat pelatihan selama 50 epoch.

### 1. Ringkasan Performa

| Arsitektur            | Akurasi Val Puncak | Akurasi Latih Akhir | Analisis Stabilitas                                                                  |
| --------------------- | ------------------ | ------------------- | ------------------------------------------------------------------------------------ |
| **MobileNetV2 (TL)**  | **80,56%**         | 84,58%              | **Paling Stabil.** Keseimbangan terbaik antara pembelajaran dan generalisasi. Ideal. |
| **Custom CNN (Base)** | 79,25%             | 88,10%              | **Cepat Belajar.** Sedikit overfitting dibanding MobileNet.                          |
| **ResNet50V2 (TL)**   | 77,38%             | 81,62%              | **Kurang Optimal.** Terlihat fluktuasi pada ukuran dataset ini.                      |

> **🧪 Insight:**
>
> - **MobileNetV2** adalah pilihan optimal untuk kasus ini. Akurasi validasi tertinggi.
> - **Custom CNN** tampil mengejutkan. Mengungguli ResNet50V2 yang lebih berat. Model sederhana bisa unggul pada dataset kecil.

---

## 💻 Antarmuka dan Deployment

Sistem ini menampilkan **Dashboard Kontrol Roastery** yang hangat dan jelas.

| Dashboard Analitik                                                            | Inspeksi Langsung (HUD)                                                            |
| ----------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| _Menyajikan perbandingan multidimensi melalui radar chart dan tren historis._ | _Klasifikasi biji secara real-time dengan unggah gambar dan overlay bounding box._ |

---

## 🚀 Instalasi dan Setup (PDM)

Proyek ini menggunakan **PDM** untuk manajemen dependensi modern. Ikuti langkah berikut.

### 1. Kloning Repositori

```bash
git clone https://github.com/haidarfaizul/Classifiaction-Quality-Coffee-beans
```

### 2. Instal PDM dan Dependensi

```bash
pip install pdm
pdm init
pdm install
```

### 3. Pulihkan File Model (Penting ⚠️)

```text
/Classifiaction-Quality-Coffee-beans
├── app.py
├── pyproject.toml
└── src/
    ├── Model/
    │   ├── model_custom_cnn.keras
    │   ├── model_mobilenetv2.keras
    │   └── model_resnet50v2.keras
    └── History/
        ├── history_custom_cnn.csv
        ├── history_mobilenetv2.csv
        └── history_resnet50v2.csv
```

### 4. Jalankan Aplikasi

```bash
pdm run streamlit run app.py
```

Akses dashboard di `http://localhost:8501`.

---

## 📦 Dependensi

- `python` >= 3.10
- `streamlit`
- `tensorflow`
- `pandas` dan `numpy`
- `plotly`
- `opencv-python-headless`
- `Pillow`
