# IPO BEI Juli 2026 — Underpricing Analysis & Return Prediction

Proyek analisis dan pemodelan prediktif untuk memprediksi return hari pertama (Return D1) dan return kumulatif 7 hari enam emiten yang melaksanakan IPO di Bursa Efek Indonesia pada Juli 2026: **RANS, PRDL, BACH, JECX, EMMI, dan JELI**.

## Tujuan

- Menganalisis karakteristik dan distribusi return IPO historis di BEI (2022–2026)
- Membangun model prediktif untuk mengklasifikasikan status Return D1 (ARA / Normal / ARB)
- Membangun model regresi untuk mengestimasi return kumulatif 7 hari
- Memvalidasi prediksi terhadap return aktual setelah masing-masing emiten listing

## Dataset

Sumber data: [e-IPO Realtime Data — Kaggle](https://www.kaggle.com/datasets/fahmirk/e-ipo-realtime-data)

Download file `e-IPO Data.csv` dan letakkan di direktori yang sama dengan notebook sebelum menjalankan analisis.

## Struktur Notebook

| File | Deskripsi |
|---|---|
| `ipo_bei_july2026_underpricing_prediction.ipynb` | EDA, preprocessing, feature engineering, pemodelan, dan validasi aktual (end-to-end) |

### Alur Analisis

**Tahap 1 — EDA & Preprocessing (Section 1–11)**
- Eksplorasi distribusi return, sektor, listing board, dan underwriter
- Deteksi dan penanganan outlier dengan batas ARB flat 15% (aturan BEI April 2025)
- Feature engineering: posisi harga vs rentang book building, log proceed, encoding kategorikal

**Tahap 2 — Pemodelan & Validasi (Section 12–17)**
- Pipeline preprocessing bebas data leakage: train (80%) / validation (20%) / test (6 emiten target)
- Perbandingan model klasifikasi: Logistic Regression vs Random Forest vs Gradient Boosting
- Perbandingan model regresi: Linear Regression vs Random Forest Regressor
- Model terpilih: **Gradient Boosting** (klasifikasi) dan **Random Forest** (regresi)
- Validasi aktual terhadap return yang terjadi di pasar

## Hasil Validasi

| Emiten | Harga IPO | Return D1 Aktual | Prediksi | Akurat |
|---|---|---|---|---|
| JELI | Rp 900 | +25.00% (ARA) | ARA | ✓ |
| JECX | Rp 1.250 | +24.80% (ARA) | Normal | ✗ |
| EMMI | Rp 470 | +20.21% (Normal) | Normal | ✓ |
| BACH | Rp 442 | +24.43% (ARA) | Normal | ✗ |
| PRDL | Rp 120 | +35.00% (ARA) | Normal | ✗ |
| RANS | Rp 170 | — | — | — |

**Distribusi Probabilitas Prediksi Model (Gradient Boosting):**

![Probabilitas Prediksi Kelas Return D1](assets/probabilitas_prediksi_d1_gradient_boosting.png.png)

Akurasi pada 5 emiten yang sudah listing: **2/5 (40%)**. Kegagalan sistematis pada prediksi kelas ARA diduga disebabkan oleh class imbalance di training data dan ketiadaan fitur oversubscription rate di dataset.

## Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
