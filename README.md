# Prediksi Durasi Pengiriman Supply Chain (DataCo)

Tugas Pengganti UTS Analitika Data, Teknik Industri UNS.

**Anggota kelompok:**
- Aditya Surya Pratama (I0324103)
- Mohammad Arief Rahman (I0324126)

## Deskripsi

Memprediksi durasi pengiriman aktual (`Days for shipping (real)`) pada dataset
DataCo Supply Chain menggunakan **Multiple Linear Regression**.

- Pembagian data: Training 64%, Validation 16%, Test 20%, dibagi per `Order Id`
  agar tidak ada kebocoran data antar subset.
- Fitur redundan dibuang berdasarkan pemeriksaan kolinearitas.
- Evaluasi mencakup baseline, perbandingan set fitur, cross-validation
  (GroupKFold), tabel OLS dengan SE klasik dan SE cluster, VIF, diagnostik
  residual, dan permutation importance.

## Struktur

| File | Keterangan |
|------|------------|
| `main (1).py` | Skrip lengkap (jalankan dari terminal) |
| `Notebook (2).ipynb` | Versi notebook dengan isi yang sama |
| `requirements.txt` | Daftar library |
| `output` | Tabel dan gambar hasil (dibuat otomatis) |

## Cara menjalankan

1. Unduh dataset **DataCoSupplyChainDataset.csv** (Kaggle) dan letakkan di folder yang sama dengan `main.py`.
2. Install library:
   ```bash
   pip install -r requirements.txt
   ```
3. Jalankan:
   ```bash
   python main.py
   ```

Semua hasil disimpan di folder `output`.
