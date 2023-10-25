# Laporan Proyek Machine Learning
### Nama : Melinda Purnama Dewi
### Nim : 211351082
### Kelas : Pagi B

## Domain Proyek

Dataset predic penyakit paru paru ini dapat digunakan memprediksi kemungkinan seseorang terkena penyakit paru-paru terutama pada individu yang memiliki riwayat merokok atau paparan asap rokok dan dapat membantu dalam pencegahan deteksi sejak dini.

## Business Understanding
Dapat menentukan apakah seseorang beresiko terkena penyakit paru-paru, berdasarkan faktor-faktor tertentu dan dapat mencegah penyakit menjadi kronis.

### Problem Statements

Ketidakmungkinan bagi seseorang menyadari aktivitas merokok secara berlebihan yang merupakan salah satu penyebab penyakit paru- paru.

### Goals

Mendiagnosa sejak awal terhadap gejala yang diderita penyakit paru - paru yang disebabkan prokok aktif dan juga orang- orang yang disekitar yang terpapar asap rokok.

Semua poin di atas harus diuraikan dengan jelas. Anda bebas menuliskan berapa pernyataan masalah dan juga goals yang diinginkan.

## Data Understanding

[Dataset predic terkena penyakit paru paru](https://www.kaggle.com/datasets/andot03bsrc/dataset-predic-terkena-penyakit-paruparu)

### Variabel-variabel pada Heart Failure Prediction Dataset adalah sebagai berikut:
- Usia : Merupakan umur pasien[muda,tua]
- Jenis kelamin: Merupakan jenis kelamin pasien meliputi[pria,wanita]
- Merokok: Merupakan tindakan menghisap dan menghirup asap yang dihasilkan oleh pembakaran tembakau [aktif,pasif]
- Bekerja: Merupakan aktivitas yang melibatkan usaha fisik atau mental yang dilakukan oleh individu atau kelompok dalam rangka mencari nafkah [ya,tidak]
- Aktivitas_Begadang: Seseorang yang melakukan kegiatan pada saat malam hari diwaktu tidur [ya,tidak]
- Aktivitas_Olahraga: Kegiatan menggerakan anggota badan yang dilakukan untuk meningkatan kebugaran fisik [jarang,seing]
- Penyakit Bawaan: penyakit atau gangguan kesehatan yang dimiliki sesorang sejak lahir[ada,tidak]

## Modeling
Model yang digunakan adalah logistik regresi
logistik regresi adalah teknik analisis data yang menggunakan matematika untuk menemukan hubungan antara dua faktor data.Tahapan projek dataset predic penyakit paru - paru ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan, anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.
membagi dataset menjadi x dan y, standarisasi yang dilakukan standart scaler dan standarisasi yang dilakukan kepada x,membagi data menjadi train dan test sama rata,model yang digunakan logistik regresi,menghitung keberhasilan model melalui akurasi train dan test.

## Evaluation

- Penjelasan mengenai metrik yang digunakan
Presisi (Precision) adalah salah satu metrik evaluasi yang digunakan dalam konteks model klasifikasi,ini mengukur sejauh mana prediksi positif yang dibuat oleh model adalah benar atau akurat.
Rumus
Presisi= TP /TP+FP
- Menjelaskan hasil proyek berdasarkan metrik evaluasi
#Library evaluasi
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#Compute performance manually
NewprediksiBenar = (predicted == Y_test).sum()
NewprediksiSalah = (predicted != Y_test).sum()

print("prediksi benar: ", NewprediksiBenar, " data")
print("prediksi salah: ", NewprediksiSalah, " data")
print("Akurasi Algoritme: ", NewprediksiBenar/(NewprediksiBenar+NewprediksiSalah)*100,"%")

CM = confusion_matrix(Y_test,predicted)

TN = CM[0][0]
FN = CM[1][0]
TP = CM[1][1]
FP = CM[0][1]
precision    = TP/(TP+FP)

Akurasi Algoritme:  94.3 %
precision = 100. %
RECALL: 88.1%
f1score:  93.7%

## Deployment
pada bagian ini anda memberikan link project yang diupload melalui streamlit share. boleh ditambahkan screen shoot halaman webnya.
https://prediksiparu-g38c8cxxgfhecd4xnurssq.streamlit.app/
https://github.com/melindapurnamadewi9/prediksi_paru/tree/main

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.