# Semantic Segmentation Self Driving Cars Menggunakan UNet, FCN8s, dan SegNet
Self driving cars telah dikembangkan untuk memenuhi kebutuhan atau memecahkan masalah manusia pada lalu lintas mulai dari menghindari kemacetan hingga kecelakaan lalu lintras yang disebabkan oleh manusia. AI (artificial intelegen) merupakan salah satu teknologi yang menyusun self driving cars, khususnya dibidang computer vision. Teknologi ini membantu dalam mengambil data visual dan mengidentifikasi objek-objek di sekitar mobil. 

Semantic segmentasi merupakan tugas computer vision yang di mana memberi kelas label wilayah tertentu atau pixel dari suatu gambar sesuai dengan apa yang ditampilkan. sehingga driving cars dapat mengetahui dan membedakan setiap objek yang ada dijalan. Project ini akan berfokus dalam semantic segmentation self driving cars yang dimana mengambil gambar RGB dan menghasilkan peta segmentasi dari hasil pembelajaran konvolusi neural network


## 1. Loading Dataset
Dataset yang kita gunakan merupakan data cityscapes yang sudah dipisah untuk proses training dan test.
<p align="center">
  <img src="https://github.com/muhammadmhmd/Semantic-Segmentation-Self-Driving-Cars/blob/main/Images/1.png" alt="data image shape"/>
</p>

### 1.1 Image
total image terdapat 468 yang dimana 367 untuk training dan 101 untuk test, begitu juga anotasi. gambar dan anotasi berukuran pixel 360x480 dengan 3 channel warna. Kumpulan file anotasi merupakan label-label objek pada setiap gambar yang dimana digunakan untuk variabel target proses pembelajaran model. Untuk membaca file menggunakan library opencv (cv2) namun library ini membaca gambar dengan urutan BGR (Blue, Red, Green), maka untuk menampilkan gambar dengan warna asli kita dapat mengkonversi ke RGB (Red, Greed, Blue)
<p align="center">
  <img src="https://github.com/muhammadmhmd/Semantic-Segmentation-Self-Driving-Cars/blob/main/Images/2.png" alt="image bgr rgb"/>
</p>

### 1.2 Anotasi
file anotasi merupakan gambar segmentasi pixel yang berisi dengan angka dari setiap kelas, contoh seperti gambar dibawah ini
<p align="center">
  <img src="https://github.com/muhammadmhmd/Semantic-Segmentation-Self-Driving-Cars/blob/main/Images/3.png" alt="input array"/>
</p>
Anotasi tidak bisa langsung divisualisasikan karena array anotasi bukan array pixel gambar, sehingga perlu dilakukan color mapping, hal ini dapat dilakukan oleh library matplotlib dengna preset color map yang sudah ada, tetapi project ini akan mendefinisikan color map baru. Menurut informasi dataset terdapat 11 kelas ditambah dengan 1 background sehingga terdapat 12 kelas dalam dataset. Setiap kelas ini didefinisikan warna channel rgb nya kemudian dapat ditampilkan
<p align="center">
  <img src="https://github.com/muhammadmhmd/Semantic-Segmentation-Self-Driving-Cars/blob/main/Images/4.png" alt="input array"/>
</p>
Diatas merupakan hasil tampilan anotasi sebelum dan sesudah di color mapping. Dengan color mapping sebagai berikut
<p align="center">
  <img src="https://github.com/muhammadmhmd/Semantic-Segmentation-Self-Driving-Cars/blob/main/Images/5.png" alt="color mapping"/>
</p>
 

## 2.	Prosesing Image
Pemrosesan gambar dilakukan untuk menghasilkan gambar yang siap untuk proses pembelajaran model sehingga dapat meningkatkan performa klasifikasi. Dalam project ini ada 3 tahap mengubah ukuran pixel, normalisasi array pixel, mengubah ke grayscale dan one hot encoding
### 2.1	Resize Image
beban komputasi tergantung pada jumlah pixel gambar waktu training, semakin besar pixel gambar maka beban komputasi akan semakin besar. mengubah ukuran image lebih kecil dapat meringankan beban komputasi, namun semakin ukuran diperkecil maka akan semakin banyak informasi yang hilang pada gambar yang mengakibatkan performa klasifikasi menurun. Pada project ini, gambar di ubah ukuran menjadi 224x224
<p align="center">
  <img src="https://github.com/muhammadmhmd/Semantic-Segmentation-Self-Driving-Cars/blob/main/Images/6.png" alt="resize img"/>
</p>
<p align="center">
  <img src="https://github.com/muhammadmhmd/Semantic-Segmentation-Self-Driving-Cars/blob/main/Images/6_1.png" alt="resize img shape"/>
</p>
 
### 2.2	Normalisasi Pixel
normalisasi pixel bertujuan memastikan bahwa setiap parameter input pixel memiliki distribusi data yang serupa yaitu dengan skala terkecil 0 dan terbesar 1. Proses ini membuat konvergensi lebih cepat saat pembelajaran jaringan.
<p align="center">
  <img src="https://github.com/muhammadmhmd/Semantic-Segmentation-Self-Driving-Cars/blob/main/Images/7.png" alt="sebelum normalisasi"/>
</p>
setelah dinormalisasikan
<p align="center">
  <img src="https://github.com/muhammadmhmd/Semantic-Segmentation-Self-Driving-Cars/blob/main/Images/8.png" alt="setelah normalisasi"/>
</p>

### 2.3	Grayscaling
Perubahan image menjadi abu-abu (grayscaling) akan mengakibatkan gambar hanya memiliki satu channel warna saja. Proses ini digunakan pada anotasi yang awalnya memiliki 3 channel dengan tujuan anotasi hanya memiliki 1 channel untuk proses selanjutnya yaitu one hot encoding
<p align="center">
  <img src="https://github.com/muhammadmhmd/Semantic-Segmentation-Self-Driving-Cars/blob/main/Images/9.png" alt="shape grayscale"/>
</p>

### 2.4	One Hot Encoding
one hot enciding merupakan teknik untuk mempresentasikan data kelas menjadi bilangan biner 0 dan 1, dimana semua elemen akan bernilai 0 kecuali satu elemen yang bernilai 1, yaitu elemen yang memiliki nilai kategori tersebut. Dalam contoh gambar gambar wanita diatas apabila dilakukan one hot encoding dapat divisualisasikan seperti gamabr dibawah ini
<p align="center">
  <img src="https://github.com/muhammadmhmd/Semantic-Segmentation-Self-Driving-Cars/blob/main/Images/10.png" alt="ohe"/>
</p>
Setiap kelas akan membentuk channel baru yang berisi array 0 dan 1, yang dimana pixel di setiap channel tersebut akan bernilai satu apabila sesuai dengan kelasnya. Channel-channel ini akan mempermudah proses klasifikasi pixel untuk setiap objek kelas dan area dari objek tersebut. Pada dataset project ini memiliki 12 kelas sehingga proses one hot encoding akan menghasilkan 12 channel
<p align="center">
  <img src="https://github.com/muhammadmhmd/Semantic-Segmentation-Self-Driving-Cars/blob/main/Images/10_1.png" alt="ohe shape"/>
</p> 
Apabila setiap channel diplot dapat divisualisasikan seperti gambar dibawah berikut ini
<p align="center">
  <img src="https://github.com/muhammadmhmd/Semantic-Segmentation-Self-Driving-Cars/blob/main/Images/11.png" alt="ohe image"/>
</p> 
 

## 3.	Modeling
dalam project ini, model neural network yang saya gunakan untuk proses training yaitu Unet, FCN8s, dan SegNet dimana saya susun dari awal menggunakan layer keras tensorflow, tidak menggunakan pre trained model

### 3.1	Unet 
<p align="center">
  <img src="https://github.com/muhammadmhmd/Semantic-Segmentation-Self-Driving-Cars/blob/main/Images/12.png" alt="unet"/>
</p> 
U-Net terdiri dari jalur kontrak dan jalur ekspansif. Jalur kontrak mengikuti arsitektur khas jaringan konvolusi. Ini terdiri dari aplikasi berulang dari dua konvolusi dan operasi pooling untuk downsampling. Pada setiap langkah downsampling, kami menggandakan jumlah saluran fitur. Setiap langkah di jalur ekspansif terdiri dari upsampling peta fitur diikuti oleh konvolusi yang membagi dua jumlah saluran fitur, penggabungan dengan peta fitur yang dipangkas sesuai dari jalur kontrak, dan dua konvolusi. Pemangkasan diperlukan karena hilangnya piksel batas di setiap konvolusi. Pada lapisan terakhir, konvolusi 1x1 digunakan untuk memetakan setiap vektor fitur ke jumlah kelas yang diinginkan. [Source](https://paperswithcode.com/method/u-net#:~:text=U%2DNet%20is%20an%20architecture,architecture%20of%20a%20convolutional%20network.)

### 3.2	FCN8s
<p align="center">
  <img src="https://github.com/muhammadmhmd/Semantic-Segmentation-Self-Driving-Cars/blob/main/Images/13.png" alt="fcn8s"/>
</p> 
FCN terdiri dari 5 blok layer konvolusi dan dekonvolusi(upsampling) untuk menghasilkan output. dekonvolusi FCN tanpa penggabungan dengan sampel konvolusi sebelumnya dinamakan network FCN32s. Dekonvolusi FCN dengan penggabungan dengan sampel konvolusi sekali (dengan layer pool4) dinamakan network FCN16s. dan Dekonvolusi FCN dengan penggabungan dengan sampel konvolusi 2 kali (dengan layer pool4 dan pool3)  dinamakan network FCN8s. Hal ini karena, fitur yang dalam dapat diperoleh saat masuk lebih dalam, informasi lokasi spasial juga hilang saat masuk lebih dalam. Itu berarti keluaran dari lapisan yang lebih dangkal memiliki lebih banyak informasi lokasi. Jika kita menggabungkan keduanya, kita dapat meningkatkan hasilnya.[Source](https://towardsdatascience.com/review-fcn-semantic-segmentation-eb8c9b50d2d1)

### 3.3	SegNet
<p align="center">
  <img src="https://github.com/muhammadmhmd/Semantic-Segmentation-Self-Driving-Cars/blob/main/Images/14.png" alt="segnet"/>
</p> 
SegNet adalah encoder-decoder convolutional, yang berturut-turut menurunkan sampel input hingga lapisan bottleneck, di mana tahap upsampling yang berurutan digunakan untuk menghasilkan resolusi yang diinginkan. Setiap lapisan konvolusi mencakup konvolusi, normalisasi batch, dan fungsi aktivasi rectified linear unit (ReLU). Blok convolutional diikuti oleh lapisan pooling/upsampling untuk mencapai arsitektur encoder-decoder. Output jaringan adalah input ke lapisan softmax untuk menghasilkan output akhir. [Source](https://www.researchgate.net/publication/349439499_Uncertainty-Aware_Deep_Learning_for_Safe_Landing_Site_Selection)

## 4. Evaluasi
Untuk mengevaluasi model kita menggunakan 4 metrik yaitu akurasi, F1 score, IoU dan MIoU. model UNet dan FCN8s didapatkan nilai perbedaan akurasi dari train dan test yang cukup kecil dari awal hingga akhir epoch, berbeda dengan SegNet yang membutuhkan beberapa kali epoch untuk mengimbangi nilai akurasi train dan test, hal ini menunjukan model segnet membutuhkan iterasi yang banyak agar model dapat belajar dengan baik, berbeda UNet dan FCN8s yang sudah dapat belajar pada iterasi awal. Hasil proses training dapat dilihat pada gambar dibawah ini
<p align="center">
  <img src="https://github.com/muhammadmhmd/Semantic-Segmentation-Self-Driving-Cars/blob/main/Images/15.png" alt="accuracy"/>
</p> 
Untuk metrik loss, model UNet dan FCN8s juga memiliki perbedaan yang rendah dari awal hingga akhir iterasi dan SegNet memiliki nilai loss yang sangat tinggi pada awal iterasi. Nilai akurasi tinggi belum pasti bahwa model dapat melakukan klasifikasi kelas dengan baik. Bisa jadi terdapat imbalance data sehingga model condong hanya mengklasifikasi kelas tertentu saja dengan sangat baik karena data kelas tersebut lebih banyak dibandingkan data kelas lain. Sehingga kita perlu melakukan evaluasi dengan metrik lainnya
<p align="center">
  <img src="https://github.com/muhammadmhmd/Semantic-Segmentation-Self-Driving-Cars/blob/main/Images/16.png" alt="IoU"/>
</p> 
IoU (Intersection over union) merupakan metrik evaluasi menghitung seberapa banyak pixel test dan pixel prediksi saling beririsan terhadap gabungan pixel test dan prediksi. Untuk MIoU (IoU rata-rata) dari gambar dihitung dengan mengambil IoU dari masing-masing kelas dan dirata-ratakan. berikut hasil tabel dari IoU per kelass untuk setiap model
<p align="center">
  <img src="https://github.com/muhammadmhmd/Semantic-Segmentation-Self-Driving-Cars/blob/main/Images/17.png" alt="IoU class"/>
</p> 
Dari hasil menunjukan bahwa model yang memiliki nilai IoU yang paling baik untuk setiap kelasnya yaitu model UNet kecuali untuk kelas 7 (fence/pagar) yaitu pada model SegNet. Untuk IoU terendah terdapat pada kelas 2 (Pole/Tiang) yang disusul oleh kelas 6 (SignSymbol/simbol lalu lintas), hal ini terjadi karena bisa saja karena ukuran pixel kelas tersebut yang kecil dan atau jumlah kelas yang sedikit untuk setiap gambarnya. Apabila dirata-rata didapatkan nilai sebagai berikut
<p align="center">
  <img src="https://github.com/muhammadmhmd/Semantic-Segmentation-Self-Driving-Cars/blob/main/Images/18.png" alt="MIoU"/>
</p> 
Walapun UNet memiliki akurasi yang besar tetapi mendapatkan MIoU hanya sebesar 0.515. apabila setiap prediksi model divisualisasikan akan didapatkan seperti berikut ini
<p align="center">
  <img src="https://github.com/muhammadmhmd/Semantic-Segmentation-Self-Driving-Cars/blob/main/Images/19.png" alt="eval"/>
</p> 
Ground truth merupakan anotasi test yang sebenarnya, untuk anotasi yang paling mendekati yaitu pada model UNet, sedangkan pada model yang lain masih banyak kelas yang belum terlabelkan atau salah dalam pelabelan.

## 5. Kesimpulan
Pada project ini telah dilakukan semantic segmentation dengan menggunakan 3 model arsitektur yaitu UNet, FCN8s, dan SegNet. Dari hasil didapatkan model terbaik pada UNet dengan akurasi 0.913 dan MIoU 0.515, dari ketiga model kelas yang memilki nilai IoU yang paling rendah  pada kelas 2 (tiang) dan 6 (simbol lalu lintas). [Full code](https://github.com/muhammadmhmd/Semantic-Segmentation-Self-Driving-Cars/blob/main/Semantic%20Segmentation%20Self%20Driving%20Cars.ipynb)

## 6. Saran
Pada project ini pembuatan model dilakukan dengan menyusun dari awal dengan layer keras tensorflor, sehingga perlu diperhatikan keseuaian setiap blok layer dalam penyusunan. Untuk memudahkan, bisa memakai tranfer learning untuk meringkas layer.

## Sumber
##### https://towardsdatascience.com/review-fcn-semantic-segmentation-eb8c9b50d2d1
##### https://paperswithcode.com/method/u-net#:~:text=U%2DNet%20is%20an%20architecture,architecture%20of%20a%20convolutional%20network. 
##### https://www.researchgate.net/publication/349439499_Uncertainty-Aware_Deep_Learning_for_Safe_Landing_Site_Selection 
##### https://www.jeremyjordan.me/semantic-segmentation/#skip_connections 
##### https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2 
