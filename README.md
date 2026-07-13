### Proje Araştırma Raporu (trex\_research)

# 



Rapor Yazarı: İlterişhan Öztekin



Bu repo, görüntü işleme, yapay zekâ, temel backend mimarisi (API) ve versiyon kontrol sistemleri (Git) üzerine yapılan araştırma ve değerlendirme sonuçlarını içermektedir. Rapor genel hatlarıyla teorik bilgileri, sektördeki karşılıklarını ve kişisel çıkarımları barındırmaktadır.







İçerik



1. Görüntü İşlemeye Giriş
2. Python ve Görüntü İşleme Ekosistemi
3. Yapay Zekâ ve Görüntü İşleme							
4. Temel API Bilgisi
5. Git ve Yazılım Geliştirme Süreci
6. Genel Değerlendirme ve Sonuç
7. Derin Öğrenme Teorik Altyapısı ve Evrişimli Sinir Ağları (CNN)
8. Haber Metninden Konu Sınıflandırma Projesi




 

1\. Görüntü İşlemeye Giriş



Görüntü İşleme (Image Processing) Nedir?



&#x20;  Görüntü işleme, dijital bir görüntünün kalitesini artırmak, üzerindeki gürültüleri temizlemek veya görüntüyü başka bir forma dönüştürmek amacıyla bilgisayar algoritmaları kullanılarak manipüle edilmesidir. Burada girdi de bir görüntüdür, çıktı da genellikle işlenmiş/değiştirilmiş bir görüntüdür (örneğin kontrast ayarı, netleştirme).



Bilgisayarlı Görü (Computer Vision) ile Görüntü İşleme Arasındaki Farklar



Sektörde bu iki kavram sıklıkla karıştırılsa da aralarında net bir felsefi ve pratik fark vardır:



* Görüntü İşleme: Görüntünün pikselleriyle oynar. Resmi iyileştirir, kırpar veya filtreler. Sistemin resmin ne olduğunu "anlamasına" gerek yoktur. (Örn: Bir fotoğrafın siyah-beyaz yapılması).



* Bilgisayarlı Görü (CV): Görüntüyü anlamlandırmaya çalışır. İnsan gözünün ve beyninin görme/yorumlama yeteneğini taklit eder. Çıktı bir görüntü değil, bir karardır. (Örn: "Bu resimde bir drone var" bilgisi).



&#x20;  Pratikte bu iki alan et tırnak gibidir. Bilgisayarlı görü algoritmalarının (örneğin bir nesne tespiti modelinin) yüksek başarıyla çalışabilmesi için, ham görüntünün önce görüntü işleme teknikleriyle (parlaklık dengeleme, boyutlandırma) optimize edilmesi gerekir.



Görüntü İşlemenin Kullanım Alanları



* Savunma Sanayii ve İHA/SİHA'lar: Otonom hedef tespiti, sınır takibi ve termal kameralardan gelen görüntülerin iyileştirilmesi.



* Tıp (Medikal Görüntüleme): MR ve BT sonuçlarındaki tümörlerin veya anomalilerin netleştirilmesi, otomatik taranması.



* Otomotiv: Sürücüsüz araçların şerit takibi yapabilmesi için yol çizgilerinin filtrelenmesi.







2\. Python ve Görüntü İşleme Ekosistemi



Görüntü İşleme Projelerinde Neden Python Tercih Edilmektedir?



&#x20;  Python'ın bu alandaki hakimiyetinin arkasında dilin sözdizimi (syntax) kolaylığından ziyade, muazzam bir kütüphane ekosistemi ve arkasındaki C/C++ optimizasyonları yatar. Python ile hızlıca prototip üretilebilir ve NumPy gibi kütüphaneler sayesinde ağır matematiksel matris işlemleri alt tarafta C hızında koşturulabilir.



Temel Kütüphaneler ve Kullanım Amaçları



|Kütüphane|Temel Amaç|Öne Çıkan Özelliği|
|-|-|-|
|OpenCV|Bilgisayarlı görü ve gerçek zamanlı görüntü işleme.|Çok hızlıdır (C++ tabanlı), video işleme ve kamera entegrasyonu güçlüdür.|
|NumPy|Çok boyutlu matris ve dizi işlemleri.|Görüntüler aslında birer \[H, W, C] matrisidir. NumPy, pikseller üzerinde doğrudan matematiksel işlem yapmayı sağlar.|
|Pillow (PIL)|Temel resim açma, kaydetme ve basit manipülasyonlar.|Hafiftir. Format dönüştürme, kırpma ve basit filtreler için idealdir.|



Diğer Sık Kullanılan Kütüphaneler



* SciPy: OpenCV'ye ek olarak daha ileri düzey sinyal ve görüntü filtreleme (Fourier dönüşümleri vb.) için kullanılır.



* Matplotlib: İşlenen görüntülerin ve piksellerin histogram grafiklerinin görselleştirilmesinde standarttır.





&#x20;  Python benim de alanım olan yapay zeka mühendisliğinin aslında temelidir diyebiliriz çünkü yukarıda da belirtmiş olduğum gibi hızlı, kullanıcı dostu ve geniş kütüphanesi ile diğer yazılım dillerine göre daha akıcı bir deneyim sunduğuna inanıyorum.

#### 



3\. Yapay Zekâ ve Görüntü İşleme



Makine Öğrenmesi (ML) ve Derin Öğrenme (DL) Arasındaki Farklar



Makine Öğrenmesi: Geleneksel ML'de öznitelik çıkarımı (feature extraction) insan eliyle yapılır. Örneğin bir resimdeki kenarları, renk dağılımlarını mühendis kendisi kodlar ve sınıflandırıcıya (örn: SVM) verir.



Derin Öğrenme: Çok katmanlı yapay sinir ağları (özellikle CNN'ler) kullanılır. Model, resmin içindeki önemli özellikleri (kenarlar, dokular, şekiller) kendi kendine, katmanlar ilerledikçe öğrenir. İnsan müdahalesine ihtiyaç duymaz, ancak çok daha fazla veri ve GPU gücü ister.



Temel Kavramlar



1\. Görüntü Sınıflandırma (Image Classification)



&#x20;  Görüntünün tamamına tek bir etiket basma işlemidir. "Bu resimde ne var?" sorusuna yanıt arar. (Örn: Resmin tamamına "Uçak" denmesi).



2\. Nesne Tespiti (Object Detection)



&#x20;  Görüntüdeki nesnelerin hem ne olduğunu bulur hem de yerlerini sınırlayıcı kutularla (bounding box) konumlandırır. "Hangi nesne, resmin neresinde?" sorusunu yanıtlar.



4\. Görüntü Segmentasyonu (Image Segmentation)



&#x20;  Resimdeki nesneleri piksel düzeyinde ayırır. Sadece kutu içine almaz, nesnenin tam sınırlarını çizer (Örn: Tıbbi bir görüntüde tümörün sınırlarının milimetrik belirlenmesi).



Popüler Modeller



* YOLO (You Only Look Once): Nesne tespiti (Object Detection) dünyasının kralıdır. Resmi tek bir seferde ağdan geçirerek hem sınırlayıcı kutuları hem de sınıfları tahmin eder. Gerçek zamanlı (real-time) sistemlerde (örneğin UCAV drone sistemlerinde) hızı nedeniyle vazgeçilmezdir.



* ResNet (Residual Networks): Görüntü sınıflandırmada devrim yaratmış derin bir CNN mimarisidir. "Skip connections" (atlama bağlantıları) sayesinde çok derin ağlarda yaşanan gradyan kaybolması (vanishing gradient) problemini çözer ve modelin çok daha derin, dolayısıyla daha zeki olmasını sağlar.





&#x20;  Makine Öğrenmesi çok katmanlı ya da karmaşık ve kompleks problemlemleri çözmekte, deep learning kadar etkili değildir. Çünkü Deep Learning'in içinde bulundurduğu çok katmanlı ve ağırlık mekanizmalı yapı karmaşık problemleri çözmede tek düzeliği bozarak daha etkili bir çözüm sunar.





4\. Temel API Bilgisi



API (Application Programming Interface) Nedir?



&#x20;  API (Uygulama Programlama Arayüzü), iki farklı yazılımın veya servisin birbiriyle konuşmasını, veri alışverişi yapmasını sağlayan bir köprüdür. Örneğin, hazırladığımız bir görüntü işleme modelinin bir web arayüzü veya mobil uygulama tarafından kullanılabilmesi için bir API arkasına gizlenmesi gerekir.



REST API Nedir ve Neden Kullanılır?



&#x20;  REST, HTTP protokolünün kurallarını kullanan, istemci-sunucu (client-server) arasındaki iletişimi hafif, stateless (durumsuz) ve standart bir yapıda kuran mimari bir tarz dır.



Neden kullanılır?



&#x20;  Platform bağımsızdır. Backend Python (FastAPI/Flask) ile yazılmış olsa bile, frontend tarafı (React, Flutter veya bir gömülü sistem) bu API'ye kolayca istek atıp yanıt alabilir.



HTTP Metodları



GET: Sunucudan veri çekmek için kullanılır. (Örn: İşlenmiş resimlerin listesini getirmek).



POST: Sunucuya yeni bir veri göndermek/yaratmak için kullanılır. (Örn: İşlenmesi için sunucuya ham bir resim upload etmek).



PUT: Sunucudaki mevcut bir veriyi güncellemek için kullanılır.



DELETE: Sunucudaki bir veriyi silmek için kullanılır.



JSON Veri Formatı Nedir?



&#x20;  JSON(JavaScript Object Notation), insan tarafından kolayca okunabilen ve makineler tarafından kolayca parse edilebilen key-value çiftlerinden oluşan hafif bir veri değişim formatıdır. API'lerde girdi parametrelerini veya model çıktılarını (örneğin tespit edilen nesnenin koordinatlarını) taşımak için standarttır.



JSON Kod Örneği:



{

&#x20; "object\_type": "drone",

&#x20; "confidence": 0.94,

&#x20; "bounding\_box": \[120, 45, 300, 250]

}





&#x20;  Http konusunda çok bilgim yok çünkü yapay zeka mühendisliği programında herhangi bir web design vb. dersi bulunmuyor. Bu nedenle bu bilgiler benim için çok yeni ve farklı. Yapay zeka mühendisliğinde web design vb. programlar yerine daha çok makine öğrenmesi, deep learning algoritmaları ağırlıktadır.





5\. Git ve Yazılım Geliştirme Süreci



Git ve GitHub Nedir?



Git: Kodunuzdaki değişiklikleri zaman içinde kaydeden, eski versiyonlara dönmenizi sağlayan lokal bir versiyon kontrol sistemidir.



GitHub: Git ile tuttuğunuz bu kayıtları (repoları) internet üzerinde saklamanızı, paylaşmanızı ve takımlarla ortak çalışmanızı sağlayan bulut tabanlı bir platformdur.



Temel Git Komutları



git clone <url>: Uzaktaki bir repoyu kendi bilgisayarınıza indirmek için kullanılır.



git add <dosya>: Değişiklikleri yerel depoya kaydetmeden önce "sahneye" (staging area) alır.



git commit -m "mesaj": Sahneye alınan değişiklikleri bir versiyon (snapshot) olarak yerel veritabanına kalıcı olarak kaydeder.



git push origin <branch>: Yereldeki commit'leri uzaktaki (GitHub) sunucuya gönderir.



git pull: Uzaktaki repoda yapılan değişiklikleri yerel bilgisayarınıza çeker ve birleştirir.



git branch <isim>: Ana kod hattını bozmadan yeni bir özellik denemek için yeni bir çalışma dalı (branch) açar.



Versiyon Kontrol Sistemlerinin Yazılım Geliştirme Sürecindeki Önemi



&#x20;  Yazılım geliştirirken kodun bir yerinde hata yaptığınızda "Geri Al (Ctrl+Z)" bir yere kadar kurtarır. Git sayesinde projenin çalışan herhangi bir geçmiş anına güvenle dönebilirsiniz. Özellikle takım çalışmalarında, iki farklı yazılımcının aynı dosya üzerinde birbirinin kodunu ezmeden, çakışmaları (conflict) çözerek eşzamanlı çalışabilmesini sağlar.





6\.Genel Değerlendirme ve Sonuç



&#x20;  Bu araştırma süreci, bir görüntü işleme projesinin sadece ham bir resmi manipüle etmekten ibaret olmadığını; Görüntü İşleme -> Bilgisayarlı Görü -> Derin Öğrenme(YOLO/ResNet) -> API(REST/JSON) -> Dağıtım/Versiyonlama(Git) adımlarıyla uçtan uca bir yazılım mimarisi gerektirdiğini açıkça göstermektedir.



&#x20;  Özellikle gerçek zamanlı sistemlerde(UCAV / SİHA hedef tespit yazılımları gibi) modelin hızı(YOLO) ve verilerin backend ile kayıpsız/hızlı iletilmesi(REST API) projenin başarısını doğrudan belirleyen kritik yapı taşlarıdır.




## 7. Derin Öğrenme Teorik Altyapısı ve Evrişimli Sinir Ağları (CNN)

### Özellik Çıkarımı (Feature Extraction): Geleneksel vs. Modern Derin Öğrenme

Görsellerdeki anlamlı bilgilerin (kenar, köşe, doku gibi özniteliklerin) matematiksel olarak ifade edilmesi sürecidir.

* **Geleneksel Yöntemler (SIFT, HOG):** SIFT (Ölçekten Bağımsız Öznitelik Dönüşümü) ve HOG (Yönlendirilmiş Gradyanların Histogramı) gibi geleneksel algoritmalar, insan eliyle tasarlanmış (hand-crafted) matematiksel kurallara dayanır. Mühendis filtreyi kendisi yazar.

* **Modern Derin Öğrenme (CNN):** Evrişimli katmanlar, görsel üzerindeki bu özellikleri (ilk katmanlarda basit kenarları, derin katmanlarda karmaşık nesne parçalarını) **otonom (kendi kendine)** öğrenir.


### CNN (Convolutional Neural Network) Mimarisi ve Temel Bileşenleri

Evrişimli(Convolutional) sinir ağları, insan görme sistemindeki lokal algılama alanlarından (local receptive fields) esinlenmiştir. En büyük avantajı, tüm resme tek tek bakmak yerine "ağırlık paylaşımı" mekanizmasıyla parametre sayısını devasa oranda azaltmasıdır.

* **Evrişim Katmanı (Convolutional Layer):** Resim üzerinde gezen filtreler (kernel) aracılığıyla öznitelik haritaları (feature maps) üretir.

* **Aktivasyon Katmanı:** Ağa doğrusalsızlık ekler (ReLU vb.). Doğrusalsızlık olmazsa ağ ne kadar derin olursa olsun sadece tek bir lineer fonksiyon gibi davranır.

* **Havuzlama Katmanı (Pooling - Max/Average):** Resmin uzamsal boyutunu (genişlik ve yükseklik) küçültür. Bilgi kaybını minimize ederken hesaplama yükünü azaltır ve "kayma ötelemelerine" karşı modeli dayanıklı kılar. Genelde en belirgin özellikleri koruyan **Max Pooling** tercih edilir.

* **Tam Bağlantılı Katman (Fully Connected Layer):** Ağın sonundaki düzleştirilmiş vektörü alır ve standart bir yapay sinir ağı gibi çalışarak nihai sınıflandırma skorlarını üretir.

### Aktivasyon Fonksiyonları ve Optimizasyon Algoritmaları

* **Aktivasyon Fonksiyonları (ReLU, LeakyReLU, Sigmoid, Softmax):** `Sigmoid` ikili sınıflandırmada, `Softmax` çoklu sınıflandırmada olasılık üretmek için en son katmanda kullanılır. Ara katmanlarda ise gradyan kaybolması (vanishing gradient) problemini çözdüğü ve hızlı hesaplandığı için **`ReLU`** standarttır. ReLU'nun negatif değerleri tamamen sıfırlayıp "nöron ölmesine" sebep olduğu durumlarda ise **`LeakyReLU`** kurtarıcıdır.

* **Optimizasyon Algoritmaları (SGD, Adam, RMSprop):** Kayıp fonksiyonunu (Loss) sıfıra yaklaştırmak için ağırlıkları günceller. `SGD` (Stokastik Gradyan İnişi) gelenekseldir ve momentum eklenerek hızlandırılır. **`Adam`** ve `RMSprop` ise her parametre için "adaptif öğrenme oranı (adaptive learning rate)" hesaplar. Günümüz projelerinde hızlı yakınsama sağladığı için genellikle ilk tercih `Adam` algoritmasıdır.

### Transfer Learning (Transferli Öğrenme)

Daha önce devasa veri kümelerinde (ImageNet gibi milyonlarca resim içeren setler) eğitilmiş dev modellerin (ResNet, EfficientNet, VGG) ağırlıklarını (bilgilerini) alıp, kendi daha küçük veri kümemize uyarlama sürecidir.

* **Avantajları:** Sıfırdan model eğitmeye kıyasla muazzam bir zaman ve GPU tasarrufu sağlar; az veriyle bile çok yüksek doğruluk elde edilebilir.

* **Dezavantajları (Negatif Transfer):** Eğer kaynak veri kümesi (örn. genel kedi/köpek resimleri) ile bizim hedef veri kümemiz (örn. çok spesifik tıp veya askeri radar resimleri) arasında aşırı bir domain uyuşmazlığı varsa, model yanlış korelasyonlar kurabilir ve başarıyı düşürebilir.

### Veri Artırımı (Data Augmentation)

Modelin eğitim verilerini sadece ezberlemesini önlemek ve genelleme yeteneğini artırmak için eldeki resimleri yapay olarak çoğaltma tekniğidir. Geometrik (Döndürme, Kırpma, Yatay/Dikey Aynalama, Ölçekleme) ve Fotometrik (Parlaklık, Kontrast, Renk Uzayı değişimleri) olarak ikiye ayrılır.

   Veri artırımı yaparken projenin mantığına dikkat edilmelidir. Örneğin, bir otonom sürüş veya drone hedef tespit projesinde resimleri dikey aynalamak (ters çevirmek) mantıklı olabilir; ancak bir plaka okuma veya el yazısı tanıma projesinde resmi ters çevirmek "b" harfini "d" yapacağı için modeli tamamen bozabilir.

### Model Değerlendirme Metrikleri

   Bir modelin başarısı sadece "Doğruluk (Accuracy)" değeriyle ölçülemez, özellikle dengesiz veri kümelerinde (örn: 1000 resmin 990'ı sağlam, 10'u kanserli hücreyse model her şeye sağlam deyip %99 accuracy alabilir ama başarısızdır). Bu yüzden Karmaşıklık Matrisi (Confusion Matrix) tabanlı metrikler kullanılır:

* **Accuracy (Doğruluk):** Toplam doğru tahminlerin tüm veri kümesine oranı.

* **Precision (Keskinlik):** Modelin "Pozitif" dediği tahminlerin ne kadarının gerçekten pozitif olduğu. (Yanlış alarm vermeme başarısı).

* **Recall (Duyarlılık):** Gerçekteki pozitiflerin ne kadarını modelin yakalayabildiği. (Gözden kaçırmama başarısı).

* **F1-Score:** Precision ve Recall değerlerinin harmonik ortalamasıdır. İkisi arasındaki dengeyi gösterir.

* **mAP (Mean Average Precision):** Özellikle YOLO gibi nesne tespiti modellerinde, sınırlayıcı kutuların (bounding box) doğruluğunu ve sınıf başarısını bir arada ölçen endüstri standardı metriktir.

### Düzenlileştirme ve Ezberleme (Overfitting/Underfitting)

* **Overfitting (Ezberleme):** Modelin eğitim verisinde sıfır hata yaparken, ilk defa gördüğü validasyon/test verisinde çuvallamasıdır. Eğitim kaybı düşerken validasyon kaybının yükselmesiyle grafiklerde kendini belli eder.

* **Underfitting (Öğrenememe):** Modelin kapasitesinin yetersiz olması veya az eğitilmesi sebebiyle hem eğitim hem validasyon setinde yüksek hata payına sahip olmasıdır.

**Çözüm Yöntemleri (Regularization):**

1.  **Batch Normalization:** Katmanların girdilerini küçük paketler (batch) halinde normalize ederek eğitimin çok daha kararlı ve hızlı olmasını sağlar.

2.  **Dropout:** Eğitim sırasında her adımda belirlenen orandaki ( örn: %30) nöronu rastgele devre dışı bırakır. Böylece ağdaki nöronlar birbirine bağımlı hale gelmez, hepsi sorumluluk alarak ezberlemeyi engeller.

3.  **L1/L2 Regularization:** Kayıp fonksiyonuna ağırlıkların büyüklüğünü ceza olarak ekler. Büyük ağırlıkları baskılayarak modelin daha esnek/sade kalmasını sağlar.





