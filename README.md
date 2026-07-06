### Proje Araştırma Raporu (trex\_research)

# 



Rapor Yazarı: İlterişhan Öztekin



Bu repo, görüntü işleme, yapay zekâ, temel backend mimarisi (API) ve versiyon kontrol sistemleri (Git) üzerine yapılan araştırma ve değerlendirme sonuçlarını içermektedir. Rapor genel hatlarıyla teorik bilgileri, sektördeki karşılıklarını ve kişisel çıkarımları barındırmaktadır.







İçerik



1\. Görüntü İşlemeye Giriş

2\. Python ve Görüntü İşleme Ekosistemi

3\. Yapay Zekâ ve Görüntü İşleme							

4\. Temel API Bilgisi

5\. Git ve Yazılım Geliştirme Süreci





&#x20;  

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





&#x20;  Kişisel düşüncem, git ve github benim için çok bir kullanım kolaylığı sağlamadığı yönünde çünkü dosyaları ve kodları kendi başıma toplamak bana daha çok güven veriyor.





6\.Genel Değerlendirme ve Sonuç



&#x20;  Bu araştırma süreci, bir görüntü işleme projesinin sadece ham bir resmi manipüle etmekten ibaret olmadığını; Görüntü İşleme -> Bilgisayarlı Görü -> Derin Öğrenme(YOLO/ResNet) -> API(REST/JSON) -> Dağıtım/Versiyonlama(Git) adımlarıyla uçtan uca bir yazılım mimarisi gerektirdiğini açıkça göstermektedir.



&#x20;  Özellikle gerçek zamanlı sistemlerde(UCAV / SİHA hedef tespit yazılımları gibi) modelin hızı(YOLO) ve verilerin backend ile kayıpsız/hızlı iletilmesi(REST API) projenin başarısını doğrudan belirleyen kritik yapı taşlarıdır.



