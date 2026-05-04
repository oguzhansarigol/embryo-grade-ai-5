

# DeepEmbryo — ConvNeXt-Base IVF Embriyo Kalite Sınıflandırma Sistemi

## Proje Özeti
5. gün (blastosist) embriyo görüntülerini Gardner sınıflandırma sistemine göre
otomatik olarak sınıflandıran, açıklanabilir yapay zeka destekli bir sistem geliştir.
Temel model olarak **ConvNeXt-Base** kullanılacak (ImageNet ön-eğitimli, transfer learning).

---

## Teknoloji Yığını
- Python 3.10+
- PyTorch + torchvision (ConvNeXt-Base)
- timm kütüphanesi (model yükleme için)
- OpenCV, Pillow (görüntü işleme)
- scikit-learn (metrikler)
- matplotlib, seaborn (görselleştirme)
- pytorch-grad-cam (Grad-CAM)
- Flask (web arayüzü)
- SQLite (veritabanı)
- ReportLab (PDF çıktı)

---

## Klasör Yapısı
Şu klasör yapısını oluştur:

deepembryo/
├── data/
│   ├── raw/                  # Ham görüntüler (sınıf klasörleri: 3AA, 4AB, 5BC vb.)
│   ├── processed/            # Ön işlenmiş görüntüler
│   └── splits/               # train / val / test JSON dosyaları
├── src/
│   ├── dataset.py            # EmbryoDataset sınıfı (augmentation dahil)
│   ├── model.py              # ConvNeXt-Base model tanımı
│   ├── train.py              # Eğitim döngüsü
│   ├── evaluate.py           # Test ve metrik hesaplama
│   ├── gradcam.py            # Grad-CAM görselleştirme
│   └── utils.py              # Yardımcı fonksiyonlar
├── app/
│   ├── app.py                # Flask uygulaması
│   ├── templates/            # HTML şablonları
│   └── static/               # CSS/JS dosyaları
├── outputs/
│   ├── models/               # .pth model ağırlıkları
│   ├── plots/                # Eğitim grafikleri
│   └── reports/              # PDF raporlar
├── requirements.txt
└── README.md

---

## 1 — Veri Seti ve Ön İşleme (dataset.py)
Şunları uygula:
- Görüntüleri 224x224 piksel boyutuna yeniden boyutlandır
- Normalizasyon: ImageNet mean/std ([0.485,0.456,0.406] / [0.229,0.224,0.225])
- Eğitim seti augmentation:
  * RandomHorizontalFlip(p=0.5)
  * RandomVerticalFlip(p=0.5)
  * RandomRotation(degrees=15)
  * ColorJitter(brightness=0.2, contrast=0.2)
- Veri bölümü: %70 eğitim / %15 validasyon / %15 test (stratified split)
- EmbryoDataset: torch Dataset sınıfı olarak yaz, label mapping JSON'a kaydet

---

## 2 — Model Mimarisi (model.py)
Şunları uygula:
- timm.create_model("convnext_base", pretrained=True) ile yükle
- Son sınıflandırıcı katmanı (head) Gardner sınıf sayısına göre değiştir
- Sınıflar: 3AA, 3AB, 3BA, 3BB, 4AA, 4AB, 4BA, 4BB, 5AA, 5AB, 5BA, 5BB
  (ve alt kalite varyantları — veri setine göre genişletilebilir)
- Çıktı: Softmax olasılık vektörü
- Modeli EmbryoClassifier sınıfı olarak sarmalayın

---

## 3 — Eğitim Döngüsü (train.py)
Şunları uygula:
- Kayıp fonksiyonu: CrossEntropyLoss (isteğe bağlı class weighting)
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-4)
- LR scheduler: CosineAnnealingLR
- Early stopping: validasyon kaybı 10 epoch iyileşmezse dur, en iyi modeli kaydet
- Her epoch sonunda: train loss, val loss, val accuracy logla
- Checkpoint: her epoch en iyi model .pth olarak outputs/models/ altına kaydet
- Eğitim tamamlanınca outputs/plots/ altına accuracy ve loss grafiklerini (train vs val) PNG olarak kaydet

---

## 4 — Değerlendirme (evaluate.py)
Test seti üzerinde şunları hesapla ve kaydet:
- Confusion Matrix (seaborn heatmap, PNG)
- Classification Report: Precision, Recall, F1 — her sınıf için ayrı + weighted avg
- Genel Accuracy
- Çıktıları hem terminale yazdır hem outputs/reports/metrics.json dosyasına kaydet

---

## 5 — Grad-CAM Görselleştirme (gradcam.py)
Şunları uygula:
- pytorch-grad-cam kütüphanesi ile GradCAM oluştur
- Hedef katman: ConvNeXt-Base'in son evrişim bloğu (stages[-1])
- Tahmin görüntüsünün yanına ısı haritasını (jet colormap) üst üste bindir
- Softmax olasılığı < 0.70 ise uyarı bayrağı ekle: "Düşük güvenilirlik — manuel kontrol gerekli"
- Çıktı: orijinal görüntü + Grad-CAM bindirmeli görüntü yan yana PNG

---

## 6 — Flask Web Arayüzü (app/app.py)
Şunları uygula:
- Ana sayfa: tekli görsel yükleme formu
- Batch sayfası: klasör/zip yükleme ile toplu işlem
- Tahmin sonuç sayfası:
  * Tahmin edilen Gardner skoru (örn. 5AA)
  * Softmax güven skoru (%)
  * Grad-CAM görüntüsü
  * Düşük güvenilirlik uyarısı (gerekiyorsa)
- Sonuçları SQLite veritabanına kaydet (timestamp, dosya adı, tahmin, güven)
- PDF/CSV indirme butonu (tüm geçmiş tahminler)

---

## 7 — Çıktı Gereksinimleri
Kodun tamamlanınca şunların var olduğunu doğrula:
- [ ] outputs/models/best_model.pth
- [ ] outputs/plots/accuracy_loss_curve.png
- [ ] outputs/plots/confusion_matrix.png
- [ ] outputs/reports/metrics.json
- [ ] Örnek bir görüntü için Grad-CAM çıktısı
- [ ] Flask uygulaması localhost:5000 üzerinde çalışıyor

---

## Kod Kalitesi
- Her modül için docstring yaz
- Type hints kullan (Python 3.10+ syntax)
- Tüm sabit değerleri (LR, batch size, epoch sayısı) config.py dosyasında topla
- requirements.txt dosyasını pin edilmiş versiyonlarla oluştur
- README.md: kurulum, veri hazırlama ve çalıştırma adımlarını anlat

---

## Başlangıç Noktası
İlk olarak klasör yapısını ve requirements.txt dosyasını oluştur,
ardından config.py → dataset.py → model.py → train.py sırasını takip et.
Projeyi Oluştur ↗
Dataset Kodu ↗
Flask Arayüzü ↗