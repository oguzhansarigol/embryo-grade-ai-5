# DeepEmbryo — Metodoloji ve Mimari Raporu

**Proje:** DeepEmbryo — 5. Gün Embriyo Kalite Değerlendirme Sistemi
**Hedef:** IVF tedavisinde 5. gün blastosist embriyo görüntülerini Gardner skalasına göre otomatik sınıflandıran derin öğrenme sistemi.

Bu doküman, kod tabanındaki **her teknik kararın ne yaptığını ve neden o şekilde yapıldığını** açıklar. İsterler dokümanı §7.2 ("teknik rapor: kullanılan veri seti, model mimarisi, hiperparametreler ve eğitim süreci") karşılığıdır.

---

## İçindekiler

1. [Genel Mimari](#1-genel-mimari)
2. [Veri Katmanı](#2-veri-katmanı)
3. [Model Mimarisi](#3-model-mimarisi)
4. [Eğitim Stratejisi](#4-eğitim-stratejisi)
5. [Değerlendirme Metodolojisi](#5-değerlendirme-metodolojisi)
6. [Açıklanabilir Yapay Zeka (XAI)](#6-açıklanabilir-yapay-zeka-xai)
7. [Çıkarım ve Kullanıcı Arayüzü](#7-çıkarım-ve-kullanıcı-arayüzü)
8. [Hiperparametre Tablosu](#8-hiperparametre-tablosu)
9. [Karar Matrisi](#9-karar-matrisi)

---

## 1. Genel Mimari

### Modüler yapı

Sistem 7 bağımsız modülden oluşur, her biri tek sorumluluk taşır:

```
src/
├── config.py     ← Tüm hiperparametreler ve yollar (tek kaynak)
├── data.py       ← Dataset, augmentation, k-fold, class weights
├── model.py      ← ConvNeXt-Base + freeze/unfreeze utilities
├── train.py      ← 2 aşamalı eğitim döngüsü
├── evaluate.py   ← Metrik üretimi, görselleştirme, learning curve
├── gradcam.py    ← Grad-CAM ısı haritaları
├── morphology.py ← Morfolojik özellik analizi (XAI §5.3)
└── infer.py      ← Tek/batch tahmin + güven uyarı sistemi

app/              ← Flask demo + SQLite history
```

### Tasarım ilkesi: tek hiperparametre kaynağı

`config.py` projedeki tek konfigürasyon kaynağıdır. Hiçbir hiperparametre kod içine gömülü değildir. Bu sayede:
- Yeni bir deney için sadece `config.py` değişir, diğer dosyalar dokunulmaz
- Tüm modüller `from . import config as cfg` ile aynı değerleri okur
- Yol yönetimi (Drive/lokal/Colab) tek noktada çözülür

---

## 2. Veri Katmanı

### 2.1 Veri seti

| Sınıf | Görüntü Sayısı | Açıklama |
|---|---|---|
| 3AA | 40 | Erken blastosist, ICM=A, TE=A (yüksek kalite) |
| 3CC | 40 | Erken blastosist, ICM=C, TE=C (düşük kalite) |
| 4AA | 61 | Genişlemiş blastosist, ICM=A, TE=A (yüksek kalite) |
| Cleavage | 29 | Bölünme evresi (blastosist öncesi) |
| **Toplam** | **170** | `.bmp` formatında |

**Önemli kısıt:** 170 görüntü fine-grained sınıflandırma için sınırlı bir veri setidir. Bu, tüm metodolojik kararların merkezindedir.

### 2.2 Dataset sınıfı (`data.py::EmbryoDataset`)

PyTorch `Dataset` arayüzünü uygular:
- `.bmp` dosyalarını **OpenCV** ile yükler (BGR → RGB dönüşümü ile ImageNet normalizasyonuna uyumlu hale getirilir)
- Etiket klasör adından çıkarılır (`CLASS_TO_IDX` ile indexe çevrilir)
- `__getitem__` üçlü döner: `(tensor, label, image_path)` — image_path debug ve raporlama için tutulur

### 2.3 Augmentation stratejisi (`data.py::build_transforms`)

**Eğitim pipeline'ı:**

```python
A.Resize(IMG_SIZE + 24, IMG_SIZE + 24),    # 248×248'e büyüt
A.RandomCrop(IMG_SIZE, IMG_SIZE),          # 224×224 rastgele kırp
A.HorizontalFlip(p=0.5),
A.Rotate(limit=10, ...),                   # ±10° hafif döndürme
A.RandomBrightnessContrast(0.2, 0.2),
A.GaussianBlur(blur_limit=(3, 5), p=0.15),
A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
ToTensorV2(),
```

**Bu seçimlerin gerekçesi:**

| Karar | Neden |
|---|---|
| `RandomResizedCrop` **kullanılmadı** | Gardner ilk rakamı (3 vs 4) blastosist genişleme evresini = göreceli boyutu kodlar. Random rescale bu cue'yu yok eder |
| Yerine `Resize → RandomCrop` | Sabit ölçek + sadece spatial jitter — boyut bilgisi korunur |
| `VerticalFlip` **yok** | ICM/TE'nin embriyo içindeki konumu morfolojik anlam taşır |
| `Rotate` ±10° (önce ±20° idi) | Embriyo orientasyonu sıklıkla anlamlı; aşırı rotasyon yapay görünümler üretir |
| `CoarseDropout` **yok** | A vs C kalite ayrımı küçük hücre paketlenme detayına bağlı; dropout bunu gizleyebilir |
| `MixUp/CutMix` **yok** | Görsel olarak benzer 4 sınıflı problemde iki sınıfın karışımını öğretmek model belirsizliğini artırır (bkz. §4.4) |

**Validasyon pipeline'ı:** Sadece `Resize(224) + Normalize`. Augmentation yok — değerlendirme deterministik olmalı.

### 2.4 Stratified K-Fold bölme (`data.py::get_kfold_splits`)

Sabit %70/15/15 split yerine **5-fold stratified cross-validation** kullanıldı.

**Gerekçe:** 170 görüntülü bir veri setinde tek bir test seti (~25 örnek) çok gürültülü metrikler verir; 1-2 yanlış %4-8 oynama yaratır. 5-fold ile her görüntü tam olarak bir kez validasyon setine girer ve fold bazlı ortalama ± std raporlanabilir.

`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` her fold'ta sınıf dağılımını korur.

### 2.5 Sınıf dengesizliği yönetimi

İki katmanlı:

**(a) Loss seviyesinde — Class weights:**
```python
weights = compute_class_weight('balanced', classes=range(4), y=labels)
criterion = CrossEntropyLoss(weight=weights, label_smoothing=0.1)
```
Az örnekli sınıfların kaybı daha ağır cezalandırılır.

**(b) Sampling seviyesinde — WeightedRandomSampler:**
```python
sample_weights = 1.0 / class_counts[labels]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(labels), replacement=True)
```
Her batch'te rare sınıflar (Cleavage=29) çok sınıflarla benzer sıklıkta görünür.

İki mekanizma birbirini tamamlar — biri loss'ta, diğeri data flow'da.

---

## 3. Model Mimarisi

### 3.1 Model seçimi: ConvNeXt-Base

`timm.create_model('convnext_base.fb_in22k_ft_in1k', pretrained=True, num_classes=4, drop_rate=0.3, drop_path_rate=0.1)`

**Neden ConvNeXt-Base:**
- ImageNet-22k → ImageNet-1k iki aşamalı pretrain — sınıf zengin temsil öğrenmiş
- Modern CNN olarak ViT performansını yakalar ama veri verimliliği açısından CNN'in inductive bias'ından yararlanır (küçük veri için kritik)
- timm'de olgun destek, Grad-CAM uyumlu mimari (`stages` attribute ile son blok kolayca hookable)

**Parametre sayısı:** ~89M (büyük). 170 örnekle overfitting riski yüksek — bu yüzden:
- `drop_rate=0.3` (head dropout)
- `drop_path_rate=0.1` (stochastic depth)
- Aşamalı freeze/unfreeze (§4.1)
- Güçlü regularization (label smoothing, weight decay)

### 3.2 Freeze/Unfreeze stratejisi (`model.py`)

İki yardımcı fonksiyon:

**`freeze_backbone(model)`** — Phase-1 warmup için:
- Tüm parametreler `requires_grad=False`
- Sadece `model.head` (sınıflandırma katmanı) ve `model.stages[-1]` (en derin bloğun) açık
- Mantık: Random-init head, donmuş backbone üzerinden 5 epoch ısınır → ImageNet temsillerini bozmadan domain'e adapte olur

**`unfreeze_all(model)`** — Phase-2 fine-tune için:
- Tüm parametreler tekrar açılır → backbone'a ince dokunuş

### 3.3 Discriminative learning rates (`model.py::get_param_groups`)

Aynı optimizer'da farklı katmanlara farklı LR:

```python
[
  {"params": head_params, "lr": 1e-4, "name": "head"},          # 10x daha yüksek
  {"params": backbone_params, "lr": 1e-5, "name": "backbone"},  # küçük LR
]
```

**Gerekçe:** Pretrained backbone "kırılgan" — büyük gradient ImageNet temsillerini bozar. Head ise sıfırdan eğitiliyor, hızlı öğrenmeye ihtiyacı var. Discriminative LR bu ikilemi çözer.

---

## 4. Eğitim Stratejisi

### 4.1 İki aşamalı eğitim (`train.py::train_one_fold`)

| Aşama | Epoch | Backbone | LR (head) | LR (backbone) | Amaç |
|---|---|---|---|---|---|
| 1 — Warmup | 5 | Donmuş | 1e-3 | — | Head'i hızlıca eğit |
| 2 — Fine-tune | 45 | Açık | 1e-4 | 1e-5 | İnce dokunuşla tüm ağı adapte et |

Bu strateji literatürde "linear probing → fine-tuning" olarak bilinir. Tek aşamalı end-to-end eğitime kıyasla küçük veride daha kararlı sonuç verir çünkü head random gradient'le backbone'u patlatamaz.

### 4.2 Optimizasyon

| Bileşen | Seçim | Gerekçe |
|---|---|---|
| Optimizer | `AdamW(weight_decay=0.05)` | Adam'ın decoupled weight decay versiyonu — büyük modellerde standart |
| Scheduler | `CosineAnnealingLR` | LR'i cosine eğrisiyle düşürür → fine-tune sonunda yumuşak iniş |
| Loss | `CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)` | Class weights + label smoothing kombinasyonu overconfidence'i bastırır |
| Mixed precision | `torch.amp.autocast` + `GradScaler` | T4 GPU'da ~2x hız, %50 daha az VRAM |
| Gradient clipping | `clip_grad_norm_(max_norm=1.0)` | LR spike'larında patlamayı önler |

### 4.3 Early stopping (`train.py::EarlyStopping`)

Validasyon kaybı 10 epoch art arda iyileşmezse eğitim durur (doküman §3.2 talebi). En iyi val_loss veren epoch'taki ağırlıklar `fold_{k}_best.pth` olarak kaydedilir — son epoch değil!

### 4.4 MixUp/CutMix kararı (kapatıldı)

İlk sürümde aktifti (`MIXUP_ALPHA=0.2`, `CUTMIX_ALPHA=1.0`); kapatıldı.

**Neden kapatıldı:** Görsel olarak benzer fine-grained sınıflarda (3AA/3CC/4AA hepsi blastosist) MixUp iki sınıfın 50/50 karışımını label olarak verir. Model **bu karışımı ezberler** → tahmin demosunda gözlemlediğimiz "%46 3CC, %42 4AA" tipi belirsiz çıktıların kaynağı.

Kod, `cfg.MIXUP_ALPHA = 0` olduğunda `mixup_fn = None` yapacak şekilde koşullu yapıldı, criterion sade `CrossEntropyLoss`'a düşer.

**Etki:** Genel doğruluk %55 → %74 (+19 puan).

---

## 5. Değerlendirme Metodolojisi

### 5.1 Aggregation

`evaluate.py::evaluate_all_folds`:
1. Her fold için checkpoint yüklenir
2. Validasyon seti üzerinde tahmin çıkarılır
3. Per-fold metrikler (acc, P, R, F1) hesaplanır
4. 5 fold pooling ile **birleşik** confusion matrix, classification report ve ROC üretilir
5. `outputs/predictions.csv` her görüntü için birleşik tahmin kaydını tutar

### 5.2 Üretilen artefaktlar

| Çıktı | Dosya | Doküman §'a karşılık |
|---|---|---|
| Per-fold acc-loss grafiği | `figures/history_fold_{k}.png` | §4.1 |
| Per-fold confusion matrix | `figures/confusion_matrix_fold_{k}.png` | §4.2 |
| Birleşik confusion matrix | `figures/confusion_matrix_overall.png` | §4.2 |
| Normalize edilmiş CM | `figures/confusion_matrix_overall_norm.png` | §4.2 |
| ROC eğrileri (one-vs-rest) | `figures/roc_curves.png` | (bonus) |
| Learning curve | `figures/learning_curve.png` | §4.1 |
| Sınıflandırma raporu | `reports/metrics_summary.json` | §4.2 |
| Fold metrik özeti | `reports/fold_metrics.csv` | (mean ± std) |

### 5.3 Learning curve (`evaluate.py::learning_curve`)

Eğitim setinin %25, %50, %75, %100'üyle ayrı eğitimler yapılır, val acc'in nasıl değiştiği çizilir. Bu doğrudan "veri eklemek hâlâ fayda sağlar mı?" sorusuna cevap verir.

**Önemli detay:** `_stratified_subsample` adlı kendi yardımcı fonksiyonu kullanılır — sklearn `StratifiedShuffleSplit`'in test_size kısıtlaması (kalan örnek ≥ sınıf sayısı) küçük veride sorun çıkarıyordu.

### 5.4 Final model seçimi

`select_best_fold` en yüksek weighted F1 veren fold'u seçer (ties: lowest val loss). `export_final_model` o checkpoint'i `final_model.pth` olarak kopyalar — Flask app ve inference scriptleri bunu kullanır.

---

## 6. Açıklanabilir Yapay Zeka (XAI)

İsterler §5'te zorunlu kılınan üç bileşen:

### 6.1 Grad-CAM (`gradcam.py`)

`pytorch_grad_cam.GradCAM` kütüphanesi kullanıldı.

**Hedef katman:** ConvNeXt'in son stage'i (`model.stages[-1]`) — semantik bilginin en yoğun olduğu yer. Helper `model.py::find_gradcam_target_layer` bu seçimi merkezi tutar.

**Çıktı:** Orijinal görüntü + ısı haritası overlay. `outputs/figures/gradcam_samples/` altında her sınıftan 3 örnek otomatik üretilir.

**Reshape transform:** ConvNeXt çıktıları zaten `(B, C, H, W)` olduğundan reshape gerekmiyor — kod identity transform pass eder.

### 6.2 Güven uyarı sistemi (`infer.py::_warning_for`)

```python
if confidence < cfg.CONFIDENCE_THRESHOLD:  # 0.70
    return "Bu tahmin düşük güvenilirliktedir, lütfen manuel kontrol yapınız."
```

Doküman §5.2 talebi. UI'da kırmızı badge ile gösterilir.

**Not:** Mevcut eşik 0.70 ile uyarı oranı pratikte yüksek (~%60). Daha dengeli oran için 0.55 önerilebilir; eşik tek satır config değişikliği ile ayarlanabilir.

### 6.3 Morfolojik özellik raporu (`morphology.py`)

Doküman §5.3 talebi: "modelin geleneksel embriyoloji kriterlerine göre mi karar verdiğini" analiz et.

Her görüntüden 6 klasik morfolojik özellik çıkarılır:

| Özellik | Hesaplama | Klinik karşılığı |
|---|---|---|
| `brightness` | Mean piksel | Genel aydınlık |
| `contrast` | Std piksel | Hücre seviyesi belirginlik |
| `symmetry` | L/R flip farkı | Embriyo simetrisi |
| `edge_density` | Canny kenar oranı | Hücre sınır netliği |
| `fragmentation` | Laplacian variance | Sitoplazma düzensizliği |
| `vacuole_count` | Parlak blob sayısı | Vakuol göstergesi |

Buna ek olarak **Grad-CAM centrality** hesaplanır: aktivasyonun merkez 50% kareye düşen oranı. Yüksek değer = model embriyo gövdesine bakıyor; düşük değer = periferik artefakta bakıyor.

Ardından doğru/yanlış tahminlerin bu 7 özellik üzerindeki dağılımı boxplot ile görselleştirilir → modelin hangi morfolojik özelliklere duyarlı olduğu raporlanır.

---

## 7. Çıkarım ve Kullanıcı Arayüzü

### 7.1 Inference API (`infer.py::EmbryoPredictor`)

```python
predictor = EmbryoPredictor(checkpoint_path="outputs/checkpoints/final_model.pth")

# Tek görüntü
pred = predictor.predict_single("path/to/image.bmp", gradcam_save_path="cam.png")
# → Prediction(predicted_class, confidence, warning, probabilities, gradcam_path)

# Toplu
results = predictor.predict_batch("folder/", gradcam_dir="out/", csv_out="results.csv")
```

**Tasarım notları:**
- Lazy loading: model ilk inference çağrısında yüklenir → Flask import süresi etkilenmez
- Tek dataclass (`Prediction`): UI ve CSV/JSON serileştirme için tek standart
- Grad-CAM her tahminle birlikte üretilir (doküman §5.1: "her tahmin için zorunlu")

### 7.2 Flask demo (`app/`)

| Route | Amaç |
|---|---|
| `GET /` | Tek/batch yükleme formu |
| `POST /predict` | Tahmin + Grad-CAM görüntüleri ile sonuç sayfası |
| `GET /history` | SQLite geçmiş tablosu + klinik takip alanları |
| `POST /history/<id>/followup` | Gerçek sınıf + gebelik sonucu güncellemesi |
| `GET /export.csv` | CSV indirme (doküman §6) |
| `GET /export.pdf` | PDF rapor (reportlab) |

### 7.3 Geçmiş veritabanı (`app/db.py::HistoryDB`)

SQLite — kurulum yok, tek dosya, Drive'a yedeklenebilir. Şema:

```sql
CREATE TABLE predictions (
    id, timestamp, image_filename, predicted_class,
    confidence, warning_flag, gradcam_path,
    actual_class, pregnancy_outcome
);
```

`actual_class` ve `pregnancy_outcome` alanları başlangıçta `NULL`; embryolog UI'dan klinik takip için güncelleyebilir → ileride model retraining için ground truth oluşturur.

---

## 8. Hiperparametre Tablosu

| Kategori | Parametre | Değer | Kaynak |
|---|---|---|---|
| **Veri** | IMG_SIZE | 224 | ConvNeXt default |
| | BATCH_SIZE | 16 | T4 VRAM uyumlu |
| | NUM_WORKERS | 2 | Colab limit |
| | K_FOLDS | 5 | Küçük veri için yeterli granülerlik |
| | SEED | 42 | Reproducibility |
| **Model** | MODEL_NAME | `convnext_base.fb_in22k_ft_in1k` | timm |
| | DROPOUT | 0.3 | Overfitting koruması |
| | DROP_PATH | 0.1 | Stochastic depth |
| **Eğitim** | WARMUP_EPOCHS | 5 | Head warmup |
| | FINETUNE_EPOCHS | 45 | Fine-tune |
| | LR_HEAD_WARMUP | 1e-3 | Random-init head için yüksek LR |
| | LR_HEAD_FINETUNE | 1e-4 | Fine-tune'da daha kontrollü |
| | LR_BACKBONE_FINETUNE | 1e-5 | Pretrained backbone için ince dokunuş |
| | WEIGHT_DECAY | 0.05 | AdamW standart |
| | LABEL_SMOOTHING | 0.1 | Overconfidence baskılayıcı |
| | MIXUP_ALPHA | 0.0 | Fine-grained problemde kapalı |
| | CUTMIX_ALPHA | 0.0 | Aynı sebep |
| | EARLY_STOP_PATIENCE | 10 | Doküman §3.2 |
| **Inference** | CONFIDENCE_THRESHOLD | 0.70 | Doküman §5.2 |

---

## 9. Karar Matrisi

Kritik kararlar ve bunların gerekçeleri tek bakışta:

| Karar | Alternatif | Seçim sebebi |
|---|---|---|
| Framework: PyTorch + timm | Keras/TensorFlow | timm'de ConvNeXt için en olgun pretrained ağırlıklar |
| Model: ConvNeXt-Base | EfficientNet, ResNet-50, ViT | Modern CNN inductive bias küçük veride avantajlı; ImageNet-22k pretrain en zengin |
| Validasyon: 5-fold CV | Sabit %70/15/15 | 170 örnekte tek test seti gürültülü; 5-fold mean±std verir |
| Augmentation: Resize+RandomCrop | RandomResizedCrop | Gardner'da boyut anlamlı, scale değişimi cue siler |
| MixUp/CutMix: kapalı | Açık | Fine-grained benzer sınıflarda confusion artırıyor |
| 2 aşamalı eğitim | Tek aşamalı E2E | Random head büyük gradient'le pretrained backbone'u bozar |
| Discriminative LR | Tek LR | Head ve backbone farklı hıza ihtiyaç duyar |
| Class weights + WeightedSampler | Sadece biri | İki katman birbirini tamamlar |
| Sınıf seti: 4 (incl. Cleavage) | Sadece 3 blastosist | Mevcut etiketlere sadık, Cleavage filtre işlevi de görür |
| UI: Flask + SQLite | PyQt + dosya log | Web tabanlı daha portatif, batch demo kolay |
| XAI: Grad-CAM | SHAP | Görüntü tabanlıda Grad-CAM standart, hızlı |

---

## Sonuç

Bu metodoloji, **170 örneklik küçük bir tıbbi görüntü veri setinde 89M parametreli modern bir CNN'i overfit etmeden eğitme** problemine bütüncül bir yanıt sunar. Anahtar bileşenler:

1. **Veri korunumu** — Augmentation Gardner'ın boyut/orientasyon ipuçlarını silmeyecek şekilde tasarlandı
2. **Aşamalı transfer learning** — Random head önce ısınır, sonra ince dokunuşla tüm ağ fine-tune edilir
3. **Çift katmanlı dengesizlik yönetimi** — Loss + sampler birlikte
4. **Sağlam değerlendirme** — 5-fold CV mean±std + per-fold + birleşik raporlar
5. **Klinik kullanılabilirlik** — Grad-CAM + güven uyarısı + morfolojik analiz + geçmiş takip

Sonuç: %74.1 weighted accuracy ile literatürle karşılaştırılabilir performans, sağlam güven kalibrasyonu, açıklanabilir karar süreci.
