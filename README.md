# DeepEmbryo — 5. Gün Embriyo Kalite Değerlendirme Sistemi

IVF tedavisinde 5. gün blastosist embriyo görüntülerini Gardner sınıflandırma
sistemine göre otomatik değerlendiren derin öğrenme tabanlı bir sistem
(Tıbbi Görüntülemede Yapay Zeka Uygulamaları dersi projesi).

- **Model:** ConvNeXt-Base (ImageNet-22k pretrained, transfer learning)
- **Veri:** 170 .bmp görüntü, 4 sınıf (3AA, 3CC, 4AA, Cleavage)
- **Strateji:** Stratified 70/15/15 train/val/test split, class weights, 2 aşamalı eğitim
- **XAI:** Grad-CAM + güven uyarı sistemi + morfolojik özellik raporu

---

## Hızlı Başlangıç

### A) Eğitim — Google Colab (önerilen)

1. **Zip oluştur:** `code-base/` klasörünü (içinde `EMBRIO GRADE DATASET/` dahil) lokalde `code-base.zip` olarak sıkıştır.
2. **Colab'i aç:** Boş bir Colab oturumu başlat → **Runtime → Change runtime type → T4 GPU** seç.
3. **Zip'i yükle:** Soldaki dosya panelinden `code-base.zip`'i `/content/` altına sürükle.
4. **Notebook'u aç:** `code-base/notebooks/DeepEmbryo_Colab.ipynb`'i Colab'de aç (veya GitHub/yerelden yükle) ve hücreleri sırayla çalıştır. Toplam süre ~30-45 dk.
5. Notebook'un son hücresi `DeepEmbryo_Teslim.zip` dosyasını otomatik indirecek. Colab oturumu kapatılırsa kaybolur — indirmeyi atlama.

### B) Sadece çıkarım (yerel makine)

```bash
pip install -r requirements.txt
python -c "from src.infer import EmbryoPredictor; \
  p = EmbryoPredictor('outputs/checkpoints/final_model.pth'); \
  print(p.predict_single('EMBRIO GRADE DATASET/4AA/4AA (2).bmp'))"
```

### C) Flask demo (yerel)

Eğitilmiş `final_model.pth` dosyasını `outputs/checkpoints/` altına koy, sonra:

```bash
python app/app.py
# tarayıcıdan: http://localhost:5000
```

---

## Proje Yapısı

```
code-base/
├── src/                       # ML pipeline
│   ├── config.py              # Hiperparametreler ve yollar
│   ├── data.py                # Dataset, transforms, train/val/test split, class weights
│   ├── model.py               # ConvNeXt-Base + freeze utilities
│   ├── train.py               # 2 aşamalı eğitim (warmup + fine-tune)
│   ├── evaluate.py            # CM, classification report, learning curve
│   ├── gradcam.py             # Grad-CAM ısı haritaları
│   ├── morphology.py          # Morfolojik özellik raporu
│   └── infer.py               # Tek/batch tahmin + güven uyarısı
├── app/                       # Flask demo + SQLite history
│   ├── app.py
│   ├── db.py
│   └── templates/
├── notebooks/
│   └── DeepEmbryo_Colab.ipynb # Uçtan uca Colab eğitim notebook'u
├── outputs/                   # Eğitim çıktıları
│   ├── checkpoints/           # best_model.pth + final_model.pth
│   ├── figures/               # Acc-loss, CM, learning curve, Grad-CAM
│   ├── reports/               # JSON + CSV metrik raporları
│   └── logs/                  # history.csv
├── requirements.txt
├── package.py                 # Teslim arşivi üretici
└── README.md
```

---

## İsterler Dokümanı Eşleştirmesi

| Doküman maddesi | Karşılayan dosya |
|---|---|
| §2.2 Ön işleme + augmentation | `src/data.py` (albumentations) |
| §3.1 ConvNeXt-Base + transfer learning | `src/model.py` |
| §3.2 Stratified 70/15/15 train/val/test split | `src/data.py::get_train_val_test_split` |
| §3.2 Early stopping, optimizer, loss | `src/train.py` |
| §4.1 Accuracy-loss + learning curve | `src/evaluate.py::plot_history`, `learning_curve` |
| §4.2 Confusion matrix + P/R/F1 | `src/evaluate.py::plot_confusion_matrix`, `classification_report` |
| §4.3 / §5.1 Grad-CAM | `src/gradcam.py`, `src/infer.py` |
| §5.2 Güven < 0.70 uyarısı | `src/infer.py::_warning_for` |
| §5.3 Morfolojik özellik raporu | `src/morphology.py` |
| §6 Tek/batch UI + DB + PDF/CSV | `app/app.py`, `app/db.py`, `app/templates/` |
| §7 Teslim arşivi | `package.py` |

---

## Hiperparametreler (özet)

| Parametre | Değer |
|---|---|
| Görüntü boyutu | 224x224 |
| Batch size | 16 |
| Warmup epoch (head only) | 5 @ LR=1e-3 |
| Fine-tune epoch | 45 (head LR=1e-4, backbone LR=1e-5) |
| Optimizer | AdamW, weight_decay=0.05 |
| Scheduler | CosineAnnealingLR |
| Loss | CE + class weights + label smoothing 0.1 |
| Early stopping patience | 10 epoch |
| Train/Val/Test | 70 / 15 / 15 (stratified, seed=42) |
| Güven eşiği (uyarı) | 0.70 |
