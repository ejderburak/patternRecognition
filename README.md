# patternRecognition — Hand Gesture Recognition (MediaPipe + Classic ML)

Bu proje, webcam görüntülerinden el hareketlerini (gesture) tanımak için uçtan uca bir akış sunar:

1. **Veri toplama**: Webcam ile belirli hareketler için fotoğraf çekme (`data_collection.ipynb`)
2. **Özellik çıkarma**: MediaPipe Hand Landmarker ile el landmark (21 nokta) çıkarıp CSV oluşturma (`feature_extraction.ipynb`)
3. **Model eğitimi ve raporlama**: RandomForest / SVM / KNN ile eğitim, metrikler ve confusion matrix (`training_and_report.ipynb`)
4. **Canlı demo**: Eğitilen SVM modelini webcam üzerinde gerçek zamanlı çalıştırma (`webcam_svm.py`)

## Repo İçeriği

- `data_collection.ipynb`  
  Webcam’den fotoğraf çekip `photos/<gesture>/` klasörüne kaydeder.

- `feature_extraction.ipynb`  
  `photos/` altındaki görsellerden **MediaPipe HandLandmarker** ile 21 el landmark’ının `(x,y)` koordinatlarını çıkarır, bileğe (wrist) göre normalize eder ve `features.csv` üretir.

- `training_and_report.ipynb`  
  `features.csv` üzerinde:
  - RandomForest
  - SVM (RBF kernel + StandardScaler pipeline)
  - KNN  
  modellerini eğitir, accuracy/F1, eğitim-tahmin sürelerini ve confusion matrix’i raporlar.  
  Demo için SVM modeli `svm_gesture_model.pkl` olarak kaydedilir.

- `webcam_svm.py`  
  `svm_gesture_model.pkl` ve `hand_landmarker.task` kullanarak webcam’den gelen görüntüde landmark çıkarır ve gesture tahmini yapar.

- `features.csv`  
  Örnek/üretim çıktısı veri seti (landmark tabanlı özellikler + label).

- `cm_*.png`  
  Modellerin confusion matrix görselleri.

## Desteklenen Gesture Sınıfları (örnek)

Çalışmada şu etiketler kullanılmıştır (veri setine göre değişebilir):
`approve`, `dissapprove`, `fingergun`, `koreanheart`, `mammamia`, `ok`, `peace`, `rocknroll`, `stop`

## Kurulum

> Not: Proje notebook ağırlıklı olduğu için en pratik kurulum bir sanal ortam (venv/conda) ile yapılır.

### 1) Python ortamı
Python 3.10+ önerilir (notebook metadata’sında 3.12 görülüyor).

### 2) Bağımlılıklar
Aşağıdaki paketler gerekir:

- `opencv-python`
- `mediapipe`
- `numpy`
- `pandas`
- `scikit-learn`
- `seaborn`
- `matplotlib`
- `joblib`

Örnek kurulum:
```bash
pip install opencv-python mediapipe numpy pandas scikit-learn seaborn matplotlib joblib
```

### 3) MediaPipe model dosyası (`hand_landmarker.task`)
`feature_extraction.ipynb` ve `webcam_svm.py` dosyaları, aynı klasörde şu dosyayı bekler:

- `hand_landmarker.task`

Bu dosya repoda yoksa indirip proje kök dizinine koymanız gerekir.

## Nasıl Çalışır? (Adım Adım)

### A) Veri Toplama (webcam’den fotoğraf çekme)
`data_collection.ipynb` içindeki ayarları düzenleyin:

- `GESTURE_NAME`: hangi gesture için toplanacak (örn: `"stop"`)
- `PERSON_TAG`: opsiyonel kişi/oturum etiketi (örn: `"burak_stop"`)
- `PHOTOS_TO_TAKE`: kaç fotoğraf (örn: `50`)
- `INTERVAL_SEC`: çekim aralığı (örn: `1.0` saniye)

Notebook’u çalıştırınca fotoğraflar şuraya kaydedilir:
`photos/<GESTURE_NAME>/...jpg`

Her gesture için ayrı ayrı veri toplayın.

### B) Özellik Çıkarma (landmark → CSV)
`feature_extraction.ipynb`:
- `DATASET_DIR = "photos"`
- `MODEL_PATH = "hand_landmarker.task"`
- `OUTPUT_CSV = "features.csv"`

Notebook:
1. `photos/` altındaki görselleri gezer
2. MediaPipe ile el landmark’larını çıkarır
3. Her örnek için 21 noktanın `(x,y)` koordinatlarını alır
4. **Bilek (landmark 0)** koordinatını referans alarak normalize eder  
   (her nokta için `lm.x - wrist_x`, `lm.y - wrist_y`)
5. Sonuçları `features.csv` olarak kaydeder.

`features.csv` kolon yapısı:
- `x0,y0,x1,y1,...,x20,y20,label`  (toplam 42 özellik + label)

### C) Model Eğitimi ve Raporlama
`training_and_report.ipynb`:
- `features.csv` okunur
- train/test split ile modeller eğitilir
- metrikler ve confusion matrix üretilir
- en sonunda demo için SVM modeli kaydedilir:
  - `svm_gesture_model.pkl`

> İpucu: Eğer gerçek hayatta genelleme istiyorsanız, train/test ayrımını “aynı oturumdan gelen benzer kareler” karışmayacak şekilde kurgulamak daha sağlıklı olur.

### D) Canlı Demo (Webcam üzerinden tahmin)
`webcam_svm.py` çalıştırmadan önce proje kökünde şunlar olmalı:
- `svm_gesture_model.pkl`
- `hand_landmarker.task`

Çalıştırma:
```bash
python webcam_svm.py
```

Script:
- webcam’i açar (`cv2.VideoCapture(0)`)
- her `PRED_EVERY_N_FRAMES` frame’de bir MediaPipe ile landmark çıkarır
- landmark’ları bileğe göre normalize eder
- SVM ile gesture tahmini yapar
- `q` ile çıkış.

## Sık Karşılaşılan Sorunlar

- **Webcam açılmıyor**: `VideoCapture(0)` yerine `1`/`2` deneyin veya OS izinlerini kontrol edin.
- **`hand_landmarker.task` bulunamadı**: Dosyayı proje köküne koyduğunuzdan emin olun.
- **MediaPipe el bulamıyor**: Işıklandırma/arka planı iyileştirin, el kadrajda net görünsün.
- **Model dosyası yok**: Önce `training_and_report.ipynb` ile modeli eğitip `svm_gesture_model.pkl` üretin.

## Geliştirme Fikirleri (Opsiyonel)

- Parametreleri `argparse` ile CLI’dan alacak hale getirme (`webcam_svm.py`)
- Daha sağlam değerlendirme (kişi/oturum bazlı split)
- Model versiyonlama ve sonuçları otomatik raporlama
- Notebook kodunu `src/` altında modüllere ayırma

---

Repo: `ejderburak/patternRecognition`
