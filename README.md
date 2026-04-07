# 🔥 Orman Yangını Büyüklük Tahmini

Meteorolojik ve çevresel faktörler kullanılarak orman yangınlarının 
büyüklük sınıfını tahmin eden makine öğrenmesi projesi.

## 📊 Veri Seti
- **Kaynak:** US Wildfire Data — Kaggle
- **Kayıt sayısı:** 55.367
- **Değişken sayısı:** 43
- **Yıllar:** 1992–2015

## 🤖 Model Sonuçları

| Model | Accuracy | F1-Score |
|---|---|---|
| ✅ Random Forest | %87.88 | 0.8954 |
| Karar Ağacı | %75.47 | 0.8177 |
| Lojistik Regresyon | %63.55 | 0.7046 |

## 🛠️ Kullanılan Kütüphaneler
pandas, numpy, scikit-learn, matplotlib, seaborn

## ▶️ Çalıştırma
pip install pandas numpy scikit-learn matplotlib seaborn
python wildfire_comparison.py
