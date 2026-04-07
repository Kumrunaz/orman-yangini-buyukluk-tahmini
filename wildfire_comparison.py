"""
=============================================================
ORMAN YANGINI BÜYÜKLÜK TAHMİNİ — MODEL KARŞILAŞTIRMASI
Modeller : Random Forest vs Karar Ağacı vs Lojistik Regresyon
Bağımlı  : fire_size_class → 3 sınıf (Küçük / Orta / Büyük)
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, f1_score
)
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("ORMAN YANGINI BÜYÜKLÜK TAHMİNİ — MODEL KARŞILAŞTIRMASI")
print("=" * 60)

# ============================================================
# 1. VERİ YÜKLEMESİ
# ============================================================
df = pd.read_csv("FW_Veg_Rem_Combined.csv")
print(f"\nToplam kayıt: {len(df)} | Toplam sütun: {df.shape[1]}")

# ============================================================
# 2. VERİ ÖN İŞLEME
# ============================================================
df_clean = df.copy()

# Gereksiz sütunları at
drop_cols = ['Unnamed: 0.1', 'Unnamed: 0', 'fire_name', 'disc_clean_date',
             'cont_clean_date', 'disc_date_final', 'cont_date_final',
             'disc_date_pre', 'wstation_usaf', 'wstation_wban',
             'wstation_byear', 'wstation_eyear', 'weather_file']
df_clean.drop(columns=[c for c in drop_cols if c in df_clean.columns], inplace=True)

# putout_time → saate çevir
df_clean['putout_time'] = pd.to_timedelta(df_clean['putout_time'], errors='coerce')
df_clean['putout_hours'] = df_clean['putout_time'].dt.total_seconds() / 3600
df_clean.drop(columns=['putout_time'], inplace=True)
df_clean['putout_hours'].fillna(df_clean['putout_hours'].median(), inplace=True)

# Aykırı değer temizleme (IQR)
for col in ['Temp_pre_30', 'Wind_pre_30', 'Hum_pre_30', 'Prec_pre_30']:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    df_clean[col] = df_clean[col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)

# Kategorik kodlama
le_cause = LabelEncoder()
le_state = LabelEncoder()
df_clean['cause_encoded'] = le_cause.fit_transform(df_clean['stat_cause_descr'])
df_clean['state_encoded'] = le_state.fit_transform(df_clean['state'])

# Ay → sayıya çevir
month_map = {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,
             'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}
df_clean['month_num'] = df_clean['discovery_month'].map(month_map).fillna(0).astype(int)

# Hedef değişkeni 3 sınıfa indir
size_map = {'B':0, 'C':0, 'D':1, 'E':1, 'F':2, 'G':2}
df_clean['fire_class_3'] = df_clean['fire_size_class'].map(size_map)
df_clean.dropna(subset=['fire_class_3'], inplace=True)
df_clean['fire_class_3'] = df_clean['fire_class_3'].astype(int)

print(f"Temiz veri: {df_clean.shape}")

# ============================================================
# 3. ÖZELLİK VE HEDEF AYIRIMI
# ============================================================
features = [
    'Temp_pre_30', 'Temp_pre_15', 'Temp_pre_7', 'Temp_cont',
    'Wind_pre_30', 'Wind_pre_15', 'Wind_pre_7', 'Wind_cont',
    'Hum_pre_30',  'Hum_pre_15',  'Hum_pre_7',  'Hum_cont',
    'Prec_pre_30', 'Prec_pre_15', 'Prec_pre_7', 'Prec_cont',
    'Vegetation', 'remoteness', 'month_num',
    'disc_pre_year', 'cause_encoded', 'state_encoded', 'putout_hours'
]

X = df_clean[features]
y = df_clean['fire_class_3']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Lojistik Regresyon için ölçekleme — önce NaN temizle
X_filled = X.fillna(X.median())
X_train_f, X_test_f, _, _ = train_test_split(
    X_filled, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_f)
X_test_scaled  = scaler.transform(X_test_f)

print(f"\nEğitim: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ============================================================
# 4. MODELLERİ EĞİT
# ============================================================
print("\n" + "=" * 60)
print("MODELLERİ EĞİTİYORUM...")
print("=" * 60)

# Model 1: Random Forest
print("\n[1/3] Random Forest eğitiliyor...")
rf_model = RandomForestClassifier(
    n_estimators=150, max_depth=12, min_samples_split=5,
    class_weight='balanced', random_state=42, n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc  = accuracy_score(y_test, rf_pred)
rf_f1   = f1_score(y_test, rf_pred, average='weighted')
print(f"    Doğruluk: %{rf_acc*100:.2f} | F1: {rf_f1:.4f}")

# Model 2: Karar Ağacı
print("\n[2/3] Karar Ağacı eğitiliyor...")
dt_model = DecisionTreeClassifier(
    max_depth=10, min_samples_split=5,
    class_weight='balanced', random_state=42
)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_acc  = accuracy_score(y_test, dt_pred)
dt_f1   = f1_score(y_test, dt_pred, average='weighted')
print(f"    Doğruluk: %{dt_acc*100:.2f} | F1: {dt_f1:.4f}")

# Model 3: Lojistik Regresyon
print("\n[3/3] Lojistik Regresyon eğitiliyor...")
lr_model = LogisticRegression(
    max_iter=1000, class_weight='balanced', random_state=42
)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_acc  = accuracy_score(y_test, lr_pred)
lr_f1   = f1_score(y_test, lr_pred, average='weighted')
print(f"    Doğruluk: %{lr_acc*100:.2f} | F1: {lr_f1:.4f}")

# ============================================================
# 5. KARŞILAŞTIRMA TABLOSU
# ============================================================
print("\n" + "=" * 60)
print("MODEL KARŞILAŞTIRMA TABLOSU")
print("=" * 60)

from sklearn.metrics import precision_score, recall_score

results = {
    'Model': ['Random Forest', 'Karar Ağacı', 'Lojistik Regresyon'],
    'Accuracy': [rf_acc, dt_acc, lr_acc],
    'Precision': [
        precision_score(y_test, rf_pred, average='weighted'),
        precision_score(y_test, dt_pred, average='weighted'),
        precision_score(y_test, lr_pred, average='weighted'),
    ],
    'Recall': [
        recall_score(y_test, rf_pred, average='weighted'),
        recall_score(y_test, dt_pred, average='weighted'),
        recall_score(y_test, lr_pred, average='weighted'),
    ],
    'F1-Score': [rf_f1, dt_f1, lr_f1]
}

df_results = pd.DataFrame(results)
df_results['Accuracy']  = df_results['Accuracy'].apply(lambda x: f"%{x*100:.2f}")
df_results['Precision'] = df_results['Precision'].apply(lambda x: f"{x:.4f}")
df_results['Recall']    = df_results['Recall'].apply(lambda x: f"{x:.4f}")
df_results['F1-Score']  = df_results['F1-Score'].apply(lambda x: f"{x:.4f}")
print(df_results.to_string(index=False))

# ============================================================
# 6. GÖRSELLEŞTİRME
# ============================================================
print("\n[Karşılaştırma grafikleri oluşturuluyor...]")

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle("Model Karşılaştırması — Random Forest vs Karar Ağacı vs Lojistik Regresyon",
             fontsize=13, fontweight='bold')

modeller   = ['Random\nForest', 'Karar\nAğacı', 'Lojistik\nRegresyon']
renkler    = ['#2E7D32', '#FF8F00', '#1565C0']
acc_vals   = [rf_acc, dt_acc, lr_acc]
f1_vals    = [rf_f1,  dt_f1,  lr_f1]

# 5.1 Accuracy karşılaştırma
bars = axes[0].bar(modeller, [v*100 for v in acc_vals], color=renkler, edgecolor='white', linewidth=1.5, width=0.5)
axes[0].set_title("Accuracy (%)", fontweight='bold', fontsize=12)
axes[0].set_ylim(0, 105)
axes[0].set_ylabel("Doğruluk (%)")
for bar, val in zip(bars, acc_vals):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"%{val*100:.2f}", ha='center', fontweight='bold', fontsize=11)

# 5.2 F1-Score karşılaştırma
bars2 = axes[1].bar(modeller, f1_vals, color=renkler, edgecolor='white', linewidth=1.5, width=0.5)
axes[1].set_title("F1-Score (Weighted)", fontweight='bold', fontsize=12)
axes[1].set_ylim(0, 1.1)
axes[1].set_ylabel("F1-Score")
for bar, val in zip(bars2, f1_vals):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.4f}", ha='center', fontweight='bold', fontsize=11)

# 5.3 Sınıf bazında F1 karşılaştırma
siniflar   = ['Küçük', 'Orta', 'Büyük']
rf_f1_cls  = f1_score(y_test, rf_pred, average=None)
dt_f1_cls  = f1_score(y_test, dt_pred, average=None)
lr_f1_cls  = f1_score(y_test, lr_pred, average=None)

x = np.arange(len(siniflar))
w = 0.25
axes[2].bar(x - w, rf_f1_cls, w, label='Random Forest',        color='#2E7D32', edgecolor='white')
axes[2].bar(x,     dt_f1_cls, w, label='Karar Ağacı',          color='#FF8F00', edgecolor='white')
axes[2].bar(x + w, lr_f1_cls, w, label='Lojistik Regresyon',   color='#1565C0', edgecolor='white')
axes[2].set_title("Sınıf Bazında F1-Score", fontweight='bold', fontsize=12)
axes[2].set_xticks(x)
axes[2].set_xticklabels(siniflar)
axes[2].set_ylabel("F1-Score")
axes[2].set_ylim(0, 1.1)
axes[2].legend(fontsize=9)
for i, (r, d, l) in enumerate(zip(rf_f1_cls, dt_f1_cls, lr_f1_cls)):
    axes[2].text(i - w, r + 0.02, f"{r:.2f}", ha='center', fontsize=8, fontweight='bold')
    axes[2].text(i,     d + 0.02, f"{d:.2f}", ha='center', fontsize=8, fontweight='bold')
    axes[2].text(i + w, l + 0.02, f"{l:.2f}", ha='center', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig("model_karsilastirma.png", dpi=150, bbox_inches='tight')
plt.close()
print("Grafik kaydedildi: model_karsilastirma.png")

# ============================================================
# 7. 3 AYRI KARISIKLIK MATRİSİ
# ============================================================
fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
fig2.suptitle("Karışıklık Matrisleri", fontsize=14, fontweight='bold')

for ax, pred, title, cmap in zip(
    axes2,
    [rf_pred, dt_pred, lr_pred],
    ['Random Forest', 'Karar Ağacı', 'Lojistik Regresyon'],
    ['Greens', 'Oranges', 'Blues']
):
    cm = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Küçük','Orta','Büyük'])
    disp.plot(ax=ax, colorbar=False, cmap=cmap)
    ax.set_title(title, fontweight='bold')

plt.tight_layout()
plt.savefig("karisiklik_matrisleri.png", dpi=150, bbox_inches='tight')
plt.close()
print("Karışıklık matrisleri kaydedildi.")

# ============================================================
# 8. ÖZET
# ============================================================
print("\n" + "=" * 60)
print("SONUÇ")
print("=" * 60)
print(f"""
  Random Forest     → Accuracy: %{rf_acc*100:.2f} | F1: {rf_f1:.4f}  ← EN İYİ
  Karar Ağacı       → Accuracy: %{dt_acc*100:.2f} | F1: {dt_f1:.4f}
  Lojistik Regresyon→ Accuracy: %{lr_acc*100:.2f} | F1: {lr_f1:.4f}
""")
