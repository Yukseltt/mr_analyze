# MR Görüntülerinden Beyin Tümörü Sınıflandırma

MR kesitlerinde tümör / tümör-yok ikili sınıflandırması. Sıfırdan tasarlanan **EytNet** CNN
ailesi ile hazır omurgalar ve hibrit mimariler aynı veri seti ve aynı protokolle karşılaştırıldı.

## Sonuçlar

12 model, F1-Score'a göre sıralı:

| # | Model | Accuracy | Precision | Recall | F1-Score | AUC |
|---|---|---|---|---|---|---|
| 1 | **EytNet_V1** *(özgün)* | 0.9950 | 0.9967 | 0.9934 | **0.9950** | 0.9994 |
| 2 | MobileNetV2 + ResNet50 *(hibrit)* | 0.9934 | 0.9978 | 0.9890 | 0.9933 | 0.9994 |
| 3 | VGG16 (augmentasyonsuz) | 0.9912 | 0.9912 | 0.9912 | 0.9912 | 0.9985 |
| 4 | VGG16 (augmentasyonlu) | 0.9906 | 0.9944 | 0.9868 | 0.9906 | 0.9985 |
| 5 | EytNet_V2 *(özgün)* | 0.9884 | 0.9922 | 0.9845 | 0.9884 | 0.9985 |
| 6 | EfficientNetB0 + ShuffleNetV2 *(hibrit)* | 0.9879 | 0.9825 | 0.9934 | 0.9879 | 0.9992 |
| 7 | EytNet_V2_Opt *(özgün)* | 0.9873 | 0.9944 | 0.9801 | 0.9872 | 0.9990 |
| 8 | EytNet_V1_High *(özgün)* | 0.9873 | 0.9966 | 0.9779 | 0.9872 | 0.9987 |
| 9 | ResNet18 + MobileNetV3Small *(hibrit)* | 0.9703 | 0.9661 | 0.9746 | 0.9703 | 0.9957 |
| 10 | DenseNet121 | 0.9510 | 0.9733 | 0.9272 | 0.9497 | 0.9909 |
| 11 | EytNet_V1_Opt *(özgün, ağır regülarizasyon)* | 0.9477 | 0.9688 | 0.9249 | 0.9464 | 0.9857 |
| 12 | ResNet50 | 0.9427 | 0.9578 | 0.9260 | 0.9416 | 0.9837 |

**En iyi sonuç sıfırdan eğitilen özgün mimariden geldi:** EytNet_V1, ImageNet ön eğitimli
ResNet50 ve DenseNet121'i belirgin farkla geçiyor.

## Çalışmanın asıl bulgusu: modern tekniklerin toplamı her zaman iyi değil

`EytNet_V1_Opt`, V1'in üzerine **Depthwise Separable Conv, SE Block, MixUp, SWA, Label Smoothing
ve progressive learning** eklenmiş varyantı. Beklenti daha iyi genelleme; sonuç tam tersi —
F1 0.9950'den **0.9464'e** düştü, listede sondan ikinci sıraya indi.

Küçük veri setinde bu tekniklerin birleşimi kapasiteyi fazla kısıtlayıp **underfitting**
yarattı. Bu, tek tek makul olan iyileştirmelerin bir arada uygulandığında neden ölçülmesi
gerektiğini gösteren somut bir örnek.

## Mimariler

**EytNet ailesi (özgün, sıfırdan eğitildi)** — ResNet tarzı artık bloklar üzerine kurulu;
V1, V1_High, V1_Opt, V2, V2_Opt varyantları farklı kapasite ve regülarizasyon seviyelerini test eder.

**Hibrit mimariler** — iki hazır omurganın özellik haritalarının birleştirilmesi:
MobileNetV2+ResNet50, EfficientNetB0+ShuffleNetV2, ResNet18+MobileNetV3Small.

**Referans omurgalar** — VGG16 (augmentasyonlu/suz), ResNet50, DenseNet121.

## Yöntem

- **PyTorch** ile eğitim, **Albumentations** ile medikal görüntüye özel artırma hattı
- Erken durdurma, en iyi modelin doğrulama başarımına göre kaydedilmesi
- Metrikler: Accuracy, Precision, Recall, Specificity, F1, Cohen's Kappa, AUC + ROC eğrileri
- Tüm modeller aynı veri bölünmesi ve aynı değerlendirme protokolüyle ölçüldü

## Depodaki defterler

| Defter | İçerik |
|---|---|
| `mr_analyze_new.ipynb` | **Güncel çalışma** — PyTorch, EytNet ailesi, 12 modelin karşılaştırması |
| `mr_analyze.ipynb` | İlk sürüm — TensorFlow/Keras ile EYTV ailesi ve hibrit mimariler, bilgi damıtma ve öz-denetimli ön görev denemeleri |

Veri seti ve model ağırlıkları depoda tutulmaz.
