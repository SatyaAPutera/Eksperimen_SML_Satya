# Eksperimen_SML_SatyaAdnyanaPutera

Repository untuk Machine Learning Experimentation: Sentiment Analysis pada Tweet Data

## Struktur Direktori

```
Eksperimen_SML_Satya/
├── .github/
│   └── workflows/
│       └── preprocessing.yml
├── sentiment_analysis_raw/
│   └── sentiment_analysis_raw.csv
├── preprocessing/
│   ├── Eksperimen_SatyaAdnyanaPutera.ipynb
│   ├── automate_SatyaAdnyanaPutera.py
│   ├── sentiment_analysis_preprocessing.csv
│   ├── train_data.csv
│   └── test_data.csv
└── README.md
```

## Deskripsi

Repository ini berisi hasil eksperimen dan otomasi preprocessing untuk dataset sentiment analysis yang mencakup tweet dari berbagai platform (Twitter, Facebook, Instagram), disusun sesuai dengan **Kriteria 1 Submission SML**.

### Dataset Karakteristik:
- **Total Records**: 499 tweets
- **Features**: Year, Month, Day, Time of Tweet, text, sentiment, Platform
- **Target Variable**: sentiment (positive, negative, neutral)
- **Sumber Data**: [Kaggle Sentiment Analysis Dataset](https://www.kaggle.com/datasets/mdismielhossenabir/sentiment-analysis)

## Pipeline Preprocessing

### 1. **Data Loading**
   - Membaca file CSV dataset raw
   - Melakukan inspeksi awal terhadap dataset

### 2. **Exploratory Data Analysis (EDA)**
   - Statistik deskriptif untuk setiap kolom
   - Analisis missing values dan duplikat
   - Distribusi sentiment dan platform
   - Analisis temporal (Year, Month, Day)
   - Text length & word count analysis

### 3. **Text Cleaning**
   - Menghilangkan URL, mention (@username), dan hashtag
   - Lowercase conversion
   - Punctuation removal
   - Whitespace normalization
   - Removal of empty text entries

### 4. **Handle Data Quality**
   - Missing values: diisi dengan median (numeric) atau mode (categorical)
   - Text columns: hapus baris dengan text kosong
   - Duplicate data removal (105 duplikat ditemukan)

### 5. **Feature Engineering**
   - Categorical encoding: sentiment, platform, time of tweet
   - TF-IDF vectorization untuk text features (max 1000 features, bigrams)
   - Feature scaling dengan MinMaxScaler

### 6. **Output**
   - Dataset yang sudah dipreprocess disimpan ke file CSV baru
   - Train/test split (80/20) dengan stratified sampling
   - Siap untuk tahap model training

## File-file Penting

### `Eksperimen_SatyaAdnyanaPutera.ipynb`
Notebook Jupyter yang berisi:
- Data Loading
- EDA (Exploratory Data Analysis)
- Text Preprocessing
- Feature Engineering
- Data Quality Checks
- Demonstrasi pemanggilan script automate

### `automate_SatyaAdnyanaPutera.py`
Script Python yang berisi:
- Class `SentimentDataPreprocessor` untuk otomasi preprocessing
- Fungsi-fungsi untuk setiap tahap preprocessing
- Main function untuk menjalankan pipeline
- Detailed logging dan error handling
- Path handling otomatis (repository-relative)

### `.github/workflows/preprocessing.yml`
GitHub Actions workflow yang:
- Otomatis trigger ketika ada push ke branch main
- Menjalankan preprocessing pipeline
- Upload artifacts hasil preprocessing
- Commit hasil preprocessing (optional)

## Cara Menggunakan

### Menggunakan Notebook (Manual Experimentation)
1. Buka `preprocessing/Eksperimen_SatyaAdnyanaPutera.ipynb` di Jupyter atau VS Code
2. Install dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn`
3. Jalankan setiap cell secara berurutan
4. Amati hasil EDA dan preprocessing di setiap step

### Menggunakan Script Python (Automated)
```python
from preprocessing.automate_SatyaAdnyanaPutera import SentimentDataPreprocessor

# Inisialisasi preprocessor
preprocessor = SentimentDataPreprocessor(verbose=True)

# Jalankan preprocessing
processed_df = preprocessor.preprocess(
    filepath='sentiment_analysis_raw/sentiment_analysis_raw.csv',
    save_output=True,
    output_filename='preprocessing/sentiment_analysis_preprocessing.csv'
)

# Dapatkan data yang sudah diproses
df_processed = preprocessor.get_processed_data()

# Dapatkan encoding mappings
mappings = preprocessor.get_mappings()
print(mappings)
```

### Menjalankan dari Command Line
```bash
# Dari project root
python preprocessing/automate_SatyaAdnyanaPutera.py
```

### Menggunakan GitHub Actions
1. Push repository ke GitHub
2. Actions akan otomatis trigger
3. Hasil preprocessing di-upload sebagai artifacts
4. Preprocessed dataset di-commit ke repository

## Output & Hasil

### Dimensi Dataset
- **Sebelum Preprocessing**: 500 rows × 7 columns
- **Setelah Preprocessing**: ~490 rows × 11 columns (tergantung missing values & duplicates)

### Kolom Output
```
- Year                    (Original)
- Month                   (Original)
- Day                     (Original)
- Time of Tweet          (Original)
- text                   (Cleaned)
- sentiment              (Original)
- Platform               (Original)
- sentiment_encoded      (New - Encoded)
- platform_encoded       (New - Encoded)
- time_encoded           (New - Encoded)
- [TF-IDF Features]      (New - 100 features)
```

### Encoding Mappings
- **Sentiment**: positive=0, negative=1, neutral=2 (sorted)
- **Platform**: Facebook, Instagram, Twitter (sorted)
- **Time**: morning, noon, night (sorted)

## Dependencies

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

Install dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Catatan

- Dataset sudah dibersihkan dan siap untuk tahap modeling
- Preprocessing bersifat konsisten baik di notebook maupun script
- GitHub Actions memastikan preprocessing selalu up-to-date
- Semua mappings disimpan untuk consistency di tahap inference

## Author

Satya
