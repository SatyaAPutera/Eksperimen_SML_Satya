"""
Automation Script untuk Sentiment Analysis Preprocessing
File ini berisi fungsi-fungsi untuk melakukan preprocessing data secara otomatis
Mengonversi langkah-langkah dari notebook eksperimen menjadi pipeline yang dapat digunakan kembali
"""

import pandas as pd
import numpy as np
import re
import string
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import warnings
import os
warnings.filterwarnings('ignore')


class SentimentDataPreprocessor:
    """
    Class untuk melakukan preprocessing otomatis pada dataset sentiment analysis
    """
    
    def __init__(self, verbose=True):
        """
        Inisialisasi preprocessor
        
        Parameters:
        -----------
        verbose : bool
            Jika True, akan menampilkan informasi progress
        """
        self.verbose = verbose
        self.df = None
        self.df_processed = None
        self.sentiment_mapping = {}
        self.platform_mapping = {}
        self.time_mapping = {}
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
    def load_data(self, filepath):
        """
        Memuat data dari file CSV
        
        Parameters:
        -----------
        filepath : str
            Path ke file CSV
            
        Returns:
        --------
        pd.DataFrame
            Dataset yang sudah dimuat
        """
        try:
            self.df = pd.read_csv(filepath)
            if self.verbose:
                print(f"Dataset berhasil dimuat!")
                print(f"  Jumlah baris: {self.df.shape[0]}")
                print(f"  Jumlah kolom: {self.df.shape[1]}")
            return self.df
        except FileNotFoundError:
            raise FileNotFoundError(f"File '{filepath}' tidak ditemukan!")
        except Exception as e:
            raise Exception(f"Error saat memuat data: {str(e)}")
    
    @staticmethod
    def clean_text(text):
        """
        Membersihkan text dari URL, mention, hashtag, dan karakter khusus
        
        Parameters:
        -----------
        text : str
            Text yang akan dibersihkan
            
        Returns:
        --------
        str
            Text yang sudah dibersihkan
        """
        if not isinstance(text, str):
            return ""
        
        # Hapus URL
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Hapus mention (@username)
        text = re.sub(r'@\w+', '', text)
        
        # Hapus hashtag
        text = re.sub(r'#\w+', '', text)
        
        # Ubah ke lowercase
        text = text.lower()
        
        # Hapus punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Hapus extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def handle_missing_values(self):
        """
        Menangani missing values dalam dataset
        """
        if self.df_processed is None:
            self.df_processed = self.df.copy()
        
        if self.verbose:
            print("HANDLE MISSING VALUES")
        
        # Cek missing values awal
        missing_count = self.df_processed.isnull().sum()
        if self.verbose and missing_count.sum() > 0:
            print(f"\nMissing values ditemukan:")
            print(missing_count[missing_count > 0])
        
        # Handle numeric columns - isi dengan median
        numeric_cols = self.df_processed.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            if self.df_processed[col].isnull().sum() > 0:
                median_value = self.df_processed[col].median()
                self.df_processed[col].fillna(median_value, inplace=True)
                if self.verbose:
                    print(f"Kolom '{col}': Filled with median ({median_value})")
        
        # Handle categorical dan text columns
        categorical_cols = self.df_processed.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if self.df_processed[col].isnull().sum() > 0:
                if col == 'text':
                    rows_before = self.df_processed.shape[0]
                    self.df_processed = self.df_processed.dropna(subset=['text'])
                    rows_removed = rows_before - self.df_processed.shape[0]
                    if self.verbose:
                        print(f"Kolom 'text': Removed {rows_removed} empty rows")
                else:
                    mode_value = self.df_processed[col].mode()[0]
                    self.df_processed[col].fillna(mode_value, inplace=True)
                    if self.verbose:
                        print(f"Kolom '{col}': Filled with mode ({mode_value})")
        
        if self.verbose:
            print(f"\nTotal missing values setelah: {self.df_processed.isnull().sum().sum()}")
    
    def handle_duplicates(self):
        """
        Menangani duplicate data dalam dataset
        """
        if self.df_processed is None:
            self.df_processed = self.df.copy()
        
        if self.verbose:
            print("HANDLE DUPLICATE DATA")
        
        duplicates_before = self.df_processed.duplicated().sum()
        if self.verbose:
            print(f"\nDuplicate rows sebelum: {duplicates_before}")
        
        if duplicates_before > 0:
            self.df_processed = self.df_processed.drop_duplicates()
            if self.verbose:
                print(f"Duplicate rows setelah: {self.df_processed.duplicated().sum()}")
                print(f"Removed {duplicates_before} duplicate rows")
        else:
            if self.verbose:
                print("Tidak ada duplicate data")
    
    def text_cleaning(self):
        """
        Melakukan text cleaning pada kolom text
        """
        if self.df_processed is None:
            self.df_processed = self.df.copy()
        
        if self.verbose:
            print("TEXT CLEANING")
        
        if 'text' in self.df_processed.columns:
            if self.verbose:
                print("\nPerforming text cleaning...")
            
            self.df_processed['text'] = self.df_processed['text'].apply(self.clean_text)
            
            # Hapus empty text setelah cleaning
            rows_before = self.df_processed.shape[0]
            self.df_processed = self.df_processed[self.df_processed['text'].str.len() > 0]
            rows_after = self.df_processed.shape[0]
            
            if self.verbose:
                print(f"Removed {rows_before - rows_after} empty text entries")
                print(f"Rows after text cleaning: {rows_after}")
    
    def encode_categorical_variables(self):
        """
        Melakukan encoding pada categorical variables
        """
        if self.df_processed is None:
            self.df_processed = self.df.copy()
        
        if self.verbose:
            print("ENCODING CATEGORICAL VARIABLES")
        
        # Encode sentiment (target variable)
        if 'sentiment' in self.df_processed.columns:
            if self.verbose:
                print("\nEncoding Sentiment...")
            
            unique_sentiments = sorted(self.df_processed['sentiment'].unique())
            for idx, sentiment in enumerate(unique_sentiments):
                self.sentiment_mapping[sentiment] = idx
            
            self.df_processed['sentiment_encoded'] = self.df_processed['sentiment'].map(self.sentiment_mapping)
            if self.verbose:
                print(f"Sentiment Mapping: {self.sentiment_mapping}")
        
        # Encode Platform
        if 'Platform' in self.df_processed.columns:
            if self.verbose:
                print("\nEncoding Platform...")
            
            self.df_processed['Platform'] = self.df_processed['Platform'].str.strip()
            unique_platforms = sorted(self.df_processed['Platform'].unique())
            for idx, platform in enumerate(unique_platforms):
                self.platform_mapping[platform] = idx
            
            self.df_processed['platform_encoded'] = self.df_processed['Platform'].map(self.platform_mapping)
            if self.verbose:
                print(f"Platform Mapping: {self.platform_mapping}")
        
        # Encode Time of Tweet
        if 'Time of Tweet' in self.df_processed.columns:
            if self.verbose:
                print("\nEncoding Time of Tweet...")
            
            unique_times = sorted([t.lower() for t in self.df_processed['Time of Tweet'].unique()])
            for idx, time in enumerate(unique_times):
                self.time_mapping[time] = idx
            
            self.df_processed['time_encoded'] = self.df_processed['Time of Tweet'].str.lower().map(self.time_mapping)
            if self.verbose:
                print(f"Time Mapping: {self.time_mapping}")
        
        if self.verbose:
            print(f"\nEncoding completed!")
    
    def feature_extraction_tfidf(self, max_features=5000, ngram_range=(1, 2)):
        """
        Melakukan feature extraction menggunakan TF-IDF
        
        Parameters:
        -----------
        max_features : int
            Jumlah maksimal features yang akan diekstrak
        ngram_range : tuple
            Range n-gram untuk TF-IDF
        """
        if self.df_processed is None:
            self.df_processed = self.df.copy()
        
        if self.verbose:
            print("FEATURE EXTRACTION (TF-IDF)")
        
        if 'text' in self.df_processed.columns:
            if self.verbose:
                print(f"\nPerforming TF-IDF vectorization...")
            
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                stop_words='english',
                min_df=2,
                max_df=0.95
            )
            
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.df_processed['text'])
            
            if self.verbose:
                print(f"TF-IDF Matrix Shape: {self.tfidf_matrix.shape}")
                print(f"Number of features: {self.tfidf_matrix.shape[1]}")
                feature_names = self.tfidf_vectorizer.get_feature_names_out()
                print(f"Top 10 features: {list(feature_names[:10])}")
        else:
            if self.verbose:
                print("Kolom 'text' tidak ditemukan; melewati TF-IDF")

    def _build_feature_matrix(self):
        """
        Bangun matriks fitur gabungan (TF-IDF + fitur numerik) dan scale fitur numerik non-TF-IDF
        """
        if self.df_processed is None:
            raise ValueError("No processed data available. Run preprocessing steps first.")

        if self.tfidf_matrix is not None:
            text_features_df = pd.DataFrame(
                self.tfidf_matrix.toarray(),
                columns=[f'tfidf_{i}' for i in range(self.tfidf_matrix.shape[1])],
                index=self.df_processed.index
            )
            numeric_features = self.df_processed.select_dtypes(include=['int64', 'float64']).copy()
            if 'sentiment_encoded' in numeric_features.columns:
                numeric_features = numeric_features.drop('sentiment_encoded', axis=1)
            X_combined = pd.concat([text_features_df, numeric_features], axis=1)
        else:
            X_combined = self.df_processed.select_dtypes(include=['int64', 'float64']).copy()

        numeric_cols_to_scale = [c for c in X_combined.columns if not c.startswith('tfidf_')]
        if len(numeric_cols_to_scale) > 0:
            scaler = MinMaxScaler()
            X_combined[numeric_cols_to_scale] = scaler.fit_transform(X_combined[numeric_cols_to_scale])

        if self.verbose:
            print(f"Feature matrix shape: {X_combined.shape}")
        return X_combined
    
    def preprocess(self, filepath, save_output=True, output_path='sentiment_analysis_preprocessing.csv', save_train_test=True, train_test_output_dir=None):
        """
        Menjalankan seluruh pipeline preprocessing
        
        Parameters:
        -----------
        filepath : str
            Path ke file CSV yang akan dipreprocess
        save_output : bool
            Jika True, akan menyimpan hasil preprocessing
        output_path : str
            Path penuh file output (folder akan dibuat otomatis)
            
        Returns:
        --------
        pd.DataFrame
            Dataset yang sudah dipreprocess
        """
        if self.verbose:
            print("SENTIMENT ANALYSIS DATA PREPROCESSING PIPELINE")
        
        # Load data
        self.load_data(filepath)
        
        # Handle missing values
        self.handle_missing_values()
        
        # Handle duplicates
        self.handle_duplicates()
        
        # Text cleaning
        self.text_cleaning()
        
        # Encode categorical variables
        self.encode_categorical_variables()
        
        # Feature extraction
        self.feature_extraction_tfidf()
        # Build features
        X_combined = self._build_feature_matrix()
        # Target
        if 'sentiment_encoded' in self.df_processed.columns:
            y = self.df_processed.loc[X_combined.index, 'sentiment_encoded']
        elif 'sentiment' in self.df_processed.columns:
            le = LabelEncoder()
            y = le.fit_transform(self.df_processed.loc[X_combined.index, 'sentiment'])
        else:
            y = None
        # Split
        if y is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                X_combined, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(X_combined, test_size=0.2, random_state=42), None, None
        
        # Summary
        if self.verbose:
            self._print_summary()
        
        # Save output
        if save_output:
            self._save_output(output_path)
        # Save train/test
        if save_train_test:
            if train_test_output_dir is None:
                # Default to model folder beside repo root
                script_dir = os.path.dirname(os.path.abspath(__file__))
                repo_root = os.path.dirname(script_dir)
                train_test_output_dir = os.path.join(repo_root, 'Membangun_model', 'sentiment_analysis_preprocessing')
            os.makedirs(train_test_output_dir, exist_ok=True)
            if y_train is not None:
                train_df = X_train.copy()
                train_df['sentiment'] = y_train
                test_df = X_test.copy()
                test_df['sentiment'] = y_test
            else:
                train_df, test_df = X_train, X_test
            train_out = os.path.join(train_test_output_dir, 'train_data.csv')
            test_out = os.path.join(train_test_output_dir, 'test_data.csv')
            train_df.to_csv(train_out, index=False)
            test_df.to_csv(test_out, index=False)
            if self.verbose:
                print(f"Train/Test saved to: {train_test_output_dir}")
        
        return self.df_processed
    
    def _print_summary(self):
        """
        Menampilkan ringkasan preprocessing
        """
        print("PREPROCESSING SUMMARY")
        
        print(f"\nOriginal dataset:")
        print(f"  Rows: {self.df.shape[0]}")
        print(f"  Columns: {self.df.shape[1]}")
        
        print(f"\nProcessed dataset:")
        print(f"  Rows: {self.df_processed.shape[0]}")
        print(f"  Columns: {self.df_processed.shape[1]}")
        
        print(f"\nChanges:")
        print(f"  Rows removed: {self.df.shape[0] - self.df_processed.shape[0]}")
        print(f"  New columns added: {self.df_processed.shape[1] - self.df.shape[1]}")
        
        print(f"\nFinal columns:")
        for col in self.df_processed.columns:
            print(f"  - {col}")
    
    def _save_output(self, output_path):
        """
        Menyimpan hasil preprocessing ke file CSV
        
        Parameters:
        -----------
        output_path : str
            Path penuh file output
        """
        if self.df_processed is None:
            raise ValueError("No processed data to save!")

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        self.df_processed.to_csv(output_path, index=False)
        if self.verbose:
            print(f"\nPreprocessed data saved to: {output_path}")
            print(f"  File size: {self.df_processed.shape[0]} rows × {self.df_processed.shape[1]} columns")
    
    def get_processed_data(self):
        """
        Mengembalikan dataset yang sudah dipreprocess
        
        Returns:
        --------
        pd.DataFrame
            Processed dataset
        """
        return self.df_processed
    
    def get_mappings(self):
        """
        Mengembalikan semua encoding mappings
        
        Returns:
        --------
        dict
            Dictionary berisi sentiment_mapping, platform_mapping, time_mapping
        """
        return {
            'sentiment_mapping': self.sentiment_mapping,
            'platform_mapping': self.platform_mapping,
            'time_mapping': self.time_mapping
        }


def main():
    """
    Main function untuk menjalankan preprocessing
    """
    # Inisialisasi preprocessor
    preprocessor = SentimentDataPreprocessor(verbose=True)
    
    # Jalankan preprocessing pipeline
    try:
        # Bangun path yang robust relatif terhadap repo root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(script_dir)
        raw_path = os.path.join(repo_root, 'sentiment_analysis_raw', 'sentiment_analysis_raw.csv')
        output_path = os.path.join(repo_root, 'preprocessing', 'sentiment_analysis_preprocessing.csv')
        model_dir = os.path.join(repo_root, 'Membangun_model', 'sentiment_analysis_preprocessing')

        processed_df = preprocessor.preprocess(
            filepath=raw_path,
            save_output=True,
            output_path=output_path,
            save_train_test=True,
            train_test_output_dir=model_dir
        )
        
        # Print success message
        print("PREPROCESSING COMPLETED SUCCESSFULLY!")
        
        # Tampilkan sample data
        print("\nSample preprocessed data (first 5 rows):")
        print(processed_df.head())
        
        return processed_df
    
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        return None


if __name__ == "__main__":
    main()
