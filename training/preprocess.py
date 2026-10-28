import pandas as pd
import numpy as np
import pickle
import os
import warnings
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = OUTPUT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'preprocessing.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
DATA_PATH = PROJECT_ROOT / "data" / "reports.csv"
MODEL_DIR = OUTPUT_DIR / "models"
REPORT_DIR = OUTPUT_DIR / "reports"
TEST_SIZE = 0.2
RANDOM_STATE = 42

warnings.filterwarnings('ignore')


def load_data():
    """Load dataset with comprehensive error handling."""
    try:
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
        
        df = pd.read_csv(DATA_PATH)
        logger.info(f"✅ Loaded dataset with shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        return df
    
    except Exception as e:
        logger.error(f"❌ Error loading data: {str(e)}")
        raise


def generate_data_quality_report(df, filename="data_quality_report.txt"):
    """Generate comprehensive data quality report."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    report_path = os.path.join(REPORT_DIR, filename)
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("DATA QUALITY REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        # Basic info
        f.write(f"Dataset Shape: {df.shape}\n")
        f.write(f"Total Records: {len(df)}\n")
        f.write(f"Total Features: {len(df.columns)}\n\n")
        
        # Missing values
        f.write("MISSING VALUES:\n")
        f.write("-" * 80 + "\n")
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        missing_df = pd.DataFrame({
            'Missing_Count': missing,
            'Percentage': missing_pct
        })
        f.write(missing_df[missing_df['Missing_Count'] > 0].to_string())
        f.write("\n\n")
        
        # Data types
        f.write("DATA TYPES:\n")
        f.write("-" * 80 + "\n")
        f.write(df.dtypes.to_string())
        f.write("\n\n")
        
        # Numeric statistics
        f.write("NUMERIC FEATURES STATISTICS:\n")
        f.write("-" * 80 + "\n")
        f.write(df.describe().to_string())
        f.write("\n\n")
        
        # Categorical features
        cat_cols = df.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            f.write("CATEGORICAL FEATURES:\n")
            f.write("-" * 80 + "\n")
            for col in cat_cols:
                f.write(f"\n{col}:\n")
                f.write(df[col].value_counts().to_string())
                f.write("\n")
        
        # Duplicates
        f.write("\nDUPLICATE RECORDS:\n")
        f.write("-" * 80 + "\n")
        duplicates = df.duplicated().sum()
        f.write(f"Total Duplicates: {duplicates}\n")
        
    logger.info(f"✅ Data quality report saved to {report_path}")


def detect_outliers(df, numeric_cols):
    """Detect outliers using IQR method."""
    outlier_info = {}
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        
        if len(outliers) > 0:
            outlier_info[col] = {
                'count': len(outliers),
                'percentage': round(len(outliers) / len(df) * 100, 2),
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
    
    return outlier_info


def clean_data(df):
    """Clean and prepare data with enhanced preprocessing."""
    df = df.copy()
    logger.info("Starting data cleaning...")
    
    # Remove duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    removed_duplicates = initial_rows - len(df)
    if removed_duplicates > 0:
        logger.info(f"Removed {removed_duplicates} duplicate rows")
    
    # Drop unnecessary columns
    cols_to_drop = ['Patient_ID', 'ID', 'Index']
    for col in cols_to_drop:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
            logger.info(f"Dropped column: {col}")
    
    # Handle numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Detect outliers before imputation
    outlier_info = detect_outliers(df, numeric_cols)
    if outlier_info:
        logger.info(f"Detected outliers in {len(outlier_info)} columns")
        for col, info in outlier_info.items():
            logger.info(f"  {col}: {info['count']} outliers ({info['percentage']}%)")
    
    # Impute numeric missing values
    for col in numeric_cols:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info(f"Imputed {missing_count} missing values in '{col}' with median: {median_val:.2f}")
    
    # Handle categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    for col in categorical_cols:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
            df[col] = df[col].fillna(mode_val)
            logger.info(f"Imputed {missing_count} missing values in '{col}' with mode: {mode_val}")
    
    # Encode categorical variables
    if 'Gender' in df.columns:
        gender_mapping = {'Male': 1, 'Female': 0, 'M': 1, 'F': 0}
        df['Gender'] = df['Gender'].map(gender_mapping)
        if df['Gender'].isnull().any():
            df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
        logger.info("Encoded 'Gender' column")
    
    # Handle other categorical columns with Label Encoding
    remaining_categorical = df.select_dtypes(include=['object']).columns.tolist()
    label_encoders = {}
    
    for col in remaining_categorical:
        if col != 'Risk_Level':  # Don't encode target if it's categorical
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
            logger.info(f"Label encoded '{col}' column")
    
    logger.info("✅ Data cleaning complete")
    return df, label_encoders


def preprocess_data(df):
    """Preprocess data with train-test split and scaling."""
    
    # Clean data first
    df_cleaned, label_encoders = clean_data(df)
    
    # Define target column
    target_col = 'Risk_Level'
    
    if target_col not in df_cleaned.columns:
        raise ValueError(f"Target column '{target_col}' not found. Available columns: {df_cleaned.columns.tolist()}")
    
    # Check target distribution
    logger.info(f"\nTarget distribution:\n{df_cleaned[target_col].value_counts()}")
    
    # Prepare features and target
    y = df_cleaned[target_col]
    X = df_cleaned.drop(columns=[target_col])
    
    logger.info(f"\nFeature matrix shape: {X.shape}")
    logger.info(f"Target vector shape: {y.shape}")
    
    # Handle target encoding if it's categorical
    target_encoder = None
    if pd.api.types.is_string_dtype(y) or y.dtype == 'object':
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y)
        logger.info("Encoded target variable")
    
    # Train-test split with stratification
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        logger.info(f"✅ Split data: Train={len(X_train)}, Test={len(X_test)}")
    except Exception as e:
        logger.warning(f"Stratified split failed: {e}. Using non-stratified split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info("✅ Applied StandardScaler to features")
    
    # Create model directory
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Save preprocessed data
    with open(os.path.join(MODEL_DIR, "train.pkl"), "wb") as f:
        pickle.dump((X_train_scaled, y_train), f)
    
    with open(os.path.join(MODEL_DIR, "test.pkl"), "wb") as f:
        pickle.dump((X_test_scaled, y_test), f)
    
    # Save preprocessing artifacts
    preprocessing_info = {
        "scaler": scaler,
        "columns": X.columns.tolist(),
        "label_encoders": label_encoders,
        "target_encoder": target_encoder,
        "feature_names": X.columns.tolist(),
        "n_features": X.shape[1],
        "n_samples": len(df_cleaned)
    }
    
    with open(os.path.join(MODEL_DIR, "preproc.pkl"), "wb") as f:
        pickle.dump(preprocessing_info, f)
    
    logger.info(f"✅ Saved preprocessed data to {MODEL_DIR}/")
    
    # Save feature importance baseline (variance)
    feature_variance = X_train.var().sort_values(ascending=False)
    feature_variance.to_csv(os.path.join(REPORT_DIR, "feature_variance.csv"))
    logger.info(f"✅ Saved feature variance to {REPORT_DIR}/feature_variance.csv")
    
    return X_train_scaled, X_test_scaled, y_train, y_test


def main():
    """Main execution function."""
    try:
        logger.info("=" * 80)
        logger.info("STARTING DATA PREPROCESSING PIPELINE")
        logger.info("=" * 80)
        
        # Load data
        df = load_data()
        
        # Generate data quality report
        generate_data_quality_report(df)
        
        # Preprocess data
        X_train, X_test, y_train, y_test = preprocess_data(df)
        
        logger.info("\n" + "=" * 80)
        logger.info("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Testing samples: {len(X_test)}")
        logger.info(f"Number of features: {X_train.shape[1]}")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
