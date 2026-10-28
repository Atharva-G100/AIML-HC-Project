import pickle
import numpy as np
import pandas as pd
import os
import json
import warnings
from datetime import datetime
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = OUTPUT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score, roc_curve,
    matthews_corrcoef, cohen_kappa_score, f1_score
)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

# Configuration
MODEL_DIR = OUTPUT_DIR / "models"
REPORT_DIR = OUTPUT_DIR / "reports"
TRAIN_PKL = os.path.join(MODEL_DIR, "train.pkl")
TEST_PKL = os.path.join(MODEL_DIR, "test.pkl")
PREPROC_PKL = os.path.join(MODEL_DIR, "preproc.pkl")
MODEL_PKL = os.path.join(MODEL_DIR, "model.pkl")
BEST_MODEL_PKL = os.path.join(MODEL_DIR, "best_model.pkl")
METRICS_JSON = os.path.join(REPORT_DIR, "metrics.json")

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


def load_pickles():
    """Load preprocessed data and artifacts."""
    required_files = [TRAIN_PKL, TEST_PKL, PREPROC_PKL]
    missing = [f for f in required_files if not os.path.exists(f)]
    
    if missing:
        raise FileNotFoundError(
            f"Missing preprocessing files: {missing}\n"
            "Please run training/preprocess.py first."
        )
    
    try:
        with open(TRAIN_PKL, "rb") as f:
            X_train, y_train = pickle.load(f)
        
        with open(TEST_PKL, "rb") as f:
            X_test, y_test = pickle.load(f)
        
        with open(PREPROC_PKL, "rb") as f:
            preproc = pickle.load(f)
        
        logger.info(f"✅ Loaded data: Train={X_train.shape}, Test={X_test.shape}")
        return X_train, y_train, X_test, y_test, preproc
    
    except Exception as e:
        logger.error(f"❌ Error loading pickles: {str(e)}")
        raise


def get_model_configurations():
    """Define models and their hyperparameter grids."""
    models = {
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0]
            }
        },
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'params': {
                'C': [0.001, 0.01, 0.1, 1, 10],
                'penalty': ['l2'],
                'solver': ['lbfgs', 'saga']
            }
        },
        'SVM': {
            'model': SVC(random_state=42, probability=True),
            'params': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
        },
        'K-Nearest Neighbors': {
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        }
    }
    
    return models


def train_single_model(model, X_train, y_train, model_name="Model"):
    """Train a single model without hyperparameter tuning."""
    logger.info(f"Training {model_name}...")
    
    model.fit(X_train, y_train)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
    logger.info(f"✅ {model_name} trained - CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    return model, cv_scores


def train_with_hyperparameter_tuning(model_config, X_train, y_train, model_name="Model"):
    """Train model with GridSearchCV for hyperparameter optimization."""
    logger.info(f"Starting hyperparameter tuning for {model_name}...")
    
    grid_search = GridSearchCV(
        estimator=model_config['model'],
        param_grid=model_config['params'],
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    logger.info(f"✅ {model_name} - Best CV Score: {grid_search.best_score_:.4f}")
    logger.info(f"Best Parameters: {grid_search.best_params_}")
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def compare_models(X_train, y_train, X_test, y_test, tune_hyperparameters=False):
    """Compare multiple models and return the best one."""
    logger.info("\n" + "=" * 80)
    logger.info("COMPARING MULTIPLE MODELS")
    logger.info("=" * 80)
    
    model_configs = get_model_configurations()
    results = {}
    
    for name, config in model_configs.items():
        try:
            if tune_hyperparameters:
                model, best_params, cv_score = train_with_hyperparameter_tuning(
                    config, X_train, y_train, name
                )
                results[name] = {
                    'model': model,
                    'best_params': best_params,
                    'cv_score': cv_score
                }
            else:
                model, cv_scores = train_single_model(config['model'], X_train, y_train, name)
                results[name] = {
                    'model': model,
                    'cv_score': cv_scores.mean()
                }
            
            # Evaluate on test set
            test_score = model.score(X_test, y_test)
            results[name]['test_score'] = test_score
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to train {name}: {str(e)}")
            continue
    
    # Find best model
    best_model_name = max(results, key=lambda x: results[x]['test_score'])
    best_model = results[best_model_name]['model']
    
    logger.info(f"\n🏆 Best Model: {best_model_name}")
    logger.info(f"Test Accuracy: {results[best_model_name]['test_score']:.4f}")
    
    # Save comparison results
    comparison_data = {
        name: {
            'cv_score': float(info['cv_score']),
            'test_score': float(info['test_score']),
            'best_params': info.get('best_params', {})
        }
        for name, info in results.items()
    }
    
    with open(os.path.join(REPORT_DIR, "model_comparison.json"), 'w') as f:
        json.dump(comparison_data, f, indent=4)
    
    return best_model, best_model_name, results


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Comprehensive model evaluation with multiple metrics."""
    logger.info("\n" + "=" * 80)
    logger.info(f"EVALUATING {model_name}")
    logger.info("=" * 80)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    
    # Basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    logger.info(f"\n📊 Classification Metrics:")
    logger.info(f"Accuracy:  {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1-Score:  {f1:.4f}")
    
    # Additional metrics
    mcc = matthews_corrcoef(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    
    logger.info(f"Matthews Correlation Coefficient: {mcc:.4f}")
    logger.info(f"Cohen's Kappa: {kappa:.4f}")
    
    # Classification report
    logger.info("\n📋 Detailed Classification Report:")
    report = classification_report(y_test, y_pred)
    logger.info(f"\n{report}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"\n🔢 Confusion Matrix:\n{cm}")
    
    # ROC-AUC for multi-class
    if y_pred_proba is not None and len(np.unique(y_test)) > 2:
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
            logger.info(f"\nROC-AUC Score: {roc_auc:.4f}")
        except:
            roc_auc = None
    elif y_pred_proba is not None and len(np.unique(y_test)) == 2:
        roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        logger.info(f"\nROC-AUC Score: {roc_auc:.4f}")
    else:
        roc_auc = None
    
    # Save metrics
    metrics = {
        'model_name': model_name,
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'matthews_corr_coef': float(mcc),
        'cohens_kappa': float(kappa),
        'roc_auc': float(roc_auc) if roc_auc else None,
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(METRICS_JSON, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    logger.info(f"\n✅ Metrics saved to {METRICS_JSON}")
    
    # Generate visualizations
    generate_visualizations(model, X_test, y_test, y_pred, cm, model_name)
    
    return y_pred, metrics


def generate_visualizations(model, X_test, y_test, y_pred, cm, model_name):
    """Generate and save evaluation visualizations."""
    
    # 1. Confusion Matrix Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, 'confusion_matrix.png'), dpi=300)
    plt.close()
    logger.info("✅ Saved confusion matrix heatmap")
    
    # 2. Feature Importance (if available)
    if hasattr(model, 'feature_importances_'):
        with open(PREPROC_PKL, 'rb') as f:
            preproc = pickle.load(f)
        
        feature_names = preproc.get('feature_names', preproc.get('columns', []))
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:20]  # Top 20 features
        
        plt.figure(figsize=(12, 8))
        plt.title(f'Top 20 Feature Importances - {model_name}')
        plt.bar(range(len(indices)), importances[indices])
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(REPORT_DIR, 'feature_importance.png'), dpi=300)
        plt.close()
        logger.info("✅ Saved feature importance plot")
        
        # Save feature importance to CSV
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        feature_importance_df.to_csv(os.path.join(REPORT_DIR, 'feature_importance.csv'), index=False)
    
    # 3. Class Distribution
    unique, counts = np.unique(y_test, return_counts=True)
    plt.figure(figsize=(10, 6))
    plt.bar(unique, counts, color='skyblue', edgecolor='black')
    plt.title('Test Set Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, 'class_distribution.png'), dpi=300)
    plt.close()
    logger.info("✅ Saved class distribution plot")


def save_model(model, filepath=MODEL_PKL):
    """Save trained model to disk."""
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"✅ Model saved to {filepath}")
    except Exception as e:
        logger.error(f"❌ Error saving model: {str(e)}")
        raise


def predict_single(sample_dict, model_path=MODEL_PKL):
    """Make prediction for a single sample with enhanced error handling."""
    try:
        # Load model and preprocessing info
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(PREPROC_PKL, 'rb') as f:
            preproc = pickle.load(f)
        
        scaler = preproc["scaler"]
        columns = preproc["columns"]
        label_encoders = preproc.get("label_encoders", {})
        target_encoder = preproc.get("target_encoder", None)
        
        # Prepare input row
        row = []
        for col in columns:
            val = sample_dict.get(col)
            
            # Handle gender encoding
            if col.lower() == "gender":
                if isinstance(val, str):
                    val = 1 if val.lower() in ['male', 'm'] else 0
            
            # Handle other categorical encodings
            elif col in label_encoders:
                if val is not None:
                    try:
                        val = label_encoders[col].transform([str(val)])[0]
                    except:
                        val = 0  # Unknown category
            
            # Convert to float
            row.append(float(val) if val is not None else 0.0)
        
        # Scale and predict
        X_scaled = scaler.transform(np.array(row).reshape(1, -1))
        pred = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0]
        
        # Get class labels
        classes = model.classes_
        
        # Decode prediction if target was encoded
        if target_encoder is not None:
            pred_decoded = target_encoder.inverse_transform([pred])[0]
            classes_decoded = target_encoder.inverse_transform(classes)
            result = {classes_decoded[i]: round(float(proba[i]), 4) for i in range(len(classes))}
        else:
            pred_decoded = pred
            result = {classes[i]: round(float(proba[i]), 4) for i in range(len(classes))}
        
        # Get confidence
        confidence = max(proba)
        
        return pred_decoded, result, confidence
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {str(e)}")
        raise


def batch_predict(samples_df, model_path=MODEL_PKL):
    """Make predictions for multiple samples."""
    predictions = []
    
    for idx, row in samples_df.iterrows():
        sample_dict = row.to_dict()
        pred, proba, conf = predict_single(sample_dict, model_path)
        predictions.append({
            'prediction': pred,
            'confidence': conf,
            'probabilities': proba
        })
    
    return pd.DataFrame(predictions)


def main(compare_multiple_models=True, tune_hyperparams=False):
    """Main training pipeline."""
    try:
        logger.info("\n" + "=" * 80)
        logger.info("STARTING MODEL TRAINING PIPELINE")
        logger.info("=" * 80)
        
        # Load data
        X_train, y_train, X_test, y_test, preproc = load_pickles()
        
        # Train model(s)
        if compare_multiple_models:
            model, model_name, comparison_results = compare_models(
                X_train, y_train, X_test, y_test, 
                tune_hyperparameters=tune_hyperparams
            )
            save_model(model, BEST_MODEL_PKL)
            save_model(model, MODEL_PKL)
        else:
            # Train single Random Forest model
            logger.info("Training Random Forest model...")
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            model_name = "Random Forest"
            save_model(model, MODEL_PKL)
        
        # Evaluate model
        y_pred, metrics = evaluate_model(model, X_test, y_test, model_name)
        
        # Example prediction
        logger.info("\n" + "=" * 80)
        logger.info("TESTING PREDICTION FUNCTION")
        logger.info("=" * 80)
        
        example = {
            "Age": 58,
            "Gender": "Male",
            "Hemoglobin": 13.2,
            "Glucose_Fasting": 140,
            "HbA1c": 7.2,
            "Cholesterol_Total": 250,
            "LDL_Cholesterol": 165,
            "HDL_Cholesterol": 38,
            "Triglycerides": 180,
            "Creatinine": 1.1,
            "TSH": 2.7,
            "Vitamin_D": 20,
            "Vitamin_B12": 310,
        }
        
        model_path = BEST_MODEL_PKL if compare_multiple_models else MODEL_PKL
        pred_label, probs, confidence = predict_single(example, model_path)
        
        logger.info(f"\n🩺 Sample Prediction Results:")
        logger.info(f"Predicted Risk Level: {pred_label}")
        logger.info(f"Confidence: {confidence:.2%}")
        logger.info(f"Class Probabilities: {probs}")
        
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"❌ Training pipeline failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    # Configuration options
    COMPARE_MODELS = True  # Set to True to compare multiple models
    TUNE_HYPERPARAMETERS = False  # Set to True for hyperparameter tuning (slower)
    
    main(compare_multiple_models=COMPARE_MODELS, tune_hyperparams=TUNE_HYPERPARAMETERS)
