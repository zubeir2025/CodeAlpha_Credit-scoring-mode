"""
Credit Scoring Model Implementation.

This module generates a synthetic credit dataset, performs feature engineering,
builds preprocessing and machine learning pipelines for multiple classifiers,
evaluates their performance with classification reports, and plots ROC curves.

Author: Zubeir Abdi
Date: June 28, 2026
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    classification_report, 
    roc_curve, 
    auc
)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def generate_synthetic_data(n_samples: int = 1000, random_seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic dataset representing past financial data.

    Parameters:
        n_samples (int): Number of financial records to generate (default 1000).
        random_seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Synthetic dataset with financial columns and target variable.
    """
    logger.info("Generating synthetic financial dataset...")
    np.random.seed(random_seed)

    # 1. Income (normally distributed around $65,000, min limit of $15,000)
    income = np.random.normal(loc=65000, scale=20000, size=n_samples)
    income = np.clip(income, a_min=15000, a_max=None)

    # 2. Total Debts (normally distributed around $25,000, min limit of $0)
    total_debts = np.random.normal(loc=25000, scale=15000, size=n_samples)
    total_debts = np.clip(total_debts, a_min=0, a_max=None)

    # 3. Missed Payments (Poisson distribution with mean 0.7)
    missed_payments = np.random.poisson(lam=0.7, size=n_samples)

    # 4. Credit Utilization Ratio (Beta distribution shifted and scaled, typical range 0.05 to 0.95)
    credit_utilization_ratio = np.random.beta(a=2, b=5, size=n_samples)
    credit_utilization_ratio = np.clip(credit_utilization_ratio, a_min=0.0, a_max=1.0)

    # Calculate latent creditworthiness probability using a logistic function
    # Normalize features to standard scale internally to compute realistic weights
    z_income = (income - 65000) / 20000
    z_debts = (total_debts - 25000) / 15000
    z_missed = (missed_payments - 0.7) / 0.83  # Std dev of Poisson(0.7) is sqrt(0.7) ≈ 0.83
    z_utilization = (credit_utilization_ratio - 0.3) / 0.15

    # Define linear relationship plus noise
    # Higher income improves creditworthiness; higher debt, missed payments, and utilization decrease it
    logit = 2.0 * z_income - 1.2 * z_debts - 2.5 * z_missed - 2.2 * z_utilization - 0.1
    probability = 1 / (1 + np.exp(-logit))

    # Binary outcome based on probability + uniform random noise
    is_creditworthy = (probability > np.random.uniform(0, 1, size=n_samples)).astype(int)

    # Create raw dataframe
    df = pd.DataFrame({
        "income": income,
        "total_debts": total_debts,
        "missed_payments": missed_payments,
        "credit_utilization_ratio": credit_utilization_ratio,
        "is_creditworthy": is_creditworthy
    })

    # Introduce ~5% missing values in 'income' and 'credit_utilization_ratio' to simulate real data issues
    missing_rate = 0.05
    for col in ["income", "credit_utilization_ratio"]:
        mask = np.random.rand(n_samples) < missing_rate
        df.loc[mask, col] = np.nan

    logger.info(f"Dataset successfully created with shape: {df.shape}")
    return df


class CreditFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to engineer features for credit scoring.
    
    Compatible with scikit-learn Pipeline architecture.
    """

    def __init__(self) -> None:
        super().__init__()

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> "CreditFeatureEngineer":
        """Fit method (no-op)."""
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate engineered financial metrics:
        1. Debt-to-Income Ratio (total_debts / (income + 1e-5))
        2. Risk Score Multiplier (credit_utilization_ratio * (missed_payments + 1))
        
        Assumes column ordering: [income, total_debts, missed_payments, credit_utilization_ratio]
        """
        # Ensure we work with numpy array
        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = np.asarray(X)

        income = X_arr[:, 0]
        total_debts = X_arr[:, 1]
        missed_payments = X_arr[:, 2]
        credit_utilization = X_arr[:, 3]

        # 1. Debt-to-Income Ratio (using a tiny epsilon to prevent division by zero)
        debt_to_income = total_debts / (income + 1e-5)

        # 2. Risk Score Multiplier (compounded risk of high card utilization combined with missed payments)
        risk_score_multiplier = credit_utilization * (missed_payments + 1.0)

        # Stack engineered features back onto the dataset
        return np.column_stack((X_arr, debt_to_income, risk_score_multiplier))

    def get_feature_names_out(self, input_features: List[str] = None) -> np.ndarray:
        """Define output feature names for scikit-learn compliance."""
        if input_features is None:
            input_features = ["income", "total_debts", "missed_payments", "credit_utilization_ratio"]
        return np.array(list(input_features) + ["debt_to_income_ratio", "risk_score_multiplier"])


def build_and_train_pipelines(X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Pipeline]:
    """
    Constructs and fits three modeling pipelines containing Imputation, Feature Engineering,
    Scaling, and the classification algorithms.

    Parameters:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.

    Returns:
        Dict[str, Pipeline]: Dictionary of trained Pipeline objects.
    """
    logger.info("Initializing modeling pipelines...")

    # Define base preprocessor pipeline
    # Impute missing values with Median, engineer new features, then scale
    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("feature_engineer", CreditFeatureEngineer()),
        ("scaler", StandardScaler())
    ])

    # Instantiate the three classification algorithms
    models = {
        "Logistic Regression": LogisticRegression(
            C=1.0, 
            random_state=42, 
            max_iter=1000
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, 
            min_samples_split=10, 
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, 
            max_depth=7, 
            min_samples_split=5, 
            random_state=42, 
            n_jobs=-1
        )
    }

    trained_pipelines = {}

    for name, model in models.items():
        logger.info(f"Training {name} pipeline...")
        # Create a unified pipeline containing preprocessing and model steps
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])
        
        # Fit the entire pipeline
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline

    logger.info("All modeling pipelines trained successfully.")
    return trained_pipelines


def evaluate_models(
    pipelines: Dict[str, Pipeline], 
    X_test: pd.DataFrame, 
    y_test: pd.Series
) -> Tuple[pd.DataFrame, Dict[str, Tuple[np.ndarray, np.ndarray, float]]]:
    """
    Evaluates the performance of trained pipelines on test data.
    Prints detailed Classification Reports for each model.

    Parameters:
        pipelines (Dict[str, Pipeline]): Dictionary of trained pipelines.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test labels.

    Returns:
        Tuple:
            - pd.DataFrame: Evaluation comparison table.
            - Dict: ROC curves data containing (fpr, tpr, roc_auc) for plotting.
    """
    logger.info("Evaluating models on test dataset...")
    metrics_summary = []
    roc_curves_data = {}

    for name, pipeline in pipelines.items():
        print(f"\n================== {name.upper()} EVALUATION ==================")
        
        # Predict class labels
        y_pred = pipeline.predict(X_test)
        
        # Predict probabilities for ROC-AUC
        if hasattr(pipeline.named_steps["classifier"], "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
        else:
            y_prob = pipeline.decision_function(X_test)

        # Print classification report
        print(classification_report(y_test, y_pred, target_names=["High Risk", "Creditworthy"]))

        # Compute classification metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        metrics_summary.append({
            "Model": name,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "ROC-AUC": roc_auc
        })

        # Calculate ROC Curve points for visualization
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_curves_data[name] = (fpr, tpr, roc_auc)
        print(f"ROC-AUC Score: {roc_auc:.4f}\n")

    # Create comparison DataFrame
    evaluation_df = pd.DataFrame(metrics_summary)
    evaluation_df = evaluation_df.set_index("Model")
    return evaluation_df, roc_curves_data


def plot_roc_curves(roc_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]], output_path: str = "roc_curves.png") -> None:
    """
    Plots the ROC Curves for all evaluated models using premium aesthetics.

    Parameters:
        roc_data (Dict): ROC curve data mapped by model name.
        output_path (str): File path where the plot image will be saved.
    """
    logger.info("Generating ROC curve plot...")
    
    # Configure premium style using seaborn
    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(9, 7), dpi=150)
    
    # Custom vibrant color palette (Indigo, Coral, Teal)
    colors = {
        "Logistic Regression": "#636EFA",
        "Decision Tree": "#EF553B",
        "Random Forest": "#00CC96"
    }

    # Plot diagonal random guess line
    plt.plot([0, 1], [0, 1], linestyle="--", color="#AAAAAA", lw=1.5, label="Random Guess (AUC = 0.50)")

    for name, (fpr, tpr, roc_auc) in roc_data.items():
        plt.plot(
            fpr, 
            tpr, 
            label=f"{name} (AUC = {roc_auc:.4f})", 
            color=colors.get(name, "#1f77b4"),
            lw=2.5
        )

    # Customize plot aesthetics
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Positive Rate (FPR)", fontsize=11, fontweight="bold", labelpad=10)
    plt.ylabel("True Positive Rate (TPR)", fontsize=11, fontweight="bold", labelpad=10)
    plt.title("ROC Curves Comparison: Credit Scoring Classifiers", fontsize=14, fontweight="bold", pad=15)
    plt.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#EAEAEA", fontsize=10)
    
    # Optimize layout and save
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info(f"ROC curves plot successfully saved to: {output_path}")


def main() -> None:
    """Main execution flow."""
    # 1. Dataset Generation
    df = generate_synthetic_data(n_samples=1000, random_seed=42)
    
    # Showcase dataset properties
    print("\n=== Dataset Sample ===")
    print(df.head())
    
    print("\n=== Missing Value Count ===")
    print(df.isnull().sum())

    # Split into features and target
    X = df.drop(columns=["is_creditworthy"])
    y = df["is_creditworthy"]

    # 2. Train-Test Split (80% Train, 20% Test)
    # Stratified split to maintain class balance in case of imbalance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    logger.info(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")

    # 3. Build & Train Pipelines
    pipelines = build_and_train_pipelines(X_train, y_train)

    # 4. Evaluate Models & Print Comparative Assessment
    evaluation_results, roc_data = evaluate_models(pipelines, X_test, y_test)
    
    print("\n================== SUMMARY MODEL COMPARISON ==================")
    print(evaluation_results.round(4).to_string())
    print("==============================================================\n")

    # 5. Plot and save ROC Curves
    plot_roc_curves(roc_data, output_path="roc_curves.png")

    # Display Feature Importances for Random Forest as a professional bonus
    rf_pipeline = pipelines["Random Forest"]
    rf_model = rf_pipeline.named_steps["classifier"]
    
    # Get feature names from preprocessor steps
    preprocessor = rf_pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.named_steps["feature_engineer"].get_feature_names_out(X.columns)
    
    importances = rf_model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)
    
    print("=== Random Forest Feature Importances ===")
    print(importance_df.to_string(index=False))
    print("=========================================\n")


if __name__ == "__main__":
    main()
