"""
House Price Prediction Mini Project
===================================

Objective: Predict house prices based on features like number of rooms, location, size, etc.
Learn basic regression techniques, feature selection, and model evaluation.

Dataset: Boston Housing Dataset (built-in in sklearn)
Target Variable: House price (median home value)

Author: FDS Mini Project
Date: September 2025
"""

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("="*60)
print("HOUSE PRICE PREDICTION MINI PROJECT")
print("="*60)

# ============================================================================
# A. DATA LOADING
# ============================================================================

print("\n" + "="*50)
print("A. DATA LOADING")
print("="*50)

# Load California Housing dataset (similar to Boston Housing but more modern)
print("Loading California Housing dataset...")
housing_data = fetch_california_housing()

# Create DataFrame
df = pd.DataFrame(housing_data.data, columns=housing_data.feature_names)
df['target'] = housing_data.target  # House prices in hundreds of thousands of dollars

print(f"Dataset shape: {df.shape}")
print(f"Features: {list(df.columns[:-1])}")
print(f"Target variable: House Price")

# Display first few rows
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Dataset summary
print("\nDataset Info:")
print(df.info())

print("\nDataset Statistical Summary:")
print(df.describe())

# ============================================================================
# B. DATA CLEANING
# ============================================================================

print("\n" + "="*50)
print("B. DATA CLEANING")
print("="*50)

# Check for missing values
print("Missing values per column:")
missing_values = df.isnull().sum()
print(missing_values)

if missing_values.sum() == 0:
    print("✓ No missing values found in the dataset!")
else:
    print("Handling missing values...")
    # Fill missing values with median for numerical columns
    df.fillna(df.median(), inplace=True)

# Check for outliers using IQR method
print("\nChecking for outliers...")
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

# Define outliers as values beyond 1.5 * IQR
outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()
print("Outliers per column:")
print(outliers)

# For this project, we'll keep outliers as they might be valid data points
print("✓ Data cleaning completed!")

# ============================================================================
# C. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

print("\n" + "="*50)
print("C. EXPLORATORY DATA ANALYSIS")
print("="*50)

# Create figure for multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Exploratory Data Analysis', fontsize=16, fontweight='bold')

# 1. Distribution of target variable (House Prices)
axes[0, 0].hist(df['target'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Distribution of House Prices')
axes[0, 0].set_xlabel('Price (hundreds of thousands $)')
axes[0, 0].set_ylabel('Frequency')

# 2. Box plot of house prices
axes[0, 1].boxplot(df['target'])
axes[0, 1].set_title('Box Plot of House Prices')
axes[0, 1].set_ylabel('Price (hundreds of thousands $)')

# 3. Scatter plot: House Age vs Price
axes[1, 0].scatter(df['HouseAge'], df['target'], alpha=0.6, color='coral')
axes[1, 0].set_title('House Age vs Price')
axes[1, 0].set_xlabel('House Age')
axes[1, 0].set_ylabel('Price (hundreds of thousands $)')

# 4. Scatter plot: Average Rooms vs Price
axes[1, 1].scatter(df['AveRooms'], df['target'], alpha=0.6, color='lightgreen')
axes[1, 1].set_title('Average Rooms vs Price')
axes[1, 1].set_xlabel('Average Rooms')
axes[1, 1].set_ylabel('Price (hundreds of thousands $)')

plt.tight_layout()
plt.show()

# Correlation Analysis
print("\nCorrelation Analysis:")
correlation_matrix = df.corr()
print("Correlation with target variable (House Price):")
target_correlation = correlation_matrix['target'].sort_values(key=abs, ascending=False)
print(target_correlation)

# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5)
plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================================================
# D. FEATURE SELECTION
# ============================================================================

print("\n" + "="*50)
print("D. FEATURE SELECTION")
print("="*50)

# Select features based on correlation with target
correlation_threshold = 0.1
selected_features = []

print(f"Selecting features with correlation > {correlation_threshold} with target:")
for feature in df.columns[:-1]:  # Exclude target column
    corr_value = abs(correlation_matrix.loc[feature, 'target'])
    if corr_value > correlation_threshold:
        selected_features.append(feature)
        print(f"✓ {feature}: {corr_value:.3f}")

print(f"\nSelected {len(selected_features)} features: {selected_features}")

# Prepare feature matrix and target vector
X = df[selected_features]
y = df['target']

print(f"Feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# ============================================================================
# E. DATA PREPROCESSING
# ============================================================================

print("\n" + "="*50)
print("E. DATA PREPROCESSING")
print("="*50)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Features scaled using StandardScaler")
print("✓ Data preprocessing completed!")

# ============================================================================
# F. MODEL BUILDING
# ============================================================================

print("\n" + "="*50)
print("F. MODEL BUILDING")
print("="*50)

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

# Train models and store results
model_results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Use scaled data for Linear Regression, original data for Random Forest
    if name == 'Linear Regression':
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
    
    # Store model and predictions
    model_results[name] = {
        'model': model,
        'predictions': predictions
    }
    
    print(f"✓ {name} training completed!")

# ============================================================================
# G. MODEL EVALUATION
# ============================================================================

print("\n" + "="*50)
print("G. MODEL EVALUATION")
print("="*50)

# Evaluate each model
evaluation_results = {}

for name, result in model_results.items():
    predictions = result['predictions']
    
    # Calculate metrics
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    evaluation_results[name] = {
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    }
    
    print(f"\n{name} Performance:")
    print(f"  Mean Squared Error (MSE): {mse:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"  R² Score: {r2:.4f}")

# Find best model
best_model_name = max(evaluation_results.keys(), 
                     key=lambda x: evaluation_results[x]['R2'])
print(f"\n🏆 Best Model: {best_model_name} (R² = {evaluation_results[best_model_name]['R2']:.4f})")

# Visualization: Actual vs Predicted Prices
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Model Performance: Actual vs Predicted Prices', fontsize=16, fontweight='bold')

for i, (name, result) in enumerate(model_results.items()):
    predictions = result['predictions']
    
    axes[i].scatter(y_test, predictions, alpha=0.6)
    axes[i].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                 'r--', linewidth=2, label='Perfect Prediction')
    axes[i].set_xlabel('Actual Prices')
    axes[i].set_ylabel('Predicted Prices')
    axes[i].set_title(f'{name}\nR² = {evaluation_results[name]["R2"]:.4f}')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Feature Importance (for Random Forest)
if 'Random Forest' in model_results:
    rf_model = model_results['Random Forest']['model']
    feature_importance = pd.DataFrame({
        'feature': selected_features,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance (Random Forest):")
    print(feature_importance)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='importance', y='feature', palette='viridis')
    plt.title('Feature Importance - Random Forest Model', fontsize=14, fontweight='bold')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()

# ============================================================================
# H. CONCLUSION
# ============================================================================

print("\n" + "="*50)
print("H. CONCLUSION")
print("="*50)

print("\n📊 PROJECT SUMMARY:")
print("-" * 30)

# Dataset summary
print(f"• Dataset: California Housing (20,640 samples, {len(selected_features)} features)")
print(f"• Target: House prices in hundreds of thousands of dollars")

# Feature insights
if 'Random Forest' in model_results:
    top_feature = feature_importance.iloc[0]['feature']
    print(f"• Most important feature: {top_feature}")

print(f"• Selected features: {', '.join(selected_features)}")

# Model performance summary
print(f"\n🎯 MODEL PERFORMANCE:")
print("-" * 25)
for name, metrics in evaluation_results.items():
    print(f"• {name}:")
    print(f"  - RMSE: ${metrics['RMSE']*100:.0f}k")
    print(f"  - R² Score: {metrics['R2']:.3f} ({metrics['R2']*100:.1f}% variance explained)")

# Best model
best_r2 = evaluation_results[best_model_name]['R2']
best_rmse = evaluation_results[best_model_name]['RMSE']

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   - Explains {best_r2*100:.1f}% of price variance")
print(f"   - Average prediction error: ${best_rmse*100:.0f}k")

# Recommendations for improvement
print(f"\n💡 POSSIBLE IMPROVEMENTS:")
print("-" * 30)
print("• Feature Engineering: Create new features (e.g., price per room, location clusters)")
print("• Advanced Models: Try XGBoost, Neural Networks, or ensemble methods")
print("• Hyperparameter Tuning: Optimize model parameters using GridSearch")
print("• More Data: Include additional features like school ratings, crime rates")
print("• Outlier Treatment: Remove or transform extreme outliers")
print("• Cross-Validation: Use k-fold CV for more robust evaluation")

print(f"\n✅ House Price Prediction Project Completed Successfully!")
print("="*60) 