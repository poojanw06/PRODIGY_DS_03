import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('default')
sns.set_palette("husl")

print("Packages imported successfully!")
print("Creating Customer Purchase Prediction Analysis with Synthetic Data...")
print("=" * 70)

# Create realistic synthetic customer data
print("Generating realistic customer dataset...")
np.random.seed(42)
n_samples = 5000

# Generate realistic customer features
data = {
    'age': np.random.normal(45, 15, n_samples).astype(int).clip(18, 80),
    'income': np.random.normal(60000, 20000, n_samples).astype(int).clip(20000, 150000),
    'credit_score': np.random.normal(650, 100, n_samples).astype(int).clip(300, 850),
    'account_balance': np.random.exponential(5000, n_samples).astype(int),
    'num_transactions': np.random.poisson(15, n_samples),
    'days_since_last_visit': np.random.exponential(30, n_samples).astype(int),
    'marketing_contacts': np.random.poisson(3, n_samples),
    'web_visits': np.random.poisson(8, n_samples),
    'customer_tenure': np.random.exponential(3, n_samples).astype(int),
    'has_loan': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
    'has_mortgage': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
}

df = pd.DataFrame(data)

# Create realistic purchase probability based on features
purchase_prob = (
    0.05 +
    0.000002 * df['income'] +
    0.0001 * df['credit_score'] -
    0.0001 * df['days_since_last_visit'] +
    0.03 * df['web_visits'] -
    0.02 * df['marketing_contacts'] +
    0.1 * df['has_loan'] -
    0.05 * df['has_mortgage'] +
    np.random.normal(0, 0.1, n_samples)
)

# Convert to binary purchase decision
df['purchased'] = (purchase_prob > 0.15).astype(int)

print(f"Dataset created with {n_samples} samples!")
print(f"Dataset shape: {df.shape}")
print(f"\nTarget distribution:")
print(df['purchased'].value_counts())
print(f"Purchase rate: {df['purchased'].mean()*100:.2f}%")

print(f"\nFirst 3 rows:")
print(df.head(3))

# Data Preprocessing
print("\nPreprocessing data...")

# Handle numerical features - scale them
numerical_cols = ['age', 'income', 'credit_score', 'account_balance', 
                 'num_transactions', 'days_since_last_visit', 
                 'marketing_contacts', 'web_visits', 'customer_tenure']

scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

print("Data preprocessing completed!")

# Split the data
X = df.drop('purchased', axis=1)
y = df['purchased']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nData split:")
print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Train target distribution: {y_train.value_counts().to_dict()}")
print(f"Test target distribution: {y_test.value_counts().to_dict()}")

# Build and train Decision Tree model
print("\nTraining Decision Tree model...")
dt_classifier = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

dt_classifier.fit(X_train, y_train)
print("Model trained successfully!")

# Make predictions
y_pred = dt_classifier.predict(X_test)
y_pred_proba = dt_classifier.predict_proba(X_test)[:, 1]

# Evaluate the model
print("\nModel Evaluation:")
print("=" * 50)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Purchase', 'Purchase'],
            yticklabels=['No Purchase', 'Purchase'])
plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Feature importance
print("\nFeature Importance Analysis:")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': dt_classifier.feature_importances_
}).sort_values('importance', ascending=False)

print("All features by importance:")
for i, row in feature_importance.iterrows():
    print(f"  {i+1:2d}. {row['feature']:20s}: {row['importance']:.4f}")

# Visualization - Feature Importance
print("\nCreating visualizations...")
plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
plt.title('Top 10 Feature Importances', fontsize=16, fontweight='bold')
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# Decision Tree visualization
plt.figure(figsize=(20, 12))
plot_tree(dt_classifier, 
          feature_names=X.columns, 
          class_names=['No Purchase', 'Purchase'],
          filled=True, 
          rounded=True,
          max_depth=3,
          fontsize=10)
plt.title('Decision Tree Visualization (First 3 Levels)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('decision_tree.png', dpi=300, bbox_inches='tight')
plt.show()

# Cross-validation
print("\nCross-validation scores:")
cv_scores = cross_val_score(dt_classifier, X, y, cv=5, scoring='accuracy')
print(f"CV Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Additional metrics
roc_auc = roc_auc_score(y_test, y_pred_proba)
precision = classification_report(y_test, y_pred, output_dict=True)['1']['precision']
recall = classification_report(y_test, y_pred, output_dict=True)['1']['recall']
f1 = classification_report(y_test, y_pred, output_dict=True)['1']['f1-score']

print(f"\nAdditional Metrics:")
print(f"ROC AUC Score: {roc_auc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Final insights
print("\n" + "=" * 70)
print("BUSINESS INSIGHTS AND RECOMMENDATIONS")
print("=" * 70)
print("1. KEY FINDINGS:")
print(f"   • Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   • Most Important Feature: {feature_importance.iloc[0]['feature']}")
print(f"   • Purchase Rate: {df['purchased'].mean()*100:.2f}% of customers")
print(f"   • Top 3 Features: {', '.join(feature_importance['feature'].iloc[:3].tolist())}")

print("\n2. BUSINESS RECOMMENDATIONS:")
print("   • Focus marketing on customers with high web engagement (web_visits)")
print("   • Target customers with better credit scores for premium products")
print("   • Avoid excessive marketing contacts to prevent customer fatigue")
print("   • Prioritize customers who visited recently (low days_since_last_visit)")

print("\n3. NEXT STEPS:")
print("   • Deploy this model for real-time customer scoring")
print("   • Implement A/B testing for different marketing strategies")
print("   • Monitor model performance and retrain quarterly")
print("   • Collect more customer data to improve prediction accuracy")

print("\nAnalysis completed successfully!")
print("Generated files:")
print("   - feature_importance.png")
print("   - decision_tree.png") 
print("   - confusion_matrix.png")
print("\nReady for customer purchase prediction!")
