# =========================
# ADNI Image Classification and Analysis
# Features + Metadata Fusion
# =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA

# Set seaborn style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 7)

# =========================
# Step 1: Load Datasets
# =========================
features_path = "resnet50_PD_selected_features_cumulative.csv"
metadata_path = "Cleaned_PD_Demographics.csv"

features_df = pd.read_csv(features_path)
metadata_df = pd.read_csv(metadata_path)

print("Features dataset:", features_df.shape)
print("Metadata dataset:", metadata_df.shape)

# =========================
# Step 2: Merge on filename
# =========================
merged_df = pd.merge(
    features_df, metadata_df,
    left_on="filename", right_on="Image Data ID",
    how="inner"
)
print("Merged dataset shape:", merged_df.shape)

# =========================
# Step 3: Prepare Features
# =========================
# Drop identifiers
drop_cols = ["filename", "Image Data ID", "Subject"]
for col in drop_cols:
    if col in merged_df.columns:
        merged_df = merged_df.drop(columns=col)

# Target label (AD, CN, Prodromal)
y = merged_df["class"] if "class" in merged_df.columns else merged_df["Group"]
X = merged_df.drop(columns=[y.name])

print("Feature matrix shape:", X.shape)
print("Labels distribution:\n", y.value_counts())

# Encode categorical columns
for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# =========================
# Step 4: Train/Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# Step 5: Scale Features
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# Step 6: Train Random Forest
# =========================
clf = RandomForestClassifier(
    n_estimators=200, random_state=42, class_weight="balanced"
)
clf.fit(X_train_scaled, y_train)

# =========================
# Step 7: Evaluate
# =========================
y_pred = clf.predict(X_test_scaled)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=clf.classes_, yticklabels=clf.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# =========================
# Step 8: Feature Importance
# =========================
feature_importances = pd.DataFrame({
    "feature": X.columns,
    "importance": clf.feature_importances_
}).sort_values("importance", ascending=False)

sns.barplot(x="importance", y="feature", data=feature_importances.head(20))
plt.title("Top 20 Feature Importances")
plt.show()

# =========================
# Step 9: PCA Visualization
# =========================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)

unique_classes = np.unique(y_train)
colors = plt.cm.Set1(np.linspace(0, 1, len(unique_classes)))

plt.figure(figsize=(10, 8))
for i, class_label in enumerate(unique_classes):
    mask = y_train == class_label
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                c=[colors[i]], label=class_label, alpha=0.7, s=50)

plt.title("PCA Visualization of Fused Features")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
