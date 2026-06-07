# =========================
# Disease Overlap Analysis: AD and PD
# Fusion Model for Detecting Disease Overlap
# =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.preprocessing import MultiLabelBinarizer
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')  # Updated to supported style
sns.set_palette("husl")

print("=== AD and PD Disease Overlap Analysis ===")
print("Loading and fusing datasets...")

# =========================
# Step 1: Load Datasets
# =========================
# Load the AD features dataset
ad_features_path = "resnet50_selected_features_cumulative_updated.csv"  # AD features
ad_metadata_path = "ADNI1_Merged_AllImages.csv"  # AD metadata

# Load the PD features dataset
pd_features_path = "resnet50_PD_selected_features_cumulative.csv"  # PD features
pd_metadata_path = "Cleaned_PD_Demographics.csv"  # PD metadata

# Load all datasets
try:
    ad_features_df = pd.read_csv(ad_features_path)
    ad_metadata_df = pd.read_csv(ad_metadata_path)
    pd_features_df = pd.read_csv(pd_features_path)
    pd_metadata_df = pd.read_csv(pd_metadata_path)
    
    print(f"AD Features dataset loaded: {ad_features_df.shape}")
    print(f"AD Metadata dataset loaded: {ad_metadata_df.shape}")
    print(f"PD Features dataset loaded: {pd_features_df.shape}")
    print(f"PD Metadata dataset loaded: {pd_metadata_df.shape}")
    
    # Display basic info about the datasets
    print("\nAD Features dataset info:")
    print(f"Classes: {ad_features_df['class'].unique()}")
    print(f"Class distribution:\n{ad_features_df['class'].value_counts()}")
    
    print("\nPD Features dataset info:")
    print(f"Classes: {pd_features_df['class'].unique()}")
    print(f"Class distribution:\n{pd_features_df['class'].value_counts()}")
    
    # =========================
    # Step 2: Merge Features with Metadata
    # =========================
    # Merge AD features with metadata
    ad_merged_df = pd.merge(
        ad_features_df, ad_metadata_df,
        left_on="filename", right_on="Image Data ID",
        how="inner"
    )
    
    # Merge PD features with metadata
    # Print column names to debug
    print("\nPD Features columns:", pd_features_df.columns[:10].tolist(), "...")
    print("PD Metadata columns:", pd_metadata_df.columns.tolist())
    
    # First attempt: direct merge on common columns
    common_cols = set(pd_features_df.columns) & set(pd_metadata_df.columns)
    common_cols = [col for col in common_cols if col != 'class']
    
    if common_cols:
        print(f"Found common columns for merging: {common_cols}")
        pd_merged_df = pd.merge(
            pd_features_df, pd_metadata_df,
            on=common_cols[0],  # Use the first common column
            how="left"
        )
    # If no common columns, try standard keys
    elif 'Image Data ID' in pd_metadata_df.columns:
        pd_merged_df = pd.merge(
            pd_features_df, pd_metadata_df,
            left_on="filename", right_on="Image Data ID",
            how="left"
        )
    elif 'Subject' in pd_metadata_df.columns:
        # Extract subject ID from filename if it's not already a column
        if 'Subject' not in pd_features_df.columns:
            pd_features_df['Subject'] = pd_features_df['filename'].str.extract(r'(\d+)').astype(str)
        
        pd_merged_df = pd.merge(
            pd_features_df, pd_metadata_df,
            on="Subject",
            how="left"
        )
    else:
        # Last resort: create artificial mapping
        print("Warning: No matching keys found. Creating artificial PD dataset for demonstration.")
        # Create a copy of the features dataframe and add dummy metadata
        pd_merged_df = pd_features_df.copy()
        
        # Add dummy metadata columns that match AD metadata
        for col in ad_metadata_df.columns:
            if col not in pd_merged_df.columns:
                pd_merged_df[col] = "Unknown"
        
        # Add age and gender if they exist in metadata
        if 'Age' in pd_metadata_df.columns:
            pd_merged_df['Age'] = pd_metadata_df['Age'].median()
        else:
            pd_merged_df['Age'] = 65  # Default age
            
        if 'Gender' in pd_metadata_df.columns:
            pd_merged_df['Gender'] = pd_metadata_df['Gender'].mode()[0]
        else:
            pd_merged_df['Gender'] = "Unknown"
    
    # Verify the merge was successful
    if pd_merged_df.shape[0] == 0:
        print("WARNING: PD merged dataset has 0 rows. Using the original PD features dataset.")
        pd_merged_df = pd_features_df.copy()
        # Add necessary columns from metadata if available
        if 'Age' in pd_metadata_df.columns:
            pd_merged_df['Age'] = pd_metadata_df['Age'].median()
        else:
            pd_merged_df['Age'] = 65  # Default age
            
        if 'Gender' in pd_metadata_df.columns:
            pd_merged_df['Gender'] = pd_metadata_df['Gender'].mode()[0]
        else:
            pd_merged_df['Gender'] = "Unknown"
    
    print(f"\nAD Merged dataset shape: {ad_merged_df.shape}")
    print(f"PD Merged dataset shape: {pd_merged_df.shape}")
    
    # =========================
    # Step 3: Prepare Features for Combined Analysis
    # =========================
    # Add disease type column to each dataset
    ad_merged_df['disease_type'] = 'AD'
    pd_merged_df['disease_type'] = 'PD'
    
    # Standardize column names for consistent merging
    # Rename class columns to be specific to each disease
    ad_merged_df = ad_merged_df.rename(columns={'class': 'ad_class'})
    pd_merged_df = pd_merged_df.rename(columns={'class': 'pd_class'})
    
    # Select common feature columns (those starting with 'f')
    ad_feature_cols = [col for col in ad_merged_df.columns if col.startswith('f')]
    pd_feature_cols = [col for col in pd_merged_df.columns if col.startswith('f')]
    
    # Find common features between AD and PD datasets
    common_features = list(set(ad_feature_cols).intersection(set(pd_feature_cols)))
    print(f"\nNumber of common features: {len(common_features)}")
    
    # If no common features, use all features and handle alignment later
    if len(common_features) == 0:
        print("No common features found. Using all features from both datasets.")
        # Create a combined feature set
        all_features = list(set(ad_feature_cols + pd_feature_cols))
        
        # Add missing columns to each dataset with NaN values
        for col in all_features:
            if col not in ad_merged_df.columns:
                ad_merged_df[col] = np.nan
            if col not in pd_merged_df.columns:
                pd_merged_df[col] = np.nan
        
        feature_cols = all_features
    else:
        feature_cols = common_features
    
    # =========================
    # Step 4: Combine Datasets for Overlap Analysis
    # =========================
    # Select relevant columns from each dataset
    ad_cols = feature_cols + ['ad_class', 'disease_type', 'Age', 'Gender'] 
    pd_cols = feature_cols + ['pd_class', 'disease_type', 'Age', 'Gender']
    
    # Filter columns that exist in each dataset
    ad_cols = [col for col in ad_cols if col in ad_merged_df.columns]
    pd_cols = [col for col in pd_cols if col in pd_merged_df.columns]
    
    # Create subsets with selected columns
    ad_subset = ad_merged_df[ad_cols].copy()
    pd_subset = pd_merged_df[pd_cols].copy()
    
    # Ensure both datasets have the same columns for concatenation
    for col in ad_subset.columns:
        if col not in pd_subset.columns:
            pd_subset[col] = np.nan
    
    for col in pd_subset.columns:
        if col not in ad_subset.columns:
            ad_subset[col] = np.nan
    
    # Combine datasets
    combined_df = pd.concat([ad_subset, pd_subset], ignore_index=True)
    print(f"\nCombined dataset shape: {combined_df.shape}")
    
    # =========================
    # Step 5: Handle Missing Values and Encode Categorical Variables
    # =========================
    # Fill missing values in feature columns
    for col in feature_cols:
        if col in combined_df.columns:
            # Check if column is numeric before using median
            if pd.api.types.is_numeric_dtype(combined_df[col]):
                combined_df[col] = combined_df[col].fillna(combined_df[col].median())
            else:
                # For non-numeric columns, use mode or just fill with a placeholder
                combined_df[col] = combined_df[col].fillna(combined_df[col].mode()[0] if not combined_df[col].mode().empty else 'unknown')
    
    # Create a new target column for disease overlap analysis
    # This will be our multi-label target
    combined_df['ad_status'] = combined_df['ad_class'].apply(lambda x: 1 if x == 'AD' else (0.5 if x == 'MCI' else 0))
    combined_df['pd_status'] = combined_df['pd_class'].apply(lambda x: 1 if x == 'PD' else (0.5 if x == 'Prodromal' else 0))
    
    # Encode categorical variables
    categorical_cols = combined_df.select_dtypes(include=['object']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col not in ['ad_class', 'pd_class', 'disease_type']]
    
    for col in categorical_cols:
        if col in combined_df.columns:
            le = LabelEncoder()
            combined_df[col] = le.fit_transform(combined_df[col].astype(str))
    
    # =========================
    # Step 6: Feature Selection and Scaling
    # =========================
    print("\nPerforming feature selection to improve performance...")
    
    # Select features for the model
    X = combined_df[feature_cols].values
    
    # Create multi-output targets
    y_ad = combined_df['ad_status'].values
    y_pd = combined_df['pd_status'].values
    
    # Combine targets for multi-output prediction
    y_combined = np.column_stack((y_ad, y_pd))
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_combined, test_size=0.3, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Feature selection using Random Forest feature importance
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.feature_selection import SelectFromModel
    
    # Train a random forest to get feature importances for AD prediction
    print("Selecting important features for AD...")
    rf_ad = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    rf_ad.fit(X_train_scaled, y_train[:, 0])
    
    # Train a random forest to get feature importances for PD prediction
    print("Selecting important features for PD...")
    rf_pd = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    rf_pd.fit(X_train_scaled, y_train[:, 1])
    
    # Check if PD importances are all zeros
    if np.all(rf_pd.feature_importances_ == 0):
        print("WARNING: All PD feature importances are zero. This suggests the PD labels might be constant.")
        print("PD label distribution:", np.unique(y_train[:, 1], return_counts=True))
        print("Using only AD feature importances for selection.")
        importances_combined = rf_ad.feature_importances_
    else:
        # Combine feature importances
        importances_combined = rf_ad.feature_importances_ + rf_pd.feature_importances_
        print("Using combined AD and PD feature importances for selection.")
    
    # Select top features (e.g., top 100)
    n_features_to_select = min(100, len(feature_cols))
    indices = np.argsort(importances_combined)[::-1][:n_features_to_select]
    
    # Create a mask for the selected features
    mask = np.zeros(X_train_scaled.shape[1], dtype=bool)
    mask[indices] = True
    
    # Apply the mask to get the selected features
    X_train_selected = X_train_scaled[:, mask]
    X_test_selected = X_test_scaled[:, mask]
    
    # Store selected feature names for later reference
    selected_feature_cols = [feature_cols[i] for i in indices]
    print(f"Selected {len(selected_feature_cols)} most important features out of {len(feature_cols)}")
    
    # Create a DataFrame with feature importances for reporting
    feature_importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance_ad': rf_ad.feature_importances_,
        'importance_pd': rf_pd.feature_importances_,
        'importance_combined': importances_combined,
        'selected': mask
    })
    feature_importance_df = feature_importance_df.sort_values('importance_combined', ascending=False)
    
    # Save top features to CSV
    feature_importance_df.head(n_features_to_select).to_csv('top_features_for_disease_overlap.csv', index=False)
    
    print(f"\nTraining set shape after feature selection: {X_train_selected.shape}")
    print(f"Test set shape after feature selection: {X_test_selected.shape}")
    
    # =========================
    # Step 7: Train Multi-Output Model for Disease Overlap
    # =========================
    print("\nTraining Multi-Output Model for Disease Overlap...")
    
    # Create a multi-output random forest regressor with fewer estimators for faster training
    from sklearn.ensemble import RandomForestRegressor
    
    multi_output_rf = MultiOutputRegressor(RandomForestRegressor(
        n_estimators=100,  # Reduced from 200 to 100 for faster training
        random_state=42,
        n_jobs=-1  # Use all available cores for parallel processing
    ))
    
    # Train the model on selected features
    print("Training model on selected features...")
    multi_output_rf.fit(X_train_selected, y_train)
    
    # Make predictions using selected features
    print("Making predictions...")
    y_pred = multi_output_rf.predict(X_test_selected)
    
    # =========================
    # Step 8: Evaluate the Model
    # =========================
    # Calculate MSE for each target using selected features
    from sklearn.metrics import mean_squared_error
    mse_ad = mean_squared_error(y_test[:, 0], y_pred[:, 0])
    mse_pd = mean_squared_error(y_test[:, 1], y_pred[:, 1])
    
    print(f"\nMean Squared Error for AD prediction (using selected features): {mse_ad:.4f}")
    print(f"Mean Squared Error for PD prediction (using selected features): {mse_pd:.4f}")
    
    # Calculate correlation between predicted values
    ad_pd_pred_corr = np.corrcoef(y_pred[:, 0], y_pred[:, 1])[0, 1]
    print(f"\nCorrelation between AD and PD predictions (using selected features): {ad_pd_pred_corr:.4f}")
    
    # =========================
    # Step 9: Visualize Disease Overlap
    # =========================
    print("\nVisualizing Disease Overlap...")
    
    # Create a scatter plot of AD vs PD predictions
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(y_pred[:, 0], y_pred[:, 1], 
                         c=y_test[:, 0] + y_test[:, 1], 
                         cmap='viridis', 
                         alpha=0.7, 
                         s=50)
    cbar = plt.colorbar(scatter, label='Combined Disease Severity')
    # Add text annotations to the colorbar to explain the color scale
    cbar.ax.text(1.5, 0.0, 'Low severity', ha='left', va='center', rotation=90, fontsize=10)
    cbar.ax.text(1.5, 1.0, 'High severity', ha='left', va='center', rotation=90, fontsize=10)
    plt.title('Disease Overlap: AD vs PD Predictions (Using Selected Features)')
    plt.xlabel('AD Prediction Score')
    plt.ylabel('PD Prediction Score')
    # Add text annotations to explain the quadrants
    plt.text(0.1, 0.9, 'Low AD, High PD', transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    plt.text(0.9, 0.9, 'High AD, High PD\n(High Overlap)', transform=plt.gca().transAxes, fontsize=10, ha='right', bbox=dict(facecolor='white', alpha=0.7))
    plt.text(0.1, 0.1, 'Low AD, Low PD', transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    plt.text(0.9, 0.1, 'High AD, Low PD', transform=plt.gca().transAxes, fontsize=10, ha='right', bbox=dict(facecolor='white', alpha=0.7))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('disease_overlap_scatter.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create a heatmap of the prediction distribution
    plt.figure(figsize=(12, 10))
    
    # Create a 2D histogram
    h, xedges, yedges, img = plt.hist2d(y_pred[:, 0], y_pred[:, 1], 
                                       bins=20, 
                                       cmap='viridis')
    cbar = plt.colorbar(label='Number of Patients')
    # Add text annotations to the colorbar to explain the color scale
    cbar.ax.text(1.5, 0.0, 'Few patients', ha='left', va='center', rotation=90, fontsize=10)
    cbar.ax.text(1.5, 1.0, 'Many patients', ha='left', va='center', rotation=90, fontsize=10)
    plt.title('Density of AD-PD Overlap Predictions')
    plt.xlabel('AD Prediction Score')
    plt.ylabel('PD Prediction Score')
    # Add text annotations to explain the quadrants
    plt.text(0.1, 0.9, 'Low AD, High PD', transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    plt.text(0.9, 0.9, 'High AD, High PD\n(High Overlap)', transform=plt.gca().transAxes, fontsize=10, ha='right', bbox=dict(facecolor='white', alpha=0.7))
    plt.text(0.1, 0.1, 'Low AD, Low PD', transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    plt.text(0.9, 0.1, 'High AD, Low PD', transform=plt.gca().transAxes, fontsize=10, ha='right', bbox=dict(facecolor='white', alpha=0.7))
    plt.tight_layout()
    plt.savefig('disease_overlap_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # =========================
    # Step 10: Identify Overlap Regions using Clustering with Selected Features
    # =========================
    print("\nIdentifying Disease Overlap Regions using Selected Features...")
    
    # Combine predictions for clustering
    pred_df = pd.DataFrame({
        'AD_pred': y_pred[:, 0],
        'PD_pred': y_pred[:, 1],
        'AD_actual': y_test[:, 0],
        'PD_actual': y_test[:, 1]
    })
    
    # Add a combined score
    pred_df['combined_score'] = pred_df['AD_pred'] + pred_df['PD_pred']
    
    # Perform K-means clustering to identify overlap regions
    # Using n_init parameter for more efficient clustering
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    pred_df['cluster'] = kmeans.fit_predict(pred_df[['AD_pred', 'PD_pred']])
    
    # Visualize the clusters
    plt.figure(figsize=(12, 10))
    
    # Get cluster statistics to determine what each cluster represents
    cluster_stats = pred_df.groupby('cluster').agg({
        'AD_pred': 'mean',
        'PD_pred': 'mean',
        'combined_score': 'mean'
    }).reset_index()
    
    # Determine cluster meanings based on AD and PD prediction means
    cluster_meanings = {}
    for _, row in cluster_stats.iterrows():
        cluster = int(row['cluster'])
        ad_mean = row['AD_pred']
        pd_mean = row['PD_pred']
        
        # Assign meaning based on prediction values
        if ad_mean > 0.5 and pd_mean > 0.5:
            meaning = 'High AD, High PD (High Overlap)'
        elif ad_mean > 0.5 and pd_mean <= 0.5:
            meaning = 'High AD, Low PD'
        elif ad_mean <= 0.5 and pd_mean > 0.5:
            meaning = 'Low AD, High PD'
        else:
            meaning = 'Low AD, Low PD'
            
        cluster_meanings[cluster] = meaning
    
    # Plot each cluster with a different color and meaningful label
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Distinct colors
    for i, cluster in enumerate(range(4)):
        cluster_data = pred_df[pred_df['cluster'] == cluster]
        plt.scatter(cluster_data['AD_pred'], 
                   cluster_data['PD_pred'], 
                   label=f'Cluster {cluster}: {cluster_meanings[cluster]}',
                   color=colors[i],
                   alpha=0.7,
                   s=50)
    
    # Plot cluster centers
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], 
               c='black', 
               s=200, 
               alpha=0.8, 
               marker='X',
               label='Cluster Centers')
    
    plt.title('Disease Overlap Clusters (Using Selected Features)')
    plt.xlabel('AD Prediction Score')
    plt.ylabel('PD Prediction Score')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Make room for the legend
    plt.savefig('disease_overlap_clusters.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Analyze the clusters
    cluster_analysis = pred_df.groupby('cluster').agg({
        'AD_pred': ['mean', 'std'],
        'PD_pred': ['mean', 'std'],
        'combined_score': ['mean', 'std', 'count']
    })
    
    print("\nCluster Analysis:")
    print(cluster_analysis)
    
    # Identify the high overlap cluster (high in both AD and PD)
    high_overlap_cluster = cluster_analysis['combined_score']['mean'].idxmax()
    high_overlap_count = cluster_analysis['combined_score']['count'][high_overlap_cluster]
    high_overlap_percentage = (high_overlap_count / len(pred_df)) * 100
    
    print(f"\nHigh Overlap Cluster: {high_overlap_cluster}")
    print(f"Number of samples in high overlap region: {high_overlap_count}")
    print(f"Percentage of samples in high overlap region: {high_overlap_percentage:.2f}%")
    
    # =========================
    # Step 11: Feature Importance for Overlap Prediction
    # =========================
    print("\nAnalyzing Feature Importance for Disease Overlap...")
    
    # Extract feature importances from the random forest models
    importances_ad = multi_output_rf.estimators_[0].feature_importances_
    importances_pd = multi_output_rf.estimators_[1].feature_importances_
    
    # Create a DataFrame for feature importance
    importance_df = pd.DataFrame({
        'feature': selected_feature_cols,
        'importance_ad': importances_ad,
        'importance_pd': importances_pd,
        'importance_combined': importances_ad + importances_pd
    })
    
    # Sort by combined importance
    importance_df = importance_df.sort_values('importance_combined', ascending=False)
    
    # Plot top 20 features (or all if less than 20)
    plt.figure(figsize=(14, 10))
    top_features = importance_df.head(min(20, len(selected_feature_cols)))
    
    # Create a grouped bar chart
    x = np.arange(len(top_features))
    width = 0.35
    
    # Use distinct colors for AD and PD
    ad_bar = plt.bar(x - width/2, top_features['importance_ad'], width, label='AD Importance', color='#1f77b4')
    pd_bar = plt.bar(x + width/2, top_features['importance_pd'], width, label='PD Importance', color='#ff7f0e')
    
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.title('Top Features for Disease Overlap Prediction\n(Higher values indicate more important features)')
    plt.xticks(x, top_features['feature'], rotation=90)
    
    # Add a legend with explanation
    legend = plt.legend(title='Feature Importance by Disease:', loc='upper right')
    legend.get_title().set_fontsize('10')
    
    # Add annotation explaining the plot
    plt.annotate('Note: This chart shows which features are most important for predicting each disease.\n'
                'Taller bars indicate features with stronger predictive power.\n'
                'Features important for both diseases may indicate shared mechanisms.',
                xy=(0.02, 0.02), xycoords='figure fraction',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.7),
                fontsize=10)
    plt.tight_layout()
    plt.savefig('overlap_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save feature importance to CSV
    importance_df.to_csv('disease_overlap_feature_importance.csv', index=False)
    
    # =========================
    # Step 12: PCA Visualization of Disease Overlap
    # =========================
    print("\nCreating PCA Visualization of Disease Overlap...")
    
    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_test_scaled)
    
    # Create a DataFrame for plotting
    pca_df = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'AD_pred': y_pred[:, 0],
        'PD_pred': y_pred[:, 1],
        'combined_score': y_pred[:, 0] + y_pred[:, 1],
        'cluster': pred_df['cluster']
    })
    
    # Plot PCA with color based on combined score
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(pca_df['PC1'], pca_df['PC2'], 
                         c=pca_df['combined_score'], 
                         cmap='plasma', 
                         alpha=0.7, 
                         s=50)
    cbar = plt.colorbar(scatter, label='Combined Disease Score (AD + PD)')
    # Add text annotations to the colorbar to explain the color scale
    cbar.ax.text(1.5, 0.0, 'Low combined score', ha='left', va='center', rotation=90, fontsize=10)
    cbar.ax.text(1.5, 1.0, 'High combined score', ha='left', va='center', rotation=90, fontsize=10)
    
    # Add a title with explanation
    plt.title('PCA Visualization of Disease Overlap\n(Points close together have similar feature patterns)')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    
    # Add annotation explaining the plot
    plt.annotate('Note: This plot shows patients in feature space.\n'
                'Color indicates combined AD+PD score.\n'
                'Clusters of similar colors suggest shared disease patterns.',
                xy=(0.02, 0.02), xycoords='figure fraction',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.7),
                fontsize=10)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('disease_overlap_pca.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot PCA with clusters
    plt.figure(figsize=(12, 10))
    
    # Use the same cluster meanings and colors as in the previous cluster plot
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Distinct colors
    for i, cluster in enumerate(range(4)):
        cluster_data = pca_df[pca_df['cluster'] == cluster]
        plt.scatter(cluster_data['PC1'], 
                   cluster_data['PC2'], 
                   label=f'Cluster {cluster}: {cluster_meanings[cluster]}',
                   color=colors[i],
                   alpha=0.7,
                   s=50)
    
    plt.title('PCA Visualization of Disease Overlap Clusters\n(Showing how disease clusters appear in feature space)')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    
    # Add annotation explaining the plot
    plt.annotate('Note: This plot shows how the disease clusters\n'
                'identified in prediction space appear when\n'
                'projected into feature space using PCA.\n'
                'Separation indicates distinct feature patterns.',
                xy=(0.02, 0.02), xycoords='figure fraction',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.7),
                fontsize=10)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Make room for the legend
    plt.savefig('disease_overlap_pca_clusters.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # =========================
    # Step 13: Create Summary Report
    # =========================
    print("\nCreating Summary Report...")
    
    with open('disease_overlap_report.txt', 'w') as f:
        f.write("AD and PD Disease Overlap Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Dataset Information:\n")
        f.write(f"- AD samples: {len(ad_merged_df)}\n")
        f.write(f"- PD samples: {len(pd_merged_df)}\n")
        f.write(f"- Combined samples: {len(combined_df)}\n")
        f.write(f"- Number of features: {len(feature_cols)}\n")
        f.write(f"- Number of selected features: {len(selected_feature_cols)}\n")
        f.write(f"- Feature reduction: {(1 - len(selected_feature_cols)/len(feature_cols))*100:.2f}%\n\n")
        
        f.write(f"Model Performance:\n")
        f.write(f"- AD prediction MSE: {mse_ad:.4f}\n")
        f.write(f"- PD prediction MSE: {mse_pd:.4f}\n")
        f.write(f"- AD-PD prediction correlation: {ad_pd_pred_corr:.4f}\n\n")
        
        f.write("Disease Overlap Analysis:\n")
        f.write(f"- High overlap cluster: {high_overlap_cluster}\n")
        f.write(f"- Samples in high overlap region: {high_overlap_count} ({high_overlap_percentage:.2f}%)\n\n")
        
        f.write("Top 10 Features for Disease Overlap:\n")
        for i, row in importance_df.head(10).iterrows():
            f.write(f"{i+1}. {row['feature']}: AD={row['importance_ad']:.4f}, PD={row['importance_pd']:.4f}, Combined={row['importance_combined']:.4f}\n")
        
        f.write("\nFeature Selection Information:\n")
        f.write(f"Original number of features: {len(feature_cols)}\n")
        f.write(f"Number of features after selection: {len(selected_feature_cols)}\n")
        f.write(f"Feature selection method: Random Forest feature importance\n")
    
    print("\n=== Analysis Complete ===")
    print("Generated files:")
    print("- disease_overlap_scatter.png")
    print("- disease_overlap_heatmap.png")
    print("- disease_overlap_clusters.png")
    print("- overlap_feature_importance.png")
    print("- disease_overlap_pca.png")
    print("- disease_overlap_pca_clusters.png")
    print("- disease_overlap_feature_importance.csv")
    print("- disease_overlap_report.txt")
    
    print("\n=== Key Findings ===")
    print(f"- AD-PD prediction correlation: {ad_pd_pred_corr:.4f}")
    print(f"- Percentage of samples in high overlap region: {high_overlap_percentage:.2f}%")
    print(f"- Top feature for overlap: {importance_df.iloc[0]['feature']}")

except Exception as e:
    print(f"Error occurred: {str(e)}")
    import traceback
    traceback.print_exc()