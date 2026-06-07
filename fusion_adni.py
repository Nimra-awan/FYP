import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("=== ADNI Image Classification and Analysis ===")
print("Loading and fusing datasets...")

# Load the datasets
try:
    # Load the features dataset
    features_df = pd.read_csv('resnet50_selected_features_cumulative_updated.csv')
    print(f"Features dataset loaded: {features_df.shape}")
    
    # Load the metadata dataset
    metadata_df = pd.read_csv('ADNI1_Merged_AllImages.csv')
    print(f"Metadata dataset loaded: {metadata_df.shape}")
    
    # Display basic info about the datasets
    print("\nFeatures dataset info:")
    print(f"Columns: {features_df.columns.tolist()[:5]}... (and {len(features_df.columns)-5} more)")
    print(f"Classes: {features_df['class'].unique()}")
    print(f"Class distribution:\n{features_df['class'].value_counts()}")
    
    print("\nMetadata dataset info:")
    print(f"Columns: {metadata_df.columns.tolist()[:10]}... (and {len(metadata_df.columns)-10} more)")
    print(f"Groups: {metadata_df['Group'].unique()}")
    print(f"Group distribution:\n{metadata_df['Group'].value_counts()}")
    
    # Fuse the datasets based on Image Data ID
    print("\nFusing datasets...")
    
    # Create a mapping from filename to Image Data ID
    # The filename in features_df is like "I101549" and Image Data ID in metadata_df is like "I112538"
    # We need to match them properly
    
    # First, let's see what the actual mapping should be
    print("\nSample filenames from features dataset:")
    print(features_df['filename'].head(10).tolist())
    
    print("\nSample Image Data IDs from metadata dataset:")
    print(metadata_df['Image Data ID'].head(10).tolist())
    
    # Create a mapping - we'll use the Image Data ID as the key
    # For now, let's assume the filename in features_df corresponds to the Image Data ID
    merged_df = features_df.merge(metadata_df, left_on='filename', right_on='Image Data ID', how='inner')
    
    print(f"\nMerged dataset shape: {merged_df.shape}")
    print(f"Number of matched records: {len(merged_df)}")
    
    if len(merged_df) == 0:
        print("No matches found. Trying alternative matching strategy...")
        # Try matching based on Subject ID
        merged_df = features_df.merge(metadata_df, left_on='filename', right_on='Subject', how='inner')
        print(f"Alternative merge shape: {merged_df.shape}")
    
    if len(merged_df) == 0:
        print("Still no matches. Creating synthetic analysis with available data...")
        # Use the features dataset directly for analysis
        analysis_df = features_df.copy()
        analysis_df['Group'] = analysis_df['class']  # Use class as group
    else:
        analysis_df = merged_df.copy()
    
    print(f"\nFinal analysis dataset shape: {analysis_df.shape}")
    
    # Data preprocessing
    print("\nPreprocessing data...")
    
    # Select features (exclude non-feature columns)
    feature_columns = [col for col in analysis_df.columns if col.startswith('f') and col != 'filename']
    print(f"Number of features: {len(feature_columns)}")
    
    # Check for non-numeric values in feature columns
    print("\nChecking for non-numeric values in features...")
    for col in feature_columns:
        non_numeric = pd.to_numeric(analysis_df[col], errors='coerce').isna().sum()
        if non_numeric > 0:
            print(f"Column {col}: {non_numeric} non-numeric values found")
            # Convert to numeric, replacing non-numeric with NaN
            analysis_df[col] = pd.to_numeric(analysis_df[col], errors='coerce')
    
    # Remove rows with NaN values in features
    initial_rows = len(analysis_df)
    analysis_df = analysis_df.dropna(subset=feature_columns)
    final_rows = len(analysis_df)
    print(f"Removed {initial_rows - final_rows} rows with NaN values")
    
    # Prepare features and labels
    X = analysis_df[feature_columns].values
    y = analysis_df['class'].values if 'class' in analysis_df.columns else analysis_df['Group'].values
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Label distribution: {np.unique(y, return_counts=True)}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Train Random Forest Classifier
    print("\nTraining Random Forest Classifier...")
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_classifier.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = rf_classifier.predict(X_test_scaled)
    y_pred_proba = rf_classifier.predict_proba(X_test_scaled)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.2%}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = rf_classifier.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Most Important Features:")
    print(feature_importance_df.head(10))
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # 1. Confusion Matrix with Accuracy
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    
    # Calculate accuracy for each class
    class_accuracy = {}
    for i, class_name in enumerate(np.unique(y)):
        true_positives = cm[i, i]
        total = np.sum(cm[i, :])
        class_accuracy[class_name] = true_positives / total if total > 0 else 0
    
    # Create annotations with counts and accuracy
    annot = np.empty_like(cm, dtype=object)
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i == j:  # Diagonal elements (correct predictions)
                annot[i, j] = f'{cm[i, j]}\n{class_accuracy[np.unique(y)[i]]:.2%}'
            else:
                annot[i, j] = f'{cm[i, j]}'
    
    # Create heatmap with custom annotations
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', 
                xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title('Confusion Matrix with Class Accuracy')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Add a note explaining the annotations
    plt.figtext(0.5, 0.01, 'Diagonal cells show: count\naccuracy', 
                ha='center', fontsize=10, bbox={'facecolor':'white', 'alpha':0.8, 'pad':5})
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Feature Importance Plot
    plt.figure(figsize=(12, 8))
    top_features = feature_importance_df.head(20)
    sns.barplot(data=top_features, x='importance', y='feature')
    plt.title('Top 20 Most Important Features')
    plt.xlabel('Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Class Distribution
    plt.figure(figsize=(10, 6))
    class_counts = pd.Series(y).value_counts()
    sns.barplot(x=class_counts.index, y=class_counts.values)
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. PCA Visualization
    print("\nCreating PCA visualization...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train_scaled)
    
    plt.figure(figsize=(12, 8))
    unique_classes = np.unique(y_train)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_classes)))
    
    for i, class_label in enumerate(unique_classes):
        mask = y_train == class_label
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   c=[colors[i]], label=class_label, alpha=0.7, s=50)
    
    plt.title('PCA Visualization of Features')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('pca_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. t-SNE Visualization (if dataset is not too large)
    if len(X_train) <= 5000:  # t-SNE is computationally expensive
        print("\nCreating t-SNE visualization...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_train)-1))
        X_tsne = tsne.fit_transform(X_train_scaled[:min(2000, len(X_train))])
        y_tsne = y_train[:min(2000, len(X_train))]
        
        plt.figure(figsize=(12, 8))
        for i, class_label in enumerate(unique_classes):
            mask = y_tsne == class_label
            plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                       c=[colors[i]], label=class_label, alpha=0.7, s=50)
        
        plt.title('t-SNE Visualization of Features')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('tsne_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 6. Model Performance Summary
    print("\nCreating performance summary...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy by class
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    class_metrics = {}
    for class_label in unique_classes:
        mask = y_test == class_label
        if np.sum(mask) > 0:
            class_acc = accuracy_score(y_test[mask], y_pred[mask])
            class_metrics[class_label] = class_acc
    
    axes[0, 0].bar(class_metrics.keys(), class_metrics.values())
    axes[0, 0].set_title('Accuracy by Class')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Feature importance distribution
    axes[0, 1].hist(feature_importance, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Feature Importance Distribution')
    axes[0, 1].set_xlabel('Importance')
    axes[0, 1].set_ylabel('Frequency')
    
    # Training vs Test accuracy
    train_accuracy = rf_classifier.score(X_train_scaled, y_train)
    test_accuracy = accuracy_score(y_test, y_pred)
    axes[1, 0].bar(['Training', 'Test'], [train_accuracy, test_accuracy])
    axes[1, 0].set_title('Training vs Test Accuracy')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_ylim(0, 1)
    
    # Class distribution in train vs test
    train_counts = pd.Series(y_train).value_counts()
    test_counts = pd.Series(y_test).value_counts()
    
    x = np.arange(len(unique_classes))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, [train_counts.get(c, 0) for c in unique_classes], 
                   width, label='Training', alpha=0.8)
    axes[1, 1].bar(x + width/2, [test_counts.get(c, 0) for c in unique_classes], 
                   width, label='Test', alpha=0.8)
    axes[1, 1].set_title('Class Distribution: Training vs Test')
    axes[1, 1].set_xlabel('Class')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(unique_classes, rotation=45)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('performance_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 7. Visualization of Border Cases (Misclassified Instances)
    print("\nVisualizing border cases (misclassified instances)...")
    
    # Get predictions and actual labels
    y_pred_test = rf_classifier.predict(X_test_scaled)
    
    # Create a DataFrame with test data, actual labels, and predictions
    misclassification_df = pd.DataFrame(X_test_scaled, columns=[f'feature_{i}' for i in range(X_test_scaled.shape[1])])
    misclassification_df['actual'] = y_test
    misclassification_df['predicted'] = y_pred_test
    misclassification_df['misclassified'] = misclassification_df['actual'] != misclassification_df['predicted']
    
    # Filter for MCI cases that were misclassified as AD or CN
    mci_to_ad = misclassification_df[(misclassification_df['actual'] == 'MCI') & 
                                     (misclassification_df['predicted'] == 'AD')]
    mci_to_cn = misclassification_df[(misclassification_df['actual'] == 'MCI') & 
                                     (misclassification_df['predicted'] == 'CN')]
    
    print(f"Number of MCI cases misclassified as AD: {len(mci_to_ad)}")
    print(f"Number of MCI cases misclassified as CN: {len(mci_to_cn)}")
    
    # Apply PCA for visualization
    pca_border = PCA(n_components=2)
    X_test_pca = pca_border.fit_transform(X_test_scaled)
    
    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({
        'PC1': X_test_pca[:, 0],
        'PC2': X_test_pca[:, 1],
        'actual': y_test,
        'predicted': y_pred_test,
        'misclassified': y_test != y_pred_test
    })
    
    # Plot the border cases
    plt.figure(figsize=(14, 10))
    
    # Plot correctly classified instances with lower alpha
    for label in np.unique(y_test):
        mask = (plot_df['actual'] == label) & (~plot_df['misclassified'])
        plt.scatter(plot_df.loc[mask, 'PC1'], plot_df.loc[mask, 'PC2'], 
                   alpha=0.3, s=30, label=f'{label} (correct)')
    
    # Highlight MCI misclassified as AD
    mask_mci_ad = (plot_df['actual'] == 'MCI') & (plot_df['predicted'] == 'AD')
    plt.scatter(plot_df.loc[mask_mci_ad, 'PC1'], plot_df.loc[mask_mci_ad, 'PC2'], 
               color='red', edgecolor='black', s=100, marker='*', 
               label='MCI misclassified as AD')
    
    # Highlight MCI misclassified as CN
    mask_mci_cn = (plot_df['actual'] == 'MCI') & (plot_df['predicted'] == 'CN')
    plt.scatter(plot_df.loc[mask_mci_cn, 'PC1'], plot_df.loc[mask_mci_cn, 'PC2'], 
               color='purple', edgecolor='black', s=100, marker='P', 
               label='MCI misclassified as CN')
    
    plt.title('PCA Visualization of Border Cases (Misclassified MCI Instances)')
    plt.xlabel(f'PC1')
    plt.ylabel(f'PC2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('border_cases_pca.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # t-SNE visualization of border cases
    if len(X_test) <= 5000:  # t-SNE is computationally expensive
        print("Creating t-SNE visualization of border cases...")
        tsne_border = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_test)-1))
        X_test_tsne = tsne_border.fit_transform(X_test_scaled)
        
        # Create a DataFrame for plotting
        tsne_plot_df = pd.DataFrame({
            'TSNE1': X_test_tsne[:, 0],
            'TSNE2': X_test_tsne[:, 1],
            'actual': y_test,
            'predicted': y_pred_test,
            'misclassified': y_test != y_pred_test
        })
        
        # Plot the border cases with t-SNE
        plt.figure(figsize=(14, 10))
        
        # Plot correctly classified instances with lower alpha
        for label in np.unique(y_test):
            mask = (tsne_plot_df['actual'] == label) & (~tsne_plot_df['misclassified'])
            plt.scatter(tsne_plot_df.loc[mask, 'TSNE1'], tsne_plot_df.loc[mask, 'TSNE2'], 
                       alpha=0.3, s=30, label=f'{label} (correct)')
        
        # Highlight MCI misclassified as AD
        mask_mci_ad = (tsne_plot_df['actual'] == 'MCI') & (tsne_plot_df['predicted'] == 'AD')
        plt.scatter(tsne_plot_df.loc[mask_mci_ad, 'TSNE1'], tsne_plot_df.loc[mask_mci_ad, 'TSNE2'], 
                   color='red', edgecolor='black', s=100, marker='*', 
                   label='MCI misclassified as AD')
        
        # Highlight MCI misclassified as CN
        mask_mci_cn = (tsne_plot_df['actual'] == 'MCI') & (tsne_plot_df['predicted'] == 'CN')
        plt.scatter(tsne_plot_df.loc[mask_mci_cn, 'TSNE1'], tsne_plot_df.loc[mask_mci_cn, 'TSNE2'], 
                   color='purple', edgecolor='black', s=100, marker='P', 
                   label='MCI misclassified as CN')
        
        plt.title('t-SNE Visualization of Border Cases (Misclassified MCI Instances)')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('border_cases_tsne.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Separate visualizations for MCI misclassified as AD and MCI misclassified as CN
    if len(mci_to_ad) > 0 or len(mci_to_cn) > 0:
        print("\nCreating separate visualizations for misclassified MCI cases...")
        
        # Get indices for different cases
        misclassified_indices = np.where(y_test != y_pred_test)[0]
        mci_to_ad_indices = np.where((y_test == 'MCI') & (y_pred_test == 'AD'))[0]
        mci_to_cn_indices = np.where((y_test == 'MCI') & (y_pred_test == 'CN'))[0]
        
        # 1. Separate visualization for MCI misclassified as AD using PCA
        if len(mci_to_ad) > 0:
            plt.figure(figsize=(14, 10))
            
            # Plot all MCI cases with lower alpha
            mci_mask = (plot_df['actual'] == 'MCI')
            plt.scatter(plot_df.loc[mci_mask, 'PC1'], plot_df.loc[mci_mask, 'PC2'], 
                       alpha=0.3, s=30, color='blue', label='MCI (all)')
            
            # Plot all AD cases with lower alpha
            ad_mask = (plot_df['actual'] == 'AD')
            plt.scatter(plot_df.loc[ad_mask, 'PC1'], plot_df.loc[ad_mask, 'PC2'], 
                       alpha=0.3, s=30, color='red', label='AD (all)')
            
            # Highlight MCI misclassified as AD
            mask_mci_ad = (plot_df['actual'] == 'MCI') & (plot_df['predicted'] == 'AD')
            plt.scatter(plot_df.loc[mask_mci_ad, 'PC1'], plot_df.loc[mask_mci_ad, 'PC2'], 
                       color='purple', edgecolor='black', s=150, marker='*', 
                       label='MCI misclassified as AD')
            
            plt.title('PCA Visualization of MCI Cases Misclassified as AD')
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('mci_misclassified_as_ad_pca.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # t-SNE visualization for MCI misclassified as AD
            if len(X_test) <= 5000 and 'tsne_plot_df' in locals():
                plt.figure(figsize=(14, 10))
                
                # Plot all MCI cases with lower alpha
                plt.scatter(tsne_plot_df.loc[mci_mask, 'TSNE1'], tsne_plot_df.loc[mci_mask, 'TSNE2'], 
                           alpha=0.3, s=30, color='blue', label='MCI (all)')
                
                # Plot all AD cases with lower alpha
                plt.scatter(tsne_plot_df.loc[ad_mask, 'TSNE1'], tsne_plot_df.loc[ad_mask, 'TSNE2'], 
                           alpha=0.3, s=30, color='red', label='AD (all)')
                
                # Highlight MCI misclassified as AD
                plt.scatter(tsne_plot_df.loc[mask_mci_ad, 'TSNE1'], tsne_plot_df.loc[mask_mci_ad, 'TSNE2'], 
                           color='purple', edgecolor='black', s=150, marker='*', 
                           label='MCI misclassified as AD')
                
                plt.title('t-SNE Visualization of MCI Cases Misclassified as AD')
                plt.xlabel('t-SNE 1')
                plt.ylabel('t-SNE 2')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig('mci_misclassified_as_ad_tsne.png', dpi=300, bbox_inches='tight')
                plt.show()
        
        # 2. Separate visualization for MCI misclassified as CN using PCA
        if len(mci_to_cn) > 0:
            plt.figure(figsize=(14, 10))
            
            # Plot all MCI cases with lower alpha
            mci_mask = (plot_df['actual'] == 'MCI')
            plt.scatter(plot_df.loc[mci_mask, 'PC1'], plot_df.loc[mci_mask, 'PC2'], 
                       alpha=0.3, s=30, color='blue', label='MCI (all)')
            
            # Plot all CN cases with lower alpha
            cn_mask = (plot_df['actual'] == 'CN')
            plt.scatter(plot_df.loc[cn_mask, 'PC1'], plot_df.loc[cn_mask, 'PC2'], 
                       alpha=0.3, s=30, color='green', label='CN (all)')
            
            # Highlight MCI misclassified as CN
            mask_mci_cn = (plot_df['actual'] == 'MCI') & (plot_df['predicted'] == 'CN')
            plt.scatter(plot_df.loc[mask_mci_cn, 'PC1'], plot_df.loc[mask_mci_cn, 'PC2'], 
                       color='orange', edgecolor='black', s=150, marker='P', 
                       label='MCI misclassified as CN')
            
            plt.title('PCA Visualization of MCI Cases Misclassified as CN')
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('mci_misclassified_as_cn_pca.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # t-SNE visualization for MCI misclassified as CN
            if len(X_test) <= 5000 and 'tsne_plot_df' in locals():
                plt.figure(figsize=(14, 10))
                
                # Plot all MCI cases with lower alpha
                plt.scatter(tsne_plot_df.loc[mci_mask, 'TSNE1'], tsne_plot_df.loc[mci_mask, 'TSNE2'], 
                           alpha=0.3, s=30, color='blue', label='MCI (all)')
                
                # Plot all CN cases with lower alpha
                plt.scatter(tsne_plot_df.loc[cn_mask, 'TSNE1'], tsne_plot_df.loc[cn_mask, 'TSNE2'], 
                           alpha=0.3, s=30, color='green', label='CN (all)')
                
                # Highlight MCI misclassified as CN
                plt.scatter(tsne_plot_df.loc[mask_mci_cn, 'TSNE1'], tsne_plot_df.loc[mask_mci_cn, 'TSNE2'], 
                           color='orange', edgecolor='black', s=150, marker='P', 
                           label='MCI misclassified as CN')
                
                plt.title('t-SNE Visualization of MCI Cases Misclassified as CN')
                plt.xlabel('t-SNE 1')
                plt.ylabel('t-SNE 2')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig('mci_misclassified_as_cn_tsne.png', dpi=300, bbox_inches='tight')
                plt.show()
    
    # Save results
    print("\nSaving results...")
    results = {
        'accuracy': accuracy,
        'feature_importance': feature_importance_df,
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': cm.tolist(),
        'top_features': feature_importance_df.head(20).to_dict('records'),
        'border_cases': {
            'mci_to_ad': len(mci_to_ad),
            'mci_to_cn': len(mci_to_cn)
        }
    }
    
    # Save feature importance to CSV
    feature_importance_df.to_csv('feature_importance_ranking.csv', index=False)
    
    # Create a summary report
    with open('classification_report.txt', 'w') as f:
        f.write("ADNI Image Classification Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dataset Information:\n")
        f.write(f"- Total samples: {len(analysis_df)}\n")
        f.write(f"- Number of features: {len(feature_columns)}\n")
        f.write(f"- Classes: {list(unique_classes)}\n")
        f.write(f"- Training samples: {len(X_train)}\n")
        f.write(f"- Test samples: {len(X_test)}\n\n")
        
        f.write(f"Model Performance:\n")
        f.write(f"- Overall Accuracy: {accuracy:.2%}\n")
        f.write(f"- Training Accuracy: {train_accuracy:.2%}\n")
        f.write(f"- Test Accuracy: {test_accuracy:.2%}\n\n")
        
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred))
        
        f.write("\nBorder Cases Analysis (Misclassified MCI Instances):\n")
        f.write(f"- MCI cases misclassified as AD: {len(mci_to_ad)}\n")
        f.write(f"- MCI cases misclassified as CN: {len(mci_to_cn)}\n")
        
        # Calculate percentages if there are MCI cases
        mci_total = np.sum(y_test == 'MCI')
        if mci_total > 0:
            mci_to_ad_pct = (len(mci_to_ad) / mci_total) * 100
            mci_to_cn_pct = (len(mci_to_cn) / mci_total) * 100
            f.write(f"- Percentage of MCI misclassified as AD: {mci_to_ad_pct:.2f}%\n")
            f.write(f"- Percentage of MCI misclassified as CN: {mci_to_cn_pct:.2f}%\n")
        
        f.write("\nTop 10 Most Important Features:\n")
        for i, row in feature_importance_df.head(10).iterrows():
            f.write(f"{i+1}. {row['feature']}: {row['importance']:.4f}\n")
    
    print("\n=== Analysis Complete ===")
    print("Generated files:")
    print("- confusion_matrix.png")
    print("- feature_importance.png") 
    print("- class_distribution.png")
    print("- pca_visualization.png")
    print("- tsne_visualization.png (if dataset size allows)")
    print("- performance_summary.png")
    print("- border_cases_pca.png")
    print("- border_cases_tsne.png (if dataset size allows)")
    print("- mci_misclassified_as_ad_pca.png (if such cases exist)")
    print("- mci_misclassified_as_ad_tsne.png (if such cases exist)")
    print("- mci_misclassified_as_cn_pca.png (if such cases exist)")
    print("- mci_misclassified_as_cn_tsne.png (if such cases exist)")
    print("- feature_importance_ranking.csv")
    print("- classification_report.txt")
    
    # Display the accuracy as a percentage
    print("\n=== Model Accuracy ===")
    print(f"Overall Test Accuracy: {accuracy:.2%}")
    print(f"Training Accuracy: {train_accuracy:.2%}")
    print(f"Test Accuracy: {test_accuracy:.2%}")
    
    print(f"\nFinal Results:")
    print(f"- Model Accuracy: {accuracy:.2%}")
    print(f"- Number of features used: {len(feature_columns)}")
    print(f"- Classes predicted: {list(unique_classes)}")
    print(f"- MCI cases misclassified as AD: {len(mci_to_ad)}")
    print(f"- MCI cases misclassified as CN: {len(mci_to_cn)}")
    
except Exception as e:
    print(f"Error occurred: {str(e)}")
    import traceback
    traceback.print_exc()
