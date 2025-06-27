# %%
import os
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

# %% [markdown]
# # --- Settings ---

# %%
# Set working directory to the project root
# script_dir = "/beegfs/scratch/ric.broccoli/kubacki.michal/scMulti_merged_matrix" # Example path
script_dir = "D:/Github/scMulti_merged_matrix"
os.chdir(script_dir)
print(f"Working directory set to: {os.getcwd()}")

# Set figure parameters
sc.settings.set_figure_params(dpi=100, facecolor='white', frameon=False)
output_dir = 'correlation_analysis'
os.makedirs(output_dir, exist_ok=True)

# %% [markdown]
# # --- Load Data ---

# %%
print("Loading normalized WT data...")
try:
    adata = sc.read_h5ad('normalized_data.h5ad')
    print("Data loaded successfully.")
    print(adata)
except FileNotFoundError:
    print("Error: 'normalized_data.h5ad' not found.")
    print("Please run the 'process_sc_data.py' script first to generate it.")
    exit()

# %% [markdown]
# # --- Correlate Genes with Snrpn Expression ---

# %%
target_gene = 'Snrpn'# %%

# %%
print("Converting expression data to a DataFrame for correlation...")
# adata.X is a sparse matrix, convert to dense array for correlation calculation
expression_df = pd.DataFrame(adata.X.toarray(), columns=adata.var_names, index=adata.obs_names)

print("Calculating Spearman correlations with Snrpn...")
# Use pandas' corrwith for an efficient way to compute correlation of one column with all others.
# Spearman is used as it's rank-based and robust to outliers, common in scRNA-seq.
correlations = expression_df.corrwith(expression_df[target_gene], method='spearman')

# Drop the target gene itself from the results
correlations = correlations.drop(target_gene)

print("Correlation calculation complete.")

# %% [markdown]
# # --- Analyze and Save Results ---

# %%
# Sort correlations to find top positive and negative associations
sorted_correlations = correlations.sort_values(ascending=False)

# Create a results DataFrame
results_df = pd.DataFrame({
    'Gene': sorted_correlations.index,
    'Spearman_Correlation': sorted_correlations.values
})

# Save all correlations to a CSV file
output_csv_path = os.path.join(output_dir, 'snrpn_correlation_all_genes.csv')
results_df.to_csv(output_csv_path, index=False)
print(f"All gene correlations with {target_gene} saved to: {output_csv_path}")

# %%
# Display top N correlated genes
n_top = 20
print(f"\n--- Top {n_top} Positively Correlated Genes with {target_gene} ---")
print(results_df.head(n_top).to_string())

print(f"\n--- Top {n_top} Negatively Correlated Genes with {target_gene} ---")
print(results_df.tail(n_top).iloc[::-1].to_string())

# %% [markdown]
# # --- Visualize Correlations ---

# %%
# Get top positively and negatively correlated genes
top_pos_gene = results_df.iloc[0:10]['Gene']
top_pos_corr = results_df.iloc[0:10]['Spearman_Correlation']

print(f"\nTop positive correlated genes:")
for gene, corr in zip(top_pos_gene, top_pos_corr):
    print(f"  {gene}: {corr:.3f}")

top_neg_gene = results_df.iloc[-10:]['Gene']
top_neg_corr = results_df.iloc[-10:]['Spearman_Correlation']

print(f"\nTop negative correlated genes:")
for gene, corr in zip(top_neg_gene, top_neg_corr):
    print(f"  {gene}: {corr:.3f}")

# %%
# Horizontal bar plot showing top positive and negative correlations
def plot_top_correlations_barplot(results_df, n_top=15, output_path=None, title_prefix=""):
    """
    Creates a horizontal bar plot showing top positive and negative correlations.
    """
    # Get top positive and negative correlations
    top_positive = results_df.head(n_top)
    top_negative = results_df.tail(n_top).iloc[::-1]
    
    # Combine them
    top_genes = pd.concat([top_positive, top_negative])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 12))
    
    # Create color map - positive correlations in red, negative in blue
    colors = ['#1f77b4' if x > 0 else '#d62728' for x in top_genes['Spearman_Correlation']]
    
    # Create horizontal bar plot
    y_pos = np.arange(len(top_genes))
    ax.barh(y_pos, top_genes['Spearman_Correlation'], color=colors, alpha=0.8)
    
    # Customize plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_genes['Gene'])
    ax.set_xlabel('Spearman Correlation with Snrpn', fontsize=12)
    base_title = f'Top {n_top} Positive and Negative Gene Correlations with Snrpn'
    full_title = f"{title_prefix}: {base_title}" if title_prefix else base_title
    ax.set_title(full_title, fontsize=14, pad=20)
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(top_genes.iterrows()):
        value = row['Spearman_Correlation']
        if value > 0:
            ax.text(value + 0.01, i, f'{value:.3f}', va='center', fontsize=9)
        else:
            ax.text(value - 0.01, i, f'{value:.3f}', va='center', ha='right', fontsize=9)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Bar plot saved to {output_path}")
    
    plt.show()

plot_top_correlations_barplot(results_df, n_top=15, 
                              output_path=os.path.join(output_dir, 'snrpn_top_correlations_barplot.png'))

# %%
# Distribution of all correlations - histogram with statistics
def plot_correlation_distribution(correlations, output_path=None, title_prefix=""):
    """
    Creates a histogram showing the distribution of all correlations.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create histogram
    n, bins, patches = ax.hist(correlations, bins=50, alpha=0.7, color='#2ca02c', edgecolor='black')
    
    # Color bins by correlation value
    for i in range(len(patches)):
        if bins[i] < -0.2:
            patches[i].set_facecolor('#d62728')  # Strong negative - red
        elif bins[i] > 0.2:
            patches[i].set_facecolor('#1f77b4')   # Strong positive - blue
        else:
            patches[i].set_facecolor('#7f7f7f')  # Weak correlation - gray
    
    # Add vertical lines for key thresholds
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1, label='No correlation')
    ax.axvline(x=0.3, color='red', linestyle='--', linewidth=1, alpha=0.5, label='r = 0.3')
    ax.axvline(x=-0.3, color='blue', linestyle='--', linewidth=1, alpha=0.5, label='r = -0.3')
    
    # Add statistics
    mean_corr = correlations.mean()
    median_corr = correlations.median()
    ax.axvline(x=mean_corr, color='green', linestyle='-', linewidth=2, label=f'Mean = {mean_corr:.3f}')
    
    # Labels and title
    ax.set_xlabel('Spearman Correlation Coefficient', fontsize=12)
    ax.set_ylabel('Number of Genes', fontsize=12)
    base_title = 'Distribution of Gene Correlations with Snrpn'
    full_title = f"{title_prefix}: {base_title}" if title_prefix else base_title
    ax.set_title(full_title, fontsize=14, pad=20)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add text box with statistics
    textstr = f'Total genes: {len(correlations)}\n'
    textstr += f'Mean: {mean_corr:.3f}\n'
    textstr += f'Median: {median_corr:.3f}\n'
    textstr += f'Genes with |r| > 0.3: {sum(abs(correlations) > 0.3)}\n'
    textstr += f'Genes with |r| > 0.5: {sum(abs(correlations) > 0.5)}'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Distribution plot saved to {output_path}")
    
    plt.show()

plot_correlation_distribution(correlations, 
                              output_path=os.path.join(output_dir, 'snrpn_correlation_distribution.png'))

# %%
# Volcano-style plot showing correlation strength vs gene expression level
def plot_correlation_volcano(expression_df, results_df, output_path=None, title_prefix=""):
    """
    Creates a volcano-style plot showing correlation vs mean expression.
    """
    # Calculate mean expression for each gene
    mean_expression = expression_df.mean()
    
    # Merge with correlation data
    volcano_df = results_df.copy()
    volcano_df['Mean_Expression'] = volcano_df['Gene'].map(mean_expression)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create scatter plot
    scatter = ax.scatter(volcano_df['Mean_Expression'], 
                         volcano_df['Spearman_Correlation'],
                         c=volcano_df['Spearman_Correlation'],
                         cmap='RdBu_r',
                         s=20,
                         alpha=0.6,
                         vmin=-0.5, vmax=0.5)
    
    # Add horizontal lines for correlation thresholds
    ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='r = 0.3')
    ax.axhline(y=-0.3, color='blue', linestyle='--', alpha=0.5, label='r = -0.3')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Label highly correlated genes
    high_corr = volcano_df[abs(volcano_df['Spearman_Correlation']) > 0.3]
    for idx, row in high_corr.iterrows():
        ax.annotate(row['Gene'], 
                    (row['Mean_Expression'], row['Spearman_Correlation']),
                    xytext=(5, 5), 
                    textcoords='offset points',
                    fontsize=8,
                    alpha=0.7)
    
    # Labels and title
    ax.set_xlabel('Mean Expression (log-normalized)', fontsize=12)
    ax.set_ylabel('Spearman Correlation with Snrpn', fontsize=12)
    base_title = 'Gene Expression Level vs Correlation with Snrpn'
    full_title = f"{title_prefix}: {base_title}" if title_prefix else base_title
    ax.set_title(full_title, fontsize=14, pad=20)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Correlation', rotation=270, labelpad=20)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Volcano plot saved to {output_path}")
    
    plt.show()

plot_correlation_volcano(expression_df, results_df,
                         output_path=os.path.join(output_dir, 'snrpn_correlation_volcano.png'))

# %%
# Improved Heatmap of top correlated genes across cells
def plot_correlation_heatmap_improved(adata, results_df, n_genes=30, subsample_cells=np.inf, output_path=None, title_prefix=""):
    """
    Creates an improved heatmap showing expression patterns of top correlated genes.
    """
    from sklearn.preprocessing import StandardScaler
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import pdist
    
    # Get top positive and negative correlated genes
    n_each = n_genes // 2
    top_pos_genes = results_df.head(n_each)['Gene'].tolist()
    top_neg_genes = results_df.tail(n_each)['Gene'].tolist()
    
    # Create ordered gene list: positive correlations first, then negative
    genes_to_plot = [target_gene] + top_pos_genes + top_neg_genes[::-1]  # Reverse negative genes
    
    # Subsample cells for cleaner visualization
    if subsample_cells and adata.n_obs > subsample_cells:
        np.random.seed(42)
        cell_indices = np.random.choice(adata.n_obs, subsample_cells, replace=False)
        adata_subset = adata[cell_indices, :]
    else:
        adata_subset = adata
    
    # Extract expression data for selected genes
    expr_matrix = adata_subset[:, genes_to_plot].X.toarray().T
    
    # Z-score normalize each gene for better visualization
    scaler = StandardScaler()
    expr_matrix_scaled = scaler.fit_transform(expr_matrix.T).T
    
    # Sort cells by Snrpn expression level (first row/column)
    snrpn_expression = expr_matrix_scaled[0, :]  # Snrpn is the first gene
    sorted_indices = np.argsort(snrpn_expression)
    expr_matrix_scaled = expr_matrix_scaled[:, sorted_indices]
    
    fig, (ax_heatmap, ax_corr) = plt.subplots(1, 2, figsize=(16, 10), 
                                               gridspec_kw={'width_ratios': [20, 1], 'wspace': 0.05})
    
    # Create heatmap
    im = ax_heatmap.imshow(expr_matrix_scaled, aspect='auto', cmap='RdBu_r', 
                           vmin=-3, vmax=3, interpolation='nearest')
    
    # Customize gene labels
    gene_labels = []
    gene_colors = []
    for i, gene in enumerate(genes_to_plot):
        if gene == target_gene:
            gene_labels.append(f">>> {gene} <<<")
            gene_colors.append('black')
        else:
            corr = results_df[results_df['Gene'] == gene]['Spearman_Correlation'].values[0]
            gene_labels.append(f"{gene} (r={corr:.2f})")
            gene_colors.append('darkred' if corr > 0 else 'darkblue')
    
    # Set labels with colors
    ax_heatmap.set_yticks(np.arange(len(genes_to_plot)))
    ax_heatmap.set_yticklabels(gene_labels, fontsize=11)
    
    # Color the labels
    for i, (label, color) in enumerate(zip(ax_heatmap.get_yticklabels(), gene_colors)):
        label.set_color(color)
        if i == 0:  # Snrpn
            label.set_weight('bold')
            label.set_fontsize(12)
    
    # Reduce number of x-axis labels for clarity
    n_xticks = 5
    xtick_positions = np.linspace(0, expr_matrix_scaled.shape[1]-1, n_xticks, dtype=int)
    ax_heatmap.set_xticks(xtick_positions)
    ax_heatmap.set_xticklabels(xtick_positions)
    ax_heatmap.set_xlabel(f'Cells (n={expr_matrix_scaled.shape[1]}, sorted by Snrpn expression)', fontsize=12)
    
    # Add title
    base_title = f'Expression Patterns: Snrpn and Top {n_genes} Correlated Genes\n(Z-score normalized)'
    full_title = f"{title_prefix}: {base_title}" if title_prefix else base_title
    ax_heatmap.set_title(full_title, 
                         fontsize=14, pad=20, weight='bold')
    
    # Add separation lines
    # Line after Snrpn
    ax_heatmap.axhline(y=0.5, color='black', linewidth=3, linestyle='-')
    # Line between positive and negative correlations
    ax_heatmap.axhline(y=n_each + 0.5, color='gray', linewidth=2, linestyle='--')
    
    # Create correlation bar on the right
    corr_values = []
    for gene in genes_to_plot:
        if gene == target_gene:
            corr_values.append(1.0)  # Snrpn correlation with itself
        else:
            corr_val = results_df[results_df['Gene'] == gene]['Spearman_Correlation'].values[0]
            corr_values.append(corr_val)

    ax_corr.set_ylim(ax_heatmap.get_ylim())
    
    # Add correlation values as text
    for i, corr in enumerate(corr_values):
        if corr == 1.0:
            text = "1.00"
        else:
            text = f"{corr:.2f}"
        ax_corr.text(0, i, text, ha='center', va='center', 
                     fontsize=9, weight='bold' if abs(corr) > 0.3 else 'normal')
    
    # Clean up correlation axis
    ax_corr.set_xticks([])
    ax_corr.set_yticks([])
    ax_corr.set_xlabel('Corr.', fontsize=10)
    ax_corr.spines['top'].set_visible(False)
    ax_corr.spines['right'].set_visible(False)
    ax_corr.spines['bottom'].set_visible(False)
    ax_corr.spines['left'].set_visible(False)
    
    # Add main colorbar
    cbar = plt.colorbar(im, ax=[ax_heatmap, ax_corr], location='right', 
                        fraction=0.02, pad=0.1)
    cbar.set_label('Z-score normalized expression', rotation=270, labelpad=20, fontsize=11)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Improved heatmap saved to {output_path}")
    
    plt.show()

plot_correlation_heatmap_improved(adata, results_df, n_genes=30, subsample_cells=300,
                                  output_path=os.path.join(output_dir, 'snrpn_expression_heatmap_improved.png'))

# %% [markdown]
# # --- Cell-Type-Specific Correlation Analysis ---

# %%
# Add cell type annotations from the 'process_sc_data.py' script
print("\n--- Annotating cell types for specific analysis ---")
cluster_mapping = {
    "Astro" : ["11"],
    "CC_upper" : ["6", "3", "2", "13", "9", "10", "7"],
    "Meningi" : ["18"],
    "Oligos" : ["16", "14"],
    "Unk" : ["8", "1", "17"],
    "CC_deeper" : ["4"],
    "GABA" : ["5"],
    "Micro" : ["12"],
    "OPCs" : ["15"]
}

if 'cluster' not in adata.obs.columns:
    print("Error: 'cluster' column not found in adata.obs. This is required for cell type annotation.")
    print("Please ensure 'normalized_data.h5ad' was generated from a script that adds cluster info.")
    exit()

# Ensure cluster column is of string type for matching
adata.obs['cluster'] = adata.obs['cluster'].astype(str)
adata.obs['cell_type'] = 'Unasigned'
for ct, clusters in cluster_mapping.items():
    adata.obs.loc[adata.obs['cluster'].isin(clusters), 'cell_type'] = ct

adata.obs['cell_type'] = adata.obs['cell_type'].astype('category')
print("Cell types annotated successfully:")
print(adata.obs['cell_type'].value_counts())

# %%
# Loop through each cell type to perform the full analysis
all_cell_types = adata.obs['cell_type'].cat.categories.tolist()
cell_types_to_analyze = [ct for ct in all_cell_types if ct not in ['Unk', 'Unasigned']]

print(f"\nStarting analysis for individual cell types: {cell_types_to_analyze}")

for cell_type in cell_types_to_analyze:
    print("\n" + "="*80)
    print(f"--- Processing cell type: {cell_type} ---")
    print("="*80)

    # 1. Filter data for the current cell type
    adata_ct = adata[adata.obs['cell_type'] == cell_type].copy()
    
    # Check for sufficient cell numbers
    if adata_ct.n_obs < 50:
        print(f"Skipping '{cell_type}' due to insufficient cells ({adata_ct.n_obs}). Minimum is 50.")
        continue
    
    print(f"Found {adata_ct.n_obs} cells for '{cell_type}'.")
    
    # Create dedicated output directory
    cell_type_dir = os.path.join(output_dir, cell_type.replace(' ', '_').replace('/', '_'))
    os.makedirs(cell_type_dir, exist_ok=True)

    # 2. Recalculate correlations within the cell type
    expression_df_ct = pd.DataFrame(adata_ct.X.toarray(), columns=adata_ct.var_names, index=adata_ct.obs_names)
    
    # Check for variance in target gene to avoid errors
    if expression_df_ct[target_gene].std() == 0:
        print(f"Skipping '{cell_type}' because target gene '{target_gene}' has zero expression variance in this subset.")
        continue
        
    correlations_ct = expression_df_ct.corrwith(expression_df_ct[target_gene], method='spearman')
    correlations_ct = correlations_ct.drop(target_gene, errors='ignore').dropna()
    
    if correlations_ct.empty:
        print(f"Skipping '{cell_type}' as no valid correlations could be calculated.")
        continue

    # 3. Save results
    results_df_ct = pd.DataFrame({
        'Gene': correlations_ct.index,
        'Spearman_Correlation': correlations_ct.values
    }).sort_values(by='Spearman_Correlation', ascending=False)
    
    output_csv_path_ct = os.path.join(cell_type_dir, f'correlation_results_{cell_type}.csv')
    results_df_ct.to_csv(output_csv_path_ct, index=False)
    print(f"Correlation results for {cell_type} saved to: {output_csv_path_ct}")

    # 4. Generate and save visualizations for the cell type
    print(f"Generating visualizations for {cell_type}...")
    
    plot_top_correlations_barplot(results_df_ct, n_top=15, title_prefix=cell_type,
                                  output_path=os.path.join(cell_type_dir, 'barplot_top_correlations.png'))

    plot_correlation_distribution(correlations_ct, title_prefix=cell_type,
                                  output_path=os.path.join(cell_type_dir, 'distribution_all_correlations.png'))

    plot_correlation_volcano(expression_df_ct, results_df_ct, title_prefix=cell_type,
                             output_path=os.path.join(cell_type_dir, 'volcano_correlation_vs_expression.png'))
    
    subsample_n = min(300, adata_ct.n_obs)
    plot_correlation_heatmap_improved(adata_ct, results_df_ct, n_genes=30, subsample_cells=subsample_n, title_prefix=cell_type,
                                      output_path=os.path.join(cell_type_dir, 'heatmap_top_genes.png'))

print("\n\nCell-type-specific analysis complete.")


