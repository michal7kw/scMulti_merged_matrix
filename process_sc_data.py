# %%
import os
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import re

# %% [markdown]
# # --- Settings ---

# %%

# script_dir = "/beegfs/scratch/ric.broccoli/kubacki.michal/scMulti_merged_matrix"
script_dir = "D:/Github/scMulti_merged_matrix"
os.chdir(script_dir)
print(f"Working directory set to: {script_dir}")

# %%
# Set verbosity to 3 to see more informative output
sc.settings.verbosity = 3
# Set figure parameters
sc.settings.set_figure_params(dpi=80, facecolor='white')

# %% [markdown]
# # --- Load Data ---

# %%
print("Loading data...")
data_dir = 'filtered_feature_bc_matrix/'
adata = sc.read_10x_mtx(
    data_dir,
    var_names='gene_symbols',
    cache=True)

# %%
adata.var_names_make_unique()

# %%
print("Data loaded successfully.")
print(adata)

# %%
##################################################### Temp #####################################################
target_gene = 'RBFOX3' # Also known as NeuN
gene_present = False

if target_gene in adata.var_names:
    print(f"Gene '{target_gene}' found in the dataset.")
    gene_present = True
else:
    # Check for case variations or common aliases if any
    print(f"Gene '{target_gene}' not found with exact match.")
    # Attempt to find variations including common synonyms and species-specific names
    possible_variations = [
        target_gene.lower(),           # rbfox3
        target_gene.upper(),           # RBFOX3
        target_gene.capitalize(),      # Rbfox3
        'Rbfox3',                      # Common mouse gene name
        'rbfox3',                      # Lowercase mouse
        'RBFOX3',                      # Human uppercase
        'Rbfox3'                       # Human/Mouse mixed case
    ]

    for var in possible_variations:
        if var in adata.var_names:
            print(f"Found alternative name: '{var}'. Using this for analysis.")
            target_gene = var
            gene_present = True
            break
    if not gene_present:
        print(f"Gene '{target_gene}' and its common variations not found in adata.var_names.")
        print("Available genes (first 100):")
        print(list(adata.var_names[:100]))
##################################################### Temp #####################################################

# %%
##################################################### Temp #####################################################
# Check if SNRPN and snord116 genes are present in the dataset
target_genes = ['SNRPN', 'snord116', 'Snrpn', 'Snord116']
target_patterns = ['SNRPN*', 'snord116*', 'Snrpn*', 'Snord116*']  # Patterns to check
available_genes = []
pattern_matches = {}

print("\nChecking for target genes in the dataset...")

# Check exact matches
for gene in target_genes:
    if gene in adata.var_names:
        available_genes.append(gene)
        print(f"Found exact match: {gene}")
    else:
        print(f"Not found: {gene}")

print("\nChecking for pattern matches...")

# Check regex patterns
for pattern in target_patterns:
    # Convert glob pattern to regex (replace * with .*)
    regex_pattern = pattern.replace('*', '.*')
    # Create regex object (case-sensitive by default)
    regex = re.compile(f'^{regex_pattern}$')
    
    # Find all matching genes
    matches = [gene for gene in adata.var_names if regex.match(gene)]
    
    if matches:
        pattern_matches[pattern] = matches
        print(f"\nPattern '{pattern}' matches:")
        for match in matches:
            print(f"  - {match}")
            if match not in available_genes:
                available_genes.append(match)
    else:
        print(f"\nNo matches for pattern: {pattern}")

print(f"\nTotal unique genes found: {len(available_genes)}")
print(f"All found genes: {available_genes}")

# Also search for partial matches
print("\nSearching for genes containing 'snrpn' or 'snord'...")
snrpn_like = adata.var_names[adata.var_names.str.contains('snrpn', case=False, na=False)]
snord_like = adata.var_names[adata.var_names.str.contains('snord', case=False, na=False)]

if len(snrpn_like) > 0:
    print(f"SNRPN-like genes found: {list(snrpn_like)}")
    available_genes.extend(snrpn_like)
if len(snord_like) > 0:
    print(f"SNORD-like genes found: {list(snord_like)}")
    available_genes.extend(snord_like)
##################################################### Temp #####################################################

# %%
available_genes = ['Snrpn', 'Rbfox3']

# %% [markdown]
# # --- Add Metadata from the Downloaded CSV File ---

# %%
metadata_csv_file = './LibraryID.csv'

# %%
print(f"\n--- Step 8: Loading metadata from '{metadata_csv_file}' ---")
metadata_df = pd.read_csv(metadata_csv_file)
metadata_df.set_index('Barcode', inplace=True)
adata.obs = adata.obs.join(metadata_df)

if 'LibraryID' not in adata.obs.columns:
    raise ValueError("'LibraryID' column not found after loading CSV.")
if adata.obs['LibraryID'].isnull().any():
    n_missing = adata.obs['LibraryID'].isnull().sum()
    print(f"WARNING: {n_missing} cells did not have a matching barcode in the CSV file.")

print("\nSuccessfully added metadata from CSV. New adata.obs:")
print(adata.obs.head())

# %%
condition_mask = adata.obs['LibraryID'] == 'WT_brain_cortex'
adata= adata[condition_mask, :]

# %%
adata

# %%
cluster_labels = pd.read_csv("GEX_Graph-Based.csv")
cluster_labels.head()

# %%
cluster_labels.set_index('Barcode', inplace=True)
adata.obs = adata.obs.join(cluster_labels)

# %%
# Rename the 'GEX Graph-based' column to 'cluster'
adata.obs = adata.obs.rename(columns={'GEX Graph-based': 'cluster'})
adata.obs.head()

# %%
# Extract only the cluster numbers from the 'cluster' column
adata.obs['cluster'] = adata.obs['cluster'].str.extract('(\d+)')
print("Updated cluster column:")
adata.obs.head()


# %% [markdown]
# # --- Preprocessing and Quality Control (QC) ---

# %%
print("\nStarting preprocessing and QC...")

# Show genes that have the highest expression in absolute counts, per cell
sc.pl.highest_expr_genes(adata, n_top=20, save='_highest_expr_genes.png')

# %%
# Calculate QC metrics
adata.var['mt'] = adata.var_names.str.startswith('MT-')  
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

# Store original cell count
n_cells_before = adata.n_obs
print(f"Starting with {n_cells_before} cells")

# %%
# ========== PERCENTILE-BASED FILTERING ==========

# Calculate percentiles for each QC metric
# Using 2.5th and 97.5th percentiles to remove extreme 5% of cells
n_genes_lower = np.percentile(adata.obs['n_genes_by_counts'], 2.5)
n_genes_upper = np.percentile(adata.obs['n_genes_by_counts'], 97.5)

# Total counts (UMIs)
counts_lower = np.percentile(adata.obs['total_counts'], 2.5)
counts_upper = np.percentile(adata.obs['total_counts'], 97.5)

# Mitochondrial percentage
mt_upper = 20

print("Percentile-based thresholds:")
print(f"  Number of genes: {n_genes_lower:.0f} - {n_genes_upper:.0f}")
print(f"  Total counts: {counts_lower:.0f} - {counts_upper:.0f}")
print(f"  Mitochondrial %: < {mt_upper:.2f}%")

# Count cells that will be filtered by each criterion
n_genes_filter = (adata.obs['n_genes_by_counts'] < n_genes_lower) | (adata.obs['n_genes_by_counts'] > n_genes_upper)
counts_filter = (adata.obs['total_counts'] < counts_lower) | (adata.obs['total_counts'] > counts_upper)
mt_filter = adata.obs['pct_counts_mt'] > mt_upper

print(f"\nCells filtered by each criterion:")
print(f"  n_genes: {n_genes_filter.sum()} cells")
print(f"  total_counts: {counts_filter.sum()} cells")
print(f"  pct_counts_mt: {mt_filter.sum()} cells")

# %%
# ========== VISUALIZATION ==========

# Save original QC metrics before filtering for visualization
orig_n_genes = adata.obs['n_genes_by_counts'].copy()
orig_total_counts = adata.obs['total_counts'].copy()
orig_pct_mt = adata.obs['pct_counts_mt'].copy()

# Apply all filters
keep_cells = (
    (adata.obs['n_genes_by_counts'] >= n_genes_lower) & 
    (adata.obs['n_genes_by_counts'] <= n_genes_upper) & 
    (adata.obs['total_counts'] >= counts_lower) & 
    (adata.obs['total_counts'] <= counts_upper) & 
    (adata.obs['pct_counts_mt'] <= mt_upper)
)

# Filter the data
adata = adata[keep_cells, :].copy()

n_cells_after = adata.n_obs
print(f"\nAfter filtering: {n_cells_after} cells")
print(f"Removed {n_cells_before - n_cells_after} cells ({100*(n_cells_before - n_cells_after)/n_cells_before:.1f}%)")

# Filter genes expressed in very few cells
min_cells = 3
print(f"\nFiltering genes expressed in < {min_cells} cells...")
n_genes_before_filter = adata.n_vars
# Keep genes that are in at least min_cells cells OR are in our list of available_genes
genes_to_keep = (adata.var['n_cells_by_counts'] >= min_cells) | (adata.var_names.isin(available_genes))
adata = adata[:, genes_to_keep].copy()
n_genes_after_filter = adata.n_vars
print(f"Removed {n_genes_before_filter - n_genes_after_filter} genes")

# Create figure with QC plots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Row 1: Before filtering
axes[0, 0].hist(orig_n_genes, bins=50, alpha=0.7)
axes[0, 0].axvline(n_genes_lower, color='red', linestyle='--', label='Lower threshold')
axes[0, 0].axvline(n_genes_upper, color='red', linestyle='--', label='Upper threshold')
axes[0, 0].set_xlabel('Number of genes')
axes[0, 0].set_ylabel('Number of cells')
axes[0, 0].set_title('Before filtering')
axes[0, 0].legend()

axes[0, 1].hist(np.log10(orig_total_counts), bins=50, alpha=0.7)
axes[0, 1].axvline(np.log10(counts_lower), color='red', linestyle='--')
axes[0, 1].axvline(np.log10(counts_upper), color='red', linestyle='--')
axes[0, 1].set_xlabel('log10(Total counts)')
axes[0, 1].set_ylabel('Number of cells')
axes[0, 1].set_title('Before filtering')

axes[0, 2].hist(orig_pct_mt, bins=50, alpha=0.7)
axes[0, 2].axvline(mt_upper, color='red', linestyle='--', label='Upper threshold')
axes[0, 2].set_xlabel('Mitochondrial gene %')
axes[0, 2].set_ylabel('Number of cells')
axes[0, 2].set_title('Before filtering')
axes[0, 2].legend()

# Row 2: After filtering
axes[1, 0].hist(adata.obs['n_genes_by_counts'], bins=50, alpha=0.7, color='green')
axes[1, 0].set_xlabel('Number of genes')
axes[1, 0].set_ylabel('Number of cells')
axes[1, 0].set_title('After filtering')

axes[1, 1].hist(np.log10(adata.obs['total_counts']), bins=50, alpha=0.7, color='green')
axes[1, 1].set_xlabel('log10(Total counts)')
axes[1, 1].set_ylabel('Number of cells')
axes[1, 1].set_title('After filtering')

axes[1, 2].hist(adata.obs['pct_counts_mt'], bins=50, alpha=0.7, color='green')
axes[1, 2].set_xlabel('Mitochondrial gene %')
axes[1, 2].set_ylabel('Number of cells')
axes[1, 2].set_title('After filtering')

plt.tight_layout()
plt.savefig('figures/qc_histograms_before_after.png', dpi=300, bbox_inches='tight')
plt.show()

# Create scatter plots to show relationships
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# MT% vs n_genes
axes[0].scatter(adata.obs['n_genes_by_counts'], adata.obs['pct_counts_mt'], 
                alpha=0.4, s=10)
axes[0].set_xlabel('Number of genes')
axes[0].set_ylabel('Mitochondrial gene %')
axes[0].set_title('MT% vs Number of genes')

# Total counts vs n_genes (log scale)
axes[1].scatter(adata.obs['total_counts'], adata.obs['n_genes_by_counts'], 
                alpha=0.4, s=10)
axes[1].set_xlabel('Total counts')
axes[1].set_ylabel('Number of genes')
axes[1].set_xscale('log')
axes[1].set_yscale('log')
axes[1].set_title('Counts vs Genes correlation')

# MT% vs total counts
axes[2].scatter(adata.obs['total_counts'], adata.obs['pct_counts_mt'], 
                alpha=0.4, s=10)
axes[2].set_xlabel('Total counts')
axes[2].set_ylabel('Mitochondrial gene %')
axes[2].set_xscale('log')
axes[2].set_title('MT% vs Total counts')

plt.tight_layout()
plt.savefig('figures/qc_scatter_filtered.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# ========== SUMMARY STATISTICS ==========

# Create summary DataFrame
summary_stats = pd.DataFrame({
    'Metric': ['Total cells (initial)', 'Total cells (filtered)', 'Cells removed', 
               'Percent cells removed', 'Total genes (initial)', 'Total genes (filtered)',
               'n_genes threshold (lower)', 'n_genes threshold (upper)',
               'total_counts threshold (lower)', 'total_counts threshold (upper)',
               'MT% threshold'],
    'Value': [n_cells_before, n_cells_after, n_cells_before - n_cells_after,
              f"{100*(n_cells_before - n_cells_after)/n_cells_before:.1f}%",
              n_genes_before_filter, n_genes_after_filter,
              f"{n_genes_lower:.0f}", f"{n_genes_upper:.0f}",
              f"{counts_lower:.0f}", f"{counts_upper:.0f}",
              f"{mt_upper:.2f}%"]
})

print("\n" + "="*50)
print("FILTERING SUMMARY")
print("="*50)
print(summary_stats.to_string(index=False))

# Save the summary to file
summary_stats.to_csv('qc_filtering_summary.csv', index=False)

# Optional: Save the filtered data
# adata.write('filtered_adata.h5ad')
print("\nFiltered data is ready for downstream analysis!")

# %% [markdown]
# # --- Normalization and Feature Selection ---

# %%
print("\nNormalizing data and selecting highly variable genes...")

# Normalize each cell by total counts over all genes, so that every cell has the same total count.
sc.pp.normalize_total(adata, target_sum=1e4)

# Logarithmize the data
sc.pp.log1p(adata)

# Identify highly-variable genes
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
sc.pl.highly_variable_genes(adata, save='_highly_variable_genes.png')

# Add available_genes to the list of highly variable genes to keep them from being filtered out
print(f"Ensuring {len(available_genes)} available genes are not filtered out as non-highly-variable.")
adata.var.loc[adata.var_names.isin(available_genes), 'highly_variable'] = True

# Slice the AnnData object to keep only highly variable genes
adata = adata[:, adata.var.highly_variable]

print("Finished normalization and feature selection.")

# %% [markdown]
# # --- 5. Dimensionality Reduction ---

# %%
print("\nPerforming dimensionality reduction...")

# Scale the data to unit variance and zero mean
sc.pp.scale(adata, max_value=10)

# Principal component analysis (PCA)
sc.tl.pca(adata, svd_solver='arpack')
sc.pl.pca_variance_ratio(adata, log=True, save='_pca_variance_ratio.png')

# %% [markdown]
# # --- 6. Clustering and Visualization ---

# %%
print("\nClustering and visualization...")

# Computing the neighborhood graph
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)

# Clustering the neighborhood graph using the Leiden algorithm
sc.tl.leiden(adata)
# sc.tl.umap(adata)

# %%
# sc.pl.umap(adata, color=['leiden'], save='_umap_leiden.png')

# %% [markdown]
# # --- 7. Add alternative UMAP projection ---

# %%
print("\nAdding alternative UMAP projection from CSV...")

# Load the external UMAP projection
umap_df = pd.read_csv('umap_projection.csv', index_col=0)

# Align the dataframes by index (barcodes)
aligned_df = umap_df.reindex(adata.obs_names)

# Add the UMAP coordinates to the AnnData object
adata.obsm['X_umap'] = aligned_df[['UMAP-1', 'UMAP-2']].values

# Plot the new UMAP projection
# sc.pl.embedding(adata, basis='X_umap_csv', color=['leiden'], save='_umap_csv_leiden.png')
sc.pl.umap(adata, color=['leiden'], save='_umap_leiden.png')

# %%
# Plot the new UMAP projection
# sc.pl.embedding(adata, basis='X_umap_csv', color=['cluster'], save='_umap_csv_cluster.png', legend_loc='on data')
sc.pl.umap(adata, color=['cluster'], save='_umap_cluster.png', legend_loc='on data')

# %%
from IPython.display import Image
Image(filename='clusters.png')

# %%
cluster_mapping = {
    "Astro" : ["11"],
    "CC_upper" : ["6", "3", "2", "13", "9", "10", "7"],
    "Meningi" : ["18"],
    "Oligos" : ["16", "14"],
    "Unk" : ["8", "1", "17"],
    "CC_deepr" : ["4"],
    "GABA" : ["5"],
    "Micro" : ["12"],
    "OPCs" : ["15"]
}

# Create cell_type column based on cluster mapping
adata.obs['cell_type'] = 'Unasigned'  # Initialize with default value

# Map clusters to cell types
for cell_type, clusters in cluster_mapping.items():
    mask = adata.obs['cluster'].isin(clusters)
    adata.obs.loc[mask, 'cell_type'] = cell_type

# Display the first few rows to verify
adata.obs[['cluster', 'cell_type']].head()


# %%
adata.obs.cell_type.value_counts()

# %%
# Plot the new UMAP projection
# sc.pl.embedding(adata, basis='X_umap_csv', color=['cell_type'], save='_umap_csv_cluster.png', legend_loc='on data')
sc.pl.umap(adata, color=['cell_type'], save='_umap_cluster.png', legend_loc='on data')

# %% [markdown]
# # --- Save Results ---

# %%
print("\nAnalysis complete. Results saved to 'figures/' directory.")
print("Final AnnData object:")
print(adata)

# Save the final AnnData object
adata.write('processed_data.h5ad')
print("\nProcessed data saved to 'processed_data.h5ad'")

# %%
print("\nCreating cell-type resolved visualizations...")
    
# UMAP plots colored by gene expression
for gene in available_genes:
    print(f"Creating UMAP plot for {gene}...")
    
    # Plot on original UMAP
    sc.pl.umap(adata, color=gene, save=f'_{gene}_expression.png',
                title=f'{gene} expression (WT nuclei)', color_map='Reds')

# %%
# print("\nCreating cell-type resolved visualizations...")
    
# # UMAP plots colored by gene expression
# for gene in available_genes:
#     print(f"Creating UMAP plot for {gene}...")
    
#     # Plot on CSV UMAP if available
#     if 'X_umap_csv' in adata.obsm.keys():
#         sc.pl.embedding(adata, basis='X_umap_csv', color=gene,
#                         save=f'_csv_{gene}_expression.png',
#                         title=f'{gene} expression (WT nuclei, CSV UMAP)', color_map='Reds')

# %%
# Create dot plot showing expression across clusters
print("\nCreating dot plot for cluster-resolved expression...")
sc.pl.dotplot(adata, available_genes, groupby='cell_type',
                save=f'_snrpn_snord_dotplot.png')

# %%
# Create violin plot
print("\nCreating violin plot for cluster-resolved expression...")
sc.settings.set_figure_params(figsize=(12, 6), dpi=80)
sc.pl.violin(adata, available_genes, groupby='cell_type',
             save=f'_snrpn_snord_violin.png')

# %%
global_max = adata[:, :].X.toarray().flatten().max()
print(f"Global Mean expression: {global_max:.3f}")

# %%
def gene_expression_summary(adata, target_gene):
    print(f"\nExpression summary statistics for {target_gene}:")

    gene_expr_overall = adata[:, target_gene].X.toarray().flatten()
    expressing_cells_overall = (gene_expr_overall > 0).sum()
    total_cells_overall = len(gene_expr_overall)
    mean_expr_overall = gene_expr_overall.mean()
    mean_expr_in_expressing_overall = gene_expr_overall[gene_expr_overall > 0].mean() if expressing_cells_overall > 0 else 0
    max_expr_overall = gene_expr_overall.max()

    print("\nOverall Dataset:")
    print(f"  Cells expressing {target_gene}: {expressing_cells_overall}/{total_cells_overall} ({100*expressing_cells_overall/total_cells_overall:.1f}%)")
    print(f"  Mean expression (all cells): {mean_expr_overall:.3f}")
    print(f"  Mean expression (expressing cells only): {mean_expr_in_expressing_overall:.3f}")
    print(f"  Max expression: {max_expr_overall:.3f}")

    if 'cell_type' in adata.obs.columns:
        print("\nBy cell_type Cluster:")
        for cluster in sorted(adata.obs['cell_type'].cat.categories):
            adata_cluster = adata[adata.obs['cell_type'] == cluster, :]
            gene_expr_cluster = adata_cluster[:, target_gene].X.toarray().flatten()
            expressing_cells_cluster = (gene_expr_cluster > 0).sum()
            total_cells_cluster = len(gene_expr_cluster)
            mean_expr_cluster = gene_expr_cluster.mean()
            mean_expr_in_expressing_cluster = gene_expr_cluster[gene_expr_cluster > 0].mean() if expressing_cells_cluster > 0 else 0
            max_expr_cluster = gene_expr_cluster.max()

            print(f"  Cluster {cluster}:")
            print(f"    Cells expressing {target_gene}: {expressing_cells_cluster}/{total_cells_cluster} ({100*expressing_cells_cluster/total_cells_cluster:.1f}%)")
            print(f"    Mean expression (all cells): {mean_expr_cluster:.3f}")
            print(f"    Mean expression (expressing cells only): {mean_expr_in_expressing_cluster:.3f}")
            print(f"    Max expression: {max_expr_cluster:.3f}")

# %%
gene_expression_summary(adata, 'Snrpn')

# %%
gene_expression_summary(adata, 'Rbfox3')

# %%
# Summary statistics
print("\nExpression summary statistics (WT nuclei only):")
for gene in available_genes:
    gene_expr = adata[:, gene].X.toarray().flatten()
    expressing_cells = (gene_expr > 0).sum()
    total_cells = len(gene_expr)
    mean_expr = gene_expr.mean()
    max_expr = gene_expr.max()
    
    print(f"{gene}:")
    print(f"  Expressing cells: {expressing_cells}/{total_cells} ({100*expressing_cells/total_cells:.1f}%)")
    print(f"  Mean expression: {mean_expr:.3f}")
    print(f"  Max expression: {max_expr:.3f}")


