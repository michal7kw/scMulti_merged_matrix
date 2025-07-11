# %%
import os
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt

# %% [markdown]
# # --- 1. Settings ---

# %%
# Ensure the working directory is set correctly
# Replace with the actual path to your project directory if different
# script_dir = "/beegfs/scratch/ric.broccoli/kubacki.michal/scMulti_merged_matrix"
script_dir = "D:/Github/scMulti_merged_matrix"
os.chdir(script_dir)
print(f"Working directory set to: {script_dir}")

target_gene = 'Rbfox3'

# %%
# Set Scanpy settings
sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.settings.set_figure_params(dpi=80, facecolor='white', frameon=False)
fig_dir = os.path.join(script_dir, 'figures_rbfox3')
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)
sc.settings.figdir = fig_dir
print(f"Figure directory set to: {fig_dir}")

# %% [markdown]
# # --- 2. Load Processed Data ---

# %%
print("Loading processed data...")
adata_file = 'processed_data.h5ad'
if not os.path.exists(adata_file):
    raise FileNotFoundError(f"The file {adata_file} was not found in {script_dir}. "
                          "Please ensure the 'process_sc_data.py' script has been run successfully.")
adata = sc.read_h5ad(adata_file)
print("Processed data loaded successfully.")
print(adata)

# %% [markdown]
# # --- 3. Analyze RBFOX3 Expression ---

# %%
print(f"\n--- Analyzing expression of {target_gene} ---")

# Ensure 'leiden' clustering is present
if 'leiden' not in adata.obs.columns:
    print("WARNING: 'leiden' clustering not found in adata.obs. Performing default Leiden clustering.")
    # Perform Leiden clustering if not present (optional, or raise error)
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40) # Ensure neighbors are computed
    sc.tl.leiden(adata)
    print("Leiden clustering performed.")

# %% [markdown]
# ## 4.1. UMAP Visualization

# %%
print(f"\nCreating UMAP plot for {target_gene} expression...")
sc.pl.umap(adata, color=target_gene, 
            save=f'_umap_{target_gene}_expression.png',
            title=f'{target_gene} Expression',
            show=True)

# Also plot on X_umap_csv if available
if 'X_umap_csv' in adata.obsm.keys():
    print(f"\nCreating UMAP plot (CSV coordinates) for {target_gene} expression...")
    sc.pl.embedding(adata, basis='X_umap_csv', color=target_gene,
                    save=f'_umap_csv_{target_gene}_expression.png',
                    title=f'{target_gene} Expression (CSV UMAP)',
                    show=True)

# %% [markdown]
# ## 4.2. Expression by Cluster

# %%
print(f"\nCreating dot plot for {target_gene} expression by Leiden cluster...")
sc.pl.dotplot(adata, [target_gene], groupby='leiden',
                save=f'_dotplot_{target_gene}_leiden.png',
                show=True)

# %%
print(f"\nCreating violin plot for {target_gene} expression by Leiden cluster...")
sc.settings.set_figure_params(figsize=(max(6, adata.obs['leiden'].nunique()*0.5), 4), dpi=80) # Adjust width based on number of clusters
sc.pl.violin(adata, [target_gene], groupby='leiden',
                save=f'_violin_{target_gene}_leiden.png',
                show=True, rotation=90)
sc.settings.set_figure_params(dpi=80, facecolor='white', frameon=False) # Reset fig params

# %% [markdown]
# ## 4.3. Summary Statistics

# %%
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

if 'leiden' in adata.obs.columns:
    print("\nBy Leiden Cluster:")
    for cluster in sorted(adata.obs['leiden'].cat.categories):
        adata_cluster = adata[adata.obs['leiden'] == cluster, :]
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
print("\n--- RBFOX3 Analysis Script Finished ---") 