# %%
import scanpy as sc
import pandas as pd
from cloupe import Cloupe
import os

# --- 1. Configuration ---
sc.settings.set_figure_params(dpi=100, frameon=False, figsize=(6, 6), facecolor='white')
data_dir = "/beegfs/scratch/ric.broccoli/kubacki.michal/scMulti_merged_matrix"
os.chdir(data_dir)

# Define the paths for your files
cloupe_file = './cloupe.cloupe'
metadata_csv_file = './LibraryID.csv'
temp_h5ad_file = './temp_adata_from_cloupe.h5ad'
h5ad_file_final = './adata_corrected_final.h5ad'

try:
    # --- 2. Convert .cloupe to a temporary .h5ad file ---
    print("--- Step 1: Converting .cloupe to a temporary .h5ad file ---")
    c = Cloupe(cloupe_file, load_csr=True)
    c.to_anndata(filename=temp_h5ad_file)
    print(f"Temporary AnnData file saved to: {temp_h5ad_file}")

    # --- 3. Load the AnnData object from the temporary file ---
    print("\n--- Step 2: Loading AnnData object from temporary file ---")
    adata = sc.read_h5ad(temp_h5ad_file)
    os.remove(temp_h5ad_file)
    print(f"Loaded AnnData object with {adata.n_obs} cells and {adata.n_vars} features.")

    # --- 4. Add Metadata from the Downloaded CSV File ---
    print(f"\n--- Step 3: Loading metadata from '{metadata_csv_file}' ---")
    metadata_df = pd.read_csv(metadata_csv_file)
    metadata_df.set_index('Barcode', inplace=True)
    adata.obs = adata.obs.join(metadata_df)
    print("\nSuccessfully added metadata from CSV. New adata.obs:")
    print(adata.obs.head())

    # --- 5. Final AnnData Setup & Analysis ---
    print("\n--- Step 4: Finalizing AnnData object ---")

    # --- THE FIX IS HERE ---
    # Convert the gene index from 'categorical' to 'string' to allow modification.
    print("Converting gene names to string type to prevent modification errors...")
    adata.var_names = adata.var_names.astype(str)
    
    # Now that the index is string-based, these operations will succeed.
    adata.var_names_make_unique()

    print(f"Saving the final, corrected AnnData object to: {h5ad_file_final}")
    adata.write_h5ad(h5ad_file_final)

    print("\n--- Step 5: Subsetting and generating visualizations ---")
    genotype_column = 'LibraryID'
    wild_type_label = 'WT_brain_cortex'

    print(f"Subsetting data for wild-type nuclei: '{wild_type_label}'")
    adata_wt = adata[adata.obs[genotype_column] == wild_type_label].copy()

    print("Wild-type subset created:")
    print(adata_wt)

    genes_to_plot = ['SNRPN']

    print(f"Generating UMAP for '{genes_to_plot[0]}' expression...")
    sc.pl.umap(adata_wt,
              color=genes_to_plot,
              basis='gex_umap',
              title=f'{genes_to_plot[0]} Expression in Wild-Type Nuclei',
              save='_umap_snrpn_expression.png',
              show=True)

    print(f"Generating Dot Plot for '{genes_to_plot[0]}' expression...")
    sc.pl.dotplot(adata_wt,
                 genes_to_plot,
                 groupby=genotype_column,
                 title=f'{genes_to_plot[0]} Expression',
                 save='_dotplot_snrpn_expression.png',
                 show=True)
                 
    print("\n--- Analysis Complete! ---")

except Exception as e:
    print(f"\nAn error occurred: {e}")
# %%
