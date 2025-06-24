# %%
from cloupe import Cloupe
import scanpy as sc
import os

script_dir = "/beegfs/scratch/ric.broccoli/kubacki.michal/scMulti_merged_matrix"
os.chdir(script_dir)

# Path to your input .cloupe file
cloupe_file = './cloupe.cloupe'

# Path for the output .h5ad file
h5ad_file = './adata_from_cloupe.h5ad'

# --- FIX IS HERE ---
# Read the .cloupe file and explicitly tell it to load the expression matrix
# by setting `load_csr=True`.
print(f"Reading {cloupe_file} and loading the sparse matrix...")
c = Cloupe(cloupe_file, load_csr=True)

# Create anndata and save it to the specified file.
# Note: The argument is 'filename' based on the library's source code.
print(f"Converting to AnnData and saving to {h5ad_file}...")
c.to_anndata(filename=h5ad_file)

print(f"\nSuccessfully converted {cloupe_file} to {h5ad_file}")

# You can now immediately load and verify the output file with scanpy
print("\nVerifying the output file...")
adata = sc.read_h5ad(h5ad_file)
print(adata)