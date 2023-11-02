# Collect/organize  SMILES strings and get descriptors

# 1) Combine the positive and negative Smiles strings into one dataframe
# 2) Get the MACCS fingerprint for each molecule/SMILES in the dataframe
# 3) Inspect the descriptor dataset;
# 4) Add row index(molecule id or index number), column names, if any, to the dataset;
# 5) Clean the dataset by removing columns with more than 10% missing values; 
#     remove columns with all zeros;
#     remove rows/columns with NaN values
# 6) Save the dataset as a CSV file

import pandas as pd

from rdkit import Chem
from rdkit.Chem import MACCSkeys



# 1) Combine the positive and negative Smiles strings into one dataframe
positive_df = pd.read_csv('data/dataset/positive_dataset.csv')
negative_df = pd.read_csv('data/dataset/negative_dataset.csv')
combined_df = pd.concat([positive_df, negative_df], ignore_index=True)


# 2) Get the MACCS fingerprint for each molecule/SMILES in the dataframe

# Create a function to calculate MACCS keys
def calculate_maccs(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        maccs_fp = MACCSkeys.GenMACCSKeys(mol)
        return maccs_fp.ToBitString()
    else:
        return None

# Apply the function to the DataFrame column
combined_df['MACCS_Keys'] = combined_df['smiles'].apply(calculate_maccs)

# Print the DataFrame
print(combined_df.head())