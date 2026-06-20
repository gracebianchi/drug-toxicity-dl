import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# Load data
df = pd.read_csv('data/tox21.csv')
print(f"Loaded {len(df)} compounds")

# Convert a single SMILES string to a Morgan fingerprint (bit vector)
def smiles_to_fingerprint(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    return np.array(fp)

# Apply to every row
df['fingerprint'] = df['smiles'].apply(smiles_to_fingerprint)

# Report and drop any rows where SMILES couldn't be parsed
invalid = df['fingerprint'].isna().sum()
print(f"Invalid SMILES dropped: {invalid}")
df = df.dropna(subset=['fingerprint'])

# Expand fingerprint array into 2048 separate columns (fp_0, fp_1, ..., fp_2047)
fp_df = pd.DataFrame(df['fingerprint'].tolist(), index=df.index)
fp_df.columns = [f'fp_{i}' for i in range(fp_df.shape[1])]

# Join fingerprint columns back to original dataframe, drop the array column
df = pd.concat([df.drop(columns=['fingerprint']), fp_df], axis=1)

print(f"Final shape: {df.shape}")
print(f"Fingerprint columns: fp_0 ... fp_{fp_df.shape[1] - 1}")

# Save
df.to_csv('data/tox21_with_fingerprints.csv', index=False)
print("Saved to data/tox21_with_fingerprints.csv")
