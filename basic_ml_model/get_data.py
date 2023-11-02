"""
    This script is for collecting compound information from the CHEMBL database
    and merge the active coumpound from the database with our inhouse compounds database.
    
    The script will output two csv files, one for positive dataset and one for negative dataset.
    The positive dataset contains the active compounds from the CHEMBL database and our inhouse database.
    The negative dataset contains the inactive compounds from the CHEMBL database.
    
    Both datasets contains the following columns:
        - comp_id: compound ID
        - smiles: SMILES string
        - bioactivity: bioactivity value
        - y_label: label for the bioactivity value, positive or negative
"""

from chembl_webresource_client.new_client import new_client
import pandas as pd



def filter_activities( standard_types, max_standard_value):
    try:
        # Filter activities based on the specified criteria
        activities = new_client.activity.filter(
            target_chembl_id='CHEMBL2107',
            standard_type__in=standard_types,
            standard_relation__in=['=', '<'],
            standard_value__lt=max_standard_value,
            standard_units__in=['uM', 'nM']
        ).only(
            "molecule_chembl_id",
            "standard_type",
            "standard_relation",
            "standard_units",
            "standard_value",
            )
        # 1. convert uM to nM
        # 2. convert pIC50 and pEC50 to normal (add unit)
        print(activities[0])

        for i, entry in enumerate(activities):
            if entry['units'] == 'uM':
                entry['value'] = float(entry['value'])*1000
                entry['units'] = 'nM'
            if entry['type'] == 'pIC50' or entry['type'] == 'pEC50':
                entry['value'] = 10**(-float(entry['value'])) * 1e9
                entry['units'] = 'nM'
                entry['type'] = entry['type'][1:]
        return activities

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return []

 
############################################################################################################
# Define the target ChEMBL ID, standard types, and maximum standard value
# target_id = "CHEMBL2107"  #CXCR4 Homo sapiens
standard_types = ["EC50", "IC50", "Ki"]

df = pd.DataFrame(filter_activities(standard_types, 10000))
df = df[['molecule_chembl_id', 'standard_relation','standard_type','standard_units','standard_value']]
df['standard_value'] = df['standard_value'].astype(float)

for column in df.columns:
    print(f"Unique values and counts in column: {column}")
    print(df[column].value_counts(dropna=False))
    print("\n")

df = df[df['standard_value']<=10000]
print(df.shape)

molecules = new_client.molecule.filter(molecule_chembl_id__in=list(df['molecule_chembl_id']))
chembl_id_to_smiles = {molecule['molecule_chembl_id']: molecule['molecule_structures']['canonical_smiles']
                       for molecule in molecules
                       if 'molecule_structures' in molecule and molecule['molecule_structures']}
smiles_df = pd.DataFrame(list(chembl_id_to_smiles.items()), columns=['molecule_chembl_id', 'smiles'])
df = df.merge(smiles_df, on='molecule_chembl_id')


# ------ deal with repeat molecules entry in the dataset ------
# action: take the median value for the repeat molecules
df_agg = df.groupby('molecule_chembl_id')['standard_value'].median().reset_index(name='median_value')
df = df_agg.merge(smiles_df, on='molecule_chembl_id')




#####################################################################
# load inhouse dataset
df2 = pd.read_csv('../data/inhouse.csv')
# df2 = df2[['comp_id', 'smiles', 'bioactivity']]




########## merge two dataset############
df = df[['molecule_chembl_id','smiles']] 
df2 = df2[['comp_id', 'smiles']]
df.columns = df2.columns
result = pd.concat([df2, df], ignore_index=True)

print(result.head())
print(result.shape)

result.to_csv('./data/smiles_active_compound.csv', index=False)



