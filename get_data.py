import pandas as pd
from chembl_webresource_client.new_client import new_client


# Read in the two CSV files
chembl_active = pd.read_csv('data/chembl-active.csv')
inhouse = pd.read_csv('data/inhouse.csv')


#Filter inactive compounds from the ChEMBL dataset
activity = new_client.activity
activities = activity.filter( standard_type__in=["EC50", "IC50"], standard_units="nM")

inactive_compounds = []
thredshhold = 1000



for i, entry in enumerate(activities):
    if i > thredshhold: break
    molecule_chembl_id = entry['molecule_chembl_id']
    molecule = new_client.molecule.get(molecule_chembl_id)
    
    if (molecule['molecule_type'] and entry['standard_value']) and ((molecule['molecule_type'] == 'Small molecule') and (molecule_chembl_id not in chembl_active.comp_id)):
        try:
            inactive_compounds.append(
                [
                molecule['molecule_chembl_id'],
                molecule['molecule_structures']['canonical_smiles'],
                entry['standard_value'],
                'negative'
                ]
            )
        except:
            pass        
    
print("Inactive Compounds:")
for compound in inactive_compounds[:5]:  # Displaying first 5 as an example
    print(compound)

print('Number of inactive compound = ', len(inactive_compounds))

chembl_rest = pd.DataFrame(inactive_compounds, columns=['comp_id', 'smiles', 'bioactivity', 'y_label'])


# assume df1 and df2 are two dataframes you want to concatenate
positive_df = pd.concat([inhouse, chembl_active],ignore_index=True)
negative_df = chembl_rest


positive_df.to_csv('data/dataset/positive_dataset.csv', index=False)
negative_df.to_csv('data/dataset/negative_dataset.csv', index=False)


