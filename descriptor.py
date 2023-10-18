import pandas as pd

from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
from rdkit.Chem import Descriptors3D, AllChem


def get_descriptors(smiles, verbose=False):
    mol = Chem.MolFromSmiles(smiles)
    # Descriptors.setupAUTOCorrDescriptors()
    
    descriptors = []
    for name, function in Descriptors.descList:
        
        try:
            descriptors.append(function(mol))
            
        except:
            descriptors.append(None)
            if verbose:
                print("Error calculating {}".format(name))
    # 585 2D descriptors    

    ############################################################################################################
    # # 3D descriptors
    
    # # Add hydrogens to the molecule
    # mol = Chem.AddHs(mol)

    # # Generate 3D coordinates
    # AllChem.EmbedMolecule(mol)

    # # (Optional) Optimize the geometry
    # try:
    #     AllChem.UFFOptimizeMolecule(mol)
    # except:
    #     pass
    
    # # List of 3D descriptors in RDKit
    # try:
    #     descriptors.append(Descriptors3D.Asphericity(mol))
    #     descriptors.append(Descriptors3D.Eccentricity(mol))
    #     descriptors.append(Descriptors3D.InertialShapeFactor(mol))
    #     descriptors.append(Descriptors3D.NPR1(mol))
    #     descriptors.append(Descriptors3D.NPR2(mol))
    #     descriptors.append(Descriptors3D.PMI1(mol))
    #     descriptors.append(Descriptors3D.PMI2(mol))
    #     descriptors.append(Descriptors3D.PMI3(mol))
    #     descriptors.append(Descriptors3D.RadiusOfGyration(mol))
    #     descriptors.append(Descriptors3D.SpherocityIndex(mol))
    # except:
    #     pass
    
    
    ############################################################################################################
    
    return descriptors

def drop_lowinfo_columns(df, threshold=0.1):
    zero_percentage = (df == 0).mean()
    filtered_df = df.loc[:, zero_percentage <= threshold]
    return filtered_df

    
def main():

    
    #load in positive and negative datasets
    positive_df = pd.read_csv('data/dataset/positive_dataset.csv')
    negative_df = pd.read_csv('data/dataset/negative_dataset.csv')
    
    #get descriptors for positive
    positive_descriptors = positive_df['smiles'].apply(get_descriptors)
    positive_descriptors  = pd.DataFrame(positive_descriptors.tolist())#, columns = [Descriptors.descList[i][0] for i in range(len(Descriptors.descList))] + ['Asphericity', 'Eccentricity', 'InertialShapeFactor', 'NPR1', 'NPR2', 'PMI1', 'PMI2', 'PMI3', 'RadiusOfGyration', 'SpherocityIndex'])
    print(positive_descriptors[:5])
   
    #get descriptors for negative
    negative_descriptors = negative_df['smiles'].apply(get_descriptors)
    negative_descriptors  = pd.DataFrame(negative_descriptors.tolist())#, columns = [Descriptors.descList[i][0] for i in range(len(Descriptors.descList))] + ['Asphericity', 'Eccentricity', 'InertialShapeFactor', 'NPR1', 'NPR2', 'PMI1', 'PMI2', 'PMI3', 'RadiusOfGyration', 'SpherocityIndex'])
    
    #write descriptors to csv file
    positive_descriptors.to_csv('data/descriptor/positive_descriptors.csv', index=False)
    negative_descriptors.to_csv('data/descriptor/negative_descriptors.csv', index=False)
    
    all_descriptors = pd.concat([positive_descriptors, negative_descriptors],ignore_index=True)
    all_descriptors_cut = drop_lowinfo_columns(all_descriptors)
    
    all_descriptors.to_csv('data/descriptor/all_descriptors.csv', index=False)
    all_descriptors_cut.to_csv('data/descriptor/all_descriptors_cut.csv', index=False)  


    #print out the dimension of the two datasets
    print('Task completed')
    print('Dimension of positive dataset', positive_descriptors.shape)
    print('Dimension of negative dataset', negative_descriptors.shape)
    print('Dimension of combined dataset', all_descriptors.shape)
    print('Dimension of combined dataset dropped', all_descriptors_cut.shape)




if __name__ == "__main__":
    main()

    
    
    