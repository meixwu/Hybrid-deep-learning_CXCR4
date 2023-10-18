import csv
from rdkit.Chem import Descriptors


# Descriptors.setupAUTOCorrDescriptors()


# Create a list with starting index 0
my_list = [x[0] for x in Descriptors.descList]
print(len(my_list))
# my_list.extend(['Asphericity', 'Eccentricity', 'InertialShapeFactor', 'NPR1', 'NPR2', 'PMI1', 'PMI2', 'PMI3', 'RadiusOfGyration', 'SpherocityIndex'])

# Write the list to a CSV file
with open('feature_names.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Index', 'Value'])
    for i, value in enumerate(my_list):
        writer.writerow([i, value])
