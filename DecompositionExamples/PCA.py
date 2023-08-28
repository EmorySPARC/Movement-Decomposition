import os
from MD_Functions import WriteC3D
from MD_Functions import WriteSTO
from MD_Functions import pca_c3d
from MD_Functions import pca_sto
import pandas as pd

file_path = 'ExampleData/OpenSim/StepAndSwing.sto'  # Replace with your actual file path

# Extract the filename without extension from the file path
base_name = os.path.basename(file_path)
filename_without_extension = os.path.splitext(base_name)[0]

VarianceThreshold = 0.90  # Make sure the threshold is in fraction form (i.e., 0.95 for 95%)

#process .c3d file
if file_path.endswith('.c3d'):
 
    # Assign a marker for re-referencing the displacement of each marker
    referenceMarker = 'spinal_cord_27'
    
    # Process the c3d file with the desired variance threshold
    reconstructions, principal_components, NumberOfPCs = pca_c3d(file_path, VarianceThreshold, referenceMarker)
    
    # Convert the principal components to a DataFrame and save to a CSV file
    pc_df = pd.DataFrame(principal_components)
    pc_csv_filename = f'{file_path}_PCs_{int(VarianceThreshold*100)}.csv'
    pc_df.to_csv(pc_csv_filename, index=False)
    
    # Write each reconstructed .c3d data using the naming pattern specified
    for i in range(NumberOfPCs):
        # Append the rank (starting from 1) to the filename
        new_filename = f'{file_path}_pc_{i+1}.c3d'
        
        # Write the reconstructed data to a new file
        WriteC3D(reconstructions[i], new_filename)

#process .sto file
elif file_path.endswith('.sto'):

    # Process the c3d file with the desired variance threshold
    reconstructions, principal_components, NumberOfPCs = pca_sto(file_path, VarianceThreshold)
    
    # Convert the principal components to a DataFrame and save to a CSV file
    pc_df = pd.DataFrame(principal_components)
    pc_csv_filename = f'{file_path}_PCs_{int(VarianceThreshold*100)}.csv'
    pc_df.to_csv(pc_csv_filename, index=False)
    
    # Write each reconstructed .c3d data using the naming pattern specified
    for i in range(NumberOfPCs):
        # Append the rank (starting from 1) to the filename
        new_filename = f'{file_path}_pc_{i+1}.sto'
        
        # Write the reconstructed data to a new file
        WriteSTO(reconstructions[i], new_filename)
        
else:
    print("Please specify a .c3d or .sto file for processing")