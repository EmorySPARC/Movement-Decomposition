import os
from MD_Functions import WriteC3D
from MD_Functions import WriteSTO
from MD_Functions import ica_c3d
from MD_Functions import ica_sto


file_path = 'ExampleData/OpenSim/StepAndSwing.sto'  # Replace with your actual file path

# Extract the filename without extension from the file path
base_name = os.path.basename(file_path)
filename_without_extension = os.path.splitext(base_name)[0]

NumberOfICs = 2

#process .c3d file
if file_path.endswith('.c3d'):
    
    # Assign a marker for re-referencing the displacement of each marker
    referenceMarker = 'spinal_cord_27'
    
    # Process the c3d file with a desired number of principal components (in this example: 2)
    ICs_df, c3d_reconstructions = ica_c3d(file_path, NumberOfICs, referenceMarker, plot = True)
    
    # Write ICs to file
    ic_csv_filename = f'{file_path}_ICs.csv'
    ICs_df.to_csv(ic_csv_filename, index=False)
    
    # Write each reconstructed .c3d data using the naming pattern specified
    for i in range (NumberOfICs):
        # Append the rank (starting from 1) to the filename
        new_filename = f'{file_path}_ic_{i+1}.c3d'
        
        # Write the reconstructed data to a new file
        WriteC3D(c3d_reconstructions[i], new_filename)

#process .sto file
elif file_path.endswith('.sto'):
    
    # Process the c3d file with a desired number of principal components (in this example: 2)
    ICs_df, reconstructions = ica_sto(file_path, NumberOfICs, plot = True)
    
    # Write ICs to file
    ic_csv_filename = f'{file_path}_ICs.csv'
    ICs_df.to_csv(ic_csv_filename, index=False)
    
    # Write each reconstructed .c3d data using the naming pattern specified
    for i in range (NumberOfICs):
        # Append the rank (starting from 1) to the filename
        new_filename = f'{file_path}_ic_{i+1}.sto'
        
        # Write the reconstructed data to a new file
        WriteSTO(reconstructions[i], new_filename)
    
else:
    print("Please specify a .c3d or .sto file for processing")