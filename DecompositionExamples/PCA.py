import os
from MD_Functions import WriteC3D
from MD_Functions import pca_c3d

file_path = 'ExampleData/SoccerHeader.c3d'  # Replace with your actual file path

# Extract the filename without extension from the file path
base_name = os.path.basename(file_path)
filename_without_extension = os.path.splitext(base_name)[0]

# Assign a marker for re-referencing the displacement of each marker
referenceMarker = 'Sacrum'

NumberOfPCs = 2

# Process the c3d file with a desired number of principal components (in this example: 2)
reconstructions = pca_c3d(file_path, NumberOfPCs, referenceMarker)

# Write each reconstructed .c3d data using the naming pattern specified
for i in range (NumberOfPCs):
    # Append the rank (starting from 1) to the filename
    new_filename = f'ExampleData/{filename_without_extension}_pc_{i+1}.c3d'
    
    # Write the reconstructed data to a new file
    WriteC3D(reconstructions[i], new_filename)