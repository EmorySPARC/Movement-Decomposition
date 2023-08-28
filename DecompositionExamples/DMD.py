import os
from MD_Functions import WriteC3D
from MD_Functions import dmd_c3d

file_path = 'ExampleData/c3d/StepAndSwing.c3d'  # Replace with your actual file path

# Extract the filename without extension from the file path
base_name = os.path.basename(file_path)
filename_without_extension = os.path.splitext(base_name)[0]

# Assign a marker for re-referencing the displacement of each marker
referenceMarker = 'spinal_cord_27'

# Assign the expected number of modes
numberOfModes = 3

# Process the c3d file with DMD
dmd, dmd_reconstructions = dmd_c3d(file_path, numberOfModes, referenceMarker, plot = True)

# Write each reconstructed .c3d data using the naming pattern specified
for i in range(numberOfModes):
    # Append the rank (starting from 1) to the filename
    new_filename = f'ExampleData/{filename_without_extension}_mode_{i+1}.c3d'

    # Write the reconstructed data to a new file
    WriteC3D(dmd_reconstructions[i], new_filename)