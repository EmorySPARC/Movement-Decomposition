import os
from C3D_Functions import WriteC3D
from SVD_Functions import svd_c3d

file_path = 'ExampleData/SoccerHeader.c3d'  # Replace with your actual file path
rank = 2  # Replace with your desired rank

# Extract the filename without extension from the file path
base_name = os.path.basename(file_path)
filename_without_extension = os.path.splitext(base_name)[0]

# Append rank to the filename
new_filename = 'ExampleData/' + f"{filename_without_extension}_rank_{rank}.c3d"

# Assign a marker for re-referencing the displacement of each marker
referenceMarker = 'Sacrum'

# Process the CSV file
approximation = svd_c3d(file_path, rank, referenceMarker)

# Write the DataFrame to a new file
WriteC3D(approximation, new_filename)


