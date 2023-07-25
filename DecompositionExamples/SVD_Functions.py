import pandas as pd
import numpy as np
from C3D_Functions import ReadC3D
from C3D_Functions import C3DToDataframe
from C3D_Functions import DataframeToC3D
from C3D_Functions import ReReference

def svd_c3d(file_path, rank, referenceMarker):
    # Read the .csv file into a DataFrame
    Data = ReadC3D(file_path)
    
    Data = ReReference(Data, referenceMarker)
    
    X = C3DToDataframe(Data)

    # Store original column names 
    original_columns = X.columns

    # Transpose the DataFrame
    X = X.transpose()

    # Convert DataFrame to numpy array for SVD
    X_np = X.values

    # Implement SVD
    U, S, VT = np.linalg.svd(X_np, full_matrices=False)

    # Convert singular values to a diagonal matrix
    S = np.diag(S)

    # Check if rank is less than or equal to the number of singular values
    if rank <= S.shape[0]:
        # Construct approximate DataFrame
        Xapprox = pd.DataFrame(U[:,:rank] @ S[0:rank,:rank] @ VT[:rank,:])
    else:
        print(f'Rank={rank} is greater than the number of singular values.')
        return None
    
    # Transpose the DataFrame back to its original orientation
    Xapprox = Xapprox.transpose()
    
    # Assign back the original column names
    Xapprox.columns = original_columns
    
    XapproxC3D = DataframeToC3D(Xapprox,Data)

    # Return the approximation
    return XapproxC3D