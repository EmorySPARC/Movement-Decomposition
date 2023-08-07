import pandas as pd
import numpy as np
from C3D_Functions import ReadC3D, C3DToDataframe, DataframeToC3D, ReReference
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA

def ica_c3d(file_path, n_components, referenceMarker):
    # Read the .c3d file into a DataFrame
    original_c3d_data = ReadC3D(file_path)
    
    # ReReference the Data based on the referenceMarker
    Data = ReReference(original_c3d_data, referenceMarker)
    
    # Convert the C3D data to DataFrame
    X = C3DToDataframe(Data)

    # Record the original mean and std for each feature
    original_mean = X.mean(axis=0)
    original_std = X.std(axis=0)

    # Normalize the features
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X) 

    # Implement ICA
    ica = FastICA(n_components=n_components)
    transformed = ica.fit_transform(X_normalized)

    c3d_reconstructions = []
    for i in range(n_components):
        # Extract the independent component
        component = ica.mixing_[:, i].reshape(-1, 1)
        
        # Project the data onto this component
        proj = transformed[:, i].reshape(-1, 1)
        
        # Reconstruct the data using this component
        recon = np.dot(proj, component.T) + ica.mean_
        
        # Rescale the reconstruction to the original range
        recon_scaled = recon * original_std.values + original_mean.values
        
        # Convert the re-scaled reconstruction to a DataFrame shape similar to 'X'
        recon_df = pd.DataFrame(recon_scaled, columns=X.columns)
        
        # Convert the dataframe back to .c3d format using the provided function
        c3d_recon = DataframeToC3D(recon_df, original_c3d_data)
        
        c3d_reconstructions.append(c3d_recon)

    return c3d_reconstructions