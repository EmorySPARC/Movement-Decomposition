from ezc3d import c3d
import pandas as pd
import copy
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from pydmd import MrDMD, DMD
import matplotlib.pyplot as plt

#read a .c3d file in
def ReadC3D(filename, showData = False):
    c = c3d(filename)
    print("C3D File Imported")
    print("Number of Markers: " + str(c['parameters']['POINT']['USED']['value'][0]));  # Print the number of markers used
    
    point_data = c['data']['points']
    
    dataRate = c['parameters']['POINT']['RATE']['value']
    print("Data Rate = " + str(int(dataRate)) + " hz")
    
    length = len(point_data[0][0])
    print (str(length) + " Samples ")
    print (str(float(length/dataRate)) + " Seconds")
    
    if (showData):      
        print(point_data)  
      
    return c

# Write a .c3d file given c3d formatted data
def WriteC3D(data,outFileName):
    print("c3d file written to: " + outFileName)
    data.write(outFileName)
    
#Create a dataframe of the marker positions in a .c3d
def C3DToDataframe(data, writeCSV = False, csv_file_path = None):
    
    df = pd.DataFrame()
    
    for k in range (len(data['data']['points'][0])):
            df[data['parameters']['POINT']['LABELS']['value'][k] + "_X"] = data['data']['points'][0,k]
            df[data['parameters']['POINT']['LABELS']['value'][k] + "_Y"] = data['data']['points'][1,k]
            df[data['parameters']['POINT']['LABELS']['value'][k] + "_Z"] = data['data']['points'][2,k]
            
    if (writeCSV):
        df.to_csv(csv_file_path, index=False)
                                                              
    return df

#Transform a dataframe to .c3d format
def DataframeToC3D(df, original_c3d_data):
    outData = copy.deepcopy(original_c3d_data)
    labels = original_c3d_data['parameters']['POINT']['LABELS']['value']

    # Pre-allocate the points array with zeros
    outData['data']['points'] = np.zeros_like(outData['data']['points'])

    for i, label in enumerate(labels):
        outData['data']['points'][0, i] = df[label + "_X"].values
        outData['data']['points'][1, i] = df[label + "_Y"].values
        outData['data']['points'][2, i] = df[label + "_Z"].values

    return outData
    
#Re-reference all markers to a given marker
def ReReference (data, refMarker):
    
    outData = copy.deepcopy(data)
    
    for i in range(len(outData['parameters']['POINT']['LABELS']['value'])):
        if (data['parameters']['POINT']['LABELS']['value'][i] == refMarker):
            refIndex = i
            
    for k in range (len(outData['data']['points'][0])):
        for j in range (len(outData['data']['points'][0][0])):
            outData['data']['points'][0,k,j] = (outData['data']['points'][0,k,j] - data['data']['points'][0,refIndex,j])
            outData['data']['points'][1,k,j] = (outData['data']['points'][1,k,j] - data['data']['points'][1,refIndex,j])
            outData['data']['points'][2,k,j] = (outData['data']['points'][2,k,j] - data['data']['points'][2,refIndex,j])
                                  
    return outData

def zero_pelvis(df):
    # Ensure the DataFrame is a pandas DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    # Set pelvis column values to 0
    df["pelvic_tx"] = 0
    df["pelvic_ty"] = 0
    df["pelvic_tz"] = 0

    return df

def STOToDataframe(file_path, zeroPelvis = True):
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            if 'endheader' in line:
                skiprows = i + 1
                break
    data = pd.read_csv(file_path, delimiter='\t', skiprows=skiprows)
    
    if (zeroPelvis):
        data = zero_pelvis(data)
        
    return data

def WriteSTO(data, file_path, csv_file_path=None):
    if csv_file_path is not None:
        data = pd.read_csv(csv_file_path)
    
    nRows = len(data)
    nColumns = len(data.columns)

    header = f"""Coordinates
    version=1
    nRows={nRows}
    nColumns={nColumns}
    inDegrees=yes

    Units are S.I. units (second, meters, Newtons, ...)
    If the header above contains a line with 'inDegrees', this indicates whether rotational values are in degrees (yes) or radians (no).

    endheader
    """

    with open(file_path, 'w') as file:
        file.write(header)
    data.to_csv(file_path, sep='\t', mode='a', index=False, float_format='%.8f')

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


def pca_c3d(file_path, variance_threshold, referenceMarker):
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

    # Implement PCA
    pca = PCA()
    pca.fit(X_normalized)
    
    # Determine the number of components required to meet the variance threshold
    explained_var = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(explained_var >= variance_threshold) + 1

    # Use the first n components to transform the data
    projected = pca.transform(X_normalized)[:, :n_components]

    c3d_reconstructions = []
    for i in range(n_components):
        # Extract the component
        component = pca.components_[i].reshape(1, -1)
        
        # Project the data onto this component
        proj = projected[:, i].reshape(-1, 1)
        
        # Reconstruct the data using this component
        recon = np.dot(proj, component) + pca.mean_

        # Rescale the reconstruction to the original range
        recon_scaled = recon * original_std.values + original_mean.values
        
        # Convert the re-scaled reconstruction to a DataFrame shape similar to 'X'
        recon_df = pd.DataFrame(recon_scaled, columns=X.columns)
        
        # Convert the dataframe back to .c3d format using the provided function
        c3d_recon = DataframeToC3D(recon_df, original_c3d_data)
        
        c3d_reconstructions.append(c3d_recon)

    # The un-reconstructed principal components
    principal_components = projected

    return c3d_reconstructions, principal_components, n_components

def pca_sto(file_path, variance_threshold):
    
    # Read the openSim file into a DataFrame
    X = STOToDataframe(file_path)
    
    # Split off the 'time' column
    time_column = X['time']
    
    # Drop the 'time' column from the DataFrame
    X = X.drop('time', axis=1)

    # Record the original mean and std for each feature
    original_mean = X.mean(axis=0)
    original_std = X.std(axis=0)

    # Normalize the features
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X) 

    # Implement PCA
    pca = PCA()
    pca.fit(X_normalized)
    
    # Determine the number of components required to meet the variance threshold
    explained_var = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(explained_var >= variance_threshold) + 1

    # Use the first n components to transform the data
    projected = pca.transform(X_normalized)[:, :n_components]

    reconstructions = []
    for i in range(n_components):
        # Extract the component
        component = pca.components_[i].reshape(1, -1)
        
        # Project the data onto this component
        proj = projected[:, i].reshape(-1, 1)
        
        # Reconstruct the data using this component
        recon = np.dot(proj, component) + pca.mean_

        # Rescale the reconstruction to the original range
        recon_scaled = recon * original_std.values + original_mean.values
        
        # Convert the re-scaled reconstruction to a DataFrame shape similar to 'X'
        recon_df = pd.DataFrame(recon_scaled, columns=X.columns)
        
        recon_df = pd.concat([time_column, recon_df], axis=1)
        
        reconstructions.append(recon_df)

    # The un-reconstructed principal components
    principal_components = projected

    return reconstructions, principal_components, n_components


def ica_c3d(file_path, NumberOfICs, referenceMarker=None, plot = False):
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
    ica = FastICA(n_components = NumberOfICs)
    transformed = ica.fit_transform(X_normalized)

    # Create a DataFrame for the independent components
    ICs_df = pd.DataFrame(transformed, columns=[f'IC_{i+1}' for i in range(NumberOfICs)])
    
    c3d_reconstructions = []
    for i in range(NumberOfICs):
        # Extract the independent component
        component = ica.mixing_[:, i].reshape(-1, 1)
        
        if plot:
            plt.plot(component)
            plt.title("Component " + str(i))
            plt.show()
        
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

    return ICs_df, c3d_reconstructions

def ica_sto(file_path, NumberOfICs, referenceMarker=None, plot = False, fft = False):

    # Convert the C3D data to DataFrame
    X = STOToDataframe(file_path)
    
    # Split off the 'time' column
    time_column = X['time']
    
    # Drop the 'time' column from the DataFrame
    X = X.drop('time', axis=1)
    
    # Record the original mean and std for each feature
    original_mean = X.mean(axis=0)
    original_std = X.std(axis=0)

    # Normalize the features
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X) 
    
    # Implement ICA
    ica = FastICA(n_components = NumberOfICs)
    transformed = ica.fit_transform(X_normalized)

    # Create a DataFrame for the independent components
    ICs_df = pd.DataFrame(transformed, columns=[f'IC_{i+1}' for i in range(NumberOfICs)])
    
    reconstructions = []
    for i in range(NumberOfICs):
                  
        # Extract the independent component
        component = ica.mixing_[:, i].reshape(-1, 1)
        
        # Project the data onto this component
        proj = transformed[:, i].reshape(-1, 1)
        
        if plot:
            plt.plot(component)
            plt.title("Component " + str(i))
            plt.show()
            
            plt.plot(proj)
            plt.title("Projection " + str(i))
            plt.show()

        # Reconstruct the data using this component
        recon = np.dot(proj, component.T) + ica.mean_
        
        # Rescale the reconstruction to the original range
        recon_scaled = recon * original_std.values + original_mean.values
        
        # Convert the re-scaled reconstruction to a DataFrame shape similar to 'X'
        recon_df = pd.DataFrame(recon_scaled, columns=X.columns)
        
        recon_df = pd.concat([time_column, recon_df], axis=1)
        
        reconstructions.append(recon_df)

    return ICs_df, reconstructions


def dmd_c3d(file_path, numberOfModes, referenceMarker=None, plot = False):
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

    # Implement DMD
    dmd = DMD(svd_rank=numberOfModes)
    dmd.fit(X_normalized.T)
    
    if plot:
        for mode in dmd.modes.T:
            plt.plot(mode.real)
            plt.title("Modes")
            plt.show()

        for dynamic in dmd.dynamics:
            plt.plot(dynamic.real)
            plt.title("Dynamics")
            plt.show()    
            
    c3d_reconstructions = []
    for i in range(numberOfModes):
        mode = dmd.modes[:,i]
        dynamic = dmd.dynamics[i,:]
        
        recon = (mode.reshape(-1, 1).dot(dynamic.reshape(1, -1))).real.T
        
        # Rescale the reconstruction to the original range
        recon_scaled = recon * original_std.values + original_mean.values

        # Convert the re-scaled reconstruction to a DataFrame shape similar to 'X'
        recon_df = pd.DataFrame(recon_scaled, columns=X.columns)

        # Convert the dataframe back to .c3d format using the provided function
        c3d_recon = DataframeToC3D(recon_df, original_c3d_data)

        c3d_reconstructions.append(c3d_recon)
    
    return dmd, c3d_reconstructions

