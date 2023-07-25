from ezc3d import c3d
import pandas as pd
import copy
import numpy as np

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
    
#Create a dataframe of the marker positions
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