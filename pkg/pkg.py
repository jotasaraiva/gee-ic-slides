import ee
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import rpy2.robjects.numpy2ri
import rpy2.robjects as robjects
from osgeo import gdal

rpy2.robjects.numpy2ri.activate()
pandas2ri.activate()

astsa = importr('astsa')
base = importr('base')
stats = importr('stats')

def add_amplitude(image, VV = "VV", VH = "VH"):
    amplitude = image\
        .expression('(VV ** 2 + VH ** 2) ** (1 / 2)', {'VV':image.select(VV), 'VH':image.select(VH)})\
        .rename('amplitude')
    return image.addBands(amplitude)

def ee_to_pandas(imagem, geometria, bandas, scale):
    imagem = imagem.addBands(ee.Image.pixelLonLat())
    
    coordenadas = imagem.select(["longitude","latitude"] + bandas)\
        .reduceRegion(reducer=ee.Reducer.toList(),
                     geometry=geometria,
                     scale=scale,
                     bestEffort=True)
    
    coordenadas = coordenadas.getInfo()
    
    return pd.DataFrame.from_dict(coordenadas)

def extract_date_string(input_string):
    pattern = r'(\d{8})(?=[a-zA-Z])'
    input_string.replace("_", " ")
    match = re.search(pattern, input_string)
    if match:
        return match.group(1)
    else:
        return None

def rename_geodf(df):
    coords = df.loc[:,['longitude', 'latitude']]
    bands = df.drop(['longitude', 'latitude'], axis=1)
    bands.columns = list(map(extract_date_string, list(bands.columns)))
    new_df = bands.join(coords)
    return new_df

def optimize(x, *args):
    if type(x) == pd.core.series.Series:
        x = np.array(x)
    arr = list(x.flatten())
    arrays = [arr] + [list(i(arr)) for i in args]
    mat = np.array(arrays).T
    spec_env = astsa.specenv(mat, real=True, plot=False)
    beta = spec_env[spec_env[:,1]==max(spec_env[:,1]), 2:].ravel()
    opt = lambda l: np.array([l] + [list(k(l)) for k in args]).T * beta
    return pd.Series(opt(arr).sum(axis=1))  

def show_tif(path, band, palette="gray"):
    raster = gdal.Open(path)
    array = raster.GetRasterBand(band).ReadAsArray()
    return plt.imshow(array, cmap=palette)