import pandas
import geopandas
import shapely
import matplotlib
import cartopy
import numpy
import rasterio
import earthpy
import rioxarray
from osgeo import gdal
import overpy
import affine

import importlib.metadata
def version_checking():

    print("pandas version: ", pandas.__version__)
    print("geopandas version: ", geopandas.__version__)
    print("shapely version: ", shapely.__version__)
    print("matplotlib version: ", matplotlib.__version__)
    print("cartopy version: ", cartopy.__version__)
    print("numpy version: ", numpy.__version__)
    print("rasterio version: ", rasterio.__version__)
    print("earthpy version: ", importlib.metadata.version("earthpy"))
    print("rioxarray version: ", rioxarray.__version__)
    print("gdal version: ", gdal.__version__)
    print("overpy version: ", overpy.__version__)
    print("affine version: ", affine.__version__)

if __name__ == '__main__':
    version_checking()
