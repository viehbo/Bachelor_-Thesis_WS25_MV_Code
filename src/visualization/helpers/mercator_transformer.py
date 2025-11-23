import numpy as np
from math import pi

R = 6378137.0  # Earth radius used by WebMercator

def lon_to_x(lon):
    return np.asarray(lon, dtype="float64") * (pi / 180.0) * R

def lat_to_y(lat):
    lat = np.asarray(lat, dtype="float64")
    lat = np.clip(lat, -85.05112878, 85.05112878)  # valid mercator range
    return R * np.log(np.tan((pi / 4.0) + (lat * (pi / 180.0) / 2.0)))

def x_to_lon(xm):
    return (np.asarray(xm)/R) * 180.0/pi

def y_to_lat(ym):
    y = np.asarray(ym)/R
    return (2.0*np.arctan(np.exp(y)) - pi/2.0) * 180.0/pi
