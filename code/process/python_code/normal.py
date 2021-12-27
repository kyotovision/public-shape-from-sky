import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import time
import cv2
from numba.experimental import jitclass
from numba import double, float32, int32, boolean, typeof
from numba import jit
from numba.types import Tuple, List

pi = np.pi

@jit
def n2pt_x(n):
    phi = np.arctan2(n[...,2],n[...,1])
    theta = np.arccos(n[...,0])
    return phi, theta

@jit
def pt2n_x(phi,theta):
    h = phi.shape[0]
    w = phi.shape[1]
    n = np.zeros((h,w,3))
    n[...,0] = np.cos(theta)
    n[...,1] = np.cos(phi)*np.sin(theta)
    n[...,2] = np.sin(phi)*np.sin(theta)
    return n

@jit
def pt2n_x_scalar(phi,theta):
    n = np.zeros((3))
    n[0] = np.cos(theta)
    n[1] = np.cos(phi)*np.sin(theta)
    n[2] = np.sin(phi)*np.sin(theta)
    return n