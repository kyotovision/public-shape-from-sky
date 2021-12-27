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

@jit(nopython=True)    
def getFresnelParamsVec_min( v, n, mu):
    A = 1/mu
    cosr = n[...,0]*v[...,0]+n[...,1]*v[...,1]+n[...,2]*v[...,2]
    sinr = np.sqrt(1-cosr**2 +1.0e-09)
    sint = A*sinr
    cost = np.sqrt(1-sint**2 +1.0e-09)
    
    # reflectance of s,p-polarized
    rs = (A*cosr-cost)/(A*cosr+cost +1.0e-09)
    rp = -(A*cost-cosr)/(A*cost+cosr +1.0e-09)
    
    # power reflectance
    Rp = rp**2 # 0-dim problem
    Rs = rs**2

    # power transmittance
    Tp = 1-Rp
    Ts = 1-Rs
    # cosr < 0 : not defined
    cosr_ = np.ravel(cosr)
    Rp_ = np.ravel(Rp)
    Rs_ = np.ravel(Rs)
    Tp_ = np.ravel(Tp)
    Ts_ = np.ravel(Ts)
    Rp_[ np.signbit(cosr_) ]=0
    Rs_[ np.signbit(cosr_) ]=0
    Tp_[ np.signbit(cosr_) ]=0
    Ts_[ np.signbit(cosr_) ]=0
    return ( Rp_.reshape(Rp.shape), Rs_.reshape(Rp.shape), Tp_.reshape(Rp.shape), Ts_.reshape(Rp.shape) )

#return rp and rs
@jit(nopython=True)    
def getFresnelParamsVec_min2( v, n, mu):
    A = 1/mu
    cosr = n[...,0]*v[...,0]+n[...,1]*v[...,1]+n[...,2]*v[...,2]
    sinr = np.sqrt(1-cosr**2 +1.0e-09)
    sint = A*sinr
    cost = np.sqrt(1-sint**2 +1.0e-09)
    
    # reflectance of s,p-polarized
    rs = (A*cosr-cost)/(A*cosr+cost +1.0e-09)
    rp = -(A*cost-cosr)/(A*cost+cosr +1.0e-09)
    
    # power reflectance
    Rp = rp**2 # 0-dim problem
    Rs = rs**2

    # power transmittance
    Tp = 1-Rp
    Ts = 1-Rs
    # cosr < 0 : not defined
    cosr_ = np.ravel(cosr)
    rp_ = np.ravel(rp)
    rs_ = np.ravel(rs)
    Rp_ = np.ravel(Rp)
    Rs_ = np.ravel(Rs)
    Tp_ = np.ravel(Tp)
    Ts_ = np.ravel(Ts)
    rp_[ np.signbit(cosr_) ]=0
    rs_[ np.signbit(cosr_) ]=0
    Rp_[ np.signbit(cosr_) ]=0
    Rs_[ np.signbit(cosr_) ]=0
    Tp_[ np.signbit(cosr_) ]=0
    Ts_[ np.signbit(cosr_) ]=0
    return ( Rp_.reshape(Rp.shape), Rs_.reshape(Rp.shape), Tp_.reshape(Rp.shape), Ts_.reshape(Rp.shape), rp_.reshape(rp.shape), rs_.reshape(rs.shape))

# obtain exitant Fresnel coefficients
@jit(nopython=True)
def getFresnelParamsVec_o_min( v, n, mu):
    A = mu
    cost = n[...,0]*v[...,0]+n[...,1]*v[...,1]+n[...,2]*v[...,2]
    sint = np.sqrt(1-cost**2 +1.0e-09)
    sinr = sint/A
    cosr = np.sqrt(1-sinr**2 +1.0e-09)
    
    # reflectance of s,p-polarized
    rs = (A*cosr-cost)/(A*cosr+cost +1.0e-09)
    rp = -(A*cost-cosr)/(A*cost+cosr +1.0e-09)
    
    # power reflectance
    Rp = rp**2 # 0-dim problem
    Rs = rs**2

    # power transmittance
    Tp = 1-Rp
    Ts = 1-Rs
    # cost < 0 : not defined
    cost_ = np.ravel(cost)
    Rp_ = np.ravel(Rp)
    Rs_ = np.ravel(Rs)
    Tp_ = np.ravel(Tp)
    Ts_ = np.ravel(Ts)
    Rp_[ np.signbit(cost_) ]=0
    Rs_[ np.signbit(cost_) ]=0
    Tp_[ np.signbit(cost_) ]=0
    Ts_[ np.signbit(cost_) ]=0
    return ( Rp_.reshape(Rp.shape), Rs_.reshape(Rp.shape), Tp_.reshape(Rp.shape), Ts_.reshape(Rp.shape) )

@jit
def getFresnelParamsVecOnePixel_min( v, n, mu):
    A = 1/mu
    cosr = n[...,0]*v[...,0]+n[...,1]*v[...,1]+n[...,2]*v[...,2]
    sinr = np.sqrt(1-cosr**2 +1.0e-09)
    sint = A*sinr
    cost = np.sqrt(1-sint**2 +1.0e-09)
    
    # reflectance of s,p-polarized
    rs = (A*cosr-cost)/(A*cosr+cost +1.0e-09)
    rp = -(A*cost-cosr)/(A*cost+cosr +1.0e-09)
    
    # power reflectance
    Rp = rp**2 # 0-dim problem
    Rs = rs**2

    # power transmittance
    Tp = 1-Rp
    Ts = 1-Rs
    # cosr < 0 : not defined
    if cosr < 0 :
        Rp = 0
        Rs = 0
        Tp = 0
        Ts = 0
    return ( Rp, Rs, Tp, Ts)

#return rp and rs
@jit(nopython=True)    
def getFresnelParamsVecOnePixel_min2( v, n, mu):
    A = 1/mu
    cosr = n[...,0]*v[...,0]+n[...,1]*v[...,1]+n[...,2]*v[...,2]
    sinr = np.sqrt(1-cosr**2 +1.0e-09)
    sint = A*sinr
    cost = np.sqrt(1-sint**2 +1.0e-09)
    
    # reflectance of s,p-polarized
    rs = (A*cosr-cost)/(A*cosr+cost +1.0e-09)
    rp = -(A*cost-cosr)/(A*cost+cosr +1.0e-09)
    
    # power reflectance
    Rp = rp**2 # 0-dim problem
    Rs = rs**2

    # power transmittance
    Tp = 1-Rp
    Ts = 1-Rs
    # cosr < 0 : not defined
    if cosr < 0 :
        rp = 0
        rs = 0
        Rp = 0
        Rs = 0
        Tp = 0
        Ts = 0
    return ( Rp, Rs, Tp, Ts, rp, rs)

# obtain exitant Fresnel coefficients
@jit(nopython=True)
def getFresnelParamsVecOnePixel_o_min( v, n, mu):
    A = mu
    cost = n[...,0]*v[...,0]+n[...,1]*v[...,1]+n[...,2]*v[...,2]
    sint = np.sqrt(1-cost**2 +1.0e-09)
    sinr = sint/A
    cosr = np.sqrt(1-sinr**2 +1.0e-09)
    
    # reflectance of s,p-polarized
    rs = (A*cosr-cost)/(A*cosr+cost +1.0e-09)
    rp = -(A*cost-cosr)/(A*cost+cosr +1.0e-09)
    
    # power reflectance
    Rp = rp**2 # 0-dim problem
    Rs = rs**2

    # power transmittance
    Tp = 1-Rp
    Ts = 1-Rs
    # cost < 0 : not defined
    if cosr < 0 :
        Rp = 0
        Rs = 0
        Tp = 0
        Ts = 0
    return ( Rp, Rs, Tp, Ts)

@jit(nopython=True)
def xyzdot( v1, v2):
    dotproduct = v1[...,0]*v2[...,0]+v1[...,1]*v2[...,1]+v1[...,2]*v2[...,2]

    ret = dotproduct

    return ret


@jit(nopython=True)
def xyzcross( v1, v2):
    ret = np.zeros(v1.shape)
    ret[...,0] = v1[...,1]*v2[...,2]-v1[...,2]*v2[...,1]
    ret[...,1] = v1[...,2]*v2[...,0]-v1[...,0]*v2[...,2]
    ret[...,2] = v1[...,0]*v2[...,1]-v1[...,1]*v2[...,0]
    return ret


@jit(nopython=True)
def xyznormalize( vect):
    vp2 = vect**2
    scale = np.sqrt(vp2[...,0]+vp2[...,1]+vp2[...,2])+1.0e-09
    vect_normalized = np.zeros(vect.shape)
    vect_normalized[...,0] = vect[...,0]/scale
    vect_normalized[...,1] = vect[...,1]/scale
    vect_normalized[...,2] = vect[...,2]/scale
    return vect_normalized

def getNormalImg( normals):
    normal_img = normals.copy()
    normal_img[:,:,1:3] = -normal_img[:,:,1:3]  
    normal_img= (normal_img+1)/2
    return normal_img

# input : angle to rotate the coordinate
# output : 4x4 rotation matrix for stokes vector
@jit
def getStokesRotation( phi):
    h = phi.shape[0]
    w = phi.shape[1]
    C = np.zeros((h,w,4,4))
    C[:,:,0,0] = 1
    C[:,:,0,1] = 0
    C[:,:,0,2] = 0
    C[:,:,0,3] = 0
    C[:,:,1,0] = 0
    C[:,:,1,1] = np.cos(2*phi)
    C[:,:,1,2] = -np.sin(2*phi)
    C[:,:,1,3] = 0
    C[:,:,2,0] = 0
    C[:,:,2,1] = np.sin(2*phi)
    C[:,:,2,2] = np.cos(2*phi)
    C[:,:,2,3] = 0
    C[:,:,3,0] = 0
    C[:,:,3,1] = 0
    C[:,:,3,2] = 0
    C[:,:,3,3] = 1
    return C

@jit
def getStokesRotationOnePixel( phi):
    C = np.zeros((4,4))
    C[0,0] = 1
    C[0,1] = 0
    C[0,2] = 0
    C[0,3] = 0
    C[1,0] = 0
    C[1,1] = np.cos(2*phi)
    C[1,2] = -np.sin(2*phi)
    C[1,3] = 0
    C[2,0] = 0
    C[2,1] = np.sin(2*phi)
    C[2,2] = np.cos(2*phi)
    C[2,3] = 0
    C[3,0] = 0
    C[3,1] = 0
    C[3,2] = 0
    C[3,3] = 1
    return C