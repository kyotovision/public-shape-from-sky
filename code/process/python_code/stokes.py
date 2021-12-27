import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys
import time
from numba.experimental import jitclass
from numba import double, float32, int32, boolean, typeof
from numba import jit
from numba.types import Tuple, List

from mpl_toolkits.axes_grid1 import make_axes_locatable
from pimg import PBayer
import glob
import json

import mueller

# rho_d is uniform
def getStokes_PerezSkyModel(n, refractive_index, sun_dir, DoLPsmax, I_zeta, I_d, I_s, rho_d, skyModelParams, zenith):
    h = n.shape[0]
    w = n.shape[1]
    view = np.array([0,0,-1])
    a = skyModelParams[0]
    b = skyModelParams[1]
    c = skyModelParams[2]
    d = skyModelParams[3]
    e = skyModelParams[4]
    # get direction of light sources for each pixel position
    l_sky = 2*np.dot( n, view )[:,:,np.newaxis] *(n) - view
    
    # get Fresnel reflectance
    Rp_i, Rs_i, Tp_i, Ts_i, rp_i, rs_i = mueller.getFresnelParamsVec_min2( view, n, refractive_index)
    Rp_o, Rs_o, Tp_o, Ts_o = mueller.getFresnelParamsVec_o_min( view, n, refractive_index)
    F_R = np.zeros((h,w,4,4))
    F_R[:,:,0,0] = (Rs_i+Rp_i)/2.0
    F_R[:,:,0,1] = (Rs_i-Rp_i)/2.0
    F_R[:,:,0,2] = 0
    F_R[:,:,0,3] = 0
    F_R[:,:,1,0] = (Rs_i-Rp_i)/2.0
    F_R[:,:,1,1] = (Rs_i+Rp_i)/2.0
    F_R[:,:,1,2] = 0
    F_R[:,:,1,3] = 0
    F_R[:,:,2,0] = 0
    F_R[:,:,2,1] = 0
    F_R[:,:,2,2] = rs_i*rp_i
    F_R[:,:,2,3] = 0
    F_R[:,:,3,0] = 0
    F_R[:,:,3,1] = 0
    F_R[:,:,3,2] = 0
    F_R[:,:,3,3] = rs_i*rp_i
    
    # get diffuse reflectance as 4x4 Mueller matrix
    F_T_o = np.zeros((h,w,4,4))
    F_T_o[:,:,0,0] = (Ts_o+Tp_o)/2.0
    F_T_o[:,:,0,1] = (Ts_o-Tp_o)/2.0
    F_T_o[:,:,0,2] = 0
    F_T_o[:,:,0,3] = 0
    F_T_o[:,:,1,0] = (Ts_o-Tp_o)/2.0
    F_T_o[:,:,1,1] = (Ts_o+Tp_o)/2.0
    F_T_o[:,:,1,2] = 0
    F_T_o[:,:,1,3] = 0
    F_T_o[:,:,2,0] = 0
    F_T_o[:,:,2,1] = 0
    F_T_o[:,:,2,2] = np.sqrt(Ts_o*Tp_o)
    F_T_o[:,:,2,3] = 0
    F_T_o[:,:,3,0] = 0
    F_T_o[:,:,3,1] = 0
    F_T_o[:,:,3,2] = 0
    F_T_o[:,:,3,3] = np.sqrt(Ts_o*Tp_o)
    
    # get rotation of observation side
    phi_o = -np.pi/2 + np.arctan2(n[:,:,1],n[:,:,0])
    C_o = mueller.getStokesRotation(phi_o)

    camera_zenith = np.array([0,-1,0])
    # get x_i y_i
    x_axis = np.cross(l_sky,camera_zenith) 
    x_axis = mueller.xyznormalize(x_axis)
    # prevent error when l_sky and camara_zenith are parallel
    x_axis[...,0] += 1e-9
    x_axis = mueller.xyznormalize(x_axis)
    
    y_axis = np.cross(l_sky,x_axis) # right hand coordinate
    y_axis = mueller.xyznormalize(y_axis)
    # get R c2i(camera to tangent plane coordinate)
    R = np.zeros((h,w,3,3))
    R[:,:,0] = x_axis
    R[:,:,1] = y_axis
    R[:,:,2] = l_sky

    #get rotation of observation side
    n_i = R @ n[...,np.newaxis]
    
    n_i[...,2,0] = 0
    phi_i = -np.pi/2 + np.arctan2(n_i[:,:,1,0],n_i[:,:,0,0])
    C_i = mueller.getStokesRotation(-phi_i)
    
    # multiply by scalar coefficient I_e later
    M_s = C_o @ F_R @ C_i
    M_d = C_o @ F_T_o

    # calculate stokes vector of the sky
    cro = np.cross(sun_dir,l_sky)
    AoLPs_vec = R @ cro[...,np.newaxis]
    AoLPs_vec[:,:,2,0] = 0
    AoLPs = np.arctan2(AoLPs_vec[:,:,1,0],AoLPs_vec[:,:,0,0])
    AoLPs[AoLPs > np.pi/2] -= np.pi
    AoLPs[AoLPs < -np.pi/2] += np.pi
    cos_gamma = np.dot(l_sky[:,:],sun_dir)
    cos_gamma = np.clip(cos_gamma,-1,1)
    DoLPs = DoLPsmax * (1-cos_gamma**2)/(1+cos_gamma**2)

    s_sky = np.zeros((h,w,4,1))
    s_sky[...,0,0] = 1
    s_sky[...,1,0] = DoLPs*np.cos(2*AoLPs)
    s_sky[...,2,0] = DoLPs*np.sin(2*AoLPs)
    s_sky[...,3,0] = 0

    # calculate radiance of the sky
    cos_theta_p = zenith[0]*l_sky[...,0] + zenith[1]*l_sky[...,1] + zenith[2]*l_sky[...,2]
    cos_theta_p = np.clip(cos_theta_p,0,1)
    gamma = np.arccos(cos_gamma)
    l_p = (1+a*np.exp(b/(cos_theta_p+1e-9))) * (1+c*np.exp(d*gamma)+e*np.cos(gamma)**2)
    I_l = I_zeta * l_p

    Rp_sun, Rs_sun, Tp_sun, Ts_sun, rp_sun, rs_sun = mueller.getFresnelParamsVec_min2( sun_dir, n, refractive_index)
    shading = np.dot(n, sun_dir)
    shading[shading < 0] = 0

    s_specular = 2*I_l[...,np.newaxis,np.newaxis] * (n[...,0]*l_sky[...,0]+n[...,1]*l_sky[...,1]+n[...,2]*l_sky[...,2])[...,np.newaxis,np.newaxis] * (M_s @ s_sky)
    s_diffuse = 2*(I_d + shading*I_s*(Ts_sun+Tp_sun)/2.0)[...,np.newaxis,np.newaxis] * rho_d * (M_d @ np.array([1.0,0,0,0]).reshape(4,1))
    s_o = s_specular + s_diffuse
    s_o = s_o.reshape(s_o.shape[0], s_o.shape[1], s_o.shape[2])    
    return s_o

# rho_d is spatially varying
def getStokes_PerezSkyModel2(n, refractive_index, sun_dir, DoLPsmax, I_zeta, I_d, I_s, rho_d, skyModelParams, zenith):
    h = n.shape[0]
    w = n.shape[1]
    view = np.array([0,0,-1])
    a = skyModelParams[0]
    b = skyModelParams[1]
    c = skyModelParams[2]
    d = skyModelParams[3]
    e = skyModelParams[4]
    # get direction of light sources for each pixel position
    l_sky = 2*np.dot( n, view )[:,:,np.newaxis] *(n) - view
    
    # get Fresnel reflectance
    Rp_i, Rs_i, Tp_i, Ts_i, rp_i, rs_i = mueller.getFresnelParamsVec_min2( view, n, refractive_index)
    Rp_o, Rs_o, Tp_o, Ts_o = mueller.getFresnelParamsVec_o_min( view, n, refractive_index)
    F_R = np.zeros((h,w,4,4))
    F_R[:,:,0,0] = (Rs_i+Rp_i)/2.0
    F_R[:,:,0,1] = (Rs_i-Rp_i)/2.0
    F_R[:,:,0,2] = 0
    F_R[:,:,0,3] = 0
    F_R[:,:,1,0] = (Rs_i-Rp_i)/2.0
    F_R[:,:,1,1] = (Rs_i+Rp_i)/2.0
    F_R[:,:,1,2] = 0
    F_R[:,:,1,3] = 0
    F_R[:,:,2,0] = 0
    F_R[:,:,2,1] = 0
    F_R[:,:,2,2] = rs_i*rp_i
    F_R[:,:,2,3] = 0
    F_R[:,:,3,0] = 0
    F_R[:,:,3,1] = 0
    F_R[:,:,3,2] = 0
    F_R[:,:,3,3] = rs_i*rp_i
    
    # get diffuse reflectance as 4x4 Mueller matrix
    F_T_o = np.zeros((h,w,4,4))
    F_T_o[:,:,0,0] = (Ts_o+Tp_o)/2.0
    F_T_o[:,:,0,1] = (Ts_o-Tp_o)/2.0
    F_T_o[:,:,0,2] = 0
    F_T_o[:,:,0,3] = 0
    F_T_o[:,:,1,0] = (Ts_o-Tp_o)/2.0
    F_T_o[:,:,1,1] = (Ts_o+Tp_o)/2.0
    F_T_o[:,:,1,2] = 0
    F_T_o[:,:,1,3] = 0
    F_T_o[:,:,2,0] = 0
    F_T_o[:,:,2,1] = 0
    F_T_o[:,:,2,2] = np.sqrt(Ts_o*Tp_o)
    F_T_o[:,:,2,3] = 0
    F_T_o[:,:,3,0] = 0
    F_T_o[:,:,3,1] = 0
    F_T_o[:,:,3,2] = 0
    F_T_o[:,:,3,3] = np.sqrt(Ts_o*Tp_o)
    
    # get rotation of observation side
    phi_o = -np.pi/2 + np.arctan2(n[:,:,1],n[:,:,0])
    C_o = mueller.getStokesRotation(phi_o)

    camera_zenith = np.array([0,-1,0])
    # get x_i y_i
    x_axis = np.cross(l_sky,camera_zenith) 
    x_axis = mueller.xyznormalize(x_axis)
    # prevent error when l_sky and camara_zenith are parallel
    x_axis[...,0] += 1e-9
    x_axis = mueller.xyznormalize(x_axis)
    
    y_axis = np.cross(l_sky,x_axis) # right hand coordinate
    y_axis = mueller.xyznormalize(y_axis)
    # get R c2i(camera to tangent plane coordinate)
    R = np.zeros((h,w,3,3))
    R[:,:,0] = x_axis
    R[:,:,1] = y_axis
    R[:,:,2] = l_sky

    #get rotation of observation side
    n_i = R @ n[...,np.newaxis]
    
    n_i[...,2,0] = 0
    phi_i = -np.pi/2 + np.arctan2(n_i[:,:,1,0],n_i[:,:,0,0])
    C_i = mueller.getStokesRotation(-phi_i)
    
    # multiply by scalar coefficient I_e later
    M_s = C_o @ F_R @ C_i
    M_d = C_o @ F_T_o

    # calculate stokes vector of the sky
    cro = np.cross(sun_dir,l_sky)
    AoLPs_vec = R @ cro[...,np.newaxis]
    AoLPs_vec[:,:,2,0] = 0
    AoLPs = np.arctan2(AoLPs_vec[:,:,1,0],AoLPs_vec[:,:,0,0])
    AoLPs[AoLPs > np.pi/2] -= np.pi
    AoLPs[AoLPs < -np.pi/2] += np.pi
    cos_gamma = np.dot(l_sky[:,:],sun_dir)
    cos_gamma = np.clip(cos_gamma,-1,1)
    DoLPs = DoLPsmax * (1-cos_gamma**2)/(1+cos_gamma**2)

    s_sky = np.zeros((h,w,4,1))
    s_sky[...,0,0] = 1
    s_sky[...,1,0] = DoLPs*np.cos(2*AoLPs)
    s_sky[...,2,0] = DoLPs*np.sin(2*AoLPs)
    s_sky[...,3,0] = 0

    # calculate radiance of the sky
    cos_theta_p = zenith[0]*l_sky[...,0] + zenith[1]*l_sky[...,1] + zenith[2]*l_sky[...,2]
    cos_theta_p = np.clip(cos_theta_p,0,1)
    gamma = np.arccos(cos_gamma)
    l_p = (1+a*np.exp(b/(cos_theta_p+1e-9))) * (1+c*np.exp(d*gamma)+e*np.cos(gamma)**2)
    I_l = I_zeta * l_p

    Rp_sun, Rs_sun, Tp_sun, Ts_sun, rp_sun, rs_sun = mueller.getFresnelParamsVec_min2( sun_dir, n, refractive_index)
    shading = np.dot(n, sun_dir)
    shading[shading < 0] = 0

    s_specular = 2*I_l[...,np.newaxis,np.newaxis] * (n[...,0]*l_sky[...,0]+n[...,1]*l_sky[...,1]+n[...,2]*l_sky[...,2])[...,np.newaxis,np.newaxis] * (M_s @ s_sky)
    s_diffuse = 2*(I_d + shading*I_s*(Ts_sun+Tp_sun)/2.0)[...,np.newaxis,np.newaxis] * rho_d[...,np.newaxis,np.newaxis] * (M_d @ np.array([1.0,0,0,0]).reshape(4,1))
    s_o = s_specular + s_diffuse
    s_o = s_o.reshape(s_o.shape[0], s_o.shape[1], s_o.shape[2])    
    return s_o

def getDoLPandAoLP(stokes):
    DoLP = np.clip(np.sqrt(stokes[...,1]**2 + stokes[...,2]**2) / (stokes[...,0] + 1e-9), 0, 1)
    AoLP = np.arctan2(stokes[...,2], stokes[...,1])
    AoLP /= 2
    return DoLP, AoLP

# sky model param doesn't depend on time.
@jit
def getOnePixelResidual_PerezSkyModel_Timevariation(n, observed_stokes, refractive_index, sun_dir, DoLPsmax, I_zeta, I_d, I_s, rho_d, skyModelParams, zenith):
    image_num = observed_stokes.shape[0]
    view = np.array([0,0,-1.0])
    # get direction of light sources for each pixel position
    l_sky = 2*np.dot( n, view ) *(n) - view
    a = skyModelParams[0]
    b = skyModelParams[1]
    c = skyModelParams[2]
    d = skyModelParams[3]
    e = skyModelParams[4]

    # get Fresnel reflectance
    Rp_i, Rs_i, Tp_i, Ts_i, rp_i, rs_i = mueller.getFresnelParamsVecOnePixel_min2( view, n, refractive_index)
    Rp_o, Rs_o, Tp_o, Ts_o = mueller.getFresnelParamsVecOnePixel_o_min( view, n, refractive_index)
    F_R = np.zeros((4,4))
    F_R[0,0] = (Rs_i+Rp_i)/2.0
    F_R[0,1] = (Rs_i-Rp_i)/2.0
    F_R[0,2] = 0
    F_R[0,3] = 0
    F_R[1,0] = (Rs_i-Rp_i)/2.0
    F_R[1,1] = (Rs_i+Rp_i)/2.0
    F_R[1,2] = 0
    F_R[1,3] = 0
    F_R[2,0] = 0
    F_R[2,1] = 0
    F_R[2,2] = rs_i*rp_i
    F_R[2,3] = 0
    F_R[3,0] = 0
    F_R[3,1] = 0
    F_R[3,2] = 0
    F_R[3,3] = rs_i*rp_i
    
    # get diffuse reflectance as 4x4 Mueller matrix
    F_T_o = np.zeros((4,4))
    F_T_o[0,0] = (Ts_o+Tp_o)/2.0
    F_T_o[0,1] = (Ts_o-Tp_o)/2.0
    F_T_o[0,2] = 0
    F_T_o[0,3] = 0
    F_T_o[1,0] = (Ts_o-Tp_o)/2.0
    F_T_o[1,1] = (Ts_o+Tp_o)/2.0
    F_T_o[1,2] = 0
    F_T_o[1,3] = 0
    F_T_o[2,0] = 0
    F_T_o[2,1] = 0
    F_T_o[2,2] = np.sqrt(Ts_o*Tp_o)
    F_T_o[2,3] = 0
    F_T_o[3,0] = 0
    F_T_o[3,1] = 0
    F_T_o[3,2] = 0
    F_T_o[3,3] = np.sqrt(Ts_o*Tp_o)
    
    # get rotation of observation side
    phi_o = -np.pi/2 + np.arctan2(n[1],n[0])
    C_o = mueller.getStokesRotationOnePixel(phi_o)

    camera_zenith = np.array([0,-1,0])
    # get x_i y_i
    x_axis = np.cross(l_sky,camera_zenith)
    x_axis = mueller.xyznormalize(x_axis)
    # prevent error when l_sky and camera_zenith are parallel
    x_axis[0] += 1e-9
    x_axis = mueller.xyznormalize(x_axis)
    
    y_axis = np.cross(l_sky,x_axis) # right hand coordinate
    y_axis = mueller.xyznormalize(y_axis)
    # get R c2i(camera to tangent plane coordinate)
    R = np.zeros((3,3))
    R[0] = x_axis
    R[1] = y_axis
    R[2] = l_sky

    #get rotation of observation side
    n_i = R @ n.reshape(3,1)
    
    n_i[2,0] = 0
    phi_i = -np.pi/2 + np.arctan2(n_i[1,0],n_i[0,0])
    C_i = mueller.getStokesRotationOnePixel(-phi_i)
    
    # multiply by scalar coefficient I_e later
    M_s = C_o @ F_R @ C_i
    M_d = C_o @ F_T_o
    
    residual = np.zeros(4*image_num)
    for i in range(0, image_num):
        I_zeta_i = I_zeta[i]
        I_d_i = I_d[i]
        I_s_i = I_s[i]
        sun_dir_i = sun_dir[i]
        DoLPsmax_i = DoLPsmax[i]
        
        # calculate stokes vector of the sky
        cro = np.cross(sun_dir_i, l_sky)
        AoLPs_vec = R @ cro.reshape(3,1)
        AoLPs_vec[2,0] = 0
        AoLPs = np.arctan2(AoLPs_vec[1,0],AoLPs_vec[0,0])
        if AoLPs > np.pi/2:
            AoLPs -= np.pi
        elif AoLPs < np.pi/2:
            AoLPs += np.pi
        cos_gamma = np.dot(l_sky, sun_dir_i)
        if cos_gamma > 1:
            cos_gamma = 1
        elif cos_gamma < -1:
            cos_gamma = -1
        DoLPs = DoLPsmax_i * (1-cos_gamma**2)/(1+cos_gamma**2)

        s_sky = np.zeros((4,1))
        s_sky[0] = 1
        s_sky[1] = DoLPs*np.cos(2*AoLPs)
        s_sky[2] = DoLPs*np.sin(2*AoLPs)
        s_sky[3] = 0

        # calculate radiance of the sky
        cos_theta_p = np.dot(zenith, l_sky)
        if cos_theta_p > 1:
            cos_theta_p = 1
        elif cos_theta_p < 0:
            cos_theta_p = 0
        gamma = np.arccos(cos_gamma)

        l_p = (1+a*np.exp(b/(cos_theta_p+1e-9))) * (1+c*np.exp(d*gamma)+e*np.cos(gamma)**2)
        I_l = I_zeta_i * l_p
        
        Rp_sun, Rs_sun, Tp_sun, Ts_sun, rp_sun, rs_sun = mueller.getFresnelParamsVecOnePixel_min2( sun_dir_i, n, refractive_index)
        shading = np.dot(n, sun_dir_i)
        if shading < 0:
            shading = 0
        
        s_specular = 2.0*I_l*np.dot(n,l_sky) * M_s @ s_sky
        s_diffuse = 2.0*(I_d_i + I_s_i*shading*(Ts_sun+Tp_sun)/2.0) * rho_d * M_d @ np.array([1.0,0,0,0]).reshape(4,1)
        s_o = s_specular + s_diffuse
        s_o = s_o.reshape(4)
        # invert y-axis
        s_o[2] = -s_o[2]
        
        residual[4*i:4*(i+1)] = observed_stokes[i] - s_o
        if observed_stokes[i,0] < 1e-9:
            residual[4*i:4*(i+1)] = 0
    return residual

@jit
def getOnePixelStokes_PerezSkyModel_Timevariation(n, refractive_index, sun_dir, DoLPsmax, I_zeta, I_d, I_s, rho_d, skyModelParams, zenith):
    image_num = sun_dir.shape[0]
    view = np.array([0,0,-1.0])
    # get direction of light sources for each pixel position
    l_sky = 2*np.dot( n, view ) *(n) - view
    a = skyModelParams[0]
    b = skyModelParams[1]
    c = skyModelParams[2]
    d = skyModelParams[3]
    e = skyModelParams[4]

    # get Fresnel reflectance
    Rp_i, Rs_i, Tp_i, Ts_i, rp_i, rs_i = mueller.getFresnelParamsVecOnePixel_min2( view, n, refractive_index)
    Rp_o, Rs_o, Tp_o, Ts_o = mueller.getFresnelParamsVecOnePixel_o_min( view, n, refractive_index)
    F_R = np.zeros((4,4))
    F_R[0,0] = (Rs_i+Rp_i)/2.0
    F_R[0,1] = (Rs_i-Rp_i)/2.0
    F_R[0,2] = 0
    F_R[0,3] = 0
    F_R[1,0] = (Rs_i-Rp_i)/2.0
    F_R[1,1] = (Rs_i+Rp_i)/2.0
    F_R[1,2] = 0
    F_R[1,3] = 0
    F_R[2,0] = 0
    F_R[2,1] = 0
    F_R[2,2] = rs_i*rp_i
    F_R[2,3] = 0
    F_R[3,0] = 0
    F_R[3,1] = 0
    F_R[3,2] = 0
    F_R[3,3] = rs_i*rp_i
    
    # get diffuse reflectance as 4x4 Mueller matrix
    F_T_o = np.zeros((4,4))
    F_T_o[0,0] = (Ts_o+Tp_o)/2.0
    F_T_o[0,1] = (Ts_o-Tp_o)/2.0
    F_T_o[0,2] = 0
    F_T_o[0,3] = 0
    F_T_o[1,0] = (Ts_o-Tp_o)/2.0
    F_T_o[1,1] = (Ts_o+Tp_o)/2.0
    F_T_o[1,2] = 0
    F_T_o[1,3] = 0
    F_T_o[2,0] = 0
    F_T_o[2,1] = 0
    F_T_o[2,2] = np.sqrt(Ts_o*Tp_o)
    F_T_o[2,3] = 0
    F_T_o[3,0] = 0
    F_T_o[3,1] = 0
    F_T_o[3,2] = 0
    F_T_o[3,3] = np.sqrt(Ts_o*Tp_o)
    
    # get rotation of observation side
    phi_o = -np.pi/2 + np.arctan2(n[1],n[0])
    C_o = mueller.getStokesRotationOnePixel(phi_o)

    camera_zenith = np.array([0,-1,0])
    # get x_i y_i
    x_axis = np.cross(l_sky,camera_zenith)
    x_axis = mueller.xyznormalize(x_axis)
    # prevent error when l_sky and camera_zenith are parallel
    x_axis[0] += 1e-9
    x_axis = mueller.xyznormalize(x_axis)
    
    y_axis = np.cross(l_sky,x_axis) # right hand coordinate
    y_axis = mueller.xyznormalize(y_axis)
    # get R c2i(camera to tangent plane coordinate)
    R = np.zeros((3,3))
    R[0] = x_axis
    R[1] = y_axis
    R[2] = l_sky

    #get rotation of observation side
    n_i = R @ n.reshape(3,1)
    
    n_i[2,0] = 0
    phi_i = -np.pi/2 + np.arctan2(n_i[1,0],n_i[0,0])
    C_i = mueller.getStokesRotationOnePixel(-phi_i)
    
    M_s = C_o @ F_R @ C_i
    M_d = C_o @ F_T_o
    
    #residual = np.zeros(4*image_num)
    ret_stokes = np.zeros((image_num,4))

    for i in range(0, image_num):
        I_zeta_i = I_zeta[i]
        I_d_i = I_d[i]
        I_s_i = I_s[i]
        sun_dir_i = sun_dir[i]
        DoLPsmax_i = DoLPsmax[i]
        
        # calculate stokes vector of the sky
        cro = np.cross(sun_dir_i, l_sky)
        AoLPs_vec = R @ cro.reshape(3,1)
        AoLPs_vec[2,0] = 0
        AoLPs = np.arctan2(AoLPs_vec[1,0],AoLPs_vec[0,0])
        if AoLPs > np.pi/2:
            AoLPs -= np.pi
        elif AoLPs < np.pi/2:
            AoLPs += np.pi
        cos_gamma = np.dot(l_sky, sun_dir_i)
        if cos_gamma > 1:
            cos_gamma = 1
        elif cos_gamma < -1:
            cos_gamma = -1
        DoLPs = DoLPsmax_i * (1-cos_gamma**2)/(1+cos_gamma**2)

        s_sky = np.zeros((4,1))
        s_sky[0] = 1
        s_sky[1] = DoLPs*np.cos(2*AoLPs)
        s_sky[2] = DoLPs*np.sin(2*AoLPs)
        s_sky[3] = 0

        # calculate radiance of the sky
        cos_theta_p = np.dot(zenith, l_sky)
        if cos_theta_p > 1:
            cos_theta_p = 1
        elif cos_theta_p < 0:
            cos_theta_p = 0
        gamma = np.arccos(cos_gamma)

        l_p = (1+a*np.exp(b/(cos_theta_p+1e-9))) * (1+c*np.exp(d*gamma)+e*np.cos(gamma)**2)
        I_l = I_zeta_i * l_p
        
        Rp_sun, Rs_sun, Tp_sun, Ts_sun, rp_sun, rs_sun = mueller.getFresnelParamsVecOnePixel_min2( sun_dir_i, n, refractive_index)
        shading = np.dot(n, sun_dir_i)
        if shading < 0:
            shading = 0
        
        s_specular = 2.0*I_l*np.dot(n,l_sky) * M_s @ s_sky
        s_diffuse = 2.0*(I_d_i + I_s_i*shading*(Ts_sun+Tp_sun)/2.0) * rho_d * M_d @ np.array([1.0,0,0,0]).reshape(4,1)
        s_o = s_specular + s_diffuse
        s_o = s_o.reshape(4)
        ret_stokes[i] = s_o
        # invert y-axis
        # s_o[2] = -s_o[2]
    return ret_stokes