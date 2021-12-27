import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys
import time
from argparse import ArgumentParser
from numba.experimental import jitclass
from numba import double, float32, int32, boolean, typeof
from numba import jit
from numba.types import Tuple, List
from scipy.optimize import least_squares

from mpl_toolkits.axes_grid1 import make_axes_locatable
from pimg import PBayer
import glob
import json

import mueller
import stokes
import normal

pi = np.pi

# variables : uniform intensity I_d, ratio of I_s to I_d t_d, diffuse albedo rho_d, scalar coefficients I_zeta, azimuth angle and zenith angle of normal
# azimuth limitation : [-pi , 0] yz-plane
# zenith limitation : [0 , pi] x-axis
@jit
def getResidual_init( variables, observed_stokes ,refractive_index, sun_dir, DoLPsmax, skyModelParams,zenith):
    image_num = observed_stokes.shape[0]
    pixel_num = observed_stokes.shape[1]
    
    offset = 0
    I_d = np.ones(image_num)
    I_d[1:image_num] = variables[0:image_num-1]
    offset += image_num-1
    
    t_d = variables[offset]
    I_s = I_d*t_d
    offset += 1
    
    I_zeta = variables[offset:offset+image_num]
    offset += image_num
    
    rho_d = variables[offset:offset+pixel_num]
    offset += pixel_num
    
    phi = variables[offset:offset+2*pixel_num:2]
    theta = variables[offset+1:offset+2*pixel_num:2]
    offset += 2*pixel_num
    
    residual = np.zeros(4*image_num*pixel_num)    
    for i in range(0,pixel_num):
        n = np.zeros((3))
        n[0] = np.cos(theta[i])
        n[1] = np.cos(phi[i])*np.sin(theta[i])
        n[2] = np.sin(phi[i])*np.sin(theta[i])
        
        residual[4*image_num*i:4*image_num*(i+1)] = stokes.getOnePixelResidual_PerezSkyModel_Timevariation(n, observed_stokes[:,i], refractive_index, sun_dir, DoLPsmax, I_zeta, I_d, I_s, rho_d[i], skyModelParams, zenith)
    return residual

# variables : I_d, t_d, I_s, scalar coefficients I_zeta
@jit
def getResidual( variables, observed_stokes, n, refractive_index, sun_dir, DoLPsmax, rho_d, skyModelParams,zenith):
    image_num = observed_stokes.shape[0]
    pixel_num = observed_stokes.shape[1]
    
    offset = 0
    I_d = np.ones(image_num)
    I_d[1:image_num] = variables[0:image_num-1]
    offset += image_num-1

    t_d = variables[image_num-1]
    I_s = I_d*t_d
    offset += 1
    
    I_zeta = variables[image_num+1:2*image_num+1]
    offset += image_num
    
    residual = np.zeros(4*image_num*pixel_num)
    for i in range(0,pixel_num):
        residual[4*image_num*i:4*image_num*(i+1)] = stokes.getOnePixelResidual_PerezSkyModel_Timevariation(n[i], observed_stokes[:,i], refractive_index, sun_dir, DoLPsmax, I_zeta, I_d, I_s, rho_d[i], skyModelParams, zenith)
    return residual

# variables : rho_d ,azimuth angle, zenith angle
# azimuth limitation : [-pi , 0] yz-plane
# zenith limitation : [0 , pi] x-axis
@jit
def getResidual2( variables, observed_stokes ,refractive_index, sun_dir, DoLPsmax, I_zeta, I_d, I_s, skyModelParams, zenith,mask):
    image_num = observed_stokes.shape[0]
    pixely_num = observed_stokes.shape[1] 
    pixelx_num = observed_stokes.shape[2]
    pixel_num = pixelx_num*pixely_num
    
    rho_d = variables[0]
    
    n = np.zeros((3))
    n[0] = np.cos(variables[2])
    n[1] = np.cos(variables[1])*np.sin(variables[2])
    n[2] = np.sin(variables[1])*np.sin(variables[2])
    
    residual = np.zeros(4*image_num*pixel_num)
    for i in range(0, pixely_num):
        for j in range(0, pixelx_num):
            iteration = i*pixelx_num + j
            if mask[i,j] >=1:
                residual[4*image_num*iteration : 4*image_num*(iteration+1)] = stokes.getOnePixelResidual_PerezSkyModel_Timevariation(n, observed_stokes[:,i,j], refractive_index, sun_dir, DoLPsmax, I_zeta, I_d, I_s, rho_d, skyModelParams, zenith)
            else:
                residual[4*image_num*iteration : 4*image_num*(iteration+1)] = 0
    return residual

def estimateNormal(step, pixel_range, n_init, rho_d_init, bounds, observed_stokes, refractive_index, sun_dir, DoLPsmax, I_zeta, I_d, I_s, skyModelParams, zenith, mask):
    phi_est, theta_est = normal.n2pt_x(n_init)
    h = n_init.shape[0]
    w = n_init.shape[1]
    rho_d_est = np.zeros((h,w))
    
    for i in range(0, h, step):
        if (i/step)%10==0:
            print('i: '+str(i)) 

        for j in range(0, w,step):
            if mask[i,j] >= 1:
                # take neighbor pixel
                up = i-pixel_range
                down = i+pixel_range+1
                left = j-pixel_range
                right = j+pixel_range+1
                if (up<0) | (down>=h) | (left<0) | (right>=w):
                    continue
                observed_stokes_ = observed_stokes[:, up:down, left:right]
                mask_ = mask[up:down, left:right]

                x0 = np.array([rho_d_init[i,j], phi_est[i,j], theta_est[i,j]])
                ret = least_squares(getResidual2,x0=x0, bounds=bounds, 
                                    args=(observed_stokes_, refractive_index, sun_dir, DoLPsmax, I_zeta, I_d, I_s, skyModelParams, zenith, mask_))

                #take pixel to update
                width = step//2
                up = np.max([i-width, 0])
                down = np.min([i+width+1, h])
                left = np.max([j-width, 0])
                right = np.min([j+width+1, w])
                rho_d_est[up:down, left:right] = ret.x[0]
                phi_est[up:down, left:right] = ret.x[1]
                theta_est[up:down, left:right] = ret.x[2]
            else:
                rho_d_est[i,j] = np.nan
                phi_est[i,j] = np.nan
                theta_est[i,j] = np.nan

    n_est = normal.pt2n_x(phi_est, theta_est)
    return n_est, rho_d_est

def get_option():
    argparser = ArgumentParser()
    argparser.add_argument('JSON_FILE')
    return argparser.parse_args()

def main():
    if len(sys.argv) < 2:
        print('USAGE : python shape_from_sky.py [json_file]')
        sys.exit()

    args = get_option()
    json_file = args.JSON_FILE

    json_open = open(json_file, 'r')
    json_load = json.load(json_open)

    IMAGE_PATH = json_load['IMAGE_PATH']
    save_file_name = json_load['save_file_name']
    
    # known parameters
    refractive_index = json_load['refractive_index']
    scene_list = json_load['scene_list']
    sun_dir_list = np.load(json_load['sun_dir_file'])
    DoLPsmax_list = np.array(json_load['DoLPsmax_list'])
    skyModelParams = np.array(json_load['sky_model_params'])
    z_phi, z_theta = json_load['zenith']
    z_phi *= np.pi/180
    z_theta *= np.pi/180
    zenith = np.array([ -np.cos(z_theta)*np.sin(z_phi), -np.cos(z_theta)*np.cos(z_phi), -np.sin(z_theta)]) # zenith for camera coordinate system

    # data index to use for estimation
    index_list = np.array(json_load['index_list'])
    # interval of pixel sampling
    pixelx_interval_1, pixely_interval_1 = json_load['pixel_interval_1']
    # step of normal estimation
    step1 = json_load['step1']
    # range of neighbor pixel e.g. range=1 is 3x3 pixels
    pixel_range1 = json_load['pixel_range1']

    # interval of pixel sampling
    pixelx_interval_2, pixely_interval_2 = json_load['pixel_interval_2']
    # step of normal estimation
    step2 = np.array(json_load['step2'])
    # range of neighbor pixel e.g. range=1 is 3x3 pixels
    pixel_range2 = np.array(json_load['pixel_range2'])
    step = step2[-1]
    iteration_num = step2.shape[0]

    # initial normal
    phi_init, theta_init = json_load['normal_init']
    phi_init *= np.pi/180 # yz-plane azimuth angle
    theta_init *= np.pi/180 # x-axis zenith angle
    # initial values
    I_d_init = json_load['I_d_init']
    t_d_init = json_load['t_d_init']
    rho_d_init = json_load['rho_d_init']
    I_zeta_init = json_load['I_zeta_init']

    t_d_min = 0
    t_d_max = np.inf

    mask_full = cv2.imread(IMAGE_PATH +'mask_skyarea.png',  cv2.IMREAD_GRAYSCALE).astype(np.float32) /(255-1.0)
    mask = mask_full[::2,::2]
    h, w = mask.shape

    ## estimate initial normal
    image_num = index_list.shape[0]
    observed_stokes = np.zeros((image_num,h,w,4))
    sun_dir = np.zeros((image_num, 3))
    DoLPsmax = np.zeros((image_num))

    for i in range(image_num):
        index = index_list[i]        
        pimg = cv2.imread(IMAGE_PATH + scene_list[index] +'/L.png', cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE).astype(np.float32) /(256*256-1.0)
        pbayer_ = PBayer(pimg)
        observed_stokes[i] = pbayer_.stokes
        
        sun_dir[i] = sun_dir_list[index]
        
        DoLPsmax[i] = DoLPsmax_list[index]

    #sampling some pixels to estimate uniform parameters
    pixel_num = 0
    max_pixel_num = int(np.ceil((h/pixely_interval_1+1)*(w/pixelx_interval_1+1)))
    sampled_observed_stokes = np.zeros((image_num, max_pixel_num, 4))
    for i in range(0,h,pixely_interval_1):
        for j in range(0,w,pixelx_interval_1):
            if mask[i,j] >= 1:
                sampled_observed_stokes[:,pixel_num,:] = observed_stokes[:,i,j,:]
                pixel_num += 1
    sampled_observed_stokes = sampled_observed_stokes[:,0:pixel_num,:]
    print("pixel_num : "+str(pixel_num))

    sep1 = image_num-1 # I_d
    sep2 = sep1+1 # t_d
    sep3 = sep2+image_num # I_zata
    sep4 = sep3+pixel_num # rho_d
    sep5 = sep4+2*pixel_num # normals
    # initial value
    x0 = np.zeros((sep5))
    x0[0:sep1] = I_d_init
    x0[sep1:sep2] = t_d_init
    x0[sep2:sep3] = I_zeta_init
    x0[sep3:sep4] = rho_d_init
    x0[sep4:sep5:2] = phi_init
    x0[sep4+1:sep5:2] = theta_init

    # boundary
    lb = np.zeros((sep5))
    lb[0:sep1] = 0
    lb[sep1:sep2] = t_d_min
    lb[sep2:sep3] =0
    lb[sep3:sep4] = 0
    lb[sep4:sep5:2] = -np.pi
    lb[sep4+1:sep5:2] = 0

    ub = np.zeros((sep5))
    ub[0:sep1] = np.inf
    ub[sep1:sep2] = t_d_max
    ub[sep2:sep3] = np.inf
    ub[sep3:sep4] = np.inf
    ub[sep4:sep5:2] = 0
    ub[sep4+1:sep5:2] = np.pi
    lb_list = lb.tolist()
    ub_list = ub.tolist()

    ret = least_squares(getResidual_init,x0=x0, bounds=(lb_list,ub_list), 
                        args=(sampled_observed_stokes, refractive_index, sun_dir, DoLPsmax, skyModelParams, zenith), max_nfev=100)
 
    I_d_est = np.zeros((image_num))
    I_d_est[0] = 1
    I_d_est[1:image_num] = ret.x[0:sep1]
    t_d_est = ret.x[sep1]
    I_zeta_est = ret.x[sep2:sep3]
    print('I_d: '+str(I_d_est))
    print('t_d: '+str(t_d_est))
    print('I_zeta: '+str(I_zeta_est))
    print('error: '+str(ret.cost))
    I_s_est = I_d_est*t_d_est

    # estimate normal
    # initial value
    n_init = np.zeros((h,w,3))
    n_init[:,:] = normal.pt2n_x_scalar(phi_init,theta_init)
    rho_d_init_matrix = np.ones((h,w))*rho_d_init
    # boundary
    lb = np.zeros((3))
    lb[0] = 0
    lb[1] = -np.pi
    lb[2] = 0
    ub = np.zeros((3))
    ub[0] = np.inf
    ub[1] = 0
    ub[2] = np.pi
    lb_list = lb.tolist()
    ub_list = ub.tolist()
    bounds = (lb_list,ub_list)

    n_est, rho_d_est = estimateNormal(step1, pixel_range1, n_init, rho_d_init_matrix, bounds, observed_stokes, refractive_index, sun_dir, DoLPsmax, I_zeta_est, I_d_est, I_s_est, skyModelParams, zenith, mask)


    ## alternating minimization
    sep1 = image_num-1 # I_d
    sep2 = sep1+1 # t_d
    sep3 = sep2+image_num # I_zata
    # first step boundary
    lb_1 = np.zeros((sep3))
    lb_1[0:sep1] = 0
    lb_1[sep1:sep2] = t_d_min
    lb_1[sep2:sep3] =0
    ub_1 = np.zeros((sep3))
    ub_1[0:sep1] = np.inf
    ub_1[sep1:sep2] = t_d_max
    ub_1[sep2:sep3] = np.inf
    lb_list_1 = lb_1.tolist()
    ub_list_1 = ub_1.tolist()

    # second step boundary
    lb_2 = np.zeros((3))
    lb_2[0] = 0 # rho_d
    lb_2[1] = -np.pi # phi
    lb_2[2] = 0 # theta
    ub_2 = np.zeros((3))
    ub_2[0] = np.inf
    ub_2[1] = 0
    ub_2[2] = np.pi
    lb_list_2 = lb_2.tolist()
    ub_list_2 = ub_2.tolist()
    bounds2 = (lb_list_2, ub_list_2)

    error = np.zeros((h,w))
    for ite in range(0,iteration_num):
        print('iteration: '+str(ite))
        
        # initial value
        x0_1 = np.zeros((sep3))
        x0_1[0:sep1] = I_d_est[1:]
        x0_1[sep1:sep2] = t_d_est
        x0_1[sep2:sep3] = I_zeta_est
        
        #sampling some pixels to estimate uniform parameters
        pixel_num = 0
        max_pixel_num = int(np.ceil((h/pixely_interval_2+1)*(w/pixelx_interval_2+1)))
        sampled_observed_stokes = np.zeros((image_num, max_pixel_num, 4))
        sampled_n_est = np.zeros((max_pixel_num, 3))
        sampled_rho_d_est = np.zeros((max_pixel_num))
        for i in range(0,h,pixely_interval_2):
            for j in range(0,w,pixelx_interval_2):
                if mask[i,j] >= 1:
                    sampled_observed_stokes[:,pixel_num,:] = observed_stokes[:,i,j,:]
                    sampled_n_est[pixel_num] = n_est[i,j]
                    sampled_rho_d_est[pixel_num] = rho_d_est[i,j]
                    pixel_num += 1
        sampled_observed_stokes = sampled_observed_stokes[:,0:pixel_num,:]
        sampled_n_est = sampled_n_est[0:pixel_num]
        sampled_rho_d_est = sampled_rho_d_est[0:pixel_num]
        print("pixel_num : "+str(pixel_num))
        
        ret = least_squares(getResidual,x0=x0_1, bounds=(lb_list_1,ub_list_1), 
                            args=(sampled_observed_stokes, sampled_n_est, refractive_index, sun_dir, DoLPsmax, sampled_rho_d_est, skyModelParams, zenith), max_nfev=100)
        
        I_d_est[0] = 1
        I_d_est[1:image_num] = ret.x[0:sep1]
        t_d_est = ret.x[sep1]
        I_zeta_est = ret.x[sep2:sep3]
        print('I_d: '+str(I_d_est))
        print('t_d: '+str(t_d_est))
        print('I_zeta: '+str(I_zeta_est))
        print('error: '+str(ret.cost))
        I_s_est = I_d_est*t_d_est
        
        # estimate normal
        n_est, rho_d_est = estimateNormal(step2[ite], pixel_range2[ite], n_est, rho_d_est, bounds2, observed_stokes, refractive_index, sun_dir, DoLPsmax, I_zeta_est, I_d_est, I_s_est, skyModelParams, zenith, mask)

    ## show result
    print('I_d: '+str(I_d_est))
    print('t_d: '+str(t_d_est))
    print('I_zeta: '+str(I_zeta_est))

    plt.figure(figsize=(5,5))
    plt.imshow(mueller.getNormalImg(n_est[::step,::step]) * (mask>=1)[::step,::step,np.newaxis])
    plt.savefig(save_file_name + '.png')
    # plt.show()

    np.save(save_file_name, n_est)

main()