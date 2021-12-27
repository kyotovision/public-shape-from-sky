import numpy as np
import scipy as sp
import scipy.optimize
import copy
import numba
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import bm3d

def aolp2rgb(aolp):
    """
    Converts 2D AoLP map to an RGB 8bit image for visualization. Angle of 0, 30, 60, 90, 120, 150, and 180 degrees are mapped to red, yellow, green, cyan, blue, magenta, and red respectively.
    """
    cmap = plt.cm.hsv
    #norm = plt.Normalize(vmin=0, vmax=np.pi)
    #img = cmap(norm((aolp+np.pi*2)%np.pi))
    norm = plt.Normalize(vmin=-np.pi/2, vmax=np.pi/2)
    #a = 2*aolp
    #a[a>np.pi/2] -= np.pi
    #a[a<-np.pi/2] += np.pi
    img = cmap(norm(aolp))
    img = np.clip(img*255, 0, 255).astype(np.uint8)
    return img


class PImage:
    """
    Polarimetric image
    """

    def __init__(self):
        self.BGR_dc = None
        self.AoLP = None
        self.DoLP = None
        self.mask = None

    def fancy_bgr(self, invalid=(1,0,1)):
        img = self.BGR_dc * (2.87, 1.0, 1.35)
        img[self.mask<0] = invalid
        return np.clip(img, 0, 1)


    def binning(self):
        # AoLP should be doubled before binning since it defines a plane, i.e., +pi/2 == -pi/2
        c0 = np.cos(self.AoLP * 2) * self.DoLP
        s0 = np.sin(self.AoLP * 2) * self.DoLP
        c1 = c0[::2,::2] + c0[0::2,1::2] + c0[1::2,::2] + c0[1::2,1::2]
        s1 = s0[::2,::2] + s0[0::2,1::2] + s0[1::2,::2] + s0[1::2,1::2]

        p = PImage()
        p.AoLP = np.arctan2(s1, c1) / 2 # halve it again
        p.DoLP = np.sqrt(c1**2 + s1**2) / 4
        p.BGR_dc = cv2.resize(self.BGR_dc, (self.BGR_dc.shape[1]//2, self.BGR_dc.shape[0]//2), interpolation=cv2.INTER_AREA)
        p.mask = self.mask[::2,::2] * self.mask[::2,1::2] * self.mask[1::2,::2] * self.mask[1::2,1::2]

        np.testing.assert_array_equal(p.AoLP.shape[:2], p.mask.shape[:2], f'{p.AoLP.shape}, {p.mask.shape}')
        np.testing.assert_array_equal(p.DoLP.shape[:2], p.mask.shape[:2], f'{p.DoLP.shape}, {p.mask.shape}')
        np.testing.assert_array_equal(p.BGR_dc.shape[:2], p.mask.shape[:2], f'{p.BGR_dc.shape}, {p.mask.shape}')

        return p

    def smooth(self, r=5):
        # AoLP should be doubled before smoothing since it defines the plane, i.e., +pi/2 == -pi/2
        ax = np.cos(self.AoLP * 2) * self.DoLP
        ay = np.sin(self.AoLP * 2) * self.DoLP

        #ax = cv2.GaussianBlur(ax, (r, r), 0)
        #ay = cv2.GaussianBlur(ay, (r, r), 0)

        #ax = cv2.bilateralFilter(ax, r, np.pi/2, r)
        #ay = cv2.bilateralFilter(ay, r, np.pi/2, r)

        ax = bm3d.bm3d(ax, sigma_psd=r/255.0, stage_arg=bm3d.BM3DStages.ALL_STAGES)
        ay = bm3d.bm3d(ay, sigma_psd=r/255.0, stage_arg=bm3d.BM3DStages.ALL_STAGES)

        #ax = cv2.ximgproc.fastGlobalSmootherFilter(self.BGR_dc.astype(np.uint8), ax, 15, 0.25)
        #ay = cv2.ximgproc.fastGlobalSmootherFilter(self.BGR_dc.astype(np.uint8), ay, 15, 0.25)

        #ax = cv2.ximgproc.guidedFilter(self.BGR_dc.astype(np.float32), ax, r, 0.01)
        #ay = cv2.ximgproc.guidedFilter(self.BGR_dc.astype(np.float32), ay, r, 0.01)

        p = copy.copy(self)
        p.AoLP = np.arctan2(ay, ax) / 2 # halve it again
        p.DoLP = np.clip(np.sqrt(ax**2 + ay**2), 0, 1)

        #self.DoLP = cv2.ximgproc.guidedFilter(self.BGR_dc.astype(np.float32), self.DoLP, 2, 0.01)

        #p.update_conf()
        return p

    def visualize_aolp(self):
        return aolp2rgb(self.AoLP)

    # def fancy_Imin(self):
    #     tmp = self.BGR_dc * (2.87, 1.0, 1.35)
    #     return np.clip(tmp, 0, 255).astype(np.uint8)

class PBayer(PImage):
    """
    Polarimetric bayer image. Each pixel has a Stokes vector.
    """
    def __init__(self, raw, demosaic_method=cv2.COLOR_BAYER_BG2BGR_EA):
        #super().__init__()
        assert raw.dtype == np.float32
#        assert np.max(raw) <= 1
#        assert np.min(raw) >= 0
        self.raw = raw.copy()
        self.mask = (raw>=0)
        self.mask = self.mask[::2,::2] * self.mask[::2,1::2] * self.mask[1::2,::2] * self.mask[1::2,1::2] # valid if all 2x2 pixels are valid

        p090 = self.raw[::2, ::2]   # | 90
        p045 = self.raw[::2, 1::2]  # / 45
        p135 = self.raw[1::2, ::2]  # \ 135
        p000 = self.raw[1::2, 1::2] # - 0

        s0 = (p000 + p090 + p045 + p135)/2
        s1 = p000 - p090
        s2 = p045 - p135

        self.stokes = np.zeros((*s0.shape, 4))
        self.stokes[...,0] = s0
        self.stokes[...,1] = s1
        self.stokes[...,2] = s2

        # Stokes vector H x W x 3
        self.svec = np.dstack([s0, s1, s2])
        # Filter angles 4
        self.filt = np.array([0, np.pi/4, np.pi/2, np.pi/4*3])
        # Polarized images H x W x 4
        self.fimg = np.dstack([p000, p045, p090, p135])

        assert self.svec.shape == (self.raw.shape[0]//2, self.raw.shape[1]//2, 3)
        assert self.fimg.shape == (self.raw.shape[0]//2, self.raw.shape[1]//2, 4)

        # H x W
        self.I_dc = s0 / 2
        self.DoLP = np.clip(np.sqrt(s1**2 + s2**2) / (s0 + 1e-9), 0, 1)
        # np.arctan2() returns [-pi:pi], not [-pi/2:pi/2] (i.e., not atan() but atan2())
        self.AoLP = np.arctan2(s2, s1)
        #self.AoLP[self.AoLP < -np.pi/2] += np.pi
        #self.AoLP[self.AoLP >  np.pi/2] -= np.pi
        #assert np.all(self.AoLP >= -np.pi/2)
        #assert np.all(self.AoLP <= np.pi/2)
        self.AoLP /= 2

        # H x W
        self.update_conf()

        self.I_max = self.I_dc + self.DoLP * self.I_dc
        self.I_min = self.I_dc - self.DoLP * self.I_dc
        #assert np.max(self.I_min) <= 1
        #assert np.min(self.I_min) >= 0

        self.BGR_dc = cv2.cvtColor((self.I_dc * 65535).astype(np.uint16), demosaic_method) / 65535.0  # fixme
        assert self.raw.shape[0] == self.BGR_dc.shape[0] * 2
        assert self.raw.shape[1] == self.BGR_dc.shape[1] * 2

    def update_conf(self):
        self.conf = []
        for i in range(4):
            self.conf.append( self.I_dc + self.DoLP * self.I_dc * np.cos(2*self.AoLP - 2 * self.filt[i]) - self.fimg[:,:,i] )
        self.conf = np.linalg.norm(np.dstack(self.conf), axis=2)
        assert self.conf.shape == (self.fimg.shape[0], self.fimg.shape[1]), self.conf.shape

    # def fit(self):
    #     phase = np.array([0, np.pi/4, np.pi/2, np.pi*3/4])

    #     @numba.jit
    #     def func(x, *args):
    #         I0, rho, phi = x
    #         return I0 + rho * I0 * np.cos(2*phase - 2*phi) - args[0]

    #     @numba.jit
    #     def dfun(x, *args):
    #         I0, rho, phi = x
    #         pp = 2*(phase - phi)
    #         return np.vstack((1 + rho * np.cos(pp), I0*np.cos(pp), 2 * rho * I0 * np.sin(pp))).T

    #     x0 = np.dstack((self.I_dc, self.DoLP, self.AoLP))
    #     for i0 in tqdm(range(self.I_dc.shape[0])):
    #         for i1 in range(self.I_dc.shape[1]):
    #             x = sp.optimize.leastsq(func, x0[i0, i1], self.fimg[i0, i1])
    #             self.I_dc[i0, i1], self.DoLP[i0, i1], self.AoLP[i0, i1] = x[0]



    # def fancy_Imin(self):
    #     tmp = self.BGR_dc * (2.87, 1.0, 1.35)
    #     return np.clip(tmp, 0, 255).astype(np.uint8)


# class PColor:
#     """ Polarimetric color image """

#     def __init__(self, pbayer):
#         assert isinstance(pbayer, PBayer)
#         self.pbayer = copy.deepcopy(pbayer)
#         # https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html
#         R = self.pbayer.fimg[::2, ::2, :]
#         G1 = self.pbayer.fimg[::2, 1::2, :]
#         G2 = self.pbayer.fimg[1::2, ::2, :]
#         B = self.pbayer.fimg[1::2, 1::2, :]
#         self.bggr = np.stack((B, G1, G2, R), axis=2)  # HxWxCxP
#         assert self.bggr.shape == (self.pbayer.raw.shape[0]//4, self.pbayer.raw.shape[1]//4, 4, 4)

#         self.height = self.pbayer.raw.shape[0]//4
#         self.width = self.pbayer.raw.shape[1]//4

#         phase = np.array([0, np.pi/4, np.pi/2, np.pi*3/4])

#         @numba.jit
#         def func(x, *args):
#             R0, G0, B0, rho, phi = x
#             c = np.cos(2*phase - 2*phi)
#             return np.concatenate((R0 + rho * R0 * c, G0 + rho * G0 * c, G0 + rho * G0 * c, B0 + rho * B0 * c)) - args[0]

#         @numba.jit
#         def dfun(x, *args):
#             B0, G0, R0, rho, phi = x
#             pp = 2*(phase - phi)
#             return np.vstack((1 + rho * np.cos(pp), I0*np.cos(pp), 2 * rho * I0 * np.sin(pp))).T

#         self.BGR_dc = np.zeros((self.height, self.width, 3))
#         self.DoLP = np.zeros((self.height, self.width))
#         self.AoLP = np.zeros((self.height, self.width))
#         self.conf = np.zeros((self.height, self.width))

#         print(self.bggr[0,0,:,:])
#         print(self.bggr[0,0,:,:].flatten())
#         for i0 in tqdm(range(self.height)):
#             for i1 in range(self.width):
#                 x = sp.optimize.leastsq(func, (self.pbayer.I_dc[i0*2, i1*2], self.pbayer.I_dc[i0*2+1, i1*2], self.pbayer.I_dc[i0*2+1, i1*2+1], self.pbayer.DoLP[i0*2, i1*2], self.pbayer.AoLP[i0*2, i1*2]), self.bggr[i0, i1, :, :].flatten(), full_output=True)
#                 self.BGR_dc[i0, i1] = x[0][:3]
#                 self.DoLP[i0, i1] = x[0][3]
#                 self.AoLP[i0, i1] = x[0][4]
#                 self.conf[i0, i1] = np.linalg.norm(x[2]['fvec'])

"""
def debayer_subsample(img):
    bgr = np.zeros((img.shape[0]//2, img.shape[1]//2, 3), img.dtype)
    bgr[:,:,0] = img[1::2,1::2] * 2.87
    bgr[:,:,1] = (img[1::2,0::2] + img[0::2,1::2]) / 2
    bgr[:,:,2] = img[0::2,0::2] * 1.35
    return np.clip(bgr, 0, 255).astype(np.uint8)
"""

