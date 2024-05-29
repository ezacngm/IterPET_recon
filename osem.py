from skimage.io import imread
from skimage import data_dir
from skimage.transform import radon, rescale, iradon
from skimage.draw import disk
# from skimage.draw import  circle
from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt



def osem(sinogram,nsub=1,niter=1,sfwhm=1.333,obj=True):


    shape = sinogram.shape
    assert shape[0] == shape[1],"osem sino must have same size"
    theta = np.linspace(0., 180., shape[0], endpoint=False)
    recon = np.zeros(shape)
    rr, cc = disk((shape[0] / 2, shape[1] / 2), shape[0] / 2 - 1)
    recon[rr, cc] = 1

    # normalization matrix
    nview = len(theta)
    norm = np.ones(shape)
    wgts = []
    for sub in range(nsub):
        views = range(sub, nview, nsub)
        wgt = iradon(norm[:, views], theta=theta[views], filter_name=None, circle=True)
        wgts.append(wgt)

    # iteration
    objs = []
    for iter in range(niter):
        # print('iter', iter)
        order = np.random.permutation(range(nsub))
        for sub in order:
            views = range(sub, nview, nsub)
            # print("recon",recon.shape)
            fp = radon(recon, theta=theta[views], circle=True)
            # print("fp",fp.shape)
            sinogram_test = sinogram[:, views]
            # print(sinogram_test.shape)
            ratio = sinogram[:, views] / (fp + 1e-6)
            bp = iradon(ratio, theta=theta[views], filter_name=None, circle=True)
            recon *= bp / (wgts[sub] + 1e-6)

            if obj:
                fp = radon(recon, theta=theta, circle=True)
                ndx = np.where(fp > 0)
                val = -(sinogram[ndx] * np.log(fp[ndx]) - fp[ndx]).sum() / 1e6
#                 print("val", val)
                objs.append(val)

    fbp = iradon(sinogram, theta=theta, circle=True)
    if sfwhm > 0:
        fbp = gaussian_filter(fbp, sfwhm / 2.355)
        recon = gaussian_filter(recon, sfwhm / 2.355)
    # plt.figure()
    # plt.suptitle('Convergence')
    # plt.plot(objs, '.-')
    # plt.xlabel('sub iteration')
    # plt.ylabel('negative log likelihood [1e+06]')
    # plt.savefig('objective.png')
    #
    # plt.figure()
    # plt.imshow(recon)
    # plt.axis('off')

    return recon,fbp,fp
