import healpy as hp
import numpy as np
from astropy.io import fits as pf
from matplotlib import pyplot as plt
from scipy.ndimage.interpolation import map_coordinates
from scipy.interpolate import interp1d,interp2d
from scipy.special import eval_legendre as leg
import time
from PIL import Image
from multiprocessing import Pool

#Input and output path:
#path_in = '/home/simone/RadioML/data/mapsim_tif/'
path_in = '/home/simone/RadioML/data/mapsim_PS/'
path_out = path_in
#Number of maps to be converted. NOTE: this assumes the same structure for the name of each input map (chage it if you like)
N_map = 10000
#Strartig map
N_start = 333
#NSIDE used for the conversion. NOTE: the higher the less artifacts should be present, but you need RAM and time...
NSIDE = 1024
#NSIDE of the output map
NSIDE_out = 1024
#Pixel size of the input map
N_x = 2000
N_y = N_x/2
#Number of processors used for the conversion
n_proc = 8
#normalizing map with values from 0 to 255?
norm = False
#2-point correlation file (first column must be theta values, other columns are the correlation function for different maps)
in_CCF = '/home/simone/RadioML/data/CCF_CNN_0_10000_label.dat'
#tag of the files
tag = 'CNN'
#theta_min and theta_max (deg)
theta_min = 0.01
theta_max = 2.0
#number of theta bins
n_theta = 10
#l_start and l_stop
l_start = 5
l_stop = 1500
#save back comverted healpix map
save_map = False
#show plots
show_plot = False

################################################################################
if tag is not '':
    tag = tag + '_'

CCF_file=np.genfromtxt(in_CCF,)[:,1:]
th_list = np.logspace(np.log10(theta_min), np.log10(theta_max), num=n_theta)
print('Theta values (deg):')
print(th_list)

ll=np.arange(l_start,l_stop+1)

binsz_x = 360./N_x  #bin size in degrees
binsz_y = 180./N_y
NPIX = hp.pixelfunc.nside2npix(NSIDE)   #number of pixel of the conversion map
iii = range(NPIX)
B, L = np.array(hp.pixelfunc.pix2ang(NSIDE,iii))  #gives b,l in radians (colatitude and longitude)
#print np.degrees(B),np.degrees(L)

pl_list = [] #Legendre polynomials
for k in range(len(th_list)):
    pl_list.append(leg(ll,np.cos(np.radians(th_list[k]))))

def normalization(moll_array):
    #moll_array[np.isinf(moll_array)] = 0.0
    moll_array = moll_array + np.abs(np.min(moll_array))
    moll_array = moll_array/(np.max(moll_array))*255.0
    return moll_array

#coordinates of the cartesian projection
l_interp=np.arange(N_x)*binsz_x
b_interp=np.arange(N_y)*binsz_y

#coordinates conversion
L,B=np.array((np.degrees(L)+180.)%360),np.array(180.-np.degrees(B))

#pixel window function for correction to the Power Spectrum
#wpix=hp.sphtfunc.pixwin(NSIDE)

n_check=NPIX/10
for j in range(N_start,N_map):
    print('map number:',j)
    iii = np.zeros(NPIX) #initialize array
    time1 = time.time()
    print('Reading image...')
    print(path_in+'msim_'+tag+str(j).zfill(4)+'_data.tif')
    m = Image.open(path_in+'msim_'+tag+str(j).zfill(4)+'_data.tif')
    m = np.array(m) #conversion to numpy array
    interp_map = interp2d(l_interp,b_interp,np.fliplr(m))

    #interpolating each healpix pixel
    print('Converting map...')
    if n_proc>1:
        def ff(x):
            return interp_map(L[x],B[x])
        p = Pool(n_proc)
        iii = p.map(ff, np.arange(NPIX, dtype=int))
        iii = np.array(iii)
        iii = iii[:,0]
    else:
        for i in range(0,len(iii)):
            if(i%n_check == 0):
                print(str(i/n_check*10).zfill(2),'%'+' completed')
                iii[i] = interp_map(L[i],B[i])

    time2=time.time()
    print('Time for building this healpix projection: ',time2-time1,' s')
    if NSIDE_out != NSIDE:
        iii = hp.ud_grade(iii, nside_out=NSIDE_out)
    hp.mollview(iii)
    plt.savefig('test_conversion_'+tag+str(j).zfill(4)+'.png')
    if show_plot:
        plt.show()
    plt.clf()

    print('Computing Power Spectrum from converted map...')
    cl = hp.anafast(iii, lmax=l_stop)
    ll = np.arange(len(cl))

    print('Reading real map...')
    print(path_in+'msim_'+tag+str(j).zfill(4)+'.fits')
    test_map = hp.read_map(path_in+'msim_'+tag+str(j).zfill(4)+'.fits')
    if norm:
        test_map = normalization(test_map)
    hp.mollview(test_map)
    plt.savefig('test_real_map_'+tag+str(j).zfill(4)+'.png')
    if show_plot:
        plt.show()
    plt.clf()

    print('Computing Power Spectrum from real map...')
    cl_real = hp.anafast(test_map, lmax=1500)
    plt.plot(ll, cl, label='back-tranformed')
    plt.plot(ll, cl_real, label='normalized-true')

    plt.xlim(xmin=10, xmax=1000)
    plt.ylim(ymin=1e-3, ymax=10)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='upper right')
    plt.savefig('test_PS_comp_'+tag+str(j).zfill(4)+'.png')
    if show_plot:
        plt.show()
    plt.clf()

    cl_trans = []
    l_start = 5
    ll = np.arange(l_start,len(cl_real))
    A = (2.*ll+1.)/(4.*np.pi)
    CCF_real = CCF_file[:,j]

    print('Converting Power Spectrum to CCF...')
    for k in range(len(th_list)):
        cl_trans.append(np.sum(cl[l_start:]*pl_list[k]*A))

    plt.plot(th_list, cl_trans, label='back-tranformed')
    plt.plot(th_list, CCF_real/CCF_real[5]*cl_trans[5], label='normalized-true')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='upper right')
    plt.savefig('test_CCF_comp_'+tag+str(j).zfill(4)+'.png')
    if show_plot:
        plt.show()
    plt.clf()

    if save_map:
        hp.write_map(path_out+'msim_'+tag+str(j).zfill(4)+'_backconv_hpxproj.fits',iii,fits_IDL=False,coord='G')
    exit()
