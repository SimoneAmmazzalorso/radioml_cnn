#!/usr/bin/env python
#
# Written by Simone Ammazzalorso
#
import healpy as hp
from healpy.visufunc import cartview
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.special import eval_legendre as leg
import time
import random
import os.path
import subprocess as sub
import argparse
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("N_start", type=int, help="start number of the map", default=0)
parser.add_argument("N_stop", type=int, help="end number of the map", default=0)
parser.add_argument("--tag", type=str, help="tag of the run", default='')
parser.add_argument("--path", type=str, help="general path", default='')
parser.add_argument("--beam_path", type=str, help="path to beam-maps", default='')
parser.add_argument("--out_dir", type=str, help="output directory", default='')
parser.add_argument("--N1h_low", type=float, help="lower normalization for 1-halo term", default=0.1)
parser.add_argument("--N1h_up", type=float, help="upper normalization for 1-halo term", default=10.0)
parser.add_argument("--N2h_low", type=float, help="lower normalization for 2-halo term", default=0.1)
parser.add_argument("--N2h_up", type=float, help="upper normalization for 2-halo term", default=10.0)
parser.add_argument("--alpha_low", type=float, help="lower power-law index for 2-halo term", default=-2.0)
parser.add_argument("--alpha_up", type=float, help="upper power-law index for 2-halo term", default=0.0)
parser.add_argument("--add_noise", action='store_true', help="apply noise term.")
parser.add_argument("--N_low", type=float, help="lower normalization for noise term", default=0.5)
parser.add_argument("--N_up", type=float, help="upper normalization for noise term", default=2.0)
parser.add_argument("--add_beam", action='store_true', help="apply beam function.")
parser.add_argument("--small_beam", action='store_true', help="Use only small patch of the beam")

parser.add_argument("--theta_min", type=float, help="theta min (deg)", default=0.01)
parser.add_argument("--theta_max", type=float, help="theta max (deg)", default=2.0)
parser.add_argument("--fact", type=float, help="normalization factor for the correlation function (default = 1.0)", default=1.0)
parser.add_argument("--norm_tif", action='store_true', help="apply normalization to tif files (values from 0 to 255).")

args = parser.parse_args()
tag = args.tag
path = args.path
beam_path = args.beam_path
out_dir = args.out_dir
N_start = args.N_start
N_stop = args.N_stop
N1h_low = args.N1h_low
N1h_up = args.N1h_up
N2h_low = args.N2h_low
N2h_up = args.N2h_up
alpha_low = args.alpha_low
alpha_up = args.alpha_up
add_noise = args.add_noise
N_low = args.N_low
N_up = args.N_up
add_beam = args.add_beam
small_beam = args.small_beam
theta_min = args.theta_min
theta_max = args.theta_max
fact = args.fact
norm_tif = args.norm_tif

if path is not '' and path[-1] is not '/':
    path = path+'/'

def normalization(moll_array):
    moll_array = moll_array + np.abs(np.min(moll_array))
    moll_array = moll_array/(np.max(moll_array))*255.0
    return moll_array

#power spectrum files; assumed to be normalized with l*(l+1)/(2 Pi)
in_1halo = 'Cl_radio_1.dat'
in_2halo = 'Cl_radio_2.dat'

###Geneal options
#Healpix size
NSIDE = 1024
#multipole range
l_start = 5
l_stop = 1500
#size of tif image
x_size = 2000
y_size = int(x_size/2)

plot_test = False
###############################################################
NPIX = 12*NSIDE**2
ll = np.arange(l_start,l_stop)
norm = 2.*np.pi/(ll*(ll+1))

th_list = np.logspace(np.log10(theta_min), np.log10(theta_max), num=10)
print('Theta values (deg):')
print(th_list)
cl_list = [np.insert(ll,0,range(l_start))]
CCF_list = [th_list]
A = (2.*ll+1.)/(4.*np.pi)
pl = []
for i in range(len(th_list)):
    pl.append(leg(ll,np.cos(np.radians(th_list[i]))))

print('factor for the correlation function:',fact)

if tag is not '':
    text = '#TAG: '+tag+'\n'+'#Maps number: '+str(N_start)+' - '+str(N_stop)+'\n'+'#N, N1h, N2h, alpha'+'\n'
    text_cl = '#TAG: '+tag+'\n'+'#Maps number: '+str(N_start)+' - '+str(N_stop)+'\n'
    text_CCF = '#TAG: '+tag+'\n'+'#Maps number: '+str(N_start)+' - '+str(N_stop)+'\n'
    tag = tag+'_'
else:
    text = '#Maps number: '+str(N_start)+' - '+str(N_stop)+'\n'+'#N, N1h, N2h, alpha'+'\n'
    text_cl = '#Maps number: '+str(N_start)+' - '+str(N_stop)+'\n'
    text_CCF = '#Maps number: '+str(N_start)+' - '+str(N_stop)+'\n'

if add_noise:
    text = text.replace('\n',', Cl_100, N_lev, N_val\n')

if out_dir is not '' and out_dir[-1] is not '/':
    out_dir = out_dir+'/'

def read_PS(path,in_1halo,in_2halo,ll,norm):
    cl1 = np.genfromtxt(path+in_1halo)
    cl1_interp = interp1d(np.log(cl1[:,0]),np.log(cl1[:,1]))
    cl1_fine = np.exp(cl1_interp(np.log(ll)))*norm
    cl2 = np.genfromtxt(path+in_2halo)
    cl2_interp = interp1d(np.log(cl2[:,0]),np.log(cl2[:,1]))
    cl2_fine = np.exp(cl2_interp(np.log(ll)))*norm
    cl_tot = cl1_fine+cl2_fine
    cl_tot = np.insert(cl_tot,0,np.zeros(l_start))
    return cl_tot,np.insert(cl1_fine,0,np.zeros(l_start)),np.insert(cl2_fine,0,np.zeros(l_start))
cl_tot,cl_1h,cl_2h = read_PS(path,in_1halo,in_2halo,ll,norm)

t_start = time.time()
out_text = out_dir+'msim_'+tag+str(N_start)+'_'+str(N_stop)+'.txt'

for i in range(N_start,N_stop+1):
    cl_trans = []

    out_text_temp = out_dir+'msim_'+tag+'back.txt'
    f = open(out_text_temp,'w')
    print('*** Map number:',i)

    print('Creating Power Spectrum...')
    N1h = np.round(10**random.uniform(np.log10(N1h_low),np.log10(N1h_up)),1)
    N2h = np.round(10**random.uniform(np.log10(N2h_low),np.log10(N2h_up)),1)
    alpha = np.round(random.uniform(alpha_low,alpha_up),1)
    text = text+'{:}'.format(i)+'  '+'{:.1e}'.format(N1h)+'  '+'{:.1e}'.format(N2h)+'  '+'{:.1e}'.format(alpha)
    print('normalization 1-halo:',N1h)
    print('normalization 2-halo:',N2h)
    print('power-law index:',alpha)
    cl_1h_temp = cl_1h*N1h
    cl_2h_temp = cl_2h*N2h
    #cl_1h_temp[l_start:] = cl_1h_temp[l_start:]#*(ll/100.)**alpha
    cl_2h_temp[l_start:] = cl_2h_temp[l_start:]*(ll/100.)**alpha
    cl_temp = cl_1h_temp+cl_2h_temp
    cl_list.append(cl_temp)
    if plot_test:
        plt.clf()
        plt.plot(range(len(cl_1h)),cl_1h,'--',linewidth=2,color='orange')
        plt.plot(range(len(cl_1h)),cl_2h,'.-',linewidth=2,color='orange')
        plt.plot(range(len(cl_1h)),cl_1h_temp,'--',linewidth=2,color='blue')
        plt.plot(range(len(cl_1h)),cl_2h_temp,'.-',linewidth=2,color='blue')
        plt.xlim(xmin=40,xmax=1500)
        plt.ylim(ymin=1.e-9,ymax=1.e-5)
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig('test_PS.png')
        plt.clf()

    print('Writing Power Spectrum...')
    out_cl = out_dir+'Cl_'+tag+str(N_start)+'_'+str(N_stop)+'_label.dat'
    np.savetxt(out_cl, np.transpose(cl_list), header=text_cl, fmt='%1.4e')

    print('Converting Power Spectrum to CCF...')
    for j in range(len(th_list)):
        cl_trans.append(np.sum(cl_temp[l_start:]*pl[j]*A))
    CCF_list.append(np.array(cl_trans)*fact)

    print('Writing CCF...')
    out_CCF = out_dir+'CCF_'+tag+str(N_start)+'_'+str(N_stop)+'_label.dat'
    np.savetxt(out_CCF, np.transpose(CCF_list), header=text_CCF, fmt='%1.4e')
    if plot_test:
        plt.plot(th_list,CCF_list[-1],'o-',linewidth=1,color='steelblue')
        plt.xlim(xmin=theta_min,xmax=theta_max)
        #plt.ylim(ymin=,ymax=)
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig('test_CCF.png')
        plt.clf()

    print('Creating map from Power Spectrum...')
    out_name = out_dir+'msim_'+tag+str(i).zfill(4)+'.fits'
    msim = hp.synfast(cl_temp,NSIDE)

    print('Saving map...')
    # hp.write_map(out_name,msim,coord='G',fits_IDL=False,overwrite=True)

    #moll_array = hp.cartview(msim, title=None, xsize=x_size, ysize=y_size, return_projected_map=True)
    #plt.savefig('/home/simone/RadioML/data/test/map_clean.png')
    if add_noise:
        print('Creating noise from random level...')
        N = np.round(10**random.uniform(np.log10(N_low),np.log10(N_up)),1)
        out_noise = 'noise/noise_'+tag+str(i).zfill(4)+'_N_'+str(N)+'.fits'
        print('Noise level:',N)
        out_name = out_name+'_N_'+str(N)
        print('Creating noise map...')
        NN = cl_tot[100]*N
        print('N:',NN)
        mnoise = hp.synfast([NN]*len(ll),NSIDE)
        text = text+'  '+'{:.3e}'.format(cl_tot[100])+'  '+str(N)+'  '+'{:.3e}'.format(NN)

        print('Combining maps...')
        msim = msim + mnoise
        print('Saving raw noise map...')
        #print(path+out_noise)
        # hp.write_map(out_dir+out_noise,mnoise,coord='G',fits_IDL=False)

    # if add_beam:
    #     b=np.round(10**random.uniform(np.log10(b_low),np.log10(b_up)),1)
    #     print('Beam level:',b)
    #     out_name = out_name+'_b_'+str(b)
    #     print('Convolving with beam...')
    #     pix_area = 4*np.pi/NPIX
    #     ang = np.sqrt(pix_area)*b
    #     print('sigma:',np.degrees(ang))
    #     msim = hp.sphtfunc.smoothing(msim,sigma=ang)
    #     text= text+'  '+str(b)+'  '+str(np.round(np.degrees(ang),5))

    print('Converting map to tif format...')
    moll_array = hp.cartview(msim, title=None, xsize=x_size, ysize=y_size, return_projected_map=True)

    if add_beam:
        # get all beam files from beam_path with '*.fits'
        import glob
        beam_ids = glob.glob(beam_path + '*.fits')

        # loading the fits-files and stacking them together
        from astropy.io import fits
        from astropy.utils.data import get_pkg_data_filename

        for k, beam_id in enumerate(beam_ids):
            with fits.open(beam_id) as hdul:
                beam_data = hdul[0].data

            beam_data = beam_data[0][0][:][:]
            if small_beam:
                # this takes only a little patch of the beam to have the code run quicker!
                num = 2288
                beam_data = beam_data[num:-num, num:-num]
            beam_data = beam_data[np.newaxis,:,:]
            if k==0:
                beams = beam_data
            else:
                beams = np.append(beams,beam_data, axis=0)
        # beams shape = (#beam, pixel-x, pixel-y)


        # convolve moll_array with each beam
        print('Convolving with beams...')
        for k in range(beams.shape[0]):
            from scipy import signal
            moll_array = signal.convolve2d(moll_array, beams[k], boundary='symm', mode='same')
        # save it with tag for inclination in it
            if norm_tif:
                print('Applying normalization to map...')
                moll_array = normalization(moll_array)
            #moll_array = np.array(moll_array, dtype=np.uint8)
            moll_image = Image.fromarray(moll_array)

            print('Saving tif map...')
            declination = beam_ids[k][len(beam_path)+47:-len('.fits')]
            out_tif = out_dir+'msim_'+tag+str(i).zfill(4)+'_'+str(declination).zfill(2)+'_data.tif'
            moll_image.save(out_tif)

    else: # If no beams are added, the map is saved once, as before
        #plt.savefig('/home/simone/RadioML/data/test/map_noise.png')
        if norm_tif:
            print('Applying normalization to map...')
            moll_array = normalization(moll_array)
        #moll_array = np.array(moll_array, dtype=np.uint8)
        moll_image = Image.fromarray(moll_array)

        print('Saving tif map...')
        out_tif = out_dir+'msim_'+tag+str(i).zfill(4)+'_data.tif'
        moll_image.save(out_tif)

    text = text+'\n'
    print('Writing parameters file...')
    f.write(text)
    f.close()
    sub.call(['cp',out_text_temp,out_text],) #shell=[bool])

    #print 'Partial time :',time.time()-t_start,'s\n'
    print('\n')

t_stop=time.time()
print('Elapsed time for create maps:',t_stop-t_start,'s')

print('Done.')
