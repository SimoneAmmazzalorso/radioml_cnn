#!/usr/bin/env python

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
parser.add_argument("--tag", type=str, help="tag of the run", default='')
parser.add_argument("--out_dir", type=str, help="output directory", default='')
parser.add_argument("N_start", type=int, help="start number of the map", default=0)
parser.add_argument("N_stop", type=int, help="end number of the map", default=0)
parser.add_argument("--N1h_low", type=float, help="lower normalization for 1-halo term", default=0.1)
parser.add_argument("--N1h_up", type=float, help="upper normalization for 1-halo term", default=10.0)
parser.add_argument("--N2h_low", type=float, help="lower normalization for 2-halo term", default=0.1)
parser.add_argument("--N2h_up", type=float, help="upper normalization for 2-halo term", default=10.0)
parser.add_argument("--alpha_low", type=float, help="lower power-law index for 2-halo term", default=-1.0)
parser.add_argument("--alpha_up", type=float, help="upper power-law index for 2-halo term", default=1.0)
parser.add_argument("--theta_min", type=float, help="theta min (deg)", default=0.01)
parser.add_argument("--theta_max", type=float, help="theta max (deg)", default=2.0)

args = parser.parse_args()
tag = args.tag
out_dir = args.out_dir
N_start = args.N_start
N_stop = args.N_stop
N1h_low = args.N1h_low
N1h_up = args.N1h_up
N2h_low = args.N2h_low
N2h_up = args.N2h_up
alpha_low = args.alpha_low
alpha_up = args.alpha_up
theta_min = args.theta_min
theta_max = args.theta_max

def normalization(moll_array):
    #moll_array[np.isinf(moll_array)] = 0.0
    moll_array = moll_array + np.abs(np.min(moll_array))
    moll_array = moll_array/(np.max(moll_array))*255.0
    return moll_array

#path
path='/home/simone/RadioML/data/'
#path = '/archive/home/sammazza/radioML/data/'
#power spectrum files; assumed to be normalized with l*(l+1)/(2 Pi)
in_1halo = 'Cl_radio_2.dat'
in_2halo = 'Cl_radio_1.dat'

###Geneal options
#Number of maps to be produced
#N_start=0
#N_stop=2
#map tag in order to distinguish them from other runs, otherwise set None
#tag='test'
#Healpix size
NSIDE = 1024
#multipole range
l_start = 5
l_stop = 1500
x_size = 2000
y_size = x_size/2

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

if tag is not '':
    text = '#TAG: '+tag+'\n'+'#Maps number: '+str(N_start)+' - '+str(N_stop)+'\n'+'#N1h, N2h, alpha'+'\n'
    text_cl = '#TAG: '+tag+'\n'+'#Maps number: '+str(N_start)+' - '+str(N_stop)+'\n'
    text_CCF = '#TAG: '+tag+'\n'+'#Maps number: '+str(N_start)+' - '+str(N_stop)+'\n'
    tag = tag+'_'
else:
    text = '#Maps number: '+str(N_start)+' - '+str(N_stop)+'\n'+'#N1h, N2h, alpha'+'\n'
    text_cl = '#Maps number: '+str(N_start)+' - '+str(N_stop)+'\n'
    text_CCF = '#Maps number: '+str(N_start)+' - '+str(N_stop)+'\n'

if out_dir is not '' and out_dir[-1] is not '/':
    out_dir = out_dir+'/'

def read_PS(path,in_1halo,in_2halo,ll,norm):
    cl1 = np.genfromtxt(path+in_1halo)
    '''
    print('l file:')
    print(cl1[:,0])
    print('l fine:')
    print(ll)
    print('norm:')
    print(norm)
    '''
    cl1_interp = interp1d(np.log(cl1[:,0]),np.log(cl1[:,1]))
    cl1_fine = np.exp(cl1_interp(np.log(ll)))*norm
    cl2 = np.genfromtxt(path+in_2halo)
    cl2_interp = interp1d(np.log(cl2[:,0]),np.log(cl2[:,1]))
    cl2_fine = np.exp(cl2_interp(np.log(ll)))*norm
    cl_tot = cl1_fine+cl2_fine
    cl_tot = np.insert(cl_tot,0,np.zeros(l_start))
    #check Cl
    #cl=1.e-11*(ll)**(-2.)+1.e-13
    #plt.plot(ll,cl)
    '''
    #TEST PS reading and interpolation
    n_1=2.*np.pi/(cl1[:,0]*(cl1[:,0]+1))
    n_2=2.*np.pi/(cl2[:,0]*(cl2[:,0]+1))
    plt.plot(cl1[:,0],cl1[:,1]*n_1)
    plt.plot(cl2[:,0],cl2[:,1]*n_2)
    plt.plot(ll,cl1_fine)
    plt.plot(ll,cl2_fine)
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(path+'test.png')
    plt.show()
    print('here')
    exit()
    '''
    return cl_tot,np.insert(cl1_fine,0,np.zeros(l_start)),np.insert(cl2_fine,0,np.zeros(l_start))
cl_tot,cl_1h,cl_2h = read_PS(path,in_1halo,in_2halo,ll,norm)

t_start = time.time()
out_text = out_dir+'msim_'+tag+str(N_start)+'_'+str(N_stop)+'.txt'

for i in range(N_start,N_stop):
    cl_trans = []

    out_text_temp = out_dir+'msim_'+tag+'back.txt'
    f = open(path+out_text_temp,'w')

    print('*** Map number:',i)

    print('Creating Power Spectrum...')
    N1h = np.round(10**random.uniform(np.log10(N1h_low),np.log10(N1h_up)),1)
    N2h = np.round(10**random.uniform(np.log10(N2h_low),np.log10(N2h_up)),1)
    alpha = np.round(random.uniform(alpha_low,alpha_up),1)
    text = text+'{:.1e}'.format(N1h)+', '+'{:.1e}'.format(N2h)+', '+'{:.1e}'.format(alpha)+'\n'
    print('normalization 1-halo:',N1h)
    print('normalization 2-halo:',N2h)
    print('power-law index:',alpha)
    cl_1h_temp = cl_1h*N1h
    cl_2h_temp = cl_2h*N2h
    cl_2h_temp[l_start:] = cl_2h_temp[l_start:]*(ll/100.)**alpha
    cl_temp = cl_1h_temp+cl_2h_temp
    cl_list.append(cl_temp)

    '''
    #TEST PS modification
    plt.plot(np.arange(len(cl_1h_temp)),cl_1h_temp,linestyle='dotted',color='orange')
    plt.plot(np.arange(len(cl_2h_temp)),cl_2h_temp,linestyle='dashed',color='orange')
    plt.plot(np.arange(len(cl_temp)),cl_temp,color='orange')
    plt.plot(np.arange(len(cl_1h)),cl_1h,linestyle='dotted',color='blue')
    plt.plot(np.arange(len(cl_2h)),cl_2h,linestyle='dashed',color='blue')
    plt.plot(np.arange(len(cl_1h)),cl_1h+cl_2h,color='blue')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(xmin=l_start)
    #plt.xlim(xmin=10,xmax=1400)
    #plt.ylim(ymin=1.e-10,ymax=5.e-6)
    #plt.show()
    plt.savefig(path+out_dir+'test.png')
    exit()
    '''

    print('Writing Power Spectrum...')
    out_cl = out_dir+'Cl_'+tag+str(N_start)+'_'+str(N_stop)+'_label.dat'
    np.savetxt(path+out_cl, np.transpose(cl_list), header=text_cl, fmt='%1.4e')

    print('Converting Power Spectrum to CCF...')
    for j in range(len(th_list)):
        cl_trans.append(np.sum(cl_temp[l_start:]*pl[j]*A))

    '''
    #TEST transform to CCF
    plt.plot(th_list,cl_trans)
    plt.xscale('log')
    plt.yscale('log')
    #plt.show()
    plt.savefig(path+out_dir+'test.png')
    exit()
    '''

    print('Writing CCF...')
    CCF_list.append(cl_trans)
    out_CCF = out_dir+'CCF_'+tag+str(N_start)+'_'+str(N_stop)+'_label.dat'
    np.savetxt(path+out_CCF, np.transpose(CCF_list), header=text_CCF, fmt='%1.4e')

    print('Creating map from Power Spectrum...')
    out_name = out_dir+'msim_'+tag+str(i).zfill(4)+'.fits'
    msim = hp.synfast(cl_temp,NSIDE)

    print('Saving map...')
    hp.write_map(path+out_name,msim,coord='G',fits_IDL=False,overwrite=True)

    '''
    #TEST PS with normalized map
    test = hp.anafast(normalization(msim),lmax=1500)
    print(test)
    print(cl_temp)
    fact=255.0/(np.max(msim))
    fact=1.0
    fact=test[100]/cl_temp[100]
    print(fact)
    plt.plot(np.arange(len(test)),test)
    plt.plot(np.arange(len(cl_temp)),cl_temp*fact,color='orange')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(xmin=l_start,xmax=1400)
    #plt.show()
    plt.savefig(path+out_dir+'test.png')
    exit()
    '''

    print('Converting map to tif format...')
    moll_array = hp.cartview(msim, title=None, xsize=x_size, ysize=y_size, return_projected_map=True)
    moll_array = normalization(moll_array)
    moll_array = np.array(moll_array, dtype=np.uint8)
    moll_image = Image.fromarray(moll_array)

    print('Saving tif map...')
    out_tif = out_dir+'msim_'+tag+str(i).zfill(4)+'_data.tif'
    moll_image.save(path+out_tif)
    #out_png = out_dir+'msim_'+tag+str(i).zfill(4)+'_data.png'
    #moll_image.save(path+out_png)

    print('Writing parameters file...')
    #print text
    f.write(text)
    f.close()
    sub.call(['cp',path+out_text_temp,path+out_text],) #shell=[bool])

    #print 'Partial time :',time.time()-t_start,'s\n'
    print '\n'

t_stop=time.time()
print 'Elapsed time for create maps:',t_stop-t_start,'s'

print 'Done.'
