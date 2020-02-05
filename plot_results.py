from matplotlib import pyplot as plt
import numpy as np
import argparse
from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams.update({'text.usetex': True})
rcParams.update({'figure.autolayout': True})

parser = argparse.ArgumentParser()
parser.add_argument("--tag_res", type=str, help="tag of the network results", default='')
parser.add_argument("--path", type=str, help="general path of results", default='/home/simone/RadioML/results/')
parser.add_argument("--path_dest", type=str, help="path of plots", default='/home/simone/RadioML/plots/')
parser.add_argument("--N_plot", type=int, help="number of plots to produce (default: 4)", default=4)

args = parser.parse_args()
tag_res = args.tag_res
path = args.path
path_dest = args.path_dest
N_plot = args.N_plot

if tag_res is not '':
    tag_res = '_' + tag_res

if path is not '' and path[-1] is not '/':
    path = path + '/'

if path_dest is not '' and path_dest[-1] is not '/':
    path_dest = path_dest + '/'

print('Creating Loss Plot:')
dpi = 300
try:
    loss = np.transpose(np.genfromtxt(path+'loss_function'+tag_res+'.txt', dtype=np.float32))

    plt.plot(loss[0], loss[1], '.-')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    # plt.ylim((0,np.max(loss[1])*1.1))
    plt.ylim((np.min(loss[1]),np.max(loss[1])*1.01))
    # plt.ylim((0,0.015))
    plt.xlim((np.min(loss[0]), np.max(loss[0])*1.01))
    # plt.grid()
    plt.savefig(path_dest+'loss'+tag_res+'.png', dpi=dpi)
    plt.close()
except Exception as e:
    print(e)

for i in range(N_plot):
    plt.figure(figsize=(12,10))
    data = np.genfromtxt(path+'2-PCF_map_'+str(i).zfill(5)+tag_res+'.txt')

    print('Percentage variation:')
    print(np.round(np.abs(1-data[:,1]/data[:,2])*100,1))

    plt.plot(data[:,0],data[:,1],'o-',label='prediction',linewidth=4,markersize=18)
    plt.plot(data[:,0],data[:,2],'o-',label='label',linewidth=4,markersize=18)

    plt.xlabel(r'$\theta \, \mathrm{[deg]}$',fontsize=35)
    plt.ylabel(r'$\xi \, (\theta) \, \mathrm{X 10^3}$',fontsize=35)
    plt.xscale('log')
    plt.yscale('log')
    plt.tick_params(direction='in', width=2, length=5, axis='both', which='major', labelsize=30, pad=7)#,length=6,width=3)
    plt.tick_params(direction='in', width=2, length=5, axis='both', which='minor', labelsize=30, pad=7)#,length=6,width=3)
    plt.legend(loc='upper right',fontsize=30,framealpha=0.5,fancybox=True)
    plt.savefig(path_dest+'2-PCF_map_'+str(i).zfill(5)+tag_res+'.png')
    #plt.show()
    plt.clf()

print('Done.')
