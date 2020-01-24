import numpy as np
import glob
from PIL import Image
import h5py

from utility import image_provider
from utility import network

# paths of images and path of the spectra.dat-file
#path = '/home/simone/RadioML/'
path = '/archive/home/sammazza/radioML/'
path_train_data = path+'data/mapsim_PS/'
path_train_label = path+'data/mapsim_PS/CCF_test_0_100_label.dat'
path_results = path+'results/'

tag='test'
tag_res='test'

if tag is not '':
   tag=tag+'_'

if tag_res is not '':
   tag_res='_'+tag_res

##########################################################
## Parameters ############################################
train = True

N_epochs = 10
batch_size = 10

model_parameters = {'learning_rate': 1E-5,      # 1E-5
                       'decay_rate': 1E-5,      # 1E-4 # i.e. lr /= (1+decay_rate) after each epoch
                      'kernel_size': (5,5), 
                        'pool_size': (4,4),
                           'stride': 1
                    }
##########################################################
##########################################################
def emptyDirectory(thePath):
    import os, shutil
    for the_file in os.listdir(thePath):
        file_path = os.path.join(thePath, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)
##########################################################
## Setting up the Generators #############################

# Creating the 'partition' dictionary that contains all the image IDs ('msim_0000_data' etc.)
# divided into a training and validation set

all_IDs = []  # to store all IDs

N_files = len(glob.glob(path_train_data+'*_data.tif'))
for i in range(N_files):
    all_IDs.append('msim_'+tag+'%04d_data'%(i))

# Splitting into training/validation/test
num1 = int(len(all_IDs)-4*batch_size)
num2 = int(len(all_IDs)-1*batch_size)

training_IDs = all_IDs[:num1]
validation_IDs = all_IDs[num1:num2]
test_IDs = all_IDs[num2:]

partition = {'train': training_IDs,
             'validation': validation_IDs,
             'test': test_IDs}

print('Training with %i/%i images'
        %(len(partition['train']), len(partition['train'])+len(partition['validation'])+len(partition['test'])))
print('Validating on %i/%i images'
        %(len(partition['validation']), len(partition['train'])+len(partition['validation'])+len(partition['test'])))
print('   Testing on %i/%i images'
        %(len(partition['test']), len(partition['train'])+len(partition['validation'])+len(partition['test'])))

# Reading the spectra.dat-file and store all spectra
all_labels = np.transpose(np.genfromtxt(path_train_label, dtype=np.float32)[:,1:])

if len(all_labels)!=len(all_IDs):
   print('lenght labels:  lenght data:')
   print(len(all_labels),len(all_IDs))
   print('ERROR: labels not matching training set.')
   exit()

# norming the labels to values from 0...y_max
y_max = 1.0
max_value_label = np.max(all_labels)
all_labels = all_labels*y_max
all_labels = all_labels/max_value_label

# Creating a dictionary that associates the right correlation function to each ID
# labels = {'msim_0000_data': [0.082, 0.20930, ....]], 
#           'msim_0001_data': [0.082, 0.20930, ....]], 
#           ....}

labels = {} 
for k,label in enumerate(all_IDs):
    labels[label] = all_labels[k]

# Number of output nodes has to be the number of points in the correlation function
N_out = len(labels['msim_'+tag+'0000_data'])

# Read a single image in order to determine the pixel-size
image = np.array(Image.open(path_train_data+partition['train'][0]+'.tif'))

n_x = image.shape[0]    # the shorter side
n_y = image.shape[1]
n_channels = 1

# data shape for the input of the CNN
data_shape = (n_x, n_y, n_channels)

# parameters for the data-generators
generator_parameters = {'path_data': path_train_data,
              'dim': (n_x, n_y),
              'N_out': N_out,
              'batch_size': batch_size,
              'norm': False}

# Definitions of the generators
print('Definition of generators...')
training_generator   = image_provider.DataGenerator(partition['train'], labels, shuffle=True, **generator_parameters)
validation_generator = image_provider.DataGenerator(partition['validation'], labels, shuffle=True, **generator_parameters)
test_generator = image_provider.DataGenerator(partition['test'], labels, shuffle=False, **generator_parameters)
# test_generator = image_provider.DataGenerator(partition['train'], labels, shuffle=False, **generator_parameters)

##########################################################
## Model and training ####################################

# Defining the learning model
print('Initializing model...')
model = network.CNN(N_out, data_shape=data_shape, **model_parameters)


if train==True:
    print('Fitting model...')
    # parameters fed into the fit-method
    fit_parameters = {'generator': training_generator,
                      'validation_data': validation_generator, 
                      'epochs': N_epochs}
    # training the model
    history = model.fit_generator(**fit_parameters)

    model.save(path+'saved_model/model'+tag_res+'.h5')
    model.save_weights(path+'saved_model/weights'+tag_res+'.csv')

    # Creating a file to store the loss function:
    epochs = np.array(range(1,1+N_epochs))
    loss_values = history.history['loss']

    #emptyDirectory(path_results)

    with open(path+'loss_function'+tag_res+'.txt','w') as stats:
        stats.write('#Epoch  Loss\n')

    for k in range(len(epochs)):
        with open(path+'loss_function'+tag_res+'.txt','a') as stats:
            stats.write('{:}    {:}\n'.format(epochs[k], loss_values[k]))
else:
    from keras.models import load_model
    print('Loading weights...')
    model = load_model(path+'saved_model/model'+tag_res+'.h5')


# Testing:
thetas = np.transpose(np.genfromtxt(path_train_label, dtype=np.float32)[:,:1])[0]
print('Running prediction:')
pred = model.predict_generator(test_generator, verbose=1)
target = np.asarray([*test_generator.labels.values()])[num2:]     # the * unpacks the dictionary_values-type

for k in range(target.shape[0]):
    # printing the outputs
    with open(path_results+'2-PCF_map_%05d.txt'%(k),'w') as stats:
        stats.write('#theta  pred    target\n')

    for i in range(len(thetas)):
        with open(path_results+'2-PCF_map_%05d.txt'%(k),'a') as stats:
            stats.write('{:}    {:}    {:}\n'.format(thetas[i], pred[k,i]*max_value_label, target[k,i]*max_value_label))






