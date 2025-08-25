import sys
sys.path.insert(0, "../")

import warnings
import os
import time

import math as m
import numpy as np
np.random.seed(1)

import matplotlib
import matplotlib.pyplot as plt
import pickle

from experiment_utils import history_todict, get_val_split
from layer_rotation_utils import LayerRotationCurves,StepwiseRotation, StepwiseLearningRateScheduler
from layca_optimizers import SGD

from import_task import import_task
from get_training_utils import get_training_schedule, get_stopping_criteria, get_optimizer, get_learning_rate_multipliers
from get_training_utils import get_optimized_training_schedule

from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator


# utilities for storing the results in pickle files
result_file = 'results_explanatory_adaptive.p'
def load_results():
    if not os.path.isfile(result_file):
        return {}
    else:
        with open(result_file,'rb') as f:
            return pickle.load(f)

def dump_results(results):
    with open(result_file,'wb') as f:
        pickle.dump(dict(results),f)

def update_results(path, new_data):
    results = load_results()
    position = results
    for p in path:
        position = position[p]
    # new_data is a dictionary with the new (key,value) pairs
    position.update(new_data)
    dump_results(results)


    # if results should be saved in the file or not
save_results = True
if not save_results:
    results = {}
# file for monitoring the experiment's progress
monitor_file = 'monitor_explanatory_adaptive.txt' 



tasks = ['C100-resnet', 'C10-CNN1']#['C10-CNN1','C100-resnet','tiny-CNN','C10-CNN2','C100-WRN']

for task in tasks:
    x_train, y_train, x_test, y_test, get_model = import_task(task)
    
    # validation set is actually not needed for this experiment... (I forgot to remove it but it doesn't matter)
    [x_train, y_train], [x_val, y_val] = get_val_split(x_train,y_train, 0.1)
    
    # creates empty dictionary if first time the task is seen
    if save_results:
        results = load_results()
        if task not in results.keys():
            update_results([],{task:{}})
    elif task not in results.keys():
        results.update({task:{}})
    
    if task == 'C10-CNN1':
        optimizers = ['SGD','RMSprop','SGD_layca']
    elif task == 'C100-resnet':
        optimizers = ['SGD','Adam','SGD_AMom_layca']
    elif task == 'tiny-CNN':
        optimizers = ['SGD','Adagrad','SGD_layca']
    elif task == 'C10-CNN2':
        optimizers = ['SGD_weight_decay','RMSprop_weight_decay','SGD_layca']
    elif task == 'C100-WRN':
        optimizers = ['SGD_weight_decay','Adam_weight_decay','SGD_AMom_layca']
    
    for optimizer in optimizers:
        start = time.time()
        model = get_model(weight_decay = 0.) if 'weight_decay' not in optimizer else get_model()

        batch_size = 128
        if 'layca' not in optimizer:
            epochs, lr, lr_scheduler = get_optimized_training_schedule(task,optimizer)
        else: # when using layca, we don't want to get best schedule, but to copy the schedule of the adaptive method
            epochs, lr, lr_scheduler = get_optimized_training_schedule(task,optimizers[1])
        verbose = 1

        # frequency at which cosine distance from initialization is computed
        batch_frequency = int((x_train.shape[0]/batch_size))+5 # higher value than # of batches per epoch means once per epoch
        ladc = LayerRotationCurves(batch_frequency = batch_frequency)
        callbacks = [lr_scheduler, ladc]

        stepwise_recordings = StepwiseRotation()
        if not 'SGD' in optimizer:
            callbacks += [stepwise_recordings]
        if 'layca' in optimizer:
            if save_results:
                results = load_results()
            stepwise_schedule = StepwiseLearningRateScheduler(schedule = results[task][optimizers[1]]['stepwise_recordings'])
            callbacks+= [stepwise_schedule]

        multipliers = get_learning_rate_multipliers(model,alpha = 0.)
        # C100-WRN + SGD is the only case where nesterov momentum is used (cfr. original implementation)
        if task == 'C100-WRN' and optimizer in ['SGD','SGD_weight_decay']: 
            opt = SGD(lr=lr, momentum=0.9, nesterov=True,multipliers = multipliers)
        else:
            opt = get_optimizer(optimizer, lr,multipliers)
        metrics = ['accuracy', 'top_k_categorical_accuracy'] if 'tiny' in task else ['accuracy']
        model.compile(loss='categorical_crossentropy',
                      optimizer= opt,
                      metrics=metrics)

        # cifar100 resnet and tinyImagenet need early stopping
        if task=='C100-resnet' or 'tiny' in task:
            weights_file = 'saved_weights/best_weights_'+str(np.random.randint(1e6))+'.h5'
            callbacks += [ModelCheckpoint(weights_file, monitor='val_acc', save_best_only=True, save_weights_only = True)]


        with warnings.catch_warnings():
            if task in ['C10-CNN2','C100-WRN']:
                # data augmentation
                datagen = ImageDataGenerator(width_shift_range=0.125,
                         height_shift_range=0.125,
                         fill_mode='reflect',
                         horizontal_flip=True)

                warnings.simplefilter("ignore") # removes warning from keras for slow callback
                history = model.fit_generator(datagen.flow(x_train, y_train,batch_size=batch_size),
                                              steps_per_epoch=x_train.shape[0] // batch_size,
                                              epochs = epochs,
                                              verbose = verbose,
                                              validation_data = (x_val, y_val),
                                              callbacks = callbacks)
            else:
                warnings.simplefilter("ignore") # removes warning from keras for slow callback
                history = model.fit(x_train,y_train,
                                    epochs = epochs,
                                    batch_size = batch_size,
                                    verbose = verbose,
                                    validation_data = (x_val, y_val),
                                    callbacks = callbacks)

        # application of early stopping
        if task=='C100-resnet' or 'tiny' in task:
            model.load_weights(weights_file)

        test_performance = model.evaluate(x_test,y_test, verbose = verbose)

        if save_results:
            update_results([task],{optimizer:{'history':history_todict(history),'ladc':ladc.memory,
                                              'stepwise_recordings': stepwise_recordings.memory,
                                              'test_performance':test_performance}})
        else:
            results[task].update({optimizer:{'history':history_todict(history),'ladc':ladc.memory,
                                             'stepwise_recordings': stepwise_recordings.memory,
                                             'test_performance':test_performance}})

        with open(monitor_file,'a') as file:
            file.write(task + ', '+optimizer+': done in '+str(time.time()-start)+' seconds.\n')