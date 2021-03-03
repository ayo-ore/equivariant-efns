#!/usr/bin/env python3

import argparse
from energyflow.utils import data_split, to_categorical
import tensorflow as tf
import numpy as np
import os
from sklearn.metrics import roc_curve

from nn import EquivariantLayer, EquivariantModel
from data import preprocess

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, help='Path to dataset.')
parser.add_argument('--model', type=str, choices=('ev-efn', 'ev-pfn', 'ev-pfn-id'), default='ev-efn', help='Equivariant archirecture to train.')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train model.')
parser.add_argument('--batch_size', type=int, default=480, help='Size of data mini-batches.')
parser.add_argument('--optimizer', type=str, default='adam', help='Keras optimisation algorithm to train model.')
parser.add_argument('--loss', type=str, default='binary_crossentropy', help='Keras loss function to optimise.')
parser.add_argument('--equi_act', type=str, default='relu', help='Keras activation applied to equivariant layers.')
parser.add_argument('--ppm_sizes', type=int, nargs='+', default=(100,100,128), help='List of layer sizes for the ppm network.')
parser.add_argument('--equi_channels', type=int, nargs='+', default=(100,100), help='List of output channels for the equivariant layers.')
parser.add_argument('--f_sizes', type=int, nargs='+', default=(100,100), help='List of layer sizes for the F network.')
parser.add_argument('--unprocessed', action='store_true', help='Whether or not to skip data preprocessing.')
parser.add_argument('--lambda_zero', action='store_true', help='Whether or not to enforce Lambda=0 in equivairant layers.')
parser.add_argument('--gamma_zero', action='store_true', help='Whether or not to enforce Gamma=0 in equivairant layers.')
parser.add_argument('--projection', type=str, choices=('sum', 'max'), default='sum', help='Projection operation to pool output of equivariant layers.')
parser.add_argument('--equi_type', type=str, choices=('sum', 'max', 'irc'), default='irc', help='Equivariant layer specification for ev-pfn(-id).')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout value to apply to layers in F.')
parser.add_argument('--bigtest', action='store_true', help='Whether or not to split data into train and test only.')
parser.add_argument('--dry', action='store_true', help='Whether or not to run without training or saving.')
parser.add_argument('--output', type=str, default=os.getcwd(), help='Directory to save training outputs.')
parser.add_argument('--filename', type=str, default='output', help='Filename for training outputs.')
args = parser.parse_args()

evefn, evpfn, evpfnid = (args.model==x for x in ('ev-efn', 'ev-pfn', 'ev-pfn-id'))
tf.keras.backend.set_floatx('float64')

# check option compatibility
if evefn:
    if args.projection != 'sum':
        print(f'WARNING: EV-EFN model must use sum projection. Switching from {args.projection}.')
        args.projection = 'sum'
    if args.equi_type != 'irc':
        print(f'WARNING: EV-EFN model must use irc equivariant layers. Switching from {args.equi_type}.')
        args.equi_type = 'irc'
if evpfn:
    if args.equi_type == 'irc':
        print(f'WARNING: EV-PFN model incompatible with irc equivariant layers. Switching to sum.')
        args.equi_type = 'sum'

print(args)


# load data
print('Loading data...')
if not args.dry:
    data_file = np.load(args.dataset)
    data = data_file['data']
    labels = data_file['labels']
    labels = to_categorical(labels, num_classes=2)
print('Done')


# preprocess data
print('Pre-processing events...')
if not args.dry:
    if not args.unprocessed:
        R = 0.2 if '500' in args.dataset.split('/')[-1] else 0.3
        preprocess(data, R, rotate=True, reflect=True)
    if evpfnid:
        print('Removing mass from data')
        data = np.delete(data,3,2) # remove mass
    else:
        print('Taking only momentum data')
        data = data[:,:,:3] # take only IRC-safe data
print('Done')


# create train/val/test splits
print('Splitting datasets...')
if not args.dry:
    val_frac, test_frac = (5, 0.25) if args.bigtest else (0.10, 0.15)
    if evpfn:
        (train, val, test,
        labels_train, labels_val, labels_test,
        _, __, indices_test)\
        = data_split(data,labels, np.arange(data.shape[0]), train=-1, val=val_frac, test=test_frac, shuffle=True)
        trainX, valX = train, val
    elif evpfnid:
        (mom_train, mom_val, mom_test,
        pid_train, pid_val, pid_test,
        labels_train, labels_val, labels_test,
        _, __, indices_test)\
        = data_split(data[:,:,:3], data[:,:,3], labels, np.arange(data.shape[0]), train=-1, val=val_frac, test=test_frac, shuffle=True)
        trainX, valX = [mom_train, pid_train]
    else:
        (z_train, z_val, z_test,
        p_train, p_val, p_test,
        labels_train, labels_val, labels_test,
        _, __, indices_test)\
        = data_split(data[:,:,0], data[:,:,1:], labels, np.arange(data.shape[0]), train=-1, val=val_frac, test=test_frac, shuffle=True)
        trainX, valX = [z_train, p_train], [z_val, p_val]
print('Done')

# build and fit model
print('Building model...')
NN = EquivariantModel(
        model=args.model,
        ppm_sizes=args.ppm_sizes,
        equi_channels=args.equi_channels,
        equi_type=args.equi_type,
        projection=args.projection,
        f_sizes=args.f_sizes,
        lambda_zero=args.lambda_zero,
        gamma_zero=args.gamma_zero,
        equi_act=args.equi_act,
        dropout=args.dropout
)
NN.compile(optimizer=args.optimizer, loss=args.loss, metrics='acc')
input_shape = [tuple([args.batch_size]) + x.shape[1:] for x in trainX] if isinstance(trainX,list) else tuple([args.batch_size]) + trainX.shape[1:]
NN.build(input_shape)
NN.summary()
print('Done')
print('Training model...')
if not args.dry:
    hc = NN.fit(trainX, labels_train, epochs=args.epochs, batch_size=args.batch_size, validation_data=(valX, labels_val))
print('Done')


# save results
print('Saving results...')
if args.dry:
    print(f"Saving model weights to {os.path.join(args.output, args.filename +'_model.h5')}")
    print(f"Saving training history to {os.path.join(args.output, args.filename +'_history.npz')}")
    print(f"Saving ROC curve to {os.path.join(args.output, args.filename + '_roc.npz')}")
else:
    NN.save_weights(os.path.join(args.output, f'{args.filename}_model.h5'))

    loss_history, val_loss_history = hc.history['loss'], hc.history['val_loss']
    acc_history, val_acc_history = hc.history['acc'], hc.history['val_acc' ]
    np.savez_compressed(os.path.join(args.output, f'{args.filename}_history'), loss=np.array(loss_history), val_loss=np.array(val_loss_history), acc=np.array(acc_history), val_acc=np.array(val_acc_history))

    if evpfn:
        preds = NN.predict(test, batch_size=args.batch_size)
    elif evpfnid:
        preds = NN.predict([mom_test, pid_test], batch_size=args.batch_size)
    else:
        preds = NN.predict([z_test, p_test], batch_size=args.batch_size)

    fp, tp, threshs = roc_curve(labels_test[:,1], preds[:,1])
    np.savez_compressed(os.path.join(args.output, f'{args.filename}_roc'), tp=tp, fp=fp)
print('Done')