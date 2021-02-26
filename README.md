# Equivariant Energy Flow Networks for jet tagging
An implementation of IRC safe permutation-equivariant layers in Energy Flow Networks.

##### References
[1] M. J. Dolan and A. Ore, _Equivariant Energy Flow Networks for jet tagging_,
[arXiv:2012.00964 [hep-ph]](https://arxiv.org/abs/2012.00964)

## Python 3 dependencies
- [numpy 1.16.2](https://numpy.org/)
- [keras ](https://keras.io/)
- [tensorflow 1.13.1](https://www.tensorflow.org/)
- [sklearn 0.22.1](https://scikit-learn.org/stable/)
- [energyflow 1.0.2](https://energyflow.network/)
- [pyjet 1.6.0](https://github.com/scikit-hep/pyjet)

## Data format
Data should be stored in a `.npz` file containing jet examples as a `numpy.ndarray` with shape `(num_jets, max_constituents, 5)` at key `'data'` and binary jet labels as a `numpy.ndarray` with shape `(num_jets,)` at key `'labels`. The last dimension of the examples array should hold constituent information in the format `(pt, y, phi, m, pid)`, where `pid` is an integer in the range `\[0,n-1\]` representing the identity of the particle from `n` categories (Absent particle represented by `-1`).

## Usage
The script `train-ev-model.py` is used to train one of the EV-EFN, EV-PFN or EV-PFN-ID models and may be passed the following arguments:

  -`dataset`: Path to the data in `.npz` format as described above. [required]
  -`model`: Equivariant archirecture to train. One of (`'ev-efn'`, `'ev-pfn'`, `'ev-pfn'id`). \[Default=`'ev-efn'`\]
  -`epochs`: Number of epochs to train model.
  -`batch_size`: Size of data mini-batches.
  -`optimizer`: Keras optimisation algorithm to use.
  -`loss`: Keras loss function to optimise.
  -`equi_act`: Keras activation applied to equivariant layers.
  -`ppm_sizes`: List of layer sizes for the Phi network.
  -`equi_channels`: List of output channels for the equivariant layers
  -`f_sizes`: List of layer sizes for the F network.
  -`lambda_zero`: Enforce Lambda=0 in equivairant layers.
  -`gamma_zero`: Enforce Gamma=0 in equivairant layers.
  -`projection`: Projection operation to pool output of equivariant layers. One of (`'sum'`, `'max'`).
  -`equi_type`: Equivariant operation specification. One of (`'sum'`, `'max'`, `'irc'`). 
  -`dropout`: Dropout value to apply to layers in F. A value of zero corresponds to no dropout.
  -`bigtest`: Split data into train and test only.
  -`dry`: Run without training or saving.
  -`output`: Directory to save training outputs.
  -`filename`: 

   - `datadir`: Directory containing the *converted* top-tagging files.
    - `maxdim`: Maximum dimensionality of tensors produced in the network.
    - `max-zf`: Maximum degree of zonal functions used in tensor decompositions.
    - `num-channels`: Number of channels per layer.
    - `num-epoch`: Number of training epochs.
    - `batch-size`: Mini-batch size.
    - `num-cg-levels`: Number of Clebsch-Gordan layers. If this is smaller than `num-channels`, the extra layers at the end will be standard multi-layer perceptrons (MLP's) acting on any Lorentz-invariants produced.
    - `lr-init`: Initial learning rate.
    - `lr-final`: Final learning rate.
    - `mlp`: Whether or not to insert MLP's acting on Lorentz-invariant scalars within the CG layers.
    - `pmu-in`: Whether or not to feed in 4-momenta themselves to the first CG layer, in addition to scalars.
    - `nobj`: Max number of jet constituents to use for entry. Constituents are ordered by decreasing `pT`, so the network uses the `nobj` leading constituents.
