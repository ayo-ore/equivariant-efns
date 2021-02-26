# Equivariant Energy Flow Networks for jet tagging
An implementation of IRC safe permutation-equivariant layers in Energy Flow Networks.

##### References
[1] M. J. Dolan and A. Ore, _Equivariant Energy Flow Networks for jet tagging_,
[arXiv:2012.00964 [hep-ph]](https://arxiv.org/abs/2012.00964)

## Python 3 dependencies
- [numpy 1.17.3](https://numpy.org/)
- [tensorflow 2.3.1](https://www.tensorflow.org/)
- [sklearn 0.22.1](https://scikit-learn.org/stable/)
- [energyflow 1.0.2](https://energyflow.network/)
- [pyjet 1.6.0](https://github.com/scikit-hep/pyjet)

## Data format
Data should be stored in a `.npz` file containing jet examples as a `numpy.ndarray` with shape `(num_jets, max_constituents, 5)` at key `'data'` and binary jet labels as a `numpy.ndarray` with shape `(num_jets,)` at key `'labels'`. The last dimension of the examples array should hold constituent information in the format `(pt, y, phi, m, pid)`, where `pid` is an integer in the range `\[0,n-1\]` representing the identity of the particle from `n` categories (Absent particle represented by `-1`).

## Usage
The script `train-ev-model.py` is used to train one of the EV-EFN, EV-PFN or EV-PFN-ID models and may be passed the following arguments:

  - `dataset`: Path to the data in `.npz` format as described above. \[required\]
  - `model`: Equivariant archirecture to train. One of (`'ev-efn'`, `'ev-pfn'`, `'ev-pfn-id'`). \[default = `'ev-efn'`\]
  - `epochs`: Number of epochs to train model. \[default = `30`\]
  - `batch_size`: Size of data mini-batches. \[default = `480`\]
  - `optimizer`: Keras optimisation algorithm to use. \[default = `'adam'`\]
  - `loss`: Keras loss function to optimise. \[default = `'binary_crossentropy'`\]
  - `equi_act`: Keras activation applied to equivariant layers. \[default = `'relu'`\]
  - `ppm_sizes`: List of layer sizes for the Phi network. \[default = `100 100 128`\]
  - `equi_channels`: List of output channels for the equivariant layers \[default = `100 100`\]
  - `f_sizes`: List of layer sizes for the F network. \[default = `100 100 100`\]
  - `projection`: Projection operation to pool output of equivariant layers. One of (`'sum'`, `'max'`). \[default = `'sum'`\]
  - `equi_type`: Equivariant operation specification. One of (`'sum'`, `'max'`, `'irc'`).  \[default = `'sum'`\]
  - `dropout`: Dropout value to apply to layers in F. A value of zero corresponds to no dropout. \[default = `0.0`\]
  - `output`: Directory to save training outputs. \[default = `os.getcwd()`\]
  - `filename`: Filename for training outputs. \[default = `'output'`\]
  - `lambda_zero`: Whether or not to enforce Lambda=0 in equivairant layers.
  - `gamma_zero`: Whether or not to enforce Gamma=0 in equivairant layers.
  - `bigtest`: Whether or not to split data into train and test only.
  - `unprocessed`: Whether or not to skip data preprocessing.
  - `dry`: Whether or not to run without training or saving.


On completion of training, the following outputs are saved:
