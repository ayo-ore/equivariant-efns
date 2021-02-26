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
Each model of 
