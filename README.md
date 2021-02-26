# Equivariant Energy Flow Networks for jet tagging
An implementation of IRC safe permutation-equivariant layers in Energy Flow Networks.

##### References
[1] M. J. Dolan and A. Ore, Equivariant Energy Flow Networks for jet tagging,
[arXiv:2012.00964 [hep-ph]](https://arxiv.org/abs/2012.00964)

## Python 3 dependencies
- [numpy 1.16.2](https://numpy.org/)
- [keras ](https://keras.io/)
- [tensorflow 1.13.1](https://www.tensorflow.org/)
- [sklearn 0.22.1](https://scikit-learn.org/stable/)
- [energyflow 1.0.2](https://energyflow.network/)
- [pyjet 1.6.0](https://github.com/scikit-hep/pyjet)

## Data format
Data should be stored in `numpy.ndarray` format with shape `(num_jets, max_constituents, 5)`. The last dimension holds constituent information in the format $(p_T,y,\phi,m,\texttt{pdg_id})$

## Usage
explain arguments
