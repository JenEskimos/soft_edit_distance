# Soft Edit Distance
[Chainer](https://github.com/chainer/chainer/)-based Python implementation of Soft Edit Distance.

## Requirement

- Python 3.6.0+
- [Chainer](https://github.com/chainer/chainer/) 4.0.0+
- [numpy](https://github.com/numpy/numpy) 1.12.1+
- [cupy](https://github.com/cupy/cupy) 4.0.0+
- matplotlib
- and their dependencies

## Example of usage
```
from chainer import Variable
import chainer_edit_distance
# definition of alphabet and sequences
alphabet = ['A', 'T', 'G', 'C']
x1 = ['ATGCCA', 'TCC']
x2 = ['TACGC', 'TCACGG']
 	
# convert strings to cupy array
x1 = chainer_edit_distance.parse_to_tensor(x1, alphabet, 6) 
x2 = chainer_edit_distance.parse_to_tensor(x2, alphabet, 6)
 	
# calculating of original edit distance
ed = chainer_edit_distance.edit_distance(x1, x2)
 	
# calculating of soft edit distance sed is a Variable object
x1 = Variable(x1)
x2 = Variable(x2)
sed = chainer_edit_distance.soft_edit_distance(x1, x2)
```
## Run test for simulated dataset clustering
```
python test_simulated.py
```

After clustering all numeric result will be saved to results_many.csv. Visualisations will be added to images folder.
