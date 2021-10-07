# Quantum Tomography Benchmarking

Python package for benchmarking quantum tomography (QT) methods.
The theory behind the software could be found in [[1]](#ref1).

## Installation

To install package download the source files and run
```bash
python setup.py install
```

## Analyze the QT method benchmarks

The following code shows a basic example of running data collection
for a 2-qubit tomography method on random pure states.
``` python
from pyqtb import Dimension
from pyqtb.analyze import collect
from pyqtb.tests import rps_test
from pyqtb.methods.proto_fmub import proto_fmub
from pyqtb.methods.est_ppi import est_ppi

dim = Dimension([2, 2])
result = collect(dim, proto_fmub(dim), est_ppi(), rps_test(dim))
```

To get and print a basic report over benchmarks one runs the following commands.
``` python
from pyqtb.analyze import report, as_table

r = report(result)
print(as_table(report))
```

One can also compare different methods.
``` python
from pyqtb.analyze import compare, as_table

r = report([result1, result2])
print(as_table(report))
```

Module `pyqtb.analyze` also contains plot functions.

## License

All code found in this repository is licensed under GPL v3

## References
<a name="ref1">[1]</a> Bantysh B.I. et al. Quantum tomography benchmarking; <a href="https://arxiv.org/abs/2012.15656">arXiv:2012.15656</a>
