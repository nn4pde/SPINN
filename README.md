# Sparse Physics-based and Interpretable Neural Networks

This repository contains the code and manuscript for research done on Sparse
Physics-based and Interpretable Neural Networks. The
[preprint](https://arxiv.org/abs/2102.13037) of this work is available on
arXiv.


## Installation

This requires pysph to be setup along with automan. See the
`requirements.txt`. To setup perform the following:

0. Setup a suitable Python distribution, using [conda](https://conda.io) or a
   [virtualenv](https://virtualenv.pypa.io/).

1. Clone this repository:
```
    $ git clone https://github.com/nn4pde/SPINN.git
```

2. Run the following from your Python environment:
```
    $ cd SPINN
    $ pip install -r requirements.txt
```


## Generating the results

The paper and the results are all automated using the
[automan](https://automan.readthedocs.io) package which should already be
installed as part of the above installation. This will perform all the
required simulations (this can take a while) and also generate all the plots
for the manuscript.

To use the automation code, do the following::

    $ python automate.py
    # or
    $ ./automate.py

By default the simulation outputs are in the ``outputs`` directory and the
final plots for the paper are in ``manuscript/figures`**.


## Building the paper

**WARNING***: while all the plots shown in the the main text of the pre-print
listed above are fully automated, there are a few files in the supplement that
are not yet fully integrated and we will be fixing this in a few days. So if
you build the paper currently you will see some errors.

Once you have generated all the figures from the automation you can easily
compile the manuscript. The manuscript is written with LaTeX and if you have
that installed you may do the following:

```
$ cd manuscript
$ latexmk spinn_manuscript.tex -pdf
```
