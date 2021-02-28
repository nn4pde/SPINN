# Sparse Physics-based and Interpretable Neural Networks for PDEs

This repository contains the code and manuscript for research done on Sparse
Physics-based and Interpretable Neural Networks for PDEs. The
[preprint](https://arxiv.org/abs/2102.13037) of this work is available on
arXiv.


## Installation

Running the code in this repository requires a few pre-requisites to be set
up. The Python packages required are in the `requirements.txt`. Here are some
instructions to help you set these up:

0. Setup a suitable Python distribution, using [conda](https://conda.io) or a
   [virtualenv](https://virtualenv.pypa.io/).

1. Clone this repository:
```
    $ git clone https://github.com/nn4pde/SPINN.git
    $ cd SPINN
```

2. If you use conda, run the following from your Python environment:
```
    $ conda env create -f environment.yml
    $ conda activate spinn
```

3. If you use a virtualenv or some other Python distribution and wish to use `pip`:
```
    $ pip install -r requirements.txt
```

Once you install the packages you should hopefully be able to run the
examples. The examples all support live-plotting of the results.
[Matplotlib](https://matplotlib.org) is required for the live plotting of any
of the 1D problems and [Mayavi](https://docs.enthought.com/mayavi/mayavi/) is
needed for any 2D/3D problems. These are already specified in the
`requirements.txt` and `environments.yml` files.


## Running the code

All the problems discussed in the paper are available in the `code`
subdirectory. The supplementary text in the paper discusses the design of the
code at a very high level.  You can run any of the problems as follows:
```
  $ cd code
  $ python ode3.py -h
```

And this will provide a variety of help options that you can use. You can see
the results live by doing:
```
  $ python ode3.py --plot
```
These require matlplotlib.

The 2D problems also feature live plotting with Mayavi if it is installed, for
example:
```
  $ python advection1d.py --plot
```
You should see the solution as well as the computational nodes.  Where
applicable you can see an exact solution as a wireframe.

If you have a GPU and it is configured to work with
[PyTorch](https://pytorch.org/), you can use it like so:
```
  $ python poisson2d_irreg_dom.py --gpu
```


## Generating the results

All the results shown in the paper are automated using the
[automan](https://automan.readthedocs.io) package which should already be
installed as part of the above installation. This will perform all the
required simulations (this can take a while) and also generate all the plots
for the manuscript.

To learn how to use the automation, do this:
```
    $ python automate.py -h
```

By default the simulation outputs are in the `outputs` directory and the
final plots for the paper are in `manuscript/figures`.

To generate all the figures in one go, run the following (this will take a while):
```
    $ python automate.py
```

If you wish to only run a particular set of problems and see those results you
can do the following:
```
   $ python automate.py PROBLEM
```

where `PROBLEM` can be any of the demonstrated problems.  For example:

```
  $ python automate.py ode1 heat cavity
```

Will only run those three problems. Please see the help output (`-h`) and look
at the code for more details.

By default we do not need to use a GPU for the automation but if you have one,
you can edit the `automate.py` and set `USE_GPU = True` to make use of your
GPU where possible.



## Building the paper

**WARNING**: while all the plots shown in the the main text of the pre-print
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
