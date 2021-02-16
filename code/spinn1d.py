'''SPINN in 1D with nodes that move and have either variable or fixed widths.
'''

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from oned import Problem1D, tensor, App1D
