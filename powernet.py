#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Patrice Bechard
@email: bechardpatrice@gmail.com
Created on Wed Nov 15 09:39:36 2017

PowerNet

"""

import torch
from torch import nn
import torch.nn.functional as F

class PowerNet(nn.Module):

	def __init__(self, input_size, output_size, reservoir_size=256):
		super(PowerNet, self).__init__()

		self.input_size = input_size
		self.output_size = output_size
		self.reservoir_size = reservoir_size

		self.fc_in = nn.Linear(input_size, reservoir_size)
		self.reservoir = nn.Linear(reservoir_size, reservoir_size)
		self.fc_out = nn.Linear(reservoir_size, output_size)

	def forward(self, x, n_steps):

		x = F.relu(self.fc_in(x))
		for i in range(n_steps):
			x = F.relu(self.reservoir(x))
		x = self.fc_out(x)

		# add weight norm so that the weights doesn't explode

		return x