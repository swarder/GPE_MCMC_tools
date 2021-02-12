import numpy as np
from matplotlib import pyplot as plt
import scipy.optimize

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

class mcmc():
	"""Class implementing simple Markov Chain Monte Carlo algorithm"""

	def __init__(self, log_posterior, observations, step_sizes, initial_guess, **lp_kwargs):
		self.log_posterior = log_posterior
		self.observations = np.array(observations)
		self.step_sizes = np.array(step_sizes)
		self.input_params = np.array(initial_guess)
		self.dimensions = len(self.step_sizes)
		self.covariance = np.diag(self.step_sizes)**2
		self.lp_kwargs = lp_kwargs
		
	def propose_next_params(self, input_params=None):
		"""
		Proposed next sample from parameter space based on random walk from current parameters
		"""
		if input_params is None:
			input_params = self.input_params
		random_step_params = np.random.multivariate_normal(np.zeros(self.dimensions), self.covariance)
		#print(input_params)
		return input_params + random_step_params

	def iteration_step(self):
		"""
		Perform one iteration of RWMH algorithm
		"""
		current_params = self.input_params
		proposed_params = self.propose_next_params(current_params)

		current_log_posterior = self.log_posterior(current_params, self.observations, **self.lp_kwargs)
		proposed_log_posterior = self.log_posterior(proposed_params, self.observations, **self.lp_kwargs)
		p_accept = min(1, np.exp(proposed_log_posterior - current_log_posterior))

		u = np.random.uniform(0, 1)

		if u < p_accept:
			self.input_params = proposed_params

		return self.input_params

	def run_algorithm(self, iterations=1000, burnin=0, print_progress_interval=-1):
		"""
		Run RWMH algorithm
		"""
		param_values = []
		for i in range(iterations+burnin):
			if print_progress_interval > 0 and i % print_progress_interval == 0:
				print(i)
			self.iteration_step()
			param_values.append(self.input_params)
		return np.array(param_values)[burnin:]

	def maximise_likelihood(self, initial_guess, bounds):
		"""
		Perform maximum likelihood estimation
		"""
		def NLL(params):
			input_params = params
			return -self.log_posterior(input_params, self.observations, **self.lp_kwargs)
		return scipy.optimize.minimize(NLL, initial_guess, method='SLSQP', bounds=bounds, options={'disp':True, 'ftol':1e-8})
