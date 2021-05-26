import numpy as np

class Prior:
	""" Base class for priors """
	
	def __init__(self, log=False):
		self.log = log

	def evaluate(self, x):
		""" Evaluate prior in non-log space"""
		pass

	def evaluate_log(self, x):
		"""Evaluate prior in log space"""
		p = self.evaluate(x)
		if p == 0:
			return -np.inf
		else:
			return np.log(p)

	def __call__(self, x):
		"""Call relevant evaluation function"""
		if self.log:
			return self.evaluate_log(x)
		else:
			return self.evaluate(x)

class FlatPrior(Prior):
	"""Flat prior, always returns 1"""

	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def evaluate(self, x):
		return 1


class UniformPrior(Prior):
	"""Uniform prior between specified min and max values"""

	def __init__(self, min_val, max_val, **kwargs):
		super().__init__(**kwargs)
		self.min_val = min_val
		self.max_val = max_val

	def evaluate(self, x):
		if x >= self.min_val and x <= self.max_val:
			return 1/(self.max_val - self.min_val)
		else:
			return 0


class GaussianPrior(Prior):
	"""Gaussian prior for specified mean and standard deviation"""

	def __init__(self, mu, sigma, **kwargs):
		super().__init__(**kwargs)
		self.mu = mu
		self.sigma = sigma

	def evaluate(self, x):
		return np.exp(-0.5 * (x - self.mu)**2 / self.sigma**2) / (self.sigma * np.sqrt(2*np.pi))

	def evaluate_log(self, x):
		return -0.5 * (x - self.mu)**2 / self.sigma**2 - np.log(self.sigma * np.sqrt(2*np.pi))


class JeffreysPrior(Prior):
	"""Jeffreys prior, to be used for hyperparameters"""

	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def evaluate(self, x):
		if x > 0:
			return 1/x**2
		else:
			return 0
