"""
Wrapper for GPy, making it easier to quickly construct single- or multi-output emulators from training input and output data
"""

import GPy
import numpy as np

class GPE:
	def __init__(self, training_inputs, training_outputs, emulator_type):
		"""
		training_inputs: array of shape (N x m), where N is the number of training samples and m is the number of input dimensions
		training_outputs: array of shape (N x p), where p is the number of output dimensions
		emulator_type: either 'single' or 'multi'
		multi takes much longer to train, but might be more accurate
		If covariance between outputs is important, use multi
		"""
		assert emulator_type in ['single', 'multi']
		self.training_inputs = training_inputs
		self.training_outputs = training_outputs
		self.emulator_type = emulator_type

		self.N, self.m = self.training_inputs.shape
		self.p = self.training_outputs.shape[1]

		self.train()

	def train(self):
		if self.emulator_type == 'single':
			self.train_single()
		else:
			self.train_multi()

	def train_single(self):
		"""
		Train one emulator per output
		"""
		all_emulators = []
		for i in range(self.p):
			kernel = GPy.kern.RBF(input_dim=self.m)
			gpe = GPy.models.GPRegression(self.training_inputs, self.training_outputs[:,i].reshape(-1,1), kernel)
			gpe.optimize()
			#gpe.optimize_restarts(num_restarts=10, parallel=4)
			all_emulators.append(gpe)
		self.emulators = all_emulators

	def train_multi(self):
		"""
		Train one multi-output emulator
		"""
		icm = GPy.util.multioutput.ICM(input_dim=self.m,num_outputs=self.p,kernel=GPy.kern.Matern52(self.m,ARD=True))
		gpe = GPy.models.GPCoregionalizedRegression([self.training_inputs for i in range(self.p)], [self.training_outputs[:,i:i+1] for i in range(self.p)], kernel=icm)
		gpe['ICM.Mat52.variance'].constrain_fixed(1.)
		gpe.optimize()
		self.emulator = gpe

	def run_single(self, inputs, cov=False):
		"""
		Run all emulators
		"""
		assert len(inputs) == self.m
		
		means = []
		covs = []
		for i in range(self.p):
			Xnew = np.array(list(inputs) + [0]).reshape(1,-1)
			Y_metadata = {'output_index': Xnew[:,self.m:].astype(int)}
			mu, c = self.emulators[i].predict(Xnew=Xnew, Y_metadata=Y_metadata, full_cov=cov)
			means.append(mu)
			covs.append(c)
		if cov:
			return np.array(means).flatten(), np.array(covs).flatten()
		else:
			return np.array(means).flatten()

	def run_multi(self, inputs, cov=False):
		"""
		Run multi-output emulator
		"""
		assert len(inputs) == self.m
		
		newx = np.concatenate([np.stack([inputs for _ in range(self.p)], axis=0), np.arange(self.p).reshape(-1,1)], axis=1)
		noise_dict = {'output_index': newx[:,self.m:].astype(int)}
		emulator_means, emulator_covariance = self.emulator.predict(Xnew=newx, Y_metadata=noise_dict, full_cov=True)
		diag_covariance = np.diag(emulator_covariance)

		if cov:
			return emulator_means.flatten(), diag_covariance
		else:
			return emulator_means.flatten()

	def run(self, *args, **kwargs):
		if self.emulator_type == 'single':
			return self.run_single(*args, **kwargs)
		else:
			return self.run_multi(*args, **kwargs)

	def __call__(self, *args, **kwargs):
		return self.run(*args, **kwargs)