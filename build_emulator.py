import GPy
import bet.sample as samp
import numpy as np
from matplotlib import pyplot as plt
import pickle
import sys
import zlib

default_gauge_names = ['LERWICK', 'WICK', 'KINLOCHBERVIE', 'STORNOWAY', 'ABERDEEN', 'TOBERMORY', 'LEITH', 'MILLPORT', 'PORTRUSH', 'PORTPATRICK', 'BANGOR', 'WORKINGTON', 'TYNE', 'PORT', 'HEYSHAM', 'WHITBY_2', 'HOLYHEAD', 'LLANDUDNO', 'IMMINGHAM_2', 'BARMOUTH', 'FISHGUARD', 'MILFORD_2', 'MUMBLES', 'NEWPORT', 'AVONMOUTH', 'HINKLEY', 'ILFRACOMBE', 'NEWLYN', 'ST.', 'DEVONPORT', 'SAINT', 'PORTSMOUTH', 'NEWHAVEN', 'DOVER', 'SHEERNESS', 'HARWICH', 'LOWESTOFT', 'CROMER']

#data_gauge_names = np.loadtxt('../observation_data/bodc/all_bodc_locations.txt', usecols=[0], dtype=str).tolist()

class SimpleEmulator:

	def __init__(self, constituent, obs_type, training_dir, gauge_names=None, emulator_name=None):
		self.constituent = constituent
		self.obs_type = obs_type
		self.training_dir = training_dir
		if gauge_names is None:
			gauge_names = default_gauge_names
		self.gauge_names = gauge_names
		if emulator_name is None:
			emulator_name = training_dir
		self.emulator_name = emulator_name
		self.load_training_data()
		self.input_dimension = self.training_inputs.shape[1]
		self.pickle_name = zlib.adler32(str.encode(''.join([self.constituent, self.obs_type, *self.gauge_names, self.emulator_name])))
		try:
			self.load_pickled_emulator()
		except FileNotFoundError:
			self.build_emulator()

	def load_training_data(self):
		training_input_samples = samp.load_sample_set('{}/lhs_samples_object'.format(self.training_dir))
		training_inputs = training_input_samples.get_values()
		training_samples = list(range(1, training_inputs.shape[0]+1))

		training_outputs = np.empty((training_inputs.shape[0], len(self.gauge_names)))

		training_available_gauge_names = np.loadtxt('{}/harmonic_analysis/training_gauges.txt'.format(self.training_dir), dtype=str).tolist()

		original_analysis_constituents = ['M2', 'S2', 'K1', 'O1']

		for si in training_samples:
			training_data = np.loadtxt('{}/harmonic_analysis/training_{:02d}_{}.txt'.format(self.training_dir, si, 'amps' if self.obs_type == 'amp' else 'phases'))
			training_outputs[si-1,:] = training_data[[training_available_gauge_names.index(g) for g in self.gauge_names], original_analysis_constituents.index(self.constituent)]

		if self.obs_type == 'phase':
			# Ensure that emulators are smoothly varying
			training_outputs[1:,:][training_outputs[1:,:] - training_outputs[0,:] >  np.pi] -= 2*np.pi
			training_outputs[1:,:][training_outputs[1:,:] - training_outputs[0,:] < -np.pi] += 2*np.pi

		self.training_inputs = training_inputs
		self.training_outputs = training_outputs

	def load_pickled_emulator(self):
		self.emulator = pickle.load(open('pickled_emulators/{}'.format(self.pickle_name), 'rb'))
		return self.emulator

	def build_emulator(self):
		m = self.input_dimension
		N = len(self.gauge_names)
		icm = GPy.util.multioutput.ICM(input_dim=m,num_outputs=N,kernel=GPy.kern.Matern52(m,ARD=True))
		gpe = GPy.models.GPCoregionalizedRegression([self.training_inputs for i in range(N)], [self.training_outputs[:,i:i+1] for i in range(N)], kernel=icm)
		gpe['ICM.Mat52.variance'].constrain_fixed(1.)
		gpe.optimize()
		pickle.dump(gpe, open('pickled_emulators/{}'.format(self.pickle_name), 'wb'))
		self.emulator = gpe

	def run_emulator(self, inputs, use_gauges=None):
		assert len(inputs) == self.input_dimension
		newx = np.concatenate([np.stack([inputs for _ in range(len(self.gauge_names))], axis=0), np.arange(len(self.gauge_names)).reshape(-1,1)], axis=1)
		noise_dict = {'output_index': newx[:,self.input_dimension:].astype(int)}
		emulator_means, emulator_covariance = self.emulator.predict(Xnew=newx, Y_metadata=noise_dict, full_cov=True)
		if use_gauges:
			use_gauges_ids = [self.gauge_names.index[g] for g in use_gauges]
			emulator_means = emulator_means[use_gauges_ids]
		return emulator_means

	def __call__(self, *args, **kwargs):
		return self.run_emulator(*args, **kwargs)


class CompositeEmulator:

	def __init__(self, constituents, training_dir, obs_type='ampphase', gauge_names=None):
		if obs_type != 'ampphase':
			raise NotImplementedError
		self.constituents = constituents
		self.training_dir = training_dir
		if gauge_names is None:
			gauge_names = default_gauge_names
		self.gauge_names = gauge_names
		self.amp_emulators = [SimpleEmulator(c, 'amp', training_dir, gauge_names=gauge_names) for c in constituents]
		self.phase_emulators = [SimpleEmulator(c, 'phase', training_dir, gauge_names=gauge_names) for c in constituents]
		self.input_dimension = self.amp_emulators[0].input_dimension

	def run_emulator(self, inputs, use_gauges=None):
		amp_means = []
		phase_means = []
		for i in range(len(self.constituents)):
			amp_means.append(self.amp_emulators[i](inputs, use_gauges).flatten())
			phase_means.append(self.phase_emulators[i](inputs, use_gauges).flatten())
		return np.array(amp_means).transpose(), np.array(phase_means).transpose()

	def __call__(self, *args, **kwargs):
		return self.run_emulator(*args, **kwargs)
