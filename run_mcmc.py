import build_emulator
import priors
import mcmc
import numpy as np
from matplotlib import pyplot as plt

def load_observation_data(gpe_constituents, gauge_names):
	tidegauge_file = '../observation_data/filtered_gauges.csv'
	obs_constituents = ['M2', 'S2']
	gauge_data_file_constituent_ids = [25, 31]
	training_constituent_IDs = [obs_constituents.index(tc) for tc in gpe_constituents]
	obs_gauge_names = np.loadtxt(tidegauge_file, skiprows=1, usecols=(0,), dtype=str, delimiter=',').tolist()
	outputs_amp = np.empty((len(gauge_names), len(gpe_constituents)))
	outputs_phase = np.empty((len(gauge_names), len(gpe_constituents)))
	for i, tc in enumerate(training_constituent_IDs):
		gauge_harmonics_data_amp = np.loadtxt(tidegauge_file, skiprows=1, usecols=(gauge_data_file_constituent_ids[tc],), delimiter=',')
		gauge_harmonics_data_phase = np.loadtxt(tidegauge_file, skiprows=1, usecols=(gauge_data_file_constituent_ids[tc]+1,), delimiter=',')*np.pi/180
		for j, g in enumerate(gauge_names):
			gi = obs_gauge_names.index(g)
			outputs_amp[j, i] = gauge_harmonics_data_amp[gi]
			outputs_phase[j, i] = gauge_harmonics_data_phase[gi]
	return outputs_amp, outputs_phase


def simple_log_likelihood(emulator_output, sigma_squared, observations):
		"""Compute log likelihood for a particular set of observations (either amp or phase, for one constituent)"""
		assert emulator_output.shape == observations.shape
		N = emulator_output.shape[0]
		covariance_matrix = np.diag(sigma_squared*np.ones(N))
		covariance_matrix_inv = np.diag(1/sigma_squared*np.ones(N))
		misfit = emulator_output - observations
		return np.log((2*np.pi)**(-N/2) * np.linalg.det(covariance_matrix)**(-0.5)) - 0.5*np.dot(misfit.transpose(), np.dot(covariance_matrix_inv, misfit))

def log_posterior(theta, observations, emulator, log_priors):
		ns = theta[:emulator.input_dimension]
		log_sigma_squareds = theta[emulator.input_dimension:]
		sigma_squareds = np.exp(log_sigma_squareds)

		emulator_outputs_amp, emulator_outputs_phase = emulator.run_emulator(ns)
		emulator_outputs_list = [emulator_outputs_amp[:,0], emulator_outputs_amp[:,1], emulator_outputs_phase[:,0], emulator_outputs_phase[:,1]]

		log_prior_ns = np.sum([logprior(n) for logprior, n in zip(log_priors, ns)])
		log_prior_sigmas = np.sum([logprior(ss) for logprior, ss in zip(log_priors, sigma_squareds)])

		log_likelihood = 0
		for e_outputs, ss, obs in zip(emulator_outputs_list, sigma_squareds, observations):
			log_likelihood += simple_log_likelihood(e_outputs, ss, obs)

		return log_likelihood + log_prior_ns + log_prior_sigmas


def build_and_run_emulator(emulator_id, burnin, iterations):
	########################################
	# Build/load emulator
	########################################
	emulator = build_emulator.CompositeEmulator(constituents=['M2', 'S2'], training_dir='../emulators/{}'.format(emulator_id))
	
	########################################
	# Load observation data
	########################################
	outputs_amp, outputs_phase = load_observation_data(emulator.constituents, emulator.gauge_names)
	outputs_list = [outputs_amp[:,0], outputs_amp[:,1], outputs_phase[:,0], outputs_phase[:,1]]
	
	########################################
	# Construct priors
	########################################
	n_log_priors = [priors.FlatPrior(log=True)]*emulator.input_dimension
	sigma_log_priors = [priors.JeffreysPrior(log=True)]*2*len(emulator.constituents)
	log_priors = n_log_priors + sigma_log_priors

	########################################
	# Define MCMC parameters
	########################################
	step_sizes = [1e-3]*emulator.input_dimension + [1e-1]*len(outputs_list)
	initial_guess = [0.025]*emulator.input_dimension + [-2]*len(outputs_list)
	
	########################################
	# Prepare MCMC
	########################################
	my_mcmc = mcmc.mcmc(log_posterior=log_posterior, observations=outputs_list, step_sizes=step_sizes, initial_guess=initial_guess, emulator=emulator, log_priors=log_priors)

	########################################
	# Perform Maximum Likelihood Estimation
	########################################
	bounds = [[0.01, 0.05]]*emulator.input_dimension + [[-10, 5]]*len(outputs_list)
	mle = my_mcmc.maximise_likelihood(initial_guess=initial_guess, bounds=bounds)
	assert mle.success
	mle_params = mle.x
	print(mle_params)

	########################################
	# Run MCMC
	########################################
	my_mcmc = mcmc.mcmc(log_posterior=log_posterior, observations=outputs_list, step_sizes=step_sizes, initial_guess=initial_guess, emulator=emulator, log_priors=log_priors)
	mcmc_chain = my_mcmc.run_algorithm(burnin=1000, iterations=1000, print_progress_interval=500)
	print(np.mean(mcmc_chain, axis=0))
	params_chain = mcmc_chain[:,:emulator.input_dimension]
	sigmas_chain = mcmc_chain[:,emulator.input_dimension:]
	return params_chain, sigmas_chain

if __name__ == '__main__':
	build_and_run_emulator('03', 1000, 9000)