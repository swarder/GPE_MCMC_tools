import unittest
import numpy as np
import scipy.stats

from gpe_mcmc_tools import priors, mcmc

from contextlib import contextmanager
import sys, os

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

np.random.seed(0)

class TestPriors(unittest.TestCase):

    def test_flat_prior(self):
        flat_prior = priors.FlatPrior()
        self.assertEqual(flat_prior(10.0), 1)
        self.assertEqual(flat_prior(0), 1)
        self.assertEqual(flat_prior(-2.3), 1)

        flat_prior = priors.FlatPrior(log=True)
        self.assertEqual(flat_prior(10.0), 0)
        self.assertEqual(flat_prior(0), 0)
        self.assertEqual(flat_prior(-2.3), 0)

    def test_uniform_prior(self):
        uniform_prior = priors.UniformPrior(2, 5)
        self.assertEqual(uniform_prior(4), 1/3)
        self.assertEqual(uniform_prior(6), 0)
        self.assertEqual(uniform_prior(2), 1/3)

        uniform_prior = priors.UniformPrior(0, 10, log=True)
        self.assertEqual(uniform_prior(5), np.log(0.1))
        self.assertEqual(uniform_prior(0), np.log(0.1))
        self.assertEqual(uniform_prior(-2.0), -np.inf)

    def test_gaussian_prior(self):
        gaussian_prior = priors.GaussianPrior(5, 2.5)
        scipy_gaussian = scipy.stats.norm(5, 2.5).pdf
        self.assertAlmostEqual(gaussian_prior(7.5), scipy_gaussian(7.5))
        self.assertAlmostEqual(gaussian_prior(20), scipy_gaussian(20))
        self.assertAlmostEqual(gaussian_prior(-4.7), scipy_gaussian(-4.7))

        gaussian_prior = priors.GaussianPrior(0, 1, log=True)
        scipy_gaussian = scipy.stats.norm(0, 1).pdf
        self.assertAlmostEqual(gaussian_prior(1), np.log(scipy_gaussian(1)))
        self.assertAlmostEqual(gaussian_prior(5.0), np.log(scipy_gaussian(5.0)))
        self.assertAlmostEqual(gaussian_prior(-3.3), np.log(scipy_gaussian(-3.3)))

    def test_jeffreys_prior(self):
        jeffreys_prior = priors.JeffreysPrior()
        self.assertEqual(jeffreys_prior(0), 0)
        self.assertEqual(jeffreys_prior(-1), 0)
        self.assertEqual(jeffreys_prior(1), 1)
        self.assertEqual(jeffreys_prior(2), 1/4)

        jeffreys_prior = priors.JeffreysPrior(log=True)
        self.assertEqual(jeffreys_prior(0), -np.inf)
        self.assertEqual(jeffreys_prior(-1), -np.inf)
        self.assertEqual(jeffreys_prior(2.5), -np.log(2.5**2))
        self.assertEqual(jeffreys_prior(10), np.log(1/100))


class TestMCMC(unittest.TestCase):

    def test_mle_1d(self):

        test_mean = 0.0
        test_std = 1.0
        initial_guess = 2.0

        def log_posterior(parameters, observations):
            return np.log(scipy.stats.norm(test_mean, test_std).pdf(parameters[0]))
        
        mcmc_tester = mcmc.mcmc(log_posterior, [], [0.01], [initial_guess])
        with suppress_stdout():
            mcmc_max_likelihood = mcmc_tester.maximise_likelihood([initial_guess], [[-10, 10]]).x
        
        self.assertAlmostEqual(mcmc_max_likelihood[0], test_mean)
    
    def test_mle_2d(self):
        
        test_means = [-0.5, 1.0]
        test_stds = [1.0, 2.0]
        initial_guess = [2.0, -4.5]
        
        def log_posterior(parameters, observations):
            return np.sum([np.log(scipy.stats.norm(test_means[i], test_stds[i]).pdf(parameters[i])) for i in range(len(test_means))])

        mcmc_tester = mcmc.mcmc(log_posterior, [], [0.01], [initial_guess])
        with suppress_stdout():
            mcmc_max_likelihood = mcmc_tester.maximise_likelihood(initial_guess, [[-10, 10]]*len(test_means)).x

        for i in range(len(test_means)):
            self.assertAlmostEqual(mcmc_max_likelihood[i], test_means[i], places=5)

    def test_mcmc_1d(self):

        test_mean = 0.0
        test_std = 1.0
        initial_guess = test_mean

        def log_posterior(parameters, observations):
            #return np.log(scipy.stats.norm(test_mean, test_std).pdf(parameters[0]))
            return -0.5*parameters[0]**2

        def posterior(x):
            return np.exp(-0.5*x**2)
        
        mcmc_tester = mcmc.mcmc(log_posterior, [], [1], [initial_guess])

        param_values = mcmc_tester.run_algorithm(burnin=0, iterations=10000, print_progress_interval=-1)

        # Check mean and standard deviation of MCMC parameter values are within suitable range
        self.assertTrue(np.abs(np.mean(param_values) - test_mean) < 0.2)
        self.assertTrue(np.abs(np.std(param_values) - test_std) < 0.1)

if __name__ == '__main__':
    unittest.main()