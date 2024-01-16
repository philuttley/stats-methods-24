---
title: >-
   MCMC for model-fitting and error estimation
teaching: 60
exercises: 60
questions:
- "How can we use MCMC methods to fit data and obtain MLEs and confidence intervals for models which may have many parameters, non-normal errors or complex posterior distributions?"
objectives:
- "Understand the application and implementation of MCMC methods using `emcee`."
- "Learn how to set up and run MCMC simulations for parameter estimation."
- "Interpret the results of MCMC simulations for confidence interval estimation."
keypoints:
- 
---

<script src="../code/math-code.js"></script>
<!-- Just one possible MathJax CDN below. You may use others. -->
<script async src="//mathjax.rstudio.com/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

## Understanding the Metropolis-Hastings MCMC Algorithm

The Metropolis-Hastings algorithm is a cornerstone of computational physics and astronomy, particularly in the context of Bayesian inference and statistical sampling. Its primary objective is to generate a sequence of sample points from a probability distribution, especially when direct sampling is challenging.

#### The Core Idea

At its core, Metropolis-Hastings is a stochastic process that constructs a Markov chain. A Markov chain is a sequence of random variables where the probability of each variable depends only on the state attained in the previous step, not on the full path that led to that state.

#### Algorithm Steps

1. **Initialization**: Start with an arbitrary point $$x_0$$ in the parameter space.

2. **Iteration**: For each step $$i$$, perform the following:

   a. **Proposal**: Generate a new candidate point $$x'$$ sampled from a proposal distribution $$q(x'|x_i)$$. This distribution is often chosen to be symmetric (e.g. a normal distribution centered on $$x_i$$) but doesn't have to be.

   b. **Acceptance Criterion**: Calculate the acceptance probability $$\alpha$$ given by:

      
$$\alpha(x'|x_i) = \min \left(1,\frac{p(x')q(x_i|x')}{p(x_i)q(x'|x_i)}\right)$$

      
      where $$p(x)$$ is the target distribution we want to sample from.

   c. **Accept or Reject**: Draw a random number $$u$$ from a uniform distribution over $$\[0, 1\]$$. If $$u \leq \alpha$$, accept $$x'$$ as the next point in the chain (set $$x_{i+1} = x'$$). Otherwise, reject $$x'$$ and set $$x_{i+1} = x_i$$.

3. **Convergence**: Repeat step 2 until the chain reaches a stationary distribution. The number of iterations required depends on the problem and the chosen proposal distribution.



## Easy Markov Chain Monte Carlo with emcee

In this tutorial we will learn how to use the `emcee` Markov Chain Monte Carlo (MCMC) Python module to obtain confidence intervals for a multi-parameter model fit. The approach is based on the example of fitting models to data, given on the `emcee` website: http://dfm.io/emcee/current/user/line/.  You should have already installed `emcee` in your Python environment (e.g. using `pip install emcee`) before you start.  Also, you should install the handy `corner` plotting module.
```python
import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
import scipy.optimize as spopt
import scipy.integrate as spint
%matplotlib notebook
```

Now we will set up the MCMC.  We first need to set up the starting positions of the 'walkers' in the 4-dimensional parameter space, which will roam the likelihood surface and (after some burn-in time) map the parameter distributions.  To do this, we follow the approach of the `emcee` example and generate the starting positions from narrow normal distributions centred on the parameter MLEs.  For the standard deviation of the distributions we can use a scaled version of the error estimates already determined from the MLEs via the covariance.  This should ensure a fairly rapid burn-in time, as the positions of the walkers are well-matched to the estimated shapes of the parameter distributions.
```python
### Import the scipy optimisation package
import scipy.optimize as op
### Now import emcee and corner
import emcee
import corner

### Now let's define our likelihood functions.
### Note that this variant of the LogLikelihood function does not take the negative of 
### log-likelihood (the -ve log-likelihood was used previously so as to work with the 
### minimisation approach of Python's optimisers, but emcee works by assuming that it is given
### the standard log-likelihood).
### These functions also read in the name of the model as a parameter my_model, so they can be
### used generically without having to always used the same model name.
      

```

Now we are ready to run the MCMC 'sampler' which lets the walkers map the likelihood surface.  But first we will make our approach more Bayesian by incorporating a prior in the probability estimate (see description of the method in the lecture, i.e. we are converting the likelihood into $\pi(\theta)$.  The prior can be used to constrain the ranges of allowed parameter values to those that are previously determined from different data sets, or it could also be neglected (this is equivalent to assuming a uniform prior from $\infty$ to $\infty$, i.e. no preference at all about what values the prior should take).  Priors may also follow other distributions: read a Bayesian statistics book for more details.  The prior should $not$ be used to constrain parameters such that they cut off the true parameter distributions (i.e. don't use prior boundaries that are comparable to a few times the parameter error bar or less).  If in doubt, don't specify a prior.
```python
## Now lets load our data
#  First read in the data.  This is a simple (single-column) list of energies:
#photens = np.genfromtxt('faint_photons.txt')
photens = np.genfromtxt('photon_energies.txt')

# Now we make our unbinned histogram.  We can keep the initial number of bins relatively large.
emin, emax = 10., 200.   # We should always use the known values that the data are sampled over 
                         # for the range used for the bins!
nbins = 50
counts, edges = np.histogram(photens, bins=nbins, range=[emin,emax], density=False)

bwidths = np.diff(edges) # calculates the width of each bin
cdens = counts/bwidths # determines the count densities
cdens_err = np.sqrt(counts)/bwidths # calculate the errors: remember the error is based on the counts, 
# not the count density, so we have to also apply the same normalisation.
energies = (edges[:-1]+edges[1:])/2.  # This calculates the energy bin centres
# Now plot the data - use a log-log scale since we are plotting a power-law
plt.figure()
plt.errorbar(energies, cdens, xerr=bwidths/2., yerr=cdens_err, fmt='o', ms=5)
plt.xlabel("Energy (GeV)", fontsize=14)
plt.ylabel("Counts/GeV", fontsize=14)
plt.tick_params(labelsize=14)
plt.yscale('linear')
plt.xscale('log')
plt.xlim(10.0,200.0)
plt.show()            
```

Now we will run the sampler to sample our `llwithprior` function (if not using a prior you could also call `LogLikelihood2` directly.  We will generate 500 samples for each walker.
```python
def pl_model(x, parm):
    '''Simple power-law function.
       Inputs:
           x - input x value(s) (can be list or single value).
           parm - parameters, list of PL normalisation (at x = 1) and power-law index.'''
    pl_norm = parm[0]  # here the function given means that the normalisation corresponds to that at a value 1.0
    pl_index = parm[1]
    return pl_norm * x**pl_index

def LogLikelihood_Pois_Integ(parm, model, ebins, counts): 
    '''Calculate the negative Poisson log-likelihood for a model integrated over bins. 
       Inputs:
           parm - model parameter list.
           model - model function name.
           ebins, counts - energy bin edges and corresponding counts per bin.
        Outputs: the negative Poisson log-likelihood'''
    i = 0
    ymod = np.zeros(len(counts))
    for energy in ebins[:-1]:
        ymod[i], ymoderr = spint.quad(lambda x: model(x, parm),ebins[i],ebins[i+1])
        # we don't normalise by bin width since the rate parameter is set by the model and needs to be 
        # in counts per bin
        i=i+1        
    pd = sps.poisson(ymod) #we define our Poisson distribution
#    print(parm,-1*sum(pd.logpmf(counts)))
    return -1*sum(pd.logpmf(counts))

parm = [2500.0, -1.5]
# We use our original counts bins.  Also, remember that the Likelihood function we have just defined uses 
# counts rather than count density
result = spopt.minimize(LogLikelihood_Pois_Integ, parm, args=(pl_model, edges, counts), method='BFGS')

err = np.sqrt(np.diag(result.hess_inv))
print("Covariance matrix:",result.hess_inv)
print("Normalisation at 1 GeV = " + str(result.x[0]) + " +/- " + str(err[0]))
print("Power-law index = " + str(result.x[1]) + " +/- " + str(err[1]))
print("Maximum log-likelihood = " + str(-1.0*result.fun))
```

Hopefully the sampler ran in a reasonable time (and note that `emcee` can also run the walkers in parallel on different cores, for a faster run-time).  How do you know if the sampler is working okay?  We can plot the time-evolution of the parameters sampled by the walkers, to see if, after some burn-in time, they each converge on some stationary (not changing with time) distribution which is (hopefully) the parameter distribution.  The output of the sampler is in the array sampler.chain, which has 3 axes: `[walker, number-of-samples, parameter]`.
```python
ndim, nwalkers = 2, 100  # The number of parameters and the number of walkers (we follow the 
# emcee example and use 100)
pos = [result.x + 0.01*result.x*sps.norm.rvs(size=ndim) for i in range(nwalkers)]  # we spread the 
# walkers around the MLEs with a standard deviation equal to 1 per cent of the previously 
# estimated MLE standard deviation
```

The distributions appear to settle down to a constant width within about 50 steps, so we can assume that this is the burn-in time.  For completeness, we note that to estimate the burn-in more formally you could calculate the auto-correlation functions of the parameter time-series and determine for how many steps they reach zero. 

Now we can use the `corner` module (see http://corner.readthedocs.io/en/latest/ for documentation) to plot a handy compilation of histogram and contour plots determined from our samples.  We discard the first 50 samples for each walker chain of samples, in order to avoid the burn-in region which will distort our results.  We use the default settings for `corner`, and also include as lines/crosses the values of the MLEs obtained from `curve_fit`, for comparison with the sampled distributions.
```python
# Our prior is uniform and just forces the gradients and break value to be positive.
def lnprior(parm):
    N, gamma = parm
    if (0.0 < N < np.inf) & (-4 <= gamma <= 0):
        return 0.0
    return -np.inf
#    return -1000.0

def llwithprior(parm, model, ebins, counts):
    lp = lnprior(parm)
    if not np.isfinite(lp):
        return -np.inf
    return lp-LogLikelihood_Pois_Integ(parm, model, ebins, counts)

```

Finally, we can use the `percentile` function in `numpy` to output the percentiles corresponding to the median and 68 per cent ($\sim 1 \sigma$) confidence intervals for each parameter, which we can compare with the estimated MLEs and their errors, obtained from curve_fit.  You should be able to see that the match is quite good!
```python
model = pl_model
sampler = emcee.EnsembleSampler(nwalkers, ndim, llwithprior, args=(model, edges, counts))
sampler.run_mcmc(pos, 1000)
```

You can see that the median values of the sampled distributions are a good match to the MLEs. In principle then, one does not need to already know the MLEs in order to find MCMC estimates of them and their confidence intervals. Knowing them in advance is useful to speed up the process, because the burn-in times are reduced.  But with sensible choices for the walker positions and reasonable priors it should be possible to use MCMC for the whole fitting process, if desired (and indeed, this may be necessary when the likelihood surface is to complicated for a standard optimiser to work).

Based on the approach in this tutorial, and with suitable background reading where appropriate, you should be able to apply `emcee` to fit and or map confidence intervals for many other data sets and models, by changing the model and the likelihood function (and prior) as appropriate.
```python
nsteps = 500  # Plots the first 400 samples for each walker - we should plot more if burn-in 
# is still not clearly reached
fig = plt.figure()
fig.clf()
for j in range(ndim):
    ax = fig.add_subplot(ndim,1,j+1)
    ax.plot(np.array([sampler.chain[:,i,j] for i in range(nsteps)]),"k", alpha = 0.3)
    ax.set_ylabel((r'$a$',r'$b_1$',r'$b_2$',r'$bk$')[j], fontsize = 25)
plt.xlabel('Steps', fontsize = 15)
fig.show()
```
