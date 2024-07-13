import torch
import pyro.distributions as dist

def empirical_gamma_cdf(x: torch.tensor, 
                        shape: torch.tensor, 
                        rate: torch.tensor) -> torch.tensor:
    '''
    Calculate the empirical cumulative distribution function of a gamma distribution.
    
    INPUTS:
    - x (torch.tensor): The value at which to calculate the CDF.
    - shape (torch.tensor): The shape parameter of the gamma distribution.
    - rate (torch.tensor): The rate parameter of the gamma distribution.
    
    OUTPUTS:
    - (torch.tensor): The empirical CDF of the gamma distribution at x.
    '''
    # Generate 1000 random samples from the gamma distribution
    # Think about setting a seed to avoid too much stochasticity
    #torch.manual_seed(3407)
    samples = torch.distributions.Gamma(shape, rate).sample((3500,))
    return (samples <= x).float().mean()

def copula_term_log(theta: torch.tensor, 
                    u: torch.tensor, 
                    v: torch.tensor) -> torch.tensor:
    '''
    Calculate the log of the copula term for a bivariate gamma distribution.

    INPUTS:
    - theta (torch.tensor): The theta parameter of the copula.
    - u (torch.tensor): The empirical CDF of the first gamma distribution.
    - v (torch.tensor): The empirical CDF of the second gamma distribution.

    OUTPUTS:
    - (torch.tensor): The log of the copula term.
    '''
    log_numerator = torch.log(theta) + torch.log(torch.exp(theta) - 1.0) + theta * (1.0 + u + v)
    denominator = (torch.exp(theta) - torch.exp(theta + theta * u) + torch.exp(theta * (u + v)) - torch.exp(theta + theta * v))**2
    log_denominator = torch.log(denominator)
    return log_numerator - log_denominator

def copulamodel_log_pdf(x: torch.tensor,
                        y: torch.tensor,
                        shape1: torch.tensor,
                        rate1: torch.tensor,
                        shape2: torch.tensor,
                        rate2: torch.tensor,
                        theta: torch.tensor) -> torch.tensor:
    '''
    Calculate the log of the probability density function of a bivariate gamma distribution with copula.
    
    INPUTS:
    - x (torch.tensor): The first observation.
    - y (torch.tensor): The second observation.
    - shape1 (torch.tensor): The shape parameter of the first gamma distribution.
    - rate1 (torch.tensor): The rate parameter of the first gamma distribution.
    - shape2 (torch.tensor): The shape parameter of the second gamma distribution.
    - rate2 (torch.tensor): The rate parameter of the second gamma distribution.
    - theta (torch.tensor): The theta parameter of the copula.
    
    OUTPUTS:
    - (torch.tensor): The log of the probability density function of the bivariate gamma distribution with copula.
    '''
    g1_lpdf= dist.Gamma(shape1,rate1).log_prob(x)
    g2_lpdf= dist.Gamma(shape2,rate2).log_prob(y)
    u= empirical_gamma_cdf(x, shape1, rate1)
    v= empirical_gamma_cdf(y, shape2, rate2)
    # Qui pensare se fare una if su u e v diversi da circa 0...in quel caso non calcolare il copula term (lo lascio nullo)
    lpdf=g1_lpdf+g2_lpdf
    if (torch.abs(u) > 1e-6) & (torch.abs(v) > 1e-6):
        lpdf += copula_term_log(theta=theta,u=u,v=v)
    return lpdf