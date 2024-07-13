import numpy as np
import torch
import pyro
from pyro import poutine
import pyro.distributions as dist
from pyro.infer import SVI, TraceEnum_ELBO
from pyro.infer.autoguide import AutoDelta
from pyro.optim import Adam
import pandas as pd
import tqdm

class BivariateHMM:

    def __init__(self, hidden_states,
                 probs_initial=None,
                 probs_x=None,
                 probs_alpha1=None,
                 probs_beta1=None,
                 probs_alpha2=None,
                 probs_beta2=None):
        self.hidden_states = hidden_states
        self.probs_initial = probs_initial
        self.probs_x = probs_x
        self.probs_alpha1 = probs_alpha1
        self.probs_beta1 = probs_beta1
        self.probs_alpha2 = probs_alpha2
        self.probs_beta2 = probs_beta2
        self.posterior = None
        self.guide = None
        self.elbo = None
        self.svi = None

    @classmethod
    def from_posterior(cls, posterior: dict):
        return cls(
            hidden_states=posterior['probs_initial'].shape[0],
            probs_initial=posterior['probs_initial'],
            probs_x=posterior['probs_x'],
            probs_alpha1=posterior['probs_alpha1'],
            probs_beta1=posterior['probs_beta1'],
            probs_alpha2=posterior['probs_alpha2'],
            probs_beta2=posterior['probs_beta2']
        )

    def pyromodel(self, sequence: torch.tensor, include_prior=True):
        '''
        Pyro Model for a Hidden Markov Model with a single univariate observation with Gamma emission distribution.
        Structure of the model taken from the Pyro documentation:
        https://pyro.ai/examples/hmm.html
        
        INPUTS:
        - sequence (torch.tensor): A 1-dimensional tensor of observations.
        - include_prior (bool): If True, include priors for the parameters of the model.
        '''
        length = sequence.shape[0]
        with poutine.mask(mask=include_prior):
            #---------------------------------------------------------------------
            # Prior for the initial state probabilities
            probs_initial = pyro.sample(
                "probs_initial",
                dist.Dirichlet(torch.ones(self.hidden_states))
            )
            #---------------------------------------------------------------------
            # Transition probabilities
            probs_x = pyro.sample(
                "probs_x",
                dist.Dirichlet(0.9 * torch.eye(self.hidden_states) + 0.1).to_event(1),
            )
            #---------------------------------------------------------------------
            # Prior for the parameters of emission probabilities 
            probs_alpha1 = pyro.sample(
                "probs_alpha1",
                dist.Gamma(concentration=15.0, rate=0.8).expand([self.hidden_states]).to_event(1)
            )

            probs_beta1 = pyro.sample(
                "probs_beta1",
                dist.Gamma(concentration=1.0, rate=1.0).expand([self.hidden_states]).to_event(1)
            )
            probs_alpha2 = pyro.sample(
                "probs_alpha2",
                dist.Gamma(concentration=15.0, rate=0.8).expand([self.hidden_states]).to_event(1)
            )

            probs_beta2 = pyro.sample(
                "probs_beta2",
                dist.Gamma(concentration=1.0, rate=1.0).expand([self.hidden_states]).to_event(1)
            )
        
        x = pyro.sample(
            "x_0",
            dist.Categorical(probs_initial),
            infer={"enumerate": "parallel"},
        )
        for t in pyro.markov(range(length)):
            if t > 0:
                x = pyro.sample(
                    f"x_{t}",
                    dist.Categorical(probs_x[x]),
                    infer={"enumerate": "parallel"},
                )
            
            pyro.sample(
                f"y1_{t}",
                dist.Gamma(probs_alpha1[x], probs_beta1[x]),
                obs=sequence[t, 0]
            )

            pyro.sample(
                f"y2_{t}",
                dist.Gamma(probs_alpha2[x], probs_beta2[x]),
                obs=sequence[t, 1]
            )

    def fit(self,sequence,training_steps=500):
        # clear fit params
        self.posterior=None
        self.guide=None
        self.elbo=None
        self.svi=None
        
        # Set the guide and clear params
        guide = AutoDelta(poutine.block(self.pyromodel, expose=["probs_initial",
                                                                     "probs_x",
                                                                     "probs_alpha1",
                                                                     "probs_beta1",
                                                                     "probs_alpha2",
                                                                     "probs_beta2"]))
        pyro.clear_param_store()

        # Optimizer
        optimizer = Adam({"lr": 0.01})

        # Inference algorithm
        self.elbo = TraceEnum_ELBO(max_plate_nesting=1)
        self.svi = SVI(self.pyromodel, guide, optimizer, loss=self.elbo)

        # Training
        tqdm_bar = tqdm.tqdm(range(training_steps))
        for step in tqdm_bar:
            loss = self.svi.step(sequence)
            tqdm_bar.set_postfix({'LOSS': loss})
            
        # Update self params
        self.posterior = guide(sequence)
        self.probs_initial = self.posterior['probs_initial']
        self.probs_x = self.posterior['probs_x']
        self.probs_alpha1 = self.posterior['probs_alpha1']
        self.probs_beta1 = self.posterior['probs_beta1']
        self.probs_alpha2 = self.posterior['probs_alpha2']
        self.probs_beta2 = self.posterior['probs_beta2']
        
    def print_estimates(self):
        print("-" * 68)
        for state in range(self.hidden_states):
            HomeMean= (self.probs_alpha1[state]/self.probs_beta1[state])*100
            HomeStd= torch.sqrt((self.probs_alpha1[state]/self.probs_beta1[state]**2))*100
            AwayMean= (self.probs_alpha2[state]/self.probs_beta2[state])*100
            AwayStd= torch.sqrt((self.probs_alpha2[state]/self.probs_beta2[state]**2))*100
            print(f"STATE {state}:")
            print(f">> Mean of the Convex Hull for home team : {HomeMean:.2f} m^2")
            print(f">> Std of the Convex Hull for home team  : {HomeStd:.2f} m^2")
            print(f">> Mean of the Convex Hull for away team : {AwayMean:.2f} m^2")
            print(f">> Std of the Convex Hull for away team  : {AwayStd:.2f} m^2")
            print("-" * 68)

    def viterbi(self,observations: torch.tensor) -> torch.tensor:
        """Viterbi algorithm to find the most probable state sequence.
        INPUTS:
        - observations (torch.tensor): A 2-dimensional tensor with the observations
        OUTPUTS:
        - (torch.tensor) Most likely sequence of states.
        """
        
        num_obs = observations.shape[0]
        
        # Initialize the dynamic programming table
        V = torch.zeros((self.hidden_states, num_obs))
        path = torch.zeros((self.hidden_states, num_obs), dtype=int)
        
        # Compute the log probabilities for the initial states
        log_initial_states_prob = torch.log(self.probs_initial)
        log_transition_matrix = torch.log(self.probs_x)
        
        # Compute the log pdf for the first observation across all states
        for s in range(self.hidden_states):
            V[s, 0] = log_initial_states_prob[s] + \
                            dist.Gamma(self.probs_alpha1[s], self.probs_beta1[s]).log_prob(observations[0, 0]) + \
                            dist.Gamma(self.probs_alpha2[s], self.probs_beta2[s]).log_prob(observations[0, 1])
            path[s, 0] = 0
        
        # Vectorized Viterbi for t > 0
        for t in range(1, num_obs):
            log_probs = torch.empty(self.hidden_states, self.hidden_states)
            for s in range(self.hidden_states):
                log_probs[:, s] = V[:, t-1] + log_transition_matrix[:, s] + \
                                    dist.Gamma(self.probs_alpha1[s], self.probs_beta1[s]).log_prob(observations[t, 0]) + \
                                    dist.Gamma(self.probs_alpha2[s], self.probs_beta2[s]).log_prob(observations[t, 1])
            V[:, t], path[:, t] = log_probs.max(dim=0)
        
        # Backtrack to find the most probable state sequence
        optimal_path = torch.zeros(num_obs, dtype=int)
        optimal_path[num_obs-1] = torch.argmax(V[:, num_obs-1])
        
        for t in range(num_obs-2, -1, -1):
            optimal_path[t] = path[optimal_path[t+1], t+1]
        
        return optimal_path
    
    def compute_log_likelihood(self,observations: torch.tensor) -> torch.tensor:
        """
        Compute the likelihood of the observations given the model parameters.
        INPUTS:
        - observations (torch.tensor): A 2-dimensional tensor with the observations.
        OUTPUTS:
        - (torch.tensor) Log likelihood of the observations given the model parameters.
        """
        
        num_obs = observations.shape[0]
    
        # Initialize the dynamic programming table
        V = torch.zeros((self.hidden_states, num_obs))
        
        # Compute the log probabilities for the initial states
        log_initial_states_prob = torch.log(self.probs_initial)
        log_transition_matrix = torch.log(self.probs_x)
        
        # Compute the log pdf for the first observation across all states
        for s in range(self.hidden_states):
            V[s, 0] = log_initial_states_prob[s] + \
                            dist.Gamma(self.probs_alpha1[s], self.probs_beta1[s]).log_prob(observations[0, 0]) + \
                            dist.Gamma(self.probs_alpha2[s], self.probs_beta2[s]).log_prob(observations[0, 1])
        
        # Vectorized Viterbi for t > 0
        for t in range(1, num_obs):
            log_probs = torch.empty(self.hidden_states, self.hidden_states)
            for s in range(self.hidden_states):
                log_probs[:, s] = V[:, t-1] + log_transition_matrix[:, s] + \
                                    dist.Gamma(self.probs_alpha1[s], self.probs_beta1[s]).log_prob(observations[t, 0]) + \
                                    dist.Gamma(self.probs_alpha2[s], self.probs_beta2[s]).log_prob(observations[t, 1])
            V[:, t] = torch.logsumexp(log_probs, dim=0)

        return torch.logsumexp(V[:, num_obs-1], dim=0)          
    
    def AIC(self,observations: torch.tensor):
        """
        Compute the Akaike Information Criterion (AIC) for the model.
        INPUTS:
        - observations (torch.tensor): A 2-dimensional tensor with the observations.
        OUTPUTS:
        - (float) The AIC value for the model.
        """
        n_params= self.hidden_states*6 + self.hidden_states**2
        return -2 * self.compute_log_likelihood(observations) + 2*n_params
    
    def predict(self,num_pred : int,initial_state: int) -> tuple:  
        # Predict the sequence of states
        states=torch.tensor([initial_state])
        for p in range(num_pred):
            x = pyro.sample(
                f"x_{p}",
                dist.Categorical(self.probs_x[states[-1],:])
            )
            states=torch.cat((states,torch.tensor([x])))
        # Predict the areas
        alpha1=self.probs_alpha1[states[1:]]
        alpha2=self.probs_alpha2[states[1:]]
        beta1=self.probs_beta1[states[1:]]
        beta2=self.probs_beta2[states[1:]]
        X= pyro.sample(
            f"X",
            dist.Gamma(alpha1, beta1)
        )
        Y= pyro.sample(
            f"Y",
            dist.Gamma(alpha2, beta2)
        )
        areas = torch.stack((X, Y), dim=1)
            
        return states[1:], areas