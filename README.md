# Research in Data Science
### Semester Project Fredrik Nestaas

## Goals:
- Understand NeuralODEs (NODEs)
- Understand ODE2ODE
- Inspect gradients of ODE2ODE
- Novel Methods

## NODEs:
Understood concept, would be good to implement.
Don't understand the negative signs in gradient computations

## ODE2ODE:
We don't need to look into the proofs since they are wrong. However, understanding the approach would be useful.
We have already established that what they were trying to solve was the wrong Ansatz, and would only indirectly
solve the problem. This ties nicely into novel methods and what we could work on

Inspect grads of ODE2ODE:
TBD

## Novel Methods:
Found new objective to minimize, namely the time derivative of the norm of the parameter gradient
Idea: use control theory to achieve the above (use 0 as a reference for this derivative)
Constant norm of state: unclear why this should help, also impossible with linear functions



## Time table (tentative):
W01:	Read and understand the papers, also further literature, make github

W02:	Implement NODEs, familiarize yourself with JAX

W03:	Implement ODE2ODE

W04:	Implement ODE2ODE / investigate the gradients, compare to other (reported) methods

W05:	slack

W06:	Explore novel methods, either self made or in the literature

W07:	Explore novel methods, either self made or in the literature

W08:	Implement some of those methods, compare performance to ODE2ODE

W09:	slack

W10:	Report / slack

W11:	Report

W12:	Report / presentation

W13:	Presentation



