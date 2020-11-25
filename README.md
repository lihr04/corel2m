# HRL and JHU's Joint Effort on Selective Plasticity 

This repo contains the code for comparing EWC, MAS, and SCP on the permuted MNIST task. In addition, I have included the code for tracking the gradients over the course of the training. 

Below are the details:

## Permuted MNIST Experiment:

* compare_mas_csh.ipynb: Comparison between CountSketch and other methods. (CountSketch implementation is in /csh_utils.)
* compare_classic_ewc_scp_10Run.ipynb: An easy to read ipynb in which we developed the comparison between different methods.
* compare_permutedMNIST.py: The packaged version of the previous ipynb
* Figure_Generation_Permuted_MNIST.ipynb: Code for replicating our  paper's figure. 

## Gradient Experiments:

In these experiments I keep track of the distribution of the absolute value of the gradient at each layer of the network.

* Gradient_Tracking_Loss.ipynb: Gradient of cross entropy loss 
* Gradient_Tracking_MAS.ipynb: Gradient of the norm-squared of the logits
* Gradient_Tracking_SCP.ipynb: Gradient of the inner product of a random vector from the d-dimensional (d=10) unit ball with the logits.

The videos are also included. 