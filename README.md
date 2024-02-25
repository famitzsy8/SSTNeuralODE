# SSTNeuralODE

A small experiment by Yuri Simantob.

## Introduction

Sea Surface Temperature (SST) prediction has seen a huge evolution in the last few decades. The new machine-learning paradigm brought forward a lot of new approaches on how to model the dynamics of SST, and thus there have been many new proposals on how to use Artificial Neural Networks for the specific task. Many different architectures were used, with the state-of-the-art architecture being recurrent architectures like LSTM and convolutional neural networks (CNN's). In this experiment, I will fit a Latent Generative Time-Series model using Neural ODE's to the SST data and evaluate its effectiveness in SST prediction. Although Latent ODE models have been proven to be most effective at predicting irregularly sampled time-series data, they have also shown their strength in modeling continuous-time dynamics and are theoretically suitable for SST prediction. This is an attempt to show this experimentally.

## The Model

### Context

In recent years, there were much more sophisticated methods proposed to predict time-series based on Neural ODEs, such as ODE-LSTM addressing the vanishing gradient problems of RNN encoders or MTGODE's, showing their strength in multivariate time-series prediction. Here however, for simplicity's sake I am building the model based on the original proposal in Section 5.

### Data

The data that was used to train the model was taken from the ERA5 Reanalysis dataset, that includes SST values from 1940 to 2024 in hourly intervals. In this experiment we took SST data at 6-day intervals between 2020 and 2023 for the training set, in order to predict the SST values of 2024 for the testing set (also in 6-day intervals). The area covered consists of the square from -8° to 8° longitude and latitude respectively, corresponding to the sea right under the West African coast. Only lagged SST data and their respective timestamps were fed to the model.

### Model Architecture

Just like in I will use a Recurrent Neural Network (in our case an LSTM) as the recognition network for the Variational Autoencoder, which gives us the mean and the variance of the distribution of possible Initial Values $z_0$:

$$f_{\text{rec}}(T_{0:n}, t_{0:n}) = (\mu, \sigma) \implies p(z_0) = \mathcal{N}(\mu, \sigma)$$

We then solve for $z(t)$, the function describing the latent space over time, which is described in terms of this ODE:
$$\frac{\partial{z(t)}}{\partial{t}} = f_{\text{latent}}(z(t), t, \theta)$$

In practice, we get back evaluations of $z(t)$ at our pre-defined timesteps $[t_1, \dots, t_n]$. These are then fed into a decoder ANN, which maps the $z(t_i)$ back to the SST predictions $T(t_i)$.

$$\tilde{T}_{0:n} = f_{\text{dec}}(z_{0:n})$$

### The Loss

In order to evaluate the performance of the model, I went for the negative Evidence Lower Bound (ELBO), since we are essentially using a VAE here. Concretely, the ELBO was calculated as:
$$
\begin{align}
    \text{ELBO} &= \mathbb{E}_{q(z  |  T )} [\ln p(T  |  z)] - \text{KL}[q(z  |  T) \ \lVert \ p(z)] \\
                &\approx -\sum_{i = 0}^n \ \lVert T(t_i) - \tilde{T}(t_i) \rVert^2 - \text{KL}[q(z  |  T) \ \lVert \ p(z)] \\
                &= -\sum_{i = 0}^n \ \lVert T(t_i) - \tilde{T}(t_i) \rVert^2 - \text{KL}[\mathcal{N}(\mu_{\text{rec}}, \Sigma_{\text{rec}}) \ \lVert \ \mathcal{N}(0, I)] \\
                &= -\sum_{i = 0}^n \ \lVert T(t_i) - \tilde{T}(t_i) \rVert^2 - \frac{1}{2}\sum_{i = 0}^n(1 + \log(\sigma_i^2) - \mu_i^2 - \sigma_i^2)
\end{align}
$$

Here we use the reconstruction loss as a surrogate for the log-likelihood of the SST's given the latent variables sampled from the recognition network's output distribution. For the KL divergence it is common to choose the standard normal as the prior $p(z)$, which acts as a regularizer for the values $z$ can attain.