---
layout: layout
title: Neural Networks
---
# Introduction

“Representation” ML-based systems figure out by themselves what features to pay attention to. Neural networks are examples of such systems.
Neural networks are able to learn nonlinear relationships between the features and the target and also to learn relationships between combinations of features and the target.


# Perceptrons

The simplest type of neural network is the perceptron. It accepts more than one binary inputs, $$x_1, x_2, x_3, \ldots$$, weights them, adds them up, and then based on a threshold you set for it — outputs a result. A simple rule is:

$$
\text{output} = 
\begin{cases}
0 & \text{if} \ \sum_j w_j x_j \leq \text{threshold} \\
1 & \text{otherwise}
\end{cases}
$$

where $$w_j$$s are real numbers.

Using the bias instead of the threshold, the perceptron rule can be rewritten as:

$$
\text{output} = 
\begin{cases}
0 & \text{if} \ w \cdot x + b \leq 0 \\
1 & \text{otherwise}
\end{cases}
$$

The bias is a measure of how easy it is to get the perceptron to fire. It turns out that we can devise learning algorithms which can automatically tune the weights and biases of a network of artificial neurons. This tuning happens in response to external stimuli, without direct intervention by a programmer.

To see how learning might work, suppose we make a small change in some weight (or bias) in the network. What we'd like is for this small change in weight to cause only a small corresponding change in the output from the network. If it were true that a small change in a weight (or bias) causes only a small change in output, then we could use this fact to modify the weights and biases to get our network to behave more in the manner we want. And then we'd repeat this, changing the weights and biases over and over to produce better and better output. The network would be learning.

The problem is that this isn't what happens when our network contains perceptrons. In fact, a small change in the weights or bias of any single perceptron in the network can sometimes cause the output of that perceptron to completely flip, say from 0 to 1. That flip may then cause the behavior of the rest of the network to completely change in some very complicated way. We can overcome this problem by introducing a new type of artificial neuron called a sigmoid neuron.

# Sigmoid Neurons

Sigmoid neurons are similar to perceptrons but modified so that small changes in their weights and bias cause only a small change in their output. That's the crucial fact that will allow a network of sigmoid neurons to learn. The output of a sigmoid neuron is $$\sigma(w \cdot x + b)$$, where $$\sigma$$ is called the sigmoid function or logistic function and is defined by:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

To understand the similarity to the perceptron model, suppose $$z \equiv w \cdot x + b$$ is a large positive number. Then $$e^{-z} \approx 0$$, and so $$\sigma(z) \approx 1$$. In other words, when $$w \cdot x + b$$ is large and positive, the output from the sigmoid neuron is approximately 1, just as it would have been for a perceptron. Suppose, on the other hand, that $$z \equiv w \cdot x + b$$ is very negative. Then $$e^{-z} \rightarrow \infty$$ and $$\sigma(z) \approx 0$$. So when $$w \cdot x + b$$ is very negative, the behavior of a sigmoid neuron also closely approximates a perceptron. It's only when $$w \cdot x + b$$ is of modest size that there's much deviation from the perceptron model.

If $$\sigma$$ had, in fact, been a step function, then the sigmoid neuron would be a perceptron, since the output would be 1 or 0 depending on whether $$w \cdot x + b$$ was positive or negative. By using the actual $$\sigma$$ function we get, as already implied above, a smoothed-out perceptron. The smoothness of $$\sigma$$ means that small changes $$\Delta w_j$$ in the weights and $$\Delta b$$ in the bias will produce a small change in the output from the neuron.

Why is using this nonlinear function a good idea? Why not the square function $f(x) = x^2$, for example? There are a couple of reasons. 
- First, we want the function we use here to be monotonic so that it “preserves” information about the numbers that were fed in. Let’s say that, two of our linear regressions produced values of $-3$ and $3$, respectively. Feeding these through the square function would then produce a value of $9$ for each, so that any function that receives these numbers as inputs after they were fed through the square function would “lose” the information that one of them was originally $-3$ and the other was $3$. 
- The second reason, of course, is that the function is nonlinear; this nonlinearity will enable our neural network to model the inherently nonlinear relationship between the features and the target.
- Finally, the sigmoid function has the nice property that its derivative can be expressed in terms of the function itself: $$\frac{\partial\sigma}{\partial u}(x) = \sigma(x) \times (1 - \sigma(x))$$ We’ll make use of this shortly when we use the sigmoid function in the backward pass of our neural network.

# Linear Basis Function Models

The simplest linear model for regression is one that involves a linear combination of the input variables:

$$y(\mathbf{x}, \mathbf{w}) = w_0 + w_1x_1 + \ldots + w_Dx_D$$

where $$x = (x_1, \ldots, x_D)^T$$. This is often simply known as linear regression. The key property of this model is that it is a linear function of the parameters $$w_0, \ldots, w_D$$. It is also, however, a linear function of the input variables $$x_i$$, and this imposes significant limitations on the model. We therefore extend the class of models by considering linear combinations of fixed nonlinear functions of the input variables, of the form:

$$y(\mathbf{x}, \mathbf{w}) = w_0 + \sum_{j=1}^{M-1} w_j \phi_j (\mathbf{x})$$

where $$\phi_j (x)$$ are known as _basis functions_. By denoting the maximum value of the index $$j$$ by $$M - 1$$, the total number of parameters in this model will be $$M$$.

The parameter $$w_0$$ allows for any fixed offset in the data and is sometimes called a bias parameter (not to be confused with ‘bias’ in a statistical sense). It is often convenient to define an additional dummy ‘basis function’ $$\phi_0(x) = 1$$ so that

$$y(\mathbf{x}, \mathbf{w}) = \sum_{j=0}^{M-1} w_j \phi_j (\mathbf{x}) = w^T \phi(\mathbf{x}) \tag{1}$$

where $$w = (w_0, \ldots, w_{M-1})^T$$ and $$\phi = (\phi_0, \ldots, \phi_{M-1})^T$$. In many practical applications of pattern recognition, we will apply some form of fixed pre-processing, or feature extraction, to the original data variables. If the original variables comprise the vector $$\mathbf{x}$$, then the features can be expressed in terms of the basis functions $$\{\phi_j (\mathbf{x})\}$$.

By using nonlinear basis functions, we allow the function $$y(\mathbf{x}, \mathbf{w})$$ to be a nonlinear function of the input vector $$x$$. Functions of the form (1) are called linear models; however, because this function is linear in $$w$$, it is this linearity in the parameters that will greatly simplify the analysis of this class of models.

The example of polynomial regression considered in <a href="https://mlbible.github.io/classical-ml/multiple-linear-regression/#introduction" target="_blank">multiple linear regression</a> is a particular example of this model in which there is a single input variable $$x$$, and the basis functions take the form of powers of $$x$$ so that $$\phi_j (x) = x^j$$. One limitation of polynomial basis functions is that they are global functions of the input variable, so that changes in one region of input space affect all other regions. This can be resolved by dividing the input space up into regions and fitting a different polynomial in each region, leading to spline functions (Hastie et al., 2001).


# Feedforward Neural Networks or Multilayer Perceptron

The idea of linear basis function models leads to the basic neural network model, which can be described as a series of functional transformations. At its core, it is nothing but a linear transformation of the input, i.e. multiplying the input by a number (the weight) and adding a constant (the bias) followed by the application of a fixed nonlinear function (referred to as the activation function). 

First, we construct $$M$$ linear combinations of the input variables $$x_1, \ldots, x_D$$ in the form

$$
a_j = \sum_{i=1}^{D} w_{ji}^{(1)} x_i + w_{j0}^{(1)}
$$

where $$j = 1, \ldots, M$$, and the superscript $$(1)$$ indicates that the corresponding parameters are in the first 'layer' of the network. We shall refer to the parameters $$w_{ji}^{(1)}$$ as _weights_ and the parameters $$w_{j0}^{(1)}$$ as _biases_. The quantities $$a_j$$ are known as _activations_. Each of them is then transformed using a differentiable, nonlinear activation function $$h(\cdot)$$ to give

$$
z_j = h(a_j).
$$

These quantities are called _hidden units_. The nonlinear functions $$h(\cdot)$$ are generally chosen to be sigmoidal functions such as the logistic sigmoid or the 'tanh' function. These hidden units are again linearly combined to give output unit activations

$$
a_k = \sum_{j=1}^{M} w_{kj}^{(2)} z_j + w_{k0}^{(2)}
$$

where $$k = 1, \ldots, K$$, and $$K$$ is the total number of outputs. This transformation corresponds to the second layer of the network, and again the $$w_{k0}^{(2)}$$ are bias parameters.

Finally, the output unit activations are transformed using an appropriate activation function to give a set of network outputs $$y_k$$. The choice of activation function is determined by the nature of the data and the assumed distribution of target variables. Thus, for standard regression problems, the activation function is the identity so that $$y_k = a_k$$. Similarly, for multiple binary classification problems, each output unit activation is transformed using a logistic sigmoid function so that $$y_k = \sigma(a_k)$$. Finally, for multiclass problems, a softmax activation function is used.

We can combine these various stages to give the overall network function that, for sigmoidal output unit activation functions, takes the form

$$
y_k(\mathbf{x}, \mathbf{w}) = \sigma\left(\sum_{j=1}^{M} w_{kj}^{(2)} h\left(\sum_{i=1}^{D} w_{ji}^{(1)} x_i + w_{j0}^{(1)}\right) + w_{k0}^{(2)}\right) \tag{2}
$$

where the set of all weight and bias parameters have been grouped together into a vector $$w$$. Thus the neural network model is simply a nonlinear function from a set of input variables $$\{x_i\}$$ to a set of output variables $$\{y_k\}$$ controlled by a vector $$w$$ of adjustable parameters.

The process of evaluating Eq. 2 can then be interpreted as a _forward propagation_ of information through the network. 

The bias parameters can be absorbed into the set of weight parameters by defining an additional input variable $$x_0$$ whose value is clamped at $$x_0 = 1$$. We can similarly absorb the second-layer biases into the second-layer weights, so that the overall network function becomesso that Eq. 2 takes the form

$$
y_k(\mathbf{x}, \mathbf{w}) = \sigma\left(\sum_{j=0}^{M} w_{kj}^{(2)} h\left(\sum_{i=0}^{D} w_{ji}^{(1)} x_i\right)\right).
$$

The idea behind neural networks is that many neurons can be joined together by communication links to carry out complex computations. Neural networks where the output from one layer is used as input to the next layer are called _feedforward neural networks_. In fact, ‘multilayer perceptron’ is really a misnomer, because the model comprises multiple layers of logistic regression models (with continuous nonlinearities) rather than multiple perceptrons (with discontinuous nonlinearities).

It is common to describe the structure of a neural network as a graph whose nodes are the neurons and each (directed) edge in the graph links the output of some neuron to the input of another neuron. We will restrict our attention to feedforward network structures in which the underlying graph does not contain cycles. This means there are no loops in the network - information is always fed forward, never fed back.

Let $$V_0, V_1, \ldots, V_{T}$$ be the layers of a neural network where $$V_0$$ is the input layer.
Layers $$V_1, \ldots, V_{T-1}$$ are often called hidden layers. The top layer, $$V_T$$, is called the output layer. In simple prediction problems, the output layer contains a single neuron whose output is the output of the network. We refer to $$T$$ as the number of layers in the network (excluding $$V_0$$), or the “depth” of the network. The size of the network is $$ \vert  V  \vert $$. The “width” of the network is $$\max_t  \vert  V_t  \vert $$. 


# MLE

We shall assume that, given the value of $$\mathbf{x}$$, the corresponding value of target $$t$$ has a Gaussian distribution with a mean equal to the value $$y_k(\mathbf{x}, \mathbf{w})$$. For the sake of simplicity, we drop the subscript $$k$$ for the time being. Thus, we have

$$
p(t \vert x, w, \beta) = \mathcal{N}(t \vert y(\mathbf{x}, \mathbf{w}), \beta^{-1})
$$

where we have defined a precision parameter $$\beta$$ corresponding to the inverse variance (the $$\Sigma$$ matrix) of the distribution. Note that the $$\beta$$'s here are not the biases. The biases are included in the $$\mathbf{w}$$ vector itself. We now use the training data to determine the values of the unknown parameters $$\mathbf{w}$$ and $$\beta$$ by maximum likelihood. If the data are assumed to be drawn independently, then the likelihood function is given by

$$
p(t \vert x, w, \beta) = \prod_{n=1}^{N} \mathcal{N}(t_n \vert y(x_n, w), \beta^{-1}). \tag{3}
$$

It is convenient to maximize the logarithm of the likelihood function which takes the form

$$
\ln p(t \vert x, w, \beta) = -\frac{\beta}{2} \sum_{n=1}^{N} \left\{y(x_n, w) - t_n\right\}^2 + \frac{N}{2} \ln \beta - \frac{N}{2} \ln(2\pi). \tag{4}
$$

Consider first the determination of the maximum likelihood solution for the coefficients $$w$$, which will be denoted by $$w_{\text{ML}}$$. These are determined by maximizing (4) with respect to $$w$$. For this purpose, we can omit the last two terms on the right-hand side of (4) because they do not depend on $$w$$. Also, we note that scaling the log likelihood by a positive constant coefficient does not alter the location of the maximum with respect to $$w$$, and so we can replace the coefficient $$\beta/2$$ with $$1/2$$. Finally, instead of maximizing the log likelihood, we can equivalently minimize the negative log likelihood. 

<blockquote style="background-color: #FFFFE0; padding: 10px;">
<b>
We therefore see that maximizing likelihood is equivalent, so far as determining $w$ is concerned, to minimizing the sum-of-squares error function. Thus the sum-of-squares error function has arisen as a consequence of maximizing likelihood under the assumption of a Gaussian noise distribution.
</b>
</blockquote>

We can also use maximum likelihood to determine the precision parameter $$\beta$$ of the Gaussian conditional distribution. Maximizing (4) with respect to $$\beta$$ gives

$$
\frac{1}{\beta_{\text{ML}}} = \frac{1}{N} \sum_{n=1}^{N} \left\{y(x_n, w_{\text{ML}}) - t_n\right\}^2
$$

Again, we can first determine the parameter vector $$w_{\text{ML}}$$ governing the mean and subsequently use this to find the precision $$\beta_{\text{ML}}$$. Having determined the parameters $$w$$ and $$\beta$$, we can now make predictions for new values of $$x$$.
