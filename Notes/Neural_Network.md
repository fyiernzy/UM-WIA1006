# Neural Network

## What is the function of activation function and hidden layer

To introduce non-linearity into the neural network.

* Activation functions introduce non-linearity into a neural network, allowing it to model complex relationships between the inputs and outputs.
* Without activation functions, a neural network would be limited to linear transformations of the input data.
* Hidden layers enable a neural network to model complex features of the input data.
* Each hidden layer extracts increasingly complex features of the input data.

## Question 23

The correct answer is `a) reshape(thetaVec(16 : 39), 4, 6)`.

To explain the answer, we need to understand how the colon operator works in MATLAB (or Octave). The colon operator creates a vector of consecutive values from a starting point to an ending point. For example, 1:5 creates a vector [1 2 3 4 5]. If we omit the starting point, it defaults to 1. For example, :5 is equivalent to 1:5.

When we use the colon operator on a matrix, it creates a vector by concatenating the columns of the matrix from left to right. For example, if A = [1 2; 3 4], then A(:) creates a vector [1; 3; 2; 4].

Now, let’s look at the question. Theta1 is a 5x3 matrix, and Theta2 is a 4x6 matrix. When we use the colon operator on them, we get vectors of length 15 and 24 respectively. When we concatenate these vectors, we get a vector of length 39. This vector is thetaVec.

To recover Theta2 from thetaVec, we need to extract the elements that correspond to Theta2 and reshape them into a 4x6 matrix. Since Theta2 comes after Theta1 in thetaVec, we need to skip the first 15 elements that belong to Theta1. Therefore, we start from the 16th element and end at the 39th element. This gives us thetaVec(16:39), which is a vector of length 24. Then, we reshape this vector into a 4x6 matrix using the reshape function. This gives us reshape(thetaVec(16:39), 4, 6), which is equal to Theta2.

## Question 25

i. For computational efficiency, after we have performed gradient checking to verify that our backpropagation code is correct, we usually disable gradient checking before using backpropagation to train the network. _True_

Gradient checking is a slow and expensive process that involves numerically approximating the gradients by perturbing the parameters slightly and measuring the change in the cost function. It is useful for debugging and verifying the correctness of the backpropagation implementation, but it is not efficient for training the network. Therefore, we usually disable gradient checking after we have confirmed that our backpropagation code is bug-free.

ii. Computing the gradient of the cost function in a neural network has the same efficiency when we use backpropagation or when we numerically compute it using the method of gradient checking. **False**

As mentioned above, gradient checking is much slower and more costly than backpropagation. Backpropagation is an efficient algorithm that exploits the chain rule of calculus to compute the gradients analytically in a single backward pass through the network. Gradient checking requires multiple forward passes through the network for each parameter, and it also introduces numerical errors due to finite precision arithmetic.

iii. Using gradient checking can help verify if one’s implementation of backpropagation is bug-free. **True**

Gradient checking can help verify if one’s implementation of backpropagation is bug-free by comparing the analytical gradients computed by backpropagation with the numerical gradients computed by gradient checking. If the two gradients are very close (up to some small tolerance), then we can be confident that our backpropagation code is correct. If they differ significantly, then there is likely a bug in our backpropagation code that needs to be fixed.

iv. Gradient checking is useful if we are using one of the advanced optimization methods (such as in fminunc) as our optimization algorithm. However, it serves little purpose if we are using gradient descent. **False**

Gradient checking is useful regardless of which optimization algorithm we are using, as long as it requires gradients as inputs. Gradient checking can help us ensure that we are providing correct gradients to the optimization algorithm, which can affect the performance and convergence of the algorithm. Gradient descent is one such optimization algorithm that requires gradients as inputs, so gradient checking can serve a purpose if we are using gradient descent.

## Question 26

The correct answer is i. and iii.

### Statement I

If we are training a neural network using gradient descent, one reasonable “debugging” step to make sure it is working is to plot 𝐽(𝜃) as a function of the number of iterations, and make sure it is decreasing (or at least non-increasing) after each iteration. **True**

* Plotting J(θ) as a function of iterations is a reasonable debugging step
* Gradient descent minimizes J(θ) by updating the parameters in the direction of the negative gradient
* J(θ) should decrease after each iteration, unless we have reached a local minimum or a saddle point
* Plotting J(θ) helps check if gradient descent is working properly and converging to a good solution

### Statement II

Suppose you have a three layer network with parameters 𝜃(1) (controlling the function mapping from the inputs to the hidden units) and 𝜃(2) (controlling the mapping from the hidden units to the outputs). If we set all the elements of 𝜃(1) to be 0, and all the elements of 𝜃(2) to be 1, then this suffices for symmetry breaking, since the neurons are no longer all computing the same function of the input. False

* Setting all the elements of 𝜃(1) to be 0 means that all the hidden units will have zero activation regardless of the input.
* This makes the hidden layer useless and reduces the network to a single layer network with parameters 𝜃(2).
* Symmetry breaking refers to the idea of initializing the parameters randomly so that each neuron computes a different function of the input.
* Random initialization helps the network to learn different features and improve its performance and generalization.

iii. Suppose you are training a neural network using gradient descent. Depending on your random initialization, your algorithm may converge to different local optima (i.e., if you run the algorithm twice with different random initializations, gradient descent may converge to two different solutions). True

This is true because gradient descent is a local optimization algorithm that depends on the initial point. The cost function of a neural network is usually non-convex and may have multiple local optima. Therefore, depending on where we start from, gradient descent may converge to different local optima that have different values of 𝐽(𝜃) and 𝜃. This also means that some initializations may lead to better solutions than others.

iv. If we initialize all the parameters of a neural network to ones instead of zeros, this will suffice for the purpose of “symmetry breaking” because the parameters are no longer symmetrically equal to zero. False

This does not suffice for symmetry breaking, because setting all the parameters to ones is equivalent to setting them to any other constant value. This means that all the neurons in each layer will still compute the same function of the input and learn the same features. This can limit the expressive power and diversity of the network. Symmetry breaking requires random initialization (usually with small values) so that each neuron computes a different function of the input and learns different features.

## Question 34

The output ˆy is 0.5.

To explain the answer, we need to understand how the sigmoid function works. The sigmoid function is a type of activation function that maps any real number to a value between 0 and 1. The sigmoid function is defined by the formula:

$$sigmoid(x) = \frac{1}{1 + e^{-x}}$$

The sigmoid function has the property that it approaches 0 as x goes to negative infinity, and it approaches 1 as x goes to positive infinity. However, when x is close to zero, the sigmoid function is close to 0.5.

Now, let's look at the question. We have a 2-layer fully connected neural network with first layer contains 4 neurones, second layer contain 3 neurones and the output layer contains 1 neurone. All activations are sigmoids and our optimizer is stochastic gradient descent. We initialize all the weights and biases to zero and forward propagate an input x ∈ Rn×1 in the network.

The forward propagation can be summarized by the following equations:

$$z_1 = W_1 x + b_1$$
$$a_1 = sigmoid(z_1)$$
$$z_2 = W_2 a_1 + b_2$$
$$a_2 = sigmoid(z_2)$$
$$z_3 = W_3 a_2 + b_3$$
$$\hat{y} = a_3 = sigmoid(z_3)$$

where W_i and b_i are the weights and biases of layer i, z_i and a_i are the linear and nonlinear activations of layer i, and \hat{y} is the output prediction.

Since we initialize all the weights and biases to zero, we can simplify these equations as follows:

$$z_1 = 0$$
$$a_1 = sigmoid(0) = 0.5$$
$$z_2 = 0$$
$$a_2 = sigmoid(0) = 0.5$$
$$z_3 = 0$$
$$\hat{y} = sigmoid(0) = 0.5$$

Therefore, the output \hat{y} is 0.5 regardless of the input x.

Source: Conversation with Bing, 09/05/2023
(1) Sigmoid function - Wikipedia. <https://en.wikipedia.org/wiki/Sigmoid_function>.
(2) Sigmoid Function Definition | DeepAI. <https://deepai.org/machine-learning-glossary-and-terms/sigmoid-function>.
(3) A Gentle Introduction To Sigmoid Function. <https://machinelearningmastery.com/a-gentle-introduction-to-sigmoid-function/>.

## Question 35

The correct answer is a) Entries of 𝑊[1] may be positive or negative.

To explain the answer, we need to understand how the backpropagation algorithm works. The backpropagation algorithm is a method for updating the weights and biases of a neural network based on the error of its predictions. The algorithm consists of two main steps: a forward pass and a backward pass.

In the forward pass, the input data is fed into the network and the output is computed using the current weights and biases. The output is then compared with the desired output (the target or label) and the error (or loss) is calculated using a loss function.

In the backward pass, the error is propagated back through the network using the chain rule of calculus. The chain rule allows us to compute the partial derivatives of the loss function with respect to each weight and bias in the network. These partial derivatives are also called gradients, and they indicate how much each parameter should change to reduce the error.

The gradients are then used to update the weights and biases by subtracting a fraction of them from the current values. This fraction is called the learning rate, and it controls how fast the network learns. The process of updating the parameters is also called gradient descent.

Now, let's look at the question. We have a model defined in question (34) with parameters initialized with zeros. 𝑊[1] denotes 
the weight matrix of the first layer. We forward propagate a batch of examples, and then 
backpropagate the gradients and update the parameters.

Since we initialize all the weights and biases to zero, we can assume that all the activations in the network are 0.5 (as explained in question (34)). This means that all the gradients in the network are also 0.5 (assuming we use a sigmoid activation function and a mean squared error loss function). Therefore, when we update the parameters using gradient descent, we have:

$$W[1] = W[1] - \alpha \frac{\partial L}{\partial W[1]}$$

where $\alpha$ is the learning rate and $\frac{\partial L}{\partial W[1]}$ is the gradient of the loss function with respect to $W[1]$. Since both $W[1]$ and $\frac{\partial L}{\partial W[1]}$ are zero matrices initially, we have:

$$W[1] = 0 - \alpha 0 = 0$$

However, this is only true for the first iteration. In subsequent iterations, $W[1]$ will change depending on the input data and the learning rate. Depending on these factors, some entries of $W[1]$ may become positive or negative after one or more iterations. Therefore, we cannot say that entries of $W[1]$ are all positive, negative, or zero after backpropagation and parameter update.
