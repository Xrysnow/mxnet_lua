-- File content is auto-generated. Do not modify.

local NDArrayBase = require('mx.ndarray._internal').NDArrayBase
local _imperative_invoke = require('mx.ndarray._internal')._imperative_invoke
local _Null = require('mx.base')._Null
---@class mx.ndarray.gen_sparse
local M = {}

--- Adds all input arguments element-wise.
--- 
--- .. math::
---    add\_n(a_1, a_2, ..., a_n) = a_1 + a_2 + ... + a_n
--- 
--- ``add_n`` is potentially more efficient than calling ``add`` by `n` times.
--- 
--- The storage type of ``add_n`` output depends on storage types of inputs
--- 
--- - add_n(row_sparse, row_sparse, ..) = row_sparse
--- - add_n(default, csr, default) = default
--- - add_n(any input combinations longer than 4 (>4) with at least one default type) = default
--- - otherwise, ``add_n`` falls all inputs back to default storage and generates default storage
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\elemwise_sum.cc:L155
---
---
---@param args any @NDArray[] | Positional input arguments
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.ElementWiseSum(...)
end

--- Maps integer indices to vector representations (embeddings).
--- 
--- This operator maps words to real-valued vectors in a high-dimensional space,
--- called word embeddings. These embeddings can capture semantic and syntactic properties of the words.
--- For example, it has been noted that in the learned embedding spaces, similar words tend
--- to be close to each other and dissimilar words far apart.
--- 
--- For an input array of shape (d1, ..., dK),
--- the shape of an output array is (d1, ..., dK, output_dim).
--- All the input values should be integers in the range [0, input_dim).
--- 
--- If the input_dim is ip0 and output_dim is op0, then shape of the embedding weight matrix must be
--- (ip0, op0).
--- 
--- By default, if any index mentioned is too large, it is replaced by the index that addresses
--- the last vector in an embedding matrix.
--- 
--- ### Examples
--- 
---   input_dim = 4
---   output_dim = 5
--- 
---   // Each row in weight matrix y represents a word. So, y = (w0,w1,w2,w3)
---   y = [[  0.,   1.,   2.,   3.,   4.],
---        [  5.,   6.,   7.,   8.,   9.],
---        [ 10.,  11.,  12.,  13.,  14.],
---        [ 15.,  16.,  17.,  18.,  19.]]
--- 
---   // Input array x represents n-grams(2-gram). So, x = [(w1,w3), (w0,w2)]
---   x = [[ 1.,  3.],
---        [ 0.,  2.]]
--- 
---   // Mapped input x to its vector representation y.
---   Embedding(x, y, 4, 5) = [[[  5.,   6.,   7.,   8.,   9.],
---                             [ 15.,  16.,  17.,  18.,  19.]],
--- 
---                            [[  0.,   1.,   2.,   3.,   4.],
---                             [ 10.,  11.,  12.,  13.,  14.]]]
--- 
--- 
--- The storage type of weight can be either row_sparse or default.
--- 
--- .. Note::
--- 
---     If "sparse_grad" is set to True, the storage type of gradient w.r.t weights will be
---     "row_sparse". Only a subset of optimizers support sparse gradients, including SGD, AdaGrad
---     and Adam. Note that by default lazy updates is turned on, which may perform differently
---     from standard updates. For more details, please check the Optimization API at:
---     https://mxnet.incubator.apache.org/api/python/optimization/optimization.html
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\indexing_op.cc:L519
---
---
---@param data any @NDArray | The input array to the embedding operator.
---@param weight any @NDArray | The embedding weight matrix.
---@param input_dim any @int, required | Vocabulary size of the input indices.
---@param output_dim any @int, required | Dimension of the embedding vectors.
---@param dtype any @{'float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8'},optional, default='float32' | Data type of weight.
---@param sparse_grad any @boolean, optional, default=0 | Compute row sparse gradient in the backward calculation. If set to True, the grad's storage type is row_sparse.
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.Embedding(data, weight, input_dim, output_dim, dtype, sparse_grad, out, name, kwargs)
end

--- Applies a linear transformation: :math:`Y = XW^T + b`.
--- 
--- If ``flatten`` is set to be true, then the shapes are:
--- 
--- - **data**: `(batch_size, x1, x2, ..., xn)`
--- - **weight**: `(num_hidden, x1 * x2 * ... * xn)`
--- - **bias**: `(num_hidden,)`
--- - **out**: `(batch_size, num_hidden)`
--- 
--- If ``flatten`` is set to be false, then the shapes are:
--- 
--- - **data**: `(x1, x2, ..., xn, input_dim)`
--- - **weight**: `(num_hidden, input_dim)`
--- - **bias**: `(num_hidden,)`
--- - **out**: `(x1, x2, ..., xn, num_hidden)`
--- 
--- The learnable parameters include both ``weight`` and ``bias``.
--- 
--- If ``no_bias`` is set to be true, then the ``bias`` term is ignored.
--- 
--- .. Note::
--- 
---     The sparse support for FullyConnected is limited to forward evaluation with `row_sparse`
---     weight and bias, where the length of `weight.indices` and `bias.indices` must be equal
---     to `num_hidden`. This could be useful for model inference with `row_sparse` weights
---     trained with importance sampling or noise contrastive estimation.
--- 
---     To compute linear transformation with 'csr' sparse data, sparse.dot is recommended instead
---     of sparse.FullyConnected.
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\nn\fully_connected.cc:L277
---
---
---@param data any @NDArray | Input data.
---@param weight any @NDArray | Weight matrix.
---@param bias any @NDArray | Bias parameter.
---@param num_hidden any @int, required | Number of hidden nodes of the output.
---@param no_bias any @boolean, optional, default=0 | Whether to disable bias parameter.
---@param flatten any @boolean, optional, default=1 | Whether to collapse all but the first axis of the input data tensor.
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.FullyConnected(data, weight, bias, num_hidden, no_bias, flatten, out, name, kwargs)
end

--- Computes and optimizes for squared loss during backward propagation.
--- Just outputs ``data`` during forward propagation.
--- 
--- If :math:`\hat{y}_i` is the predicted value of the i-th sample, and :math:`y_i` is the corresponding target value,
--- then the squared loss estimated over :math:`n` samples is defined as
--- 
--- :math:`\text{SquaredLoss}(\textbf{Y}, \hat{\textbf{Y}} ) = \frac{1}{n} \sum_{i=0}^{n-1} \lVert  \textbf{y}_i - \hat{\textbf{y}}_i  \rVert_2`
--- 
--- .. note::
---    Use the LinearRegressionOutput as the final output layer of a net.
--- 
--- The storage type of ``label`` can be ``default`` or ``csr``
--- 
--- - LinearRegressionOutput(default, default) = default
--- - LinearRegressionOutput(default, csr) = default
--- 
--- By default, gradients of this loss function are scaled by factor `1/m`, where m is the number of regression outputs of a training example.
--- The parameter `grad_scale` can be used to change this scale to `grad_scale/m`.
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\regression_output.cc:L92
---
---
---@param data any @NDArray | Input data to the function.
---@param label any @NDArray | Input label to the function.
---@param grad_scale any @float, optional, default=1 | Scale the gradient by a float factor
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.LinearRegressionOutput(data, label, grad_scale, out, name, kwargs)
end

--- Applies a logistic function to the input.
--- 
--- The logistic function, also known as the sigmoid function, is computed as
--- :math:`\frac{1}{1+exp(-\textbf{x})}`.
--- 
--- Commonly, the sigmoid is used to squash the real-valued output of a linear model
--- :math:`wTx+b` into the [0,1] range so that it can be interpreted as a probability.
--- It is suitable for binary classification or probability prediction tasks.
--- 
--- .. note::
---    Use the LogisticRegressionOutput as the final output layer of a net.
--- 
--- The storage type of ``label`` can be ``default`` or ``csr``
--- 
--- - LogisticRegressionOutput(default, default) = default
--- - LogisticRegressionOutput(default, csr) = default
--- 
--- The loss function used is the Binary Cross Entropy Loss:
--- 
--- :math:`-{(y\log(p) + (1 - y)\log(1 - p))}`
--- 
--- Where `y` is the ground truth probability of positive outcome for a given example, and `p` the probability predicted by the model. By default, gradients of this loss function are scaled by factor `1/m`, where m is the number of regression outputs of a training example.
--- The parameter `grad_scale` can be used to change this scale to `grad_scale/m`.
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\regression_output.cc:L152
---
---
---@param data any @NDArray | Input data to the function.
---@param label any @NDArray | Input label to the function.
---@param grad_scale any @float, optional, default=1 | Scale the gradient by a float factor
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.LogisticRegressionOutput(data, label, grad_scale, out, name, kwargs)
end

--- Computes mean absolute error of the input.
--- 
--- MAE is a risk metric corresponding to the expected value of the absolute error.
--- 
--- If :math:`\hat{y}_i` is the predicted value of the i-th sample, and :math:`y_i` is the corresponding target value,
--- then the mean absolute error (MAE) estimated over :math:`n` samples is defined as
--- 
--- :math:`\text{MAE}(\textbf{Y}, \hat{\textbf{Y}} ) = \frac{1}{n} \sum_{i=0}^{n-1} \lVert \textbf{y}_i - \hat{\textbf{y}}_i \rVert_1`
--- 
--- .. note::
---    Use the MAERegressionOutput as the final output layer of a net.
--- 
--- The storage type of ``label`` can be ``default`` or ``csr``
--- 
--- - MAERegressionOutput(default, default) = default
--- - MAERegressionOutput(default, csr) = default
--- 
--- By default, gradients of this loss function are scaled by factor `1/m`, where m is the number of regression outputs of a training example.
--- The parameter `grad_scale` can be used to change this scale to `grad_scale/m`.
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\regression_output.cc:L120
---
---
---@param data any @NDArray | Input data to the function.
---@param label any @NDArray | Input label to the function.
---@param grad_scale any @float, optional, default=1 | Scale the gradient by a float factor
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.MAERegressionOutput(data, label, grad_scale, out, name, kwargs)
end

--- Returns element-wise absolute value of the input.
--- 
--- ### Example
--- 
---    abs([-2, 0, 3]) = [2, 0, 3]
--- 
--- The storage type of ``abs`` output depends upon the input storage type:
--- 
---    - abs(default) = default
---    - abs(row_sparse) = row_sparse
---    - abs(csr) = csr
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\elemwise_unary_op_basic.cc:L708
---
---
---@param data any @NDArray | The input array.
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.abs(data, out, name, kwargs)
end

--- Update function for AdaGrad optimizer.
--- 
--- Referenced from *Adaptive Subgradient Methods for Online Learning and Stochastic Optimization*,
--- and available at http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf.
--- 
--- Updates are applied by::
--- 
---     rescaled_grad = clip(grad * rescale_grad, clip_gradient)
---     history = history + square(rescaled_grad)
---     w = w - learning_rate * rescaled_grad / sqrt(history + epsilon)
--- 
--- Note that non-zero values for the weight decay option are not supported.
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\optimizer_op.cc:L907
---
---
---@param weight any @NDArray | Weight
---@param grad any @NDArray | Gradient
---@param history any @NDArray | History
---@param lr any @float, required | Learning rate
---@param epsilon any @float, optional, default=1.00000001e-07 | epsilon
---@param wd any @float, optional, default=0 | weight decay
---@param rescale_grad any @float, optional, default=1 | Rescale gradient to grad = rescale_grad*grad.
---@param clip_gradient any @float, optional, default=-1 | Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.adagrad_update(weight, grad, history, lr, epsilon, wd, rescale_grad, clip_gradient, out, name, kwargs)
end

--- Update function for Adam optimizer. Adam is seen as a generalization
--- of AdaGrad.
--- 
--- Adam update consists of the following steps, where g represents gradient and m, v
--- are 1st and 2nd order moment estimates (mean and variance).
--- 
--- .. math::
--- 
---  g_t = \nabla J(W_{t-1})\\
---  m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t\\
---  v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2\\
---  W_t = W_{t-1} - \alpha \frac{ m_t }{ \sqrt{ v_t } + \epsilon }
--- 
--- It updates the weights using::
--- 
---  m = beta1*m + (1-beta1)*grad
---  v = beta2*v + (1-beta2)*(grad**2)
---  w += - learning_rate * m / (sqrt(v) + epsilon)
--- 
--- However, if grad's storage type is ``row_sparse``, ``lazy_update`` is True and the storage
--- type of weight is the same as those of m and v,
--- only the row slices whose indices appear in grad.indices are updated (for w, m and v)::
--- 
---  for row in grad.indices:
---      m[row] = beta1*m[row] + (1-beta1)*grad[row]
---      v[row] = beta2*v[row] + (1-beta2)*(grad[row]**2)
---      w[row] += - learning_rate * m[row] / (sqrt(v[row]) + epsilon)
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\optimizer_op.cc:L686
---
---
---@param weight any @NDArray | Weight
---@param grad any @NDArray | Gradient
---@param mean any @NDArray | Moving mean
---@param var any @NDArray | Moving variance
---@param lr any @float, required | Learning rate
---@param beta1 any @float, optional, default=0.899999976 | The decay rate for the 1st moment estimates.
---@param beta2 any @float, optional, default=0.999000013 | The decay rate for the 2nd moment estimates.
---@param epsilon any @float, optional, default=9.99999994e-09 | A small constant for numerical stability.
---@param wd any @float, optional, default=0 | Weight decay augments the objective function with a regularization term that penalizes large weights. The penalty scales with the square of the magnitude of each weight.
---@param rescale_grad any @float, optional, default=1 | Rescale gradient to grad = rescale_grad*grad.
---@param clip_gradient any @float, optional, default=-1 | Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).
---@param lazy_update any @boolean, optional, default=1 | If true, lazy updates are applied if gradient's stype is row_sparse and all of w, m and v have the same stype
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.adam_update(weight, grad, mean, var, lr, beta1, beta2, epsilon, wd, rescale_grad, clip_gradient, lazy_update, out, name, kwargs)
end

--- Adds all input arguments element-wise.
--- 
--- .. math::
---    add\_n(a_1, a_2, ..., a_n) = a_1 + a_2 + ... + a_n
--- 
--- ``add_n`` is potentially more efficient than calling ``add`` by `n` times.
--- 
--- The storage type of ``add_n`` output depends on storage types of inputs
--- 
--- - add_n(row_sparse, row_sparse, ..) = row_sparse
--- - add_n(default, csr, default) = default
--- - add_n(any input combinations longer than 4 (>4) with at least one default type) = default
--- - otherwise, ``add_n`` falls all inputs back to default storage and generates default storage
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\elemwise_sum.cc:L155
---
---
---@param args any @NDArray[] | Positional input arguments
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.add_n(...)
end

--- Returns element-wise inverse cosine of the input array.
--- 
--- The input should be in range `[-1, 1]`.
--- The output is in the closed interval :math:`[0, \pi]`
--- 
--- .. math::
---    arccos([-1, -.707, 0, .707, 1]) = [\pi, 3\pi/4, \pi/2, \pi/4, 0]
--- 
--- The storage type of ``arccos`` output is always dense
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\elemwise_unary_op_trig.cc:L179
---
---
---@param data any @NDArray | The input array.
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.arccos(data, out, name, kwargs)
end

--- Returns the element-wise inverse hyperbolic cosine of the input array, \
--- computed element-wise.
--- 
--- The storage type of ``arccosh`` output is always dense
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\elemwise_unary_op_trig.cc:L320
---
---
---@param data any @NDArray | The input array.
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.arccosh(data, out, name, kwargs)
end

--- Returns element-wise inverse sine of the input array.
--- 
--- The input should be in the range `[-1, 1]`.
--- The output is in the closed interval of [:math:`-\pi/2`, :math:`\pi/2`].
--- 
--- .. math::
---    arcsin([-1, -.707, 0, .707, 1]) = [-\pi/2, -\pi/4, 0, \pi/4, \pi/2]
--- 
--- The storage type of ``arcsin`` output depends upon the input storage type:
--- 
---    - arcsin(default) = default
---    - arcsin(row_sparse) = row_sparse
---    - arcsin(csr) = csr
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\elemwise_unary_op_trig.cc:L160
---
---
---@param data any @NDArray | The input array.
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.arcsin(data, out, name, kwargs)
end

--- Returns the element-wise inverse hyperbolic sine of the input array, \
--- computed element-wise.
--- 
--- The storage type of ``arcsinh`` output depends upon the input storage type:
--- 
---    - arcsinh(default) = default
---    - arcsinh(row_sparse) = row_sparse
---    - arcsinh(csr) = csr
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\elemwise_unary_op_trig.cc:L306
---
---
---@param data any @NDArray | The input array.
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.arcsinh(data, out, name, kwargs)
end

--- Returns element-wise inverse tangent of the input array.
--- 
--- The output is in the closed interval :math:`[-\pi/2, \pi/2]`
--- 
--- .. math::
---    arctan([-1, 0, 1]) = [-\pi/4, 0, \pi/4]
--- 
--- The storage type of ``arctan`` output depends upon the input storage type:
--- 
---    - arctan(default) = default
---    - arctan(row_sparse) = row_sparse
---    - arctan(csr) = csr
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\elemwise_unary_op_trig.cc:L200
---
---
---@param data any @NDArray | The input array.
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.arctan(data, out, name, kwargs)
end

--- Returns the element-wise inverse hyperbolic tangent of the input array, \
--- computed element-wise.
--- 
--- The storage type of ``arctanh`` output depends upon the input storage type:
--- 
---    - arctanh(default) = default
---    - arctanh(row_sparse) = row_sparse
---    - arctanh(csr) = csr
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\elemwise_unary_op_trig.cc:L337
---
---
---@param data any @NDArray | The input array.
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.arctanh(data, out, name, kwargs)
end

--- Returns element-wise sum of the input arrays with broadcasting.
--- 
--- `broadcast_plus` is an alias to the function `broadcast_add`.
--- 
--- ### Example
--- 
---    x = [[ 1.,  1.,  1.],
---         [ 1.,  1.,  1.]]
--- 
---    y = [[ 0.],
---         [ 1.]]
--- 
---    broadcast_add(x, y) = [[ 1.,  1.,  1.],
---                           [ 2.,  2.,  2.]]
--- 
---    broadcast_plus(x, y) = [[ 1.,  1.,  1.],
---                            [ 2.,  2.,  2.]]
--- 
--- Supported sparse operations:
--- 
---    broadcast_add(csr, dense(1D)) = dense
---    broadcast_add(dense(1D), csr) = dense
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_basic.cc:L58
---
---
---@param lhs any @NDArray | First input to the function
---@param rhs any @NDArray | Second input to the function
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.broadcast_add(lhs, rhs, out, name, kwargs)
end

--- Returns element-wise division of the input arrays with broadcasting.
--- 
--- ### Example
--- 
---    x = [[ 6.,  6.,  6.],
---         [ 6.,  6.,  6.]]
--- 
---    y = [[ 2.],
---         [ 3.]]
--- 
---    broadcast_div(x, y) = [[ 3.,  3.,  3.],
---                           [ 2.,  2.,  2.]]
--- 
--- Supported sparse operations:
--- 
---    broadcast_div(csr, dense(1D)) = csr
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_basic.cc:L187
---
---
---@param lhs any @NDArray | First input to the function
---@param rhs any @NDArray | Second input to the function
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.broadcast_div(lhs, rhs, out, name, kwargs)
end

--- Returns element-wise difference of the input arrays with broadcasting.
--- 
--- `broadcast_minus` is an alias to the function `broadcast_sub`.
--- 
--- ### Example
--- 
---    x = [[ 1.,  1.,  1.],
---         [ 1.,  1.,  1.]]
--- 
---    y = [[ 0.],
---         [ 1.]]
--- 
---    broadcast_sub(x, y) = [[ 1.,  1.,  1.],
---                           [ 0.,  0.,  0.]]
--- 
---    broadcast_minus(x, y) = [[ 1.,  1.,  1.],
---                             [ 0.,  0.,  0.]]
--- 
--- Supported sparse operations:
--- 
---    broadcast_sub/minus(csr, dense(1D)) = dense
---    broadcast_sub/minus(dense(1D), csr) = dense
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_basic.cc:L106
---
---
---@param lhs any @NDArray | First input to the function
---@param rhs any @NDArray | Second input to the function
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.broadcast_minus(lhs, rhs, out, name, kwargs)
end

--- Returns element-wise product of the input arrays with broadcasting.
--- 
--- ### Example
--- 
---    x = [[ 1.,  1.,  1.],
---         [ 1.,  1.,  1.]]
--- 
---    y = [[ 0.],
---         [ 1.]]
--- 
---    broadcast_mul(x, y) = [[ 0.,  0.,  0.],
---                           [ 1.,  1.,  1.]]
--- 
--- Supported sparse operations:
--- 
---    broadcast_mul(csr, dense(1D)) = csr
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_basic.cc:L146
---
---
---@param lhs any @NDArray | First input to the function
---@param rhs any @NDArray | Second input to the function
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.broadcast_mul(lhs, rhs, out, name, kwargs)
end

--- Returns element-wise sum of the input arrays with broadcasting.
--- 
--- `broadcast_plus` is an alias to the function `broadcast_add`.
--- 
--- ### Example
--- 
---    x = [[ 1.,  1.,  1.],
---         [ 1.,  1.,  1.]]
--- 
---    y = [[ 0.],
---         [ 1.]]
--- 
---    broadcast_add(x, y) = [[ 1.,  1.,  1.],
---                           [ 2.,  2.,  2.]]
--- 
---    broadcast_plus(x, y) = [[ 1.,  1.,  1.],
---                            [ 2.,  2.,  2.]]
--- 
--- Supported sparse operations:
--- 
---    broadcast_add(csr, dense(1D)) = dense
---    broadcast_add(dense(1D), csr) = dense
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_basic.cc:L58
---
---
---@param lhs any @NDArray | First input to the function
---@param rhs any @NDArray | Second input to the function
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.broadcast_plus(lhs, rhs, out, name, kwargs)
end

--- Returns element-wise difference of the input arrays with broadcasting.
--- 
--- `broadcast_minus` is an alias to the function `broadcast_sub`.
--- 
--- ### Example
--- 
---    x = [[ 1.,  1.,  1.],
---         [ 1.,  1.,  1.]]
--- 
---    y = [[ 0.],
---         [ 1.]]
--- 
---    broadcast_sub(x, y) = [[ 1.,  1.,  1.],
---                           [ 0.,  0.,  0.]]
--- 
---    broadcast_minus(x, y) = [[ 1.,  1.,  1.],
---                             [ 0.,  0.,  0.]]
--- 
--- Supported sparse operations:
--- 
---    broadcast_sub/minus(csr, dense(1D)) = dense
---    broadcast_sub/minus(dense(1D), csr) = dense
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_basic.cc:L106
---
---
---@param lhs any @NDArray | First input to the function
---@param rhs any @NDArray | Second input to the function
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.broadcast_sub(lhs, rhs, out, name, kwargs)
end

--- Casts tensor storage type to the new type.
--- 
--- When an NDArray with default storage type is cast to csr or row_sparse storage,
--- the result is compact, which means:
--- 
--- - for csr, zero values will not be retained
--- - for row_sparse, row slices of all zeros will not be retained
--- 
--- The storage type of ``cast_storage`` output depends on stype parameter:
--- 
--- - cast_storage(csr, 'default') = default
--- - cast_storage(row_sparse, 'default') = default
--- - cast_storage(default, 'csr') = csr
--- - cast_storage(default, 'row_sparse') = row_sparse
--- - cast_storage(csr, 'csr') = csr
--- - cast_storage(row_sparse, 'row_sparse') = row_sparse
--- 
--- ### Example
--- 
---     dense = [[ 0.,  1.,  0.],
---              [ 2.,  0.,  3.],
---              [ 0.,  0.,  0.],
---              [ 0.,  0.,  0.]]
--- 
---     # cast to row_sparse storage type
---     rsp = cast_storage(dense, 'row_sparse')
---     rsp.indices = [0, 1]
---     rsp.values = [[ 0.,  1.,  0.],
---                   [ 2.,  0.,  3.]]
--- 
---     # cast to csr storage type
---     csr = cast_storage(dense, 'csr')
---     csr.indices = [1, 0, 2]
---     csr.values = [ 1.,  2.,  3.]
---     csr.indptr = [0, 1, 3, 3, 3]
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\cast_storage.cc:L71
---
---
---@param data any @NDArray | The input.
---@param stype any @{'csr', 'default', 'row_sparse'}, required | Output storage type.
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.cast_storage(data, stype, out, name, kwargs)
end

--- Returns element-wise cube-root value of the input.
--- 
--- .. math::
---    cbrt(x) = \sqrt[3]{x}
--- 
--- ### Example
--- 
---    cbrt([1, 8, -125]) = [1, 2, -5]
--- 
--- The storage type of ``cbrt`` output depends upon the input storage type:
--- 
---    - cbrt(default) = default
---    - cbrt(row_sparse) = row_sparse
---    - cbrt(csr) = csr
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\elemwise_unary_op_basic.cc:L950
---
---
---@param data any @NDArray | The input array.
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.cbrt(data, out, name, kwargs)
end

--- Returns element-wise ceiling of the input.
--- 
--- The ceil of the scalar x is the smallest integer i, such that i >= x.
--- 
--- ### Example
--- 
---    ceil([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-2., -1.,  2.,  2.,  3.]
--- 
--- The storage type of ``ceil`` output depends upon the input storage type:
--- 
---    - ceil(default) = default
---    - ceil(row_sparse) = row_sparse
---    - ceil(csr) = csr
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\elemwise_unary_op_basic.cc:L786
---
---
---@param data any @NDArray | The input array.
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.ceil(data, out, name, kwargs)
end

--- Clips (limits) the values in an array.
--- 
--- Given an interval, values outside the interval are clipped to the interval edges.
--- Clipping ``x`` between `a_min` and `a_x` would be::
--- 
---    clip(x, a_min, a_max) = max(min(x, a_max), a_min))
--- 
--- ### Example
--- 
---     x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
--- 
---     clip(x,1,8) = [ 1.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  8.]
--- 
--- The storage type of ``clip`` output depends on storage types of inputs and the a_min, a_max \
--- parameter values:
--- 
---    - clip(default) = default
---    - clip(row_sparse, a_min <= 0, a_max >= 0) = row_sparse
---    - clip(csr, a_min <= 0, a_max >= 0) = csr
---    - clip(row_sparse, a_min < 0, a_max < 0) = default
---    - clip(row_sparse, a_min > 0, a_max > 0) = default
---    - clip(csr, a_min < 0, a_max < 0) = csr
---    - clip(csr, a_min > 0, a_max > 0) = csr
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\matrix_op.cc:L725
---
---
---@param data any @NDArray | Input array.
---@param a_min any @float, required | Minimum value
---@param a_max any @float, required | Maximum value
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.clip(data, a_min, a_max, out, name, kwargs)
end

--- Joins input arrays along a given axis.
--- 
--- ### Note: `Concat` is deprecated. Use `concat` instead.
--- 
--- The dimensions of the input arrays should be the same except the axis along
--- which they will be concatenated.
--- The dimension of the output array along the concatenated axis will be equal
--- to the sum of the corresponding dimensions of the input arrays.
--- 
--- The storage type of ``concat`` output depends on storage types of inputs
--- 
--- - concat(csr, csr, ..., csr, dim=0) = csr
--- - otherwise, ``concat`` generates output with default storage
--- 
--- ### Example
--- 
---    x = [[1,1],[2,2]]
---    y = [[3,3],[4,4],[5,5]]
---    z = [[6,6], [7,7],[8,8]]
--- 
---    concat(x,y,z,dim=0) = [[ 1.,  1.],
---                           [ 2.,  2.],
---                           [ 3.,  3.],
---                           [ 4.,  4.],
---                           [ 5.,  5.],
---                           [ 6.,  6.],
---                           [ 7.,  7.],
---                           [ 8.,  8.]]
--- 
---    Note that you cannot concat x,y,z along dimension 1 since dimension
---    0 is not the same for all the input arrays.
--- 
---    concat(y,z,dim=1) = [[ 3.,  3.,  6.,  6.],
---                          [ 4.,  4.,  7.,  7.],
---                          [ 5.,  5.,  8.,  8.]]
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\nn\concat.cc:L371
---
---
---@param data any @NDArray[] | List of arrays to concatenate
---@param dim any @int, optional, default='1' | the dimension to be concated.
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.concat(...)
end

--- Computes the element-wise cosine of the input array.
--- 
--- The input should be in radians (:math:`2\pi` rad equals 360 degrees).
--- 
--- .. math::
---    cos([0, \pi/4, \pi/2]) = [1, 0.707, 0]
--- 
--- The storage type of ``cos`` output is always dense
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\elemwise_unary_op_trig.cc:L89
---
---
---@param data any @NDArray | The input array.
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.cos(data, out, name, kwargs)
end

--- Returns the hyperbolic cosine  of the input array, computed element-wise.
--- 
--- .. math::
---    cosh(x) = 0.5\times(exp(x) + exp(-x))
--- 
--- The storage type of ``cosh`` output is always dense
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\elemwise_unary_op_trig.cc:L272
---
---
---@param data any @NDArray | The input array.
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.cosh(data, out, name, kwargs)
end

--- Converts each element of the input array from radians to degrees.
--- 
--- .. math::
---    degrees([0, \pi/2, \pi, 3\pi/2, 2\pi]) = [0, 90, 180, 270, 360]
--- 
--- The storage type of ``degrees`` output depends upon the input storage type:
--- 
---    - degrees(default) = default
---    - degrees(row_sparse) = row_sparse
---    - degrees(csr) = csr
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\elemwise_unary_op_trig.cc:L219
---
---
---@param data any @NDArray | The input array.
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.degrees(data, out, name, kwargs)
end

--- Dot product of two arrays.
--- 
--- ``dot``'s behavior depends on the input array dimensions:
--- 
--- - 1-D arrays: inner product of vectors
--- - 2-D arrays: matrix multiplication
--- - N-D arrays: a sum product over the last axis of the first input and the first
---   axis of the second input
--- 
---   For example, given 3-D ``x`` with shape `(n,m,k)` and ``y`` with shape `(k,r,s)`, the
---   result array will have shape `(n,m,r,s)`. It is computed by::
--- 
---     dot(x,y)[i,j,a,b] = sum(x[i,j,:]*y[:,a,b])
--- 
---   Example::
--- 
---     x = reshape([0,1,2,3,4,5,6,7], shape=(2,2,2))
---     y = reshape([7,6,5,4,3,2,1,0], shape=(2,2,2))
---     dot(x,y)[0,0,1,1] = 0
---     sum(x[0,0,:]*y[:,1,1]) = 0
--- 
--- The storage type of ``dot`` output depends on storage types of inputs, transpose option and
--- forward_stype option for output storage type. Implemented sparse operations include:
--- 
--- - dot(default, default, transpose_a=True/False, transpose_b=True/False) = default
--- - dot(csr, default, transpose_a=True) = default
--- - dot(csr, default, transpose_a=True) = row_sparse
--- - dot(csr, default) = default
--- - dot(csr, row_sparse) = default
--- - dot(default, csr) = csr (CPU only)
--- - dot(default, csr, forward_stype='default') = default
--- - dot(default, csr, transpose_b=True, forward_stype='default') = default
--- 
--- If the combination of input storage types and forward_stype does not match any of the
--- above patterns, ``dot`` will fallback and generate output with default storage.
--- 
--- .. Note::
--- 
---     If the storage type of the lhs is "csr", the storage type of gradient w.r.t rhs will be
---     "row_sparse". Only a subset of optimizers support sparse gradients, including SGD, AdaGrad
---     and Adam. Note that by default lazy updates is turned on, which may perform differently
---     from standard updates. For more details, please check the Optimization API at:
---     https://mxnet.incubator.apache.org/api/python/optimization/optimization.html
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\dot.cc:L77
---
---
---@param lhs any @NDArray | The first input
---@param rhs any @NDArray | The second input
---@param transpose_a any @boolean, optional, default=0 | If true then transpose the first input before dot.
---@param transpose_b any @boolean, optional, default=0 | If true then transpose the second input before dot.
---@param forward_stype any @{None, 'csr', 'default', 'row_sparse'},optional, default='None' | The desired storage type of the forward output given by user, if thecombination of input storage types and this hint does not matchany implemented ones, the dot operator will perform fallback operationand still produce an output of the desired storage type.
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.dot(lhs, rhs, transpose_a, transpose_b, forward_stype, out, name, kwargs)
end

--- Adds arguments element-wise.
--- 
--- The storage type of ``elemwise_add`` output depends on storage types of inputs
--- 
---    - elemwise_add(row_sparse, row_sparse) = row_sparse
---    - elemwise_add(csr, csr) = csr
---    - elemwise_add(default, csr) = default
---    - elemwise_add(csr, default) = default
---    - elemwise_add(default, rsp) = default
---    - elemwise_add(rsp, default) = default
---    - otherwise, ``elemwise_add`` generates output with default storage
--- 
--- 
---
---
---@param lhs any @NDArray | first input
---@param rhs any @NDArray | second input
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.elemwise_add(lhs, rhs, out, name, kwargs)
end

--- Divides arguments element-wise.
--- 
--- The storage type of ``elemwise_div`` output is always dense
--- 
--- 
---
---
---@param lhs any @NDArray | first input
---@param rhs any @NDArray | second input
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.elemwise_div(lhs, rhs, out, name, kwargs)
end

--- Multiplies arguments element-wise.
--- 
--- The storage type of ``elemwise_mul`` output depends on storage types of inputs
--- 
---    - elemwise_mul(default, default) = default
---    - elemwise_mul(row_sparse, row_sparse) = row_sparse
---    - elemwise_mul(default, row_sparse) = row_sparse
---    - elemwise_mul(row_sparse, default) = row_sparse
---    - elemwise_mul(csr, csr) = csr
---    - otherwise, ``elemwise_mul`` generates output with default storage
--- 
--- 
---
---
---@param lhs any @NDArray | first input
---@param rhs any @NDArray | second input
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.elemwise_mul(lhs, rhs, out, name, kwargs)
end

--- Subtracts arguments element-wise.
--- 
--- The storage type of ``elemwise_sub`` output depends on storage types of inputs
--- 
---    - elemwise_sub(row_sparse, row_sparse) = row_sparse
---    - elemwise_sub(csr, csr) = csr
---    - elemwise_sub(default, csr) = default
---    - elemwise_sub(csr, default) = default
---    - elemwise_sub(default, rsp) = default
---    - elemwise_sub(rsp, default) = default
---    - otherwise, ``elemwise_sub`` generates output with default storage
--- 
--- 
---
---
---@param lhs any @NDArray | first input
---@param rhs any @NDArray | second input
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.elemwise_sub(lhs, rhs, out, name, kwargs)
end

--- Returns element-wise exponential value of the input.
--- 
--- .. math::
---    exp(x) = e^x \approx 2.718^x
--- 
--- ### Example
--- 
---    exp([0, 1, 2]) = [1., 2.71828175, 7.38905621]
--- 
--- The storage type of ``exp`` output is always dense
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\elemwise_unary_op_basic.cc:L1044
---
---
---@param data any @NDArray | The input array.
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.exp(data, out, name, kwargs)
end

--- Returns ``exp(x) - 1`` computed element-wise on the input.
--- 
--- This function provides greater precision than ``exp(x) - 1`` for small values of ``x``.
--- 
--- The storage type of ``expm1`` output depends upon the input storage type:
--- 
---    - expm1(default) = default
---    - expm1(row_sparse) = row_sparse
---    - expm1(csr) = csr
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\elemwise_unary_op_basic.cc:L1189
---
---
---@param data any @NDArray | The input array.
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.expm1(data, out, name, kwargs)
end

--- Returns element-wise rounded value to the nearest \
--- integer towards zero of the input.
--- 
--- ### Example
--- 
---    fix([-2.1, -1.9, 1.9, 2.1]) = [-2., -1.,  1., 2.]
--- 
--- The storage type of ``fix`` output depends upon the input storage type:
--- 
---    - fix(default) = default
---    - fix(row_sparse) = row_sparse
---    - fix(csr) = csr
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\elemwise_unary_op_basic.cc:L843
---
---
---@param data any @NDArray | The input array.
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.fix(data, out, name, kwargs)
end

--- Returns element-wise floor of the input.
--- 
--- The floor of the scalar x is the largest integer i, such that i <= x.
--- 
--- ### Example
--- 
---    floor([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-3., -2.,  1.,  1.,  2.]
--- 
--- The storage type of ``floor`` output depends upon the input storage type:
--- 
---    - floor(default) = default
---    - floor(row_sparse) = row_sparse
---    - floor(csr) = csr
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\elemwise_unary_op_basic.cc:L805
---
---
---@param data any @NDArray | The input array.
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.floor(data, out, name, kwargs)
end

--- Update function for Ftrl optimizer.
--- Referenced from *Ad Click Prediction: a View from the Trenches*, available at
--- http://dl.acm.org/citation.cfm?id=2488200.
--- 
--- It updates the weights using::
--- 
---  rescaled_grad = clip(grad * rescale_grad, clip_gradient)
---  z += rescaled_grad - (sqrt(n + rescaled_grad**2) - sqrt(n)) * weight / learning_rate
---  n += rescaled_grad**2
---  w = (sign(z) * lamda1 - z) / ((beta + sqrt(n)) / learning_rate + wd) * (abs(z) > lamda1)
--- 
--- If w, z and n are all of ``row_sparse`` storage type,
--- only the row slices whose indices appear in grad.indices are updated (for w, z and n)::
--- 
---  for row in grad.indices:
---      rescaled_grad[row] = clip(grad[row] * rescale_grad, clip_gradient)
---      z[row] += rescaled_grad[row] - (sqrt(n[row] + rescaled_grad[row]**2) - sqrt(n[row])) * weight[row] / learning_rate
---      n[row] += rescaled_grad[row]**2
---      w[row] = (sign(z[row]) * lamda1 - z[row]) / ((beta + sqrt(n[row])) / learning_rate + wd) * (abs(z[row]) > lamda1)
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\optimizer_op.cc:L874
---
---
---@param weight any @NDArray | Weight
---@param grad any @NDArray | Gradient
---@param z any @NDArray | z
---@param n any @NDArray | Square of grad
---@param lr any @float, required | Learning rate
---@param lamda1 any @float, optional, default=0.00999999978 | The L1 regularization coefficient.
---@param beta any @float, optional, default=1 | Per-Coordinate Learning Rate beta.
---@param wd any @float, optional, default=0 | Weight decay augments the objective function with a regularization term that penalizes large weights. The penalty scales with the square of the magnitude of each weight.
---@param rescale_grad any @float, optional, default=1 | Rescale gradient to grad = rescale_grad*grad.
---@param clip_gradient any @float, optional, default=-1 | Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.ftrl_update(weight, grad, z, n, lr, lamda1, beta, wd, rescale_grad, clip_gradient, out, name, kwargs)
end

--- Returns the gamma function (extension of the factorial function \
--- to the reals), computed element-wise on the input array.
--- 
--- The storage type of ``gamma`` output is always dense
--- 
--- 
---
---
---@param data any @NDArray | The input array.
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.gamma(data, out, name, kwargs)
end

--- Returns element-wise log of the absolute value of the gamma function \
--- of the input.
--- 
--- The storage type of ``gammaln`` output is always dense
--- 
--- 
---
---
---@param data any @NDArray | The input array.
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.gammaln(data, out, name, kwargs)
end

--- Returns element-wise Natural logarithmic value of the input.
--- 
--- The natural logarithm is logarithm in base *e*, so that ``log(exp(x)) = x``
--- 
--- The storage type of ``log`` output is always dense
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\elemwise_unary_op_basic.cc:L1057
---
---
---@param data any @NDArray | The input array.
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.log(data, out, name, kwargs)
end

--- Returns element-wise Base-10 logarithmic value of the input.
--- 
--- ``10**log10(x) = x``
--- 
--- The storage type of ``log10`` output is always dense
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\elemwise_unary_op_basic.cc:L1074
---
---
---@param data any @NDArray | The input array.
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.log10(data, out, name, kwargs)
end

--- Returns element-wise ``log(1 + x)`` value of the input.
--- 
--- This function is more accurate than ``log(1 + x)``  for small ``x`` so that
--- :math:`1+x\approx 1`
--- 
--- The storage type of ``log1p`` output depends upon the input storage type:
--- 
---    - log1p(default) = default
---    - log1p(row_sparse) = row_sparse
---    - log1p(csr) = csr
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\elemwise_unary_op_basic.cc:L1171
---
---
---@param data any @NDArray | The input array.
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.log1p(data, out, name, kwargs)
end

--- Returns element-wise Base-2 logarithmic value of the input.
--- 
--- ``2**log2(x) = x``
--- 
--- The storage type of ``log2`` output is always dense
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\elemwise_unary_op_basic.cc:L1086
---
---
---@param data any @NDArray | The input array.
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.log2(data, out, name, kwargs)
end

--- Make your own loss function in network construction.
--- 
--- This operator accepts a customized loss function symbol as a terminal loss and
--- the symbol should be an operator with no backward dependency.
--- The output of this function is the gradient of loss with respect to the input data.
--- 
--- For example, if you are a making a cross entropy loss function. Assume ``out`` is the
--- predicted output and ``label`` is the true label, then the cross entropy can be defined as::
--- 
---   cross_entropy = label * log(out) + (1 - label) * log(1 - out)
---   loss = make_loss(cross_entropy)
--- 
--- We will need to use ``make_loss`` when we are creating our own loss function or we want to
--- combine multiple loss functions. Also we may want to stop some variables' gradients
--- from backpropagation. See more detail in ``BlockGrad`` or ``stop_gradient``.
--- 
--- The storage type of ``make_loss`` output depends upon the input storage type:
--- 
---    - make_loss(default) = default
---    - make_loss(row_sparse) = row_sparse
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\elemwise_unary_op_basic.cc:L332
---
---
---@param data any @NDArray | The input array.
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.make_loss(data, out, name, kwargs)
end

--- Computes the mean of array elements over given axes.
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\broadcast_reduce_op_value.cc:L132
---
---
---@param data any @NDArray | The input
---@param axis any @Shape or None, optional, default=None | The axis or axes along which to perform the reduction.      The default, `axis=()`, will compute over all elements into a      scalar array with shape `(1,)`.      If `axis` is int, a reduction is performed on a particular axis.      If `axis` is a tuple of ints, a reduction is performed on all the axes      specified in the tuple.      If `exclude` is true, reduction will be performed on the axes that are      NOT in axis instead.      Negative values means indexing from right to left.
---@param keepdims any @boolean, optional, default=0 | If this is set to `True`, the reduced axes are left in the result as dimension with size one.
---@param exclude any @boolean, optional, default=0 | Whether to perform reduction on axis that are NOT in axis instead.
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.mean(data, axis, keepdims, exclude, out, name, kwargs)
end

--- Numerical negative of the argument, element-wise.
--- 
--- The storage type of ``negative`` output depends upon the input storage type:
--- 
---    - negative(default) = default
---    - negative(row_sparse) = row_sparse
---    - negative(csr) = csr
--- 
--- 
---
---
---@param data any @NDArray | The input array.
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.negative(data, out, name, kwargs)
end

--- Computes the norm on an NDArray.
--- 
--- This operator computes the norm on an NDArray with the specified axis, depending
--- on the value of the ord parameter. By default, it computes the L2 norm on the entire
--- array. Currently only ord=2 supports sparse ndarrays.
--- 
--- ### Examples
--- 
---   x = [[[1, 2],
---         [3, 4]],
---        [[2, 2],
---         [5, 6]]]
--- 
---   norm(x, ord=2, axis=1) = [[3.1622777 4.472136 ]
---                             [5.3851647 6.3245554]]
--- 
---   norm(x, ord=1, axis=1) = [[4., 6.],
---                             [7., 8.]]
--- 
---   rsp = x.cast_storage('row_sparse')
--- 
---   norm(rsp) = [5.47722578]
--- 
---   csr = x.cast_storage('csr')
--- 
---   norm(csr) = [5.47722578]
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\broadcast_reduce_op_value.cc:L350
---
---
---@param data any @NDArray | The input
---@param ord any @int, optional, default='2' | Order of the norm. Currently ord=1 and ord=2 is supported.
---@param axis any @Shape or None, optional, default=None | The axis or axes along which to perform the reduction.      The default, `axis=()`, will compute over all elements into a      scalar array with shape `(1,)`.      If `axis` is int, a reduction is performed on a particular axis.      If `axis` is a 2-tuple, it specifies the axes that hold 2-D matrices,      and the matrix norms of these matrices are computed.
---@param out_dtype any @{None, 'float16', 'float32', 'float64', 'int32', 'int64', 'int8'},optional, default='None' | The data type of the output.
---@param keepdims any @boolean, optional, default=0 | If this is set to `True`, the reduced axis is left in the result as dimension with size one.
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.norm(data, ord, axis, out_dtype, keepdims, out, name, kwargs)
end

--- Converts each element of the input array from degrees to radians.
--- 
--- .. math::
---    radians([0, 90, 180, 270, 360]) = [0, \pi/2, \pi, 3\pi/2, 2\pi]
--- 
--- The storage type of ``radians`` output depends upon the input storage type:
--- 
---    - radians(default) = default
---    - radians(row_sparse) = row_sparse
---    - radians(csr) = csr
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\elemwise_unary_op_trig.cc:L238
---
---
---@param data any @NDArray | The input array.
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.radians(data, out, name, kwargs)
end

--- Computes rectified linear activation.
--- 
--- .. math::
---    max(features, 0)
--- 
--- The storage type of ``relu`` output depends upon the input storage type:
--- 
---    - relu(default) = default
---    - relu(row_sparse) = row_sparse
---    - relu(csr) = csr
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\elemwise_unary_op_basic.cc:L85
---
---
---@param data any @NDArray | The input array.
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.relu(data, out, name, kwargs)
end

--- pick rows specified by user input index array from a row sparse matrix
--- and save them in the output sparse matrix.
--- 
--- ### Example
--- 
---   data = [[1, 2], [3, 4], [5, 6]]
---   indices = [0, 1, 3]
---   shape = (4, 2)
---   rsp_in = row_sparse(data, indices)
---   to_retain = [0, 3]
---   rsp_out = retain(rsp_in, to_retain)
---   rsp_out.values = [[1, 2], [5, 6]]
---   rsp_out.indices = [0, 3]
--- 
--- The storage type of ``retain`` output depends on storage types of inputs
--- 
--- - retain(row_sparse, default) = row_sparse
--- - otherwise, ``retain`` is not supported
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\sparse_retain.cc:L53
---
---
---@param data any @NDArray | The input array for sparse_retain operator.
---@param indices any @NDArray | The index array of rows ids that will be retained.
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.retain(data, indices, out, name, kwargs)
end

--- Returns element-wise rounded value to the nearest integer of the input.
--- 
--- .. note::
---    - For input ``n.5`` ``rint`` returns ``n`` while ``round`` returns ``n+1``.
---    - For input ``-n.5`` both ``rint`` and ``round`` returns ``-n-1``.
--- 
--- ### Example
--- 
---    rint([-1.5, 1.5, -1.9, 1.9, 2.1]) = [-2.,  1., -2.,  2.,  2.]
--- 
--- The storage type of ``rint`` output depends upon the input storage type:
--- 
---    - rint(default) = default
---    - rint(row_sparse) = row_sparse
---    - rint(csr) = csr
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\elemwise_unary_op_basic.cc:L767
---
---
---@param data any @NDArray | The input array.
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.rint(data, out, name, kwargs)
end

--- Returns element-wise rounded value to the nearest integer of the input.
--- 
--- ### Example
--- 
---    round([-1.5, 1.5, -1.9, 1.9, 2.1]) = [-2.,  2., -2.,  2.,  2.]
--- 
--- The storage type of ``round`` output depends upon the input storage type:
--- 
---   - round(default) = default
---   - round(row_sparse) = row_sparse
---   - round(csr) = csr
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\elemwise_unary_op_basic.cc:L746
---
---
---@param data any @NDArray | The input array.
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.round(data, out, name, kwargs)
end

--- Returns element-wise inverse square-root value of the input.
--- 
--- .. math::
---    rsqrt(x) = 1/\sqrt{x}
--- 
--- ### Example
--- 
---    rsqrt([4,9,16]) = [0.5, 0.33333334, 0.25]
--- 
--- The storage type of ``rsqrt`` output is always dense
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\elemwise_unary_op_basic.cc:L927
---
---
---@param data any @NDArray | The input array.
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.rsqrt(data, out, name, kwargs)
end

--- Momentum update function for Stochastic Gradient Descent (SGD) optimizer.
--- 
--- Momentum update has better convergence rates on neural networks. Mathematically it looks
--- like below:
--- 
--- .. math::
--- 
---   v_1 = \alpha * \nabla J(W_0)\\
---   v_t = \gamma v_{t-1} - \alpha * \nabla J(W_{t-1})\\
---   W_t = W_{t-1} + v_t
--- 
--- It updates the weights using::
--- 
---   v = momentum * v - learning_rate * gradient
---   weight += v
--- 
--- Where the parameter ``momentum`` is the decay rate of momentum estimates at each epoch.
--- 
--- However, if grad's storage type is ``row_sparse``, ``lazy_update`` is True and weight's storage
--- type is the same as momentum's storage type,
--- only the row slices whose indices appear in grad.indices are updated (for both weight and momentum)::
--- 
---   for row in gradient.indices:
---       v[row] = momentum[row] * v[row] - learning_rate * gradient[row]
---       weight[row] += v[row]
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\optimizer_op.cc:L563
---
---
---@param weight any @NDArray | Weight
---@param grad any @NDArray | Gradient
---@param mom any @NDArray | Momentum
---@param lr any @float, required | Learning rate
---@param momentum any @float, optional, default=0 | The decay rate of momentum estimates at each epoch.
---@param wd any @float, optional, default=0 | Weight decay augments the objective function with a regularization term that penalizes large weights. The penalty scales with the square of the magnitude of each weight.
---@param rescale_grad any @float, optional, default=1 | Rescale gradient to grad = rescale_grad*grad.
---@param clip_gradient any @float, optional, default=-1 | Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).
---@param lazy_update any @boolean, optional, default=1 | If true, lazy updates are applied if gradient's stype is row_sparse and both weight and momentum have the same stype
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.sgd_mom_update(weight, grad, mom, lr, momentum, wd, rescale_grad, clip_gradient, lazy_update, out, name, kwargs)
end

--- Update function for Stochastic Gradient Descent (SGD) optimizer.
--- 
--- It updates the weights using::
--- 
---  weight = weight - learning_rate * (gradient + wd * weight)
--- 
--- However, if gradient is of ``row_sparse`` storage type and ``lazy_update`` is True,
--- only the row slices whose indices appear in grad.indices are updated::
--- 
---  for row in gradient.indices:
---      weight[row] = weight[row] - learning_rate * (gradient[row] + wd * weight[row])
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\optimizer_op.cc:L522
---
---
---@param weight any @NDArray | Weight
---@param grad any @NDArray | Gradient
---@param lr any @float, required | Learning rate
---@param wd any @float, optional, default=0 | Weight decay augments the objective function with a regularization term that penalizes large weights. The penalty scales with the square of the magnitude of each weight.
---@param rescale_grad any @float, optional, default=1 | Rescale gradient to grad = rescale_grad*grad.
---@param clip_gradient any @float, optional, default=-1 | Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).
---@param lazy_update any @boolean, optional, default=1 | If true, lazy updates are applied if gradient's stype is row_sparse.
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.sgd_update(weight, grad, lr, wd, rescale_grad, clip_gradient, lazy_update, out, name, kwargs)
end

--- Computes sigmoid of x element-wise.
--- 
--- .. math::
---    y = 1 / (1 + exp(-x))
--- 
--- The storage type of ``sigmoid`` output is always dense
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\elemwise_unary_op_basic.cc:L119
---
---
---@param data any @NDArray | The input array.
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.sigmoid(data, out, name, kwargs)
end

--- Returns element-wise sign of the input.
--- 
--- ### Example
--- 
---    sign([-2, 0, 3]) = [-1, 0, 1]
--- 
--- The storage type of ``sign`` output depends upon the input storage type:
--- 
---    - sign(default) = default
---    - sign(row_sparse) = row_sparse
---    - sign(csr) = csr
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\elemwise_unary_op_basic.cc:L727
---
---
---@param data any @NDArray | The input array.
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.sign(data, out, name, kwargs)
end

--- Computes the element-wise sine of the input array.
--- 
--- The input should be in radians (:math:`2\pi` rad equals 360 degrees).
--- 
--- .. math::
---    sin([0, \pi/4, \pi/2]) = [0, 0.707, 1]
--- 
--- The storage type of ``sin`` output depends upon the input storage type:
--- 
---    - sin(default) = default
---    - sin(row_sparse) = row_sparse
---    - sin(csr) = csr
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\elemwise_unary_op_trig.cc:L46
---
---
---@param data any @NDArray | The input array.
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.sin(data, out, name, kwargs)
end

--- Returns the hyperbolic sine of the input array, computed element-wise.
--- 
--- .. math::
---    sinh(x) = 0.5\times(exp(x) - exp(-x))
--- 
--- The storage type of ``sinh`` output depends upon the input storage type:
--- 
---    - sinh(default) = default
---    - sinh(row_sparse) = row_sparse
---    - sinh(csr) = csr
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\elemwise_unary_op_trig.cc:L257
---
---
---@param data any @NDArray | The input array.
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.sinh(data, out, name, kwargs)
end

--- Slices a region of the array.
--- 
--- ### Note: ``crop`` is deprecated. Use ``slice`` instead.
--- 
--- This function returns a sliced array between the indices given
--- by `begin` and `end` with the corresponding `step`.
--- 
--- For an input array of ``shape=(d_0, d_1, ..., d_n-1)``,
--- slice operation with ``begin=(b_0, b_1...b_m-1)``,
--- ``end=(e_0, e_1, ..., e_m-1)``, and ``step=(s_0, s_1, ..., s_m-1)``,
--- where m <= n, results in an array with the shape
--- ``(|e_0-b_0|/|s_0|, ..., |e_m-1-b_m-1|/|s_m-1|, d_m, ..., d_n-1)``.
--- 
--- The resulting array's *k*-th dimension contains elements
--- from the *k*-th dimension of the input array starting
--- from index ``b_k`` (inclusive) with step ``s_k``
--- until reaching ``e_k`` (exclusive).
--- 
--- If the *k*-th elements are `None` in the sequence of `begin`, `end`,
--- and `step`, the following rule will be used to set default values.
--- If `s_k` is `None`, set `s_k=1`. If `s_k > 0`, set `b_k=0`, `e_k=d_k`;
--- else, set `b_k=d_k-1`, `e_k=-1`.
--- 
--- The storage type of ``slice`` output depends on storage types of inputs
--- 
--- - slice(csr) = csr
--- - otherwise, ``slice`` generates output with default storage
--- 
--- ### Note: When input data storage type is csr, it only supports
---    step=(), or step=(None,), or step=(1,) to generate a csr output.
---    For other step parameter values, it falls back to slicing
---    a dense tensor.
--- 
--- ### Example
--- 
---   x = [[  1.,   2.,   3.,   4.],
---        [  5.,   6.,   7.,   8.],
---        [  9.,  10.,  11.,  12.]]
--- 
---   slice(x, begin=(0,1), end=(2,4)) = [[ 2.,  3.,  4.],
---                                      [ 6.,  7.,  8.]]
---   slice(x, begin=(None, 0), end=(None, 3), step=(-1, 2)) = [[9., 11.],
---                                                             [5.,  7.],
---                                                             [1.,  3.]]
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\matrix_op.cc:L508
---
---
---@param data any @NDArray | Source input
---@param begin any @Shape(tuple), required | starting indices for the slice operation, supports negative indices.
---@param end_ any @Shape(tuple), required | ending indices for the slice operation, supports negative indices.
---@param step any @Shape(tuple), optional, default=[] | step for the slice operation, supports negative values.
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.slice(data, begin, end_, step, out, name, kwargs)
end

--- Returns element-wise square-root value of the input.
--- 
--- .. math::
---    \textrm{sqrt}(x) = \sqrt{x}
--- 
--- ### Example
--- 
---    sqrt([4, 9, 16]) = [2, 3, 4]
--- 
--- The storage type of ``sqrt`` output depends upon the input storage type:
--- 
---    - sqrt(default) = default
---    - sqrt(row_sparse) = row_sparse
---    - sqrt(csr) = csr
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\elemwise_unary_op_basic.cc:L907
---
---
---@param data any @NDArray | The input array.
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.sqrt(data, out, name, kwargs)
end

--- Returns element-wise squared value of the input.
--- 
--- .. math::
---    square(x) = x^2
--- 
--- ### Example
--- 
---    square([2, 3, 4]) = [4, 9, 16]
--- 
--- The storage type of ``square`` output depends upon the input storage type:
--- 
---    - square(default) = default
---    - square(row_sparse) = row_sparse
---    - square(csr) = csr
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\elemwise_unary_op_basic.cc:L883
---
---
---@param data any @NDArray | The input array.
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.square(data, out, name, kwargs)
end

--- Stops gradient computation.
--- 
--- Stops the accumulated gradient of the inputs from flowing through this operator
--- in the backward direction. In other words, this operator prevents the contribution
--- of its inputs to be taken into account for computing gradients.
--- 
--- ### Example
--- 
---   v1 = [1, 2]
---   v2 = [0, 1]
---   a = Variable('a')
---   b = Variable('b')
---   b_stop_grad = stop_gradient(3 * b)
---   loss = MakeLoss(b_stop_grad + a)
--- 
---   executor = loss.simple_bind(ctx=cpu(), a=(1,2), b=(1,2))
---   executor.forward(is_train=True, a=v1, b=v2)
---   executor.outputs
---   [ 1.  5.]
--- 
---   executor.backward()
---   executor.grad_arrays
---   [ 0.  0.]
---   [ 1.  1.]
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\elemwise_unary_op_basic.cc:L299
---
---
---@param data any @NDArray | The input array.
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.stop_gradient(data, out, name, kwargs)
end

--- Computes the sum of array elements over given axes.
--- 
--- .. Note::
--- 
---   `sum` and `sum_axis` are equivalent.
---   For ndarray of csr storage type summation along axis 0 and axis 1 is supported.
---   Setting keepdims or exclude to True will cause a fallback to dense operator.
--- 
--- ### Example
--- 
---   data = [[[1, 2], [2, 3], [1, 3]],
---           [[1, 4], [4, 3], [5, 2]],
---           [[7, 1], [7, 2], [7, 3]]]
--- 
---   sum(data, axis=1)
---   [[  4.   8.]
---    [ 10.   9.]
---    [ 21.   6.]]
--- 
---   sum(data, axis=[1,2])
---   [ 12.  19.  27.]
--- 
---   data = [[1, 2, 0],
---           [3, 0, 1],
---           [4, 1, 0]]
--- 
---   csr = cast_storage(data, 'csr')
--- 
---   sum(csr, axis=0)
---   [ 8.  3.  1.]
--- 
---   sum(csr, axis=1)
---   [ 3.  4.  5.]
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\broadcast_reduce_op_value.cc:L116
---
---
---@param data any @NDArray | The input
---@param axis any @Shape or None, optional, default=None | The axis or axes along which to perform the reduction.      The default, `axis=()`, will compute over all elements into a      scalar array with shape `(1,)`.      If `axis` is int, a reduction is performed on a particular axis.      If `axis` is a tuple of ints, a reduction is performed on all the axes      specified in the tuple.      If `exclude` is true, reduction will be performed on the axes that are      NOT in axis instead.      Negative values means indexing from right to left.
---@param keepdims any @boolean, optional, default=0 | If this is set to `True`, the reduced axes are left in the result as dimension with size one.
---@param exclude any @boolean, optional, default=0 | Whether to perform reduction on axis that are NOT in axis instead.
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.sum(data, axis, keepdims, exclude, out, name, kwargs)
end

--- Computes the element-wise tangent of the input array.
--- 
--- The input should be in radians (:math:`2\pi` rad equals 360 degrees).
--- 
--- .. math::
---    tan([0, \pi/4, \pi/2]) = [0, 1, -inf]
--- 
--- The storage type of ``tan`` output depends upon the input storage type:
--- 
---    - tan(default) = default
---    - tan(row_sparse) = row_sparse
---    - tan(csr) = csr
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\elemwise_unary_op_trig.cc:L139
---
---
---@param data any @NDArray | The input array.
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.tan(data, out, name, kwargs)
end

--- Returns the hyperbolic tangent of the input array, computed element-wise.
--- 
--- .. math::
---    tanh(x) = sinh(x) / cosh(x)
--- 
--- The storage type of ``tanh`` output depends upon the input storage type:
--- 
---    - tanh(default) = default
---    - tanh(row_sparse) = row_sparse
---    - tanh(csr) = csr
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\elemwise_unary_op_trig.cc:L290
---
---
---@param data any @NDArray | The input array.
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.tanh(data, out, name, kwargs)
end

--- Return the element-wise truncated value of the input.
--- 
--- The truncated value of the scalar x is the nearest integer i which is closer to
--- zero than x is. In short, the fractional part of the signed number x is discarded.
--- 
--- ### Example
--- 
---    trunc([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-2., -1.,  1.,  1.,  2.]
--- 
--- The storage type of ``trunc`` output depends upon the input storage type:
--- 
---    - trunc(default) = default
---    - trunc(row_sparse) = row_sparse
---    - trunc(csr) = csr
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\elemwise_unary_op_basic.cc:L825
---
---
---@param data any @NDArray | The input array.
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.trunc(data, out, name, kwargs)
end

--- Return the elements, either from x or y, depending on the condition.
--- 
--- Given three ndarrays, condition, x, and y, return an ndarray with the elements from x or y,
--- depending on the elements from condition are true or false. x and y must have the same shape.
--- If condition has the same shape as x, each element in the output array is from x if the
--- corresponding element in the condition is true, and from y if false.
--- 
--- If condition does not have the same shape as x, it must be a 1D array whose size is
--- the same as x's first dimension size. Each row of the output array is from x's row
--- if the corresponding element from condition is true, and from y's row if false.
--- 
--- Note that all non-zero values are interpreted as ``True`` in condition.
--- 
--- ### Examples
--- 
---   x = [[1, 2], [3, 4]]
---   y = [[5, 6], [7, 8]]
---   cond = [[0, 1], [-1, 0]]
--- 
---   where(cond, x, y) = [[5, 2], [3, 8]]
--- 
---   csr_cond = cast_storage(cond, 'csr')
--- 
---   where(csr_cond, x, y) = [[5, 2], [3, 8]]
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\control_flow_op.cc:L57
---
---
---@param condition any @NDArray | condition array
---@param x any @NDArray
---@param y any @NDArray
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.where(condition, x, y, out, name, kwargs)
end

--- Return an array of zeros with the same shape, type and storage type
--- as the input array.
--- 
--- The storage type of ``zeros_like`` output depends on the storage type of the input
--- 
--- - zeros_like(row_sparse) = row_sparse
--- - zeros_like(csr) = csr
--- - zeros_like(default) = default
--- 
--- ### Examples
--- 
---   x = [[ 1.,  1.,  1.],
---        [ 1.,  1.,  1.]]
--- 
---   zeros_like(x) = [[ 0.,  0.,  0.],
---                    [ 0.,  0.,  0.]]
--- 
--- 
---
---
---@param data any @NDArray | The input
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.zeros_like(data, out, name, kwargs)
end


return M