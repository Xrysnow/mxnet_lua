-- File content is auto-generated. Do not modify.

local SymbolBase = require('mx._ctypes.symbol').SymbolBase
local _symbol_creator = require('mx._ctypes.symbol')._symbol_creator
local NameManager = require('mx.name').NameManager
local AttrScope = require('mx.attribute').AttrScope
local _Null = require('mx.base')._Null
---@class mx.symbol.gen__internal
local M = {}

--- 
--- This function support variable length of positional input.
---
---
---@param data any @Symbol[] | input data list
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._CachedOp(...)
end

--- Special op to copy data cross device
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._CrossDeviceCopy(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._CustomFunction(name, attr, out, kwargs)
end

--- Divides arguments element-wise.
--- 
--- The storage type of ``elemwise_div`` output is always dense
--- 
--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._Div(lhs, rhs, name, attr, out, kwargs)
end

--- Divide an array with a scalar.
--- 
--- ``_div_scalar`` only operates on data array of input if input is sparse.
--- 
--- For example, if input of shape (100, 100) has only 2 non zero elements,
--- i.e. input.data = [5, 6], scalar = nan,
--- it will result output.data = [nan, nan] instead of 10000 nans.
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\elemwise_binary_scalar_op_basic.cc:L171
--- This function support variable length of positional input.
---
---
---@param data any @Symbol | source input
---@param scalar any @float | scalar input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._DivScalar(data, scalar, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._Equal(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param data any @Symbol | source input
---@param scalar any @float | scalar input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._EqualScalar(data, scalar, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._Greater(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param data any @Symbol | source input
---@param scalar any @float | scalar input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._GreaterEqualScalar(data, scalar, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param data any @Symbol | source input
---@param scalar any @float | scalar input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._GreaterScalar(data, scalar, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._Greater_Equal(lhs, rhs, name, attr, out, kwargs)
end

--- Given the "legs" of a right triangle, return its hypotenuse.
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\elemwise_binary_op_extended.cc:L79
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._Hypot(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param data any @Symbol | source input
---@param scalar any @float | scalar input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._HypotScalar(data, scalar, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._Lesser(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param data any @Symbol | source input
---@param scalar any @float | scalar input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._LesserEqualScalar(data, scalar, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param data any @Symbol | source input
---@param scalar any @float | scalar input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._LesserScalar(data, scalar, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._Lesser_Equal(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param data any @Symbol | source input
---@param scalar any @float | scalar input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._LogicalAndScalar(data, scalar, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param data any @Symbol | source input
---@param scalar any @float | scalar input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._LogicalOrScalar(data, scalar, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param data any @Symbol | source input
---@param scalar any @float | scalar input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._LogicalXorScalar(data, scalar, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._Logical_And(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._Logical_Or(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._Logical_Xor(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._Maximum(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param data any @Symbol | source input
---@param scalar any @float | scalar input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._MaximumScalar(data, scalar, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._Minimum(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param data any @Symbol | source input
---@param scalar any @float | scalar input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._MinimumScalar(data, scalar, name, attr, out, kwargs)
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
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._Minus(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param data any @Symbol | source input
---@param scalar any @float | scalar input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._MinusScalar(data, scalar, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._Mod(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param data any @Symbol | source input
---@param scalar any @float | scalar input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._ModScalar(data, scalar, name, attr, out, kwargs)
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
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._Mul(lhs, rhs, name, attr, out, kwargs)
end

--- Multiply an array with a scalar.
--- 
--- ``_mul_scalar`` only operates on data array of input if input is sparse.
--- 
--- For example, if input of shape (100, 100) has only 2 non zero elements,
--- i.e. input.data = [5, 6], scalar = nan,
--- it will result output.data = [nan, nan] instead of 10000 nans.
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\elemwise_binary_scalar_op_basic.cc:L149
--- This function support variable length of positional input.
---
---
---@param data any @Symbol | source input
---@param scalar any @float | scalar input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._MulScalar(data, scalar, name, attr, out, kwargs)
end

--- Stub for implementing an operator implemented in native frontend language with ndarray.
--- This function support variable length of positional input.
---
---
---@param data any @Symbol[] | Input data for the custom operator.
---@param info any @ptr, required
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._NDArray(...)
end

--- Stub for implementing an operator implemented in native frontend language.
--- This function support variable length of positional input.
---
---
---@param data any @Symbol[] | Input data for the custom operator.
---@param info any @ptr, required
---@param need_top_grad any @boolean, optional, default=1 | Whether this layer needs out grad for backward. Should be false for loss layers.
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._Native(...)
end

--- Place holder for variable who cannot perform gradient
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._NoGradient(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param data any @Symbol | source input
---@param scalar any @float | scalar input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._NotEqualScalar(data, scalar, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._Not_Equal(lhs, rhs, name, attr, out, kwargs)
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
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._Plus(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param data any @Symbol | source input
---@param scalar any @float | scalar input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._PlusScalar(data, scalar, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._Power(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param data any @Symbol | source input
---@param scalar any @float | scalar input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._PowerScalar(data, scalar, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param data any @Symbol | source input
---@param scalar any @float | scalar input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._RDivScalar(data, scalar, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param data any @Symbol | source input
---@param scalar any @float | scalar input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._RMinusScalar(data, scalar, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param data any @Symbol | source input
---@param scalar any @float | scalar input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._RModScalar(data, scalar, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param data any @Symbol | source input
---@param scalar any @float | scalar input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._RPowerScalar(data, scalar, name, attr, out, kwargs)
end

--- Update function for AdamW optimizer. AdamW is seen as a modification of
--- Adam by decoupling the weight decay from the optimization steps taken w.r.t. the loss function.
--- 
--- Adam update consists of the following steps, where g represents gradient and m, v
--- are 1st and 2nd order moment estimates (mean and variance).
--- 
--- .. math::
--- 
---  g_t = \nabla J(W_{t-1})\\
---  m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t\\
---  v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2\\
---  W_t = W_{t-1} - \eta_t (\alpha \frac{ m_t }{ \sqrt{ v_t } + \epsilon } + wd W_{t-1})
--- 
--- It updates the weights using::
--- 
---  m = beta1*m + (1-beta1)*grad
---  v = beta2*v + (1-beta2)*(grad**2)
---  w -= eta * (learning_rate * m / (sqrt(v) + epsilon) + w * wd)
--- 
--- Note that gradient is rescaled to grad = rescale_grad * grad. If rescale_grad is NaN, Inf, or 0,
--- the update is skipped.
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\contrib\adamw.cc:L120
--- This function support variable length of positional input.
---
---
---@param weight any @Symbol | Weight
---@param grad any @Symbol | Gradient
---@param mean any @Symbol | Moving mean
---@param var any @Symbol | Moving variance
---@param rescale_grad any @Symbol | Rescale gradient to rescale_grad * grad. If NaN, Inf, or 0, the update is skipped.
---@param lr any @float, required | Learning rate
---@param beta1 any @float, optional, default=0.899999976 | The decay rate for the 1st moment estimates.
---@param beta2 any @float, optional, default=0.999000013 | The decay rate for the 2nd moment estimates.
---@param epsilon any @float, optional, default=9.99999994e-09 | A small constant for numerical stability.
---@param wd any @float, optional, default=0 | Weight decay augments the objective function with a regularization term that penalizes large weights. The penalty scales with the square of the magnitude of each weight.
---@param eta any @float, required | Learning rate schedule multiplier
---@param clip_gradient any @float, optional, default=-1 | Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._adamw_update(weight, grad, mean, var, rescale_grad, lr, beta1, beta2, epsilon, wd, eta, clip_gradient, name, attr, out, kwargs)
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
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._add(lhs, rhs, name, attr, out, kwargs)
end

--- Return evenly spaced values within a given interval. Similar to Numpy
--- This function support variable length of positional input.
---
---
---@param start any @double, required | Start of interval. The interval includes this value. The default start value is 0.
---@param stop any @double or None, optional, default=None | End of interval. The interval does not include this value, except in some cases where step is not an integer and floating point round-off affects the length of out.
---@param step any @double, optional, default=1 | Spacing between values.
---@param repeat_ any @int, optional, default='1' | The repeating time of all elements. E.g repeat=3, the element a will be repeated three times --> a, a, a.
---@param infer_range any @boolean, optional, default=0 | When set to True, infer the stop position from the start, step, repeat, and output tensor size.
---@param ctx any @string, optional, default='' | Context of output, in format [cpu|gpu|cpu_pinned](n).Only used for imperative calls.
---@param dtype any @{'float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8'},optional, default='float32' | Target data type.
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._arange(start, stop, step, repeat_, infer_range, ctx, dtype, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_Activation(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_BatchNorm(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_BatchNorm_v1(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_BilinearSampler(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_CachedOp(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_Concat(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_Convolution(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_Convolution_v1(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_Correlation(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_Crop(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_CuDNNBatchNorm(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_Custom(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_CustomFunction(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_Deconvolution(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_Dropout(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_Embedding(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_FullyConnected(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_GridGenerator(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_IdentityAttachKLSparseReg(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_InstanceNorm(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_L2Normalization(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_LRN(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_LayerNorm(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_LeakyReLU(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_MakeLoss(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_Pad(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_Pooling(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_Pooling_v1(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_RNN(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_ROIAlign(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_ROIPooling(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_SVMOutput(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_SequenceLast(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_SequenceMask(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_SequenceReverse(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_SliceChannel(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_SoftmaxActivation(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_SoftmaxOutput(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_SparseEmbedding(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_SpatialTransformer(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_SwapAxis(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_UpSampling(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward__CrossDeviceCopy(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward__NDArray(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward__Native(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward__contrib_DeformableConvolution(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward__contrib_DeformablePSROIPooling(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward__contrib_MultiBoxDetection(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward__contrib_MultiBoxPrior(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward__contrib_MultiBoxTarget(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward__contrib_MultiProposal(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward__contrib_PSROIPooling(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward__contrib_Proposal(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward__contrib_SyncBatchNorm(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward__contrib_count_sketch(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward__contrib_fft(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward__contrib_ifft(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_abs(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_add(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_amp_cast(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param grad any @Symbol[] | Gradients
---@param num_outputs any @int, required | Number of input/output pairs to be casted to the widest type.
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_amp_multicast(...)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_arccos(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_arccosh(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_arcsin(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_arcsinh(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_arctan(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_arctanh(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_batch_dot(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_broadcast_add(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_broadcast_div(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_broadcast_hypot(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_broadcast_maximum(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_broadcast_minimum(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_broadcast_mod(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_broadcast_mul(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_broadcast_power(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_broadcast_sub(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_cast(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_cbrt(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_clip(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_cond(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_contrib_AdaptiveAvgPooling2D(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_contrib_BilinearResize2D(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param is_ascend any @boolean, optional, default=0 | Use ascend order for scores instead of descending. Please set threshold accordingly.
---@param threshold any @float, required | Ignore matching when score < thresh, if is_ascend=false, or ignore score > thresh, if is_ascend=true.
---@param topk any @int, optional, default='-1' | Limit the number of matches to topk, set -1 for no limit
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_contrib_bipartite_matching(is_ascend, threshold, topk, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param axis any @int, optional, default='0' | An integer that represents the axis in NDArray to mask from.
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_contrib_boolean_mask(axis, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param format any @{'center', 'corner'},optional, default='corner' | The box encoding type.  "corner" means boxes are encoded as [xmin, ymin, xmax, ymax], "center" means boxes are encodes as [x, y, width, height].
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_contrib_box_iou(format, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param overlap_thresh any @float, optional, default=0.5 | Overlapping(IoU) threshold to suppress object with smaller score.
---@param valid_thresh any @float, optional, default=0 | Filter input boxes to those whose scores greater than valid_thresh.
---@param topk any @int, optional, default='-1' | Apply nms to topk boxes with descending scores, -1 to no restriction.
---@param coord_start any @int, optional, default='2' | Start index of the consecutive 4 coordinates.
---@param score_index any @int, optional, default='1' | Index of the scores/confidence of boxes.
---@param id_index any @int, optional, default='-1' | Optional, index of the class categories, -1 to disable.
---@param background_id any @int, optional, default='-1' | Optional, id of the background class which will be ignored in nms.
---@param force_suppress any @boolean, optional, default=0 | Optional, if set false and id_index is provided, nms will only apply to boxes belongs to the same category
---@param in_format any @{'center', 'corner'},optional, default='corner' | The input box encoding type.  "corner" means boxes are encoded as [xmin, ymin, xmax, ymax], "center" means boxes are encodes as [x, y, width, height].
---@param out_format any @{'center', 'corner'},optional, default='corner' | The output box encoding type.  "corner" means boxes are encoded as [xmin, ymin, xmax, ymax], "center" means boxes are encodes as [x, y, width, height].
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_contrib_box_nms(overlap_thresh, valid_thresh, topk, coord_start, score_index, id_index, background_id, force_suppress, in_format, out_format, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_copy(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_cos(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_cosh(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_ctc_loss(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_degrees(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_diag(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_div(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param data any @Symbol | source input
---@param scalar any @float | scalar input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_div_scalar(data, scalar, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param transpose_a any @boolean, optional, default=0 | If true then transpose the first input before dot.
---@param transpose_b any @boolean, optional, default=0 | If true then transpose the second input before dot.
---@param forward_stype any @{None, 'csr', 'default', 'row_sparse'},optional, default='None' | The desired storage type of the forward output given by user, if thecombination of input storage types and this hint does not matchany implemented ones, the dot operator will perform fallback operationand still produce an output of the desired storage type.
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_dot(transpose_a, transpose_b, forward_stype, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_erf(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_erfinv(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_expm1(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_foreach(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_gamma(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_gammaln(lhs, rhs, name, attr, out, kwargs)
end

--- Accumulates data according to indices and get the result. It's the backward of
--- `gather_nd`.
--- 
--- Given `data` with shape `(Y_0, ..., Y_{K-1}, X_M, ..., X_{N-1})` and indices with shape
--- `(M, Y_0, ..., Y_{K-1})`, the output will have shape `(X_0, X_1, ..., X_{N-1})`,
--- where `M <= N`. If `M == N`, data shape should simply be `(Y_0, ..., Y_{K-1})`.
--- 
--- The elements in output is defined as follows::
--- 
---   output[indices[0, y_0, ..., y_{K-1}],
---          ...,
---          indices[M-1, y_0, ..., y_{K-1}],
---          x_M, ..., x_{N-1}] += data[y_0, ..., y_{K-1}, x_M, ..., x_{N-1}]
--- 
--- all other entries in output are 0 or the original value if AddTo is triggered.
--- 
--- ### Examples
--- 
---   data = [2, 3, 0]
---   indices = [[1, 1, 0], [0, 1, 0]]
---   shape = (2, 2)
---   _backward_gather_nd(data, indices, shape) = [[0, 0], [2, 3]] # Same as scatter_nd
--- 
---   -- The difference between scatter_nd and scatter_nd_acc is the latter will accumulate
---   --  the values that point to the same index.
--- 
---   data = [2, 3, 0]
---   indices = [[1, 1, 0], [1, 1, 0]]
---   shape = (2, 2)
---   _backward_gather_nd(data, indices, shape) = [[0, 0], [0, 5]]
--- 
--- 
--- This function support variable length of positional input.
---
---
---@param data any @Symbol | data
---@param indices any @Symbol | indices
---@param shape any @Shape(tuple), required | Shape of output.
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_gather_nd(data, indices, shape, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_hard_sigmoid(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_hypot(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param scalar any @float | scalar value
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_hypot_scalar(lhs, rhs, scalar, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_image_crop(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_image_normalize(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_linalg_extractdiag(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_linalg_extracttrian(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_linalg_gelqf(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_linalg_gemm(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_linalg_gemm2(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_linalg_inverse(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_linalg_makediag(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_linalg_maketrian(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_linalg_potrf(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_linalg_potri(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_linalg_sumlogdiag(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_linalg_syevd(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_linalg_syrk(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_linalg_trmm(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_linalg_trsm(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_linear_reg_out(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_log(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_log10(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_log1p(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_log2(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param args any @Symbol[] | Positional input arguments
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_log_softmax(...)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_logistic_reg_out(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_mae_reg_out(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_max(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_maximum(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param scalar any @float | scalar value
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_maximum_scalar(lhs, rhs, scalar, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_mean(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_min(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_minimum(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param scalar any @float | scalar value
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_minimum_scalar(lhs, rhs, scalar, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_mod(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param scalar any @float | scalar value
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_mod_scalar(lhs, rhs, scalar, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_moments(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_mul(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param data any @Symbol | source input
---@param scalar any @float | scalar input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_mul_scalar(data, scalar, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_nanprod(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_nansum(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_norm(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_pick(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_power(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param scalar any @float | scalar value
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_power_scalar(lhs, rhs, scalar, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_prod(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_radians(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_rcbrt(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param scalar any @float | scalar value
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_rdiv_scalar(lhs, rhs, scalar, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_reciprocal(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_relu(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_repeat(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_reshape(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_reverse(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param scalar any @float | scalar value
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_rmod_scalar(lhs, rhs, scalar, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param scalar any @float | scalar value
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_rpower_scalar(lhs, rhs, scalar, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_rsqrt(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_sample_multinomial(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_sigmoid(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_sign(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_sin(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_sinh(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_slice(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_slice_axis(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_slice_like(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_smooth_l1(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param args any @Symbol[] | Positional input arguments
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_softmax(...)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_softmax_cross_entropy(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param args any @Symbol[] | Positional input arguments
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_softmin(...)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_softsign(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_sparse_retain(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_sqrt(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_square(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_square_sum(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_squeeze(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_stack(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_sub(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_sum(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_take(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_tan(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_tanh(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_tile(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_topk(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_where(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._backward_while_loop(name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._broadcast_backward(name, attr, out, kwargs)
end

--- Run a if-then-else using user-defined condition and computation
--- 
--- From:C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\control_flow.cc:1212
--- This function support variable length of positional input.
---
---
---@param cond any @Symbol | Input graph for the condition.
---@param then_branch any @Symbol | Input graph for the then branch.
---@param else_branch any @Symbol | Input graph for the else branch.
---@param data any @Symbol[] | The input arrays that include data arrays and states.
---@param num_outputs any @int, required | The number of outputs of the subgraph.
---@param cond_input_locs any @, required | The locations of cond's inputs in the given inputs.
---@param then_input_locs any @, required | The locations of then's inputs in the given inputs.
---@param else_input_locs any @, required | The locations of else's inputs in the given inputs.
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._cond(...)
end

--- Returns a copy of the input.
--- 
--- From:C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\elemwise_unary_op_basic.cc:218
--- This function support variable length of positional input.
---
---
---@param data any @Symbol | The input array.
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._copy(data, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param data any @NDArray | input data
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._copyto(data, name, attr, out, kwargs)
end

--- Assign the rhs to a cropped subset of lhs.
--- 
--- Requirements
--- ------------
--- - output should be explicitly given and be the same as lhs.
--- - lhs and rhs are of the same data type, and on the same device.
--- 
--- 
--- From:C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\matrix_op.cc:537
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | Source input
---@param rhs any @Symbol | value to assign
---@param begin any @Shape(tuple), required | starting indices for the slice operation, supports negative indices.
---@param end_ any @Shape(tuple), required | ending indices for the slice operation, supports negative indices.
---@param step any @Shape(tuple), optional, default=[] | step for the slice operation, supports negative values.
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._crop_assign(lhs, rhs, begin, end_, step, name, attr, out, kwargs)
end

--- (Assign the scalar to a cropped subset of the input.
--- 
--- Requirements
--- ------------
--- - output should be explicitly given and be the same as input
--- )
--- 
--- From:C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\matrix_op.cc:562
--- This function support variable length of positional input.
---
---
---@param data any @Symbol | Source input
---@param scalar any @double, optional, default=0 | The scalar value for assignment.
---@param begin any @Shape(tuple), required | starting indices for the slice operation, supports negative indices.
---@param end_ any @Shape(tuple), required | ending indices for the slice operation, supports negative indices.
---@param step any @Shape(tuple), optional, default=[] | step for the slice operation, supports negative values.
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._crop_assign_scalar(data, scalar, begin, end_, step, name, attr, out, kwargs)
end

--- Pad image border with OpenCV. 
--- 
--- This function support variable length of positional input.
---
---
---@param src any @NDArray | source image
---@param top any @int, required | Top margin.
---@param bot any @int, required | Bottom margin.
---@param left any @int, required | Left margin.
---@param right any @int, required | Right margin.
---@param type any @int, optional, default='0' | Filling type (default=cv2.BORDER_CONSTANT).
---@param value any @double, optional, default=0 | (Deprecated! Use ``values`` instead.) Fill with single value.
---@param values any @, optional, default=[] | Fill with value(RGB[A] or gray), up to 4 channels.
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._cvcopyMakeBorder(src, top, bot, left, right, type, value, values, name, attr, out, kwargs)
end

--- Decode image with OpenCV. 
--- Note: return image in RGB by default, instead of OpenCV's default BGR.
--- This function support variable length of positional input.
---
---
---@param buf any @NDArray | Buffer containing binary encoded image
---@param flag any @int, optional, default='1' | Convert decoded image to grayscale (0) or color (1).
---@param to_rgb any @boolean, optional, default=1 | Whether to convert decoded image to mxnet's default RGB format (instead of opencv's default BGR).
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._cvimdecode(buf, flag, to_rgb, name, attr, out, kwargs)
end

--- Read and decode image with OpenCV. 
--- Note: return image in RGB by default, instead of OpenCV's default BGR.
--- This function support variable length of positional input.
---
---
---@param filename any @string, required | Name of the image file to be loaded.
---@param flag any @int, optional, default='1' | Convert decoded image to grayscale (0) or color (1).
---@param to_rgb any @boolean, optional, default=1 | Whether to convert decoded image to mxnet's default RGB format (instead of opencv's default BGR).
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._cvimread(filename, flag, to_rgb, name, attr, out, kwargs)
end

--- Resize image with OpenCV. 
--- 
--- This function support variable length of positional input.
---
---
---@param src any @NDArray | source image
---@param w any @int, required | Width of resized image.
---@param h any @int, required | Height of resized image.
---@param interp any @int, optional, default='1' | Interpolation method (default=cv2.INTER_LINEAR).
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._cvimresize(src, w, h, interp, name, attr, out, kwargs)
end

--- Divides arguments element-wise.
--- 
--- The storage type of ``elemwise_div`` output is always dense
--- 
--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._div(lhs, rhs, name, attr, out, kwargs)
end

--- Divide an array with a scalar.
--- 
--- ``_div_scalar`` only operates on data array of input if input is sparse.
--- 
--- For example, if input of shape (100, 100) has only 2 non zero elements,
--- i.e. input.data = [5, 6], scalar = nan,
--- it will result output.data = [nan, nan] instead of 10000 nans.
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\elemwise_binary_scalar_op_basic.cc:L171
--- This function support variable length of positional input.
---
---
---@param data any @Symbol | source input
---@param scalar any @float | scalar input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._div_scalar(data, scalar, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._equal(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param data any @Symbol | source input
---@param scalar any @float | scalar input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._equal_scalar(data, scalar, name, attr, out, kwargs)
end

--- Return a 2-D array with ones on the diagonal and zeros elsewhere.
--- This function support variable length of positional input.
---
---
---@param N any @, required | Number of rows in the output.
---@param M any @, optional, default=0 | Number of columns in the output. If 0, defaults to N
---@param k any @, optional, default=0 | Index of the diagonal. 0 (the default) refers to the main diagonal.A positive value refers to an upper diagonal.A negative value to a lower diagonal.
---@param ctx any @string, optional, default='' | Context of output, in format [cpu|gpu|cpu_pinned](n).Only used for imperative calls.
---@param dtype any @{'float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8'},optional, default='float32' | Target data type.
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._eye(N, M, k, ctx, dtype, name, attr, out, kwargs)
end

--- Run a for loop over an NDArray with user-defined computation
--- 
--- From:C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\control_flow.cc:1090
--- This function support variable length of positional input.
---
---
---@param fn any @Symbol | Input graph.
---@param data any @Symbol[] | The input arrays that include data arrays and states.
---@param num_outputs any @int, required | The number of outputs of the subgraph.
---@param num_out_data any @int, required | The number of output data of the subgraph.
---@param in_state_locs any @, required | The locations of loop states among the inputs.
---@param in_data_locs any @, required | The locations of input data among the inputs.
---@param remain_locs any @, required | The locations of remaining data among the inputs.
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._foreach(...)
end

--- fill target with a scalar value
--- This function support variable length of positional input.
---
---
---@param shape any @Shape(tuple), optional, default=None | The shape of the output
---@param ctx any @string, optional, default='' | Context of output, in format [cpu|gpu|cpu_pinned](n).Only used for imperative calls.
---@param dtype any @{'float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8'},optional, default='float32' | Target data type.
---@param value any @double, required | Value with which to fill newly created tensor
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._full(shape, ctx, dtype, value, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._grad_add(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._greater(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._greater_equal(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param data any @Symbol | source input
---@param scalar any @float | scalar input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._greater_equal_scalar(data, scalar, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param data any @Symbol | source input
---@param scalar any @float | scalar input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._greater_scalar(data, scalar, name, attr, out, kwargs)
end

--- This operators implements the histogram function.
--- 
--- ### Example
---   x = [[0, 1], [2, 2], [3, 4]]
---   histo, bin_edges = histogram(data=x, bin_bounds=[], bin_cnt=5, range=(0,5))
---   histo = [1, 1, 2, 1, 1]
---   bin_edges = [0., 1., 2., 3., 4.]
---   histo, bin_edges = histogram(data=x, bin_bounds=[0., 2.1, 3.])
---   histo = [4, 1]
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\histogram.cc:L136
--- This function support variable length of positional input.
---
---
---@param data any @Symbol | Input ndarray
---@param bins any @Symbol | Input ndarray
---@param bin_cnt any @int or None, optional, default='None' | Number of bins for uniform case
---@param range any @, optional, default=None | The lower and upper range of the bins. if not provided, range is simply (a.min(), a.max()). values outside the range are ignored. the first element of the range must be less than or equal to the second. range affects the automatic bin computation as well. while bin width is computed to be optimal based on the actual data within range, the bin count will fill the entire range including portions containing no data.
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._histogram(data, bins, bin_cnt, range, name, attr, out, kwargs)
end

--- Given the "legs" of a right triangle, return its hypotenuse.
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\elemwise_binary_op_extended.cc:L79
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._hypot(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param data any @Symbol | source input
---@param scalar any @float | scalar input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._hypot_scalar(data, scalar, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | First input.
---@param rhs any @Symbol | Second input.
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._identity_with_attr_like_rhs(lhs, rhs, name, attr, out, kwargs)
end

--- Decode an image, clip to (x0, y0, x1, y1), subtract mean, and write to buffer
--- This function support variable length of positional input.
---
---
---@param mean any @Symbol | image mean
---@param index any @int | buffer position for output
---@param x0 any @int | x0
---@param y0 any @int | y0
---@param x1 any @int | x1
---@param y1 any @int | y1
---@param c any @int | channel
---@param size any @int | length of str_img
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._imdecode(mean, index, x0, y0, x1, y1, c, size, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._lesser(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._lesser_equal(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param data any @Symbol | source input
---@param scalar any @float | scalar input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._lesser_equal_scalar(data, scalar, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param data any @Symbol | source input
---@param scalar any @float | scalar input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._lesser_scalar(data, scalar, name, attr, out, kwargs)
end

--- Return evenly spaced numbers over a specified interval. Similar to Numpy
--- This function support variable length of positional input.
---
---
---@param start any @double, required | Start of interval. The interval includes this value. The default start value is 0.
---@param stop any @double or None, optional, default=None | End of interval. The interval does not include this value, except in some cases where step is not an integer and floating point round-off affects the length of out.
---@param step any @double, optional, default=1 | Spacing between values.
---@param repeat_ any @int, optional, default='1' | The repeating time of all elements. E.g repeat=3, the element a will be repeated three times --> a, a, a.
---@param infer_range any @boolean, optional, default=0 | When set to True, infer the stop position from the start, step, repeat, and output tensor size.
---@param ctx any @string, optional, default='' | Context of output, in format [cpu|gpu|cpu_pinned](n).Only used for imperative calls.
---@param dtype any @{'float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8'},optional, default='float32' | Target data type.
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._linspace(start, stop, step, repeat_, infer_range, ctx, dtype, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._logical_and(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param data any @Symbol | source input
---@param scalar any @float | scalar input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._logical_and_scalar(data, scalar, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._logical_or(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param data any @Symbol | source input
---@param scalar any @float | scalar input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._logical_or_scalar(data, scalar, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._logical_xor(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param data any @Symbol | source input
---@param scalar any @float | scalar input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._logical_xor_scalar(data, scalar, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._maximum(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param data any @Symbol | source input
---@param scalar any @float | scalar input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._maximum_scalar(data, scalar, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._minimum(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param data any @Symbol | source input
---@param scalar any @float | scalar input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._minimum_scalar(data, scalar, name, attr, out, kwargs)
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
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._minus(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param data any @Symbol | source input
---@param scalar any @float | scalar input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._minus_scalar(data, scalar, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._mod(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param data any @Symbol | source input
---@param scalar any @float | scalar input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._mod_scalar(data, scalar, name, attr, out, kwargs)
end

--- Update function for multi-precision AdamW optimizer.
--- 
--- AdamW is seen as a modification of Adam by decoupling the weight decay from the
--- optimization steps taken w.r.t. the loss function.
--- 
--- Adam update consists of the following steps, where g represents gradient and m, v
--- are 1st and 2nd order moment estimates (mean and variance).
--- 
--- .. math::
--- 
---  g_t = \nabla J(W_{t-1})\\
---  m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t\\
---  v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2\\
---  W_t = W_{t-1} - \eta_t (\alpha \frac{ m_t }{ \sqrt{ v_t } + \epsilon } + wd W_{t-1})
--- 
--- It updates the weights using::
--- 
---  m = beta1*m + (1-beta1)*grad
---  v = beta2*v + (1-beta2)*(grad**2)
---  w -= eta * (learning_rate * m / (sqrt(v) + epsilon) + w * wd)
--- 
--- Note that gradient is rescaled to grad = rescale_grad * grad. If rescale_grad is NaN, Inf, or 0,
--- the update is skipped.
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\contrib\adamw.cc:L77
--- This function support variable length of positional input.
---
---
---@param weight any @Symbol | Weight
---@param grad any @Symbol | Gradient
---@param mean any @Symbol | Moving mean
---@param var any @Symbol | Moving variance
---@param weight32 any @Symbol | Weight32
---@param rescale_grad any @Symbol | Rescale gradient to rescale_grad * grad. If NaN, Inf, or 0, the update is skipped.
---@param lr any @float, required | Learning rate
---@param beta1 any @float, optional, default=0.899999976 | The decay rate for the 1st moment estimates.
---@param beta2 any @float, optional, default=0.999000013 | The decay rate for the 2nd moment estimates.
---@param epsilon any @float, optional, default=9.99999994e-09 | A small constant for numerical stability.
---@param wd any @float, optional, default=0 | Weight decay augments the objective function with a regularization term that penalizes large weights. The penalty scales with the square of the magnitude of each weight.
---@param eta any @float, required | Learning rate schedule multiplier
---@param clip_gradient any @float, optional, default=-1 | Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._mp_adamw_update(weight, grad, mean, var, weight32, rescale_grad, lr, beta1, beta2, epsilon, wd, eta, clip_gradient, name, attr, out, kwargs)
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
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._mul(lhs, rhs, name, attr, out, kwargs)
end

--- Multiply an array with a scalar.
--- 
--- ``_mul_scalar`` only operates on data array of input if input is sparse.
--- 
--- For example, if input of shape (100, 100) has only 2 non zero elements,
--- i.e. input.data = [5, 6], scalar = nan,
--- it will result output.data = [nan, nan] instead of 10000 nans.
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\elemwise_binary_scalar_op_basic.cc:L149
--- This function support variable length of positional input.
---
---
---@param data any @Symbol | source input
---@param scalar any @float | scalar input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._mul_scalar(data, scalar, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._not_equal(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param data any @Symbol | source input
---@param scalar any @float | scalar input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._not_equal_scalar(data, scalar, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @NDArray | Left operand to the function.
---@param rhs any @NDArray | Right operand to the function.
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._onehot_encode(lhs, rhs, name, attr, out, kwargs)
end

--- fill target with ones
--- This function support variable length of positional input.
---
---
---@param shape any @Shape(tuple), optional, default=[] | The shape of the output
---@param ctx any @string, optional, default='' | Context of output, in format [cpu|gpu|cpu_pinned](n).Only used for imperative calls.
---@param dtype any @{'float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8'},optional, default='float32' | Target data type.
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._ones(shape, ctx, dtype, name, attr, out, kwargs)
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
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._plus(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param data any @Symbol | source input
---@param scalar any @float | scalar input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._plus_scalar(data, scalar, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._power(lhs, rhs, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param data any @Symbol | source input
---@param scalar any @float | scalar input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._power_scalar(data, scalar, name, attr, out, kwargs)
end

--- Converts a batch of index arrays into an array of flat indices. The operator follows numpy conventions so a single multi index is given by a column of the input matrix. The leading dimension may be left unspecified by using -1 as placeholder.  
--- 
--- ### Examples
---    
---    A = [[3,6,6],[4,5,1]]
---    ravel(A, shape=(7,6)) = [22,41,37]
---    ravel(A, shape=(-1,6)) = [22,41,37]
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\ravel.cc:L42
--- This function support variable length of positional input.
---
---
---@param data any @Symbol | Batch of multi-indices
---@param shape any @Shape(tuple), optional, default=None | Shape of the array into which the multi-indices apply.
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._ravel_multi_index(data, shape, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param data any @Symbol | source input
---@param scalar any @float | scalar input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._rdiv_scalar(data, scalar, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param data any @Symbol | source input
---@param scalar any @float | scalar input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._rminus_scalar(data, scalar, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param data any @Symbol | source input
---@param scalar any @float | scalar input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._rmod_scalar(data, scalar, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param data any @Symbol[] | List of arrays to concatenate
---@param dim any @int, optional, default='1' | the dimension to be concated.
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._rnn_param_concat(...)
end

--- 
--- This function support variable length of positional input.
---
---
---@param data any @Symbol | source input
---@param scalar any @float | scalar input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._rpower_scalar(data, scalar, name, attr, out, kwargs)
end

--- Concurrent sampling from multiple
--- exponential distributions with parameters lambda (rate).
--- 
--- The parameters of the distributions are provided as an input array.
--- Let *[s]* be the shape of the input array, *n* be the dimension of *[s]*, *[t]*
--- be the shape specified as the parameter of the operator, and *m* be the dimension
--- of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape *[s]x[t]*.
--- 
--- For any valid *n*-dimensional index *i* with respect to the input array, *output[i]*
--- will be an *m*-dimensional array that holds randomly drawn samples from the distribution
--- which is parameterized by the input value at index *i*. If the shape parameter of the
--- operator is not set, then one sample will be drawn per distribution and the output array
--- has the same shape as the input array.
--- 
--- ### Examples
--- 
---    lam = [ 1.0, 8.5 ]
--- 
---    // Draw a single sample for each distribution
---    sample_exponential(lam) = [ 0.51837951,  0.09994757]
--- 
---    // Draw a vector containing two samples for each distribution
---    sample_exponential(lam, shape=(2)) = [[ 0.51837951,  0.19866663],
---                                          [ 0.09994757,  0.50447971]]
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\random\multisample_op.cc:L284
--- This function support variable length of positional input.
---
---
---@param lam any @Symbol | Lambda (rate) parameters of the distributions.
---@param shape any @Shape(tuple), optional, default=None | Shape to be sampled from each random distribution.
---@param dtype any @{'None', 'float16', 'float32', 'float64'},optional, default='None' | DType of the output in case this can't be inferred. Defaults to float32 if not defined (dtype=None).
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._sample_exponential(lam, shape, dtype, name, attr, out, kwargs)
end

--- Concurrent sampling from multiple
--- gamma distributions with parameters *alpha* (shape) and *beta* (scale).
--- 
--- The parameters of the distributions are provided as input arrays.
--- Let *[s]* be the shape of the input arrays, *n* be the dimension of *[s]*, *[t]*
--- be the shape specified as the parameter of the operator, and *m* be the dimension
--- of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape *[s]x[t]*.
--- 
--- For any valid *n*-dimensional index *i* with respect to the input arrays, *output[i]*
--- will be an *m*-dimensional array that holds randomly drawn samples from the distribution
--- which is parameterized by the input values at index *i*. If the shape parameter of the
--- operator is not set, then one sample will be drawn per distribution and the output array
--- has the same shape as the input arrays.
--- 
--- ### Examples
--- 
---    alpha = [ 0.0, 2.5 ]
---    beta = [ 1.0, 0.7 ]
--- 
---    // Draw a single sample for each distribution
---    sample_gamma(alpha, beta) = [ 0.        ,  2.25797319]
--- 
---    // Draw a vector containing two samples for each distribution
---    sample_gamma(alpha, beta, shape=(2)) = [[ 0.        ,  0.        ],
---                                            [ 2.25797319,  1.70734084]]
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\random\multisample_op.cc:L282
--- This function support variable length of positional input.
---
---
---@param alpha any @Symbol | Alpha (shape) parameters of the distributions.
---@param shape any @Shape(tuple), optional, default=None | Shape to be sampled from each random distribution.
---@param dtype any @{'None', 'float16', 'float32', 'float64'},optional, default='None' | DType of the output in case this can't be inferred. Defaults to float32 if not defined (dtype=None).
---@param beta any @Symbol | Beta (scale) parameters of the distributions.
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._sample_gamma(alpha, beta, shape, dtype, name, attr, out, kwargs)
end

--- Concurrent sampling from multiple
--- generalized negative binomial distributions with parameters *mu* (mean) and *alpha* (dispersion).
--- 
--- The parameters of the distributions are provided as input arrays.
--- Let *[s]* be the shape of the input arrays, *n* be the dimension of *[s]*, *[t]*
--- be the shape specified as the parameter of the operator, and *m* be the dimension
--- of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape *[s]x[t]*.
--- 
--- For any valid *n*-dimensional index *i* with respect to the input arrays, *output[i]*
--- will be an *m*-dimensional array that holds randomly drawn samples from the distribution
--- which is parameterized by the input values at index *i*. If the shape parameter of the
--- operator is not set, then one sample will be drawn per distribution and the output array
--- has the same shape as the input arrays.
--- 
--- Samples will always be returned as a floating point data type.
--- 
--- ### Examples
--- 
---    mu = [ 2.0, 2.5 ]
---    alpha = [ 1.0, 0.1 ]
--- 
---    // Draw a single sample for each distribution
---    sample_generalized_negative_binomial(mu, alpha) = [ 0.,  3.]
--- 
---    // Draw a vector containing two samples for each distribution
---    sample_generalized_negative_binomial(mu, alpha, shape=(2)) = [[ 0.,  3.],
---                                                                  [ 3.,  1.]]
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\random\multisample_op.cc:L293
--- This function support variable length of positional input.
---
---
---@param mu any @Symbol | Means of the distributions.
---@param shape any @Shape(tuple), optional, default=None | Shape to be sampled from each random distribution.
---@param dtype any @{'None', 'float16', 'float32', 'float64'},optional, default='None' | DType of the output in case this can't be inferred. Defaults to float32 if not defined (dtype=None).
---@param alpha any @Symbol | Alpha (dispersion) parameters of the distributions.
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._sample_generalized_negative_binomial(mu, alpha, shape, dtype, name, attr, out, kwargs)
end

--- Concurrent sampling from multiple multinomial distributions.
--- 
--- *data* is an *n* dimensional array whose last dimension has length *k*, where
--- *k* is the number of possible outcomes of each multinomial distribution. This
--- operator will draw *shape* samples from each distribution. If shape is empty
--- one sample will be drawn from each distribution.
--- 
--- If *get_prob* is true, a second array containing log likelihood of the drawn
--- samples will also be returned. This is usually used for reinforcement learning
--- where you can provide reward as head gradient for this array to estimate
--- gradient.
--- 
--- Note that the input distribution must be normalized, i.e. *data* must sum to
--- 1 along its last axis.
--- 
--- ### Examples
--- 
---    probs = [[0, 0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1, 0]]
--- 
---    // Draw a single sample for each distribution
---    sample_multinomial(probs) = [3, 0]
--- 
---    // Draw a vector containing two samples for each distribution
---    sample_multinomial(probs, shape=(2)) = [[4, 2],
---                                            [0, 0]]
--- 
---    // requests log likelihood
---    sample_multinomial(probs, get_prob=True) = [2, 1], [0.2, 0.3]
--- 
--- This function support variable length of positional input.
---
---
---@param data any @Symbol | Distribution probabilities. Must sum to one on the last axis.
---@param shape any @Shape(tuple), optional, default=[] | Shape to be sampled from each random distribution.
---@param get_prob any @boolean, optional, default=0 | Whether to also return the log probability of sampled result. This is usually used for differentiating through stochastic variables, e.g. in reinforcement learning.
---@param dtype any @{'float16', 'float32', 'float64', 'int32', 'uint8'},optional, default='int32' | DType of the output in case this can't be inferred.
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._sample_multinomial(data, shape, get_prob, dtype, name, attr, out, kwargs)
end

--- Concurrent sampling from multiple
--- negative binomial distributions with parameters *k* (failure limit) and *p* (failure probability).
--- 
--- The parameters of the distributions are provided as input arrays.
--- Let *[s]* be the shape of the input arrays, *n* be the dimension of *[s]*, *[t]*
--- be the shape specified as the parameter of the operator, and *m* be the dimension
--- of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape *[s]x[t]*.
--- 
--- For any valid *n*-dimensional index *i* with respect to the input arrays, *output[i]*
--- will be an *m*-dimensional array that holds randomly drawn samples from the distribution
--- which is parameterized by the input values at index *i*. If the shape parameter of the
--- operator is not set, then one sample will be drawn per distribution and the output array
--- has the same shape as the input arrays.
--- 
--- Samples will always be returned as a floating point data type.
--- 
--- ### Examples
--- 
---    k = [ 20, 49 ]
---    p = [ 0.4 , 0.77 ]
--- 
---    // Draw a single sample for each distribution
---    sample_negative_binomial(k, p) = [ 15.,  16.]
--- 
---    // Draw a vector containing two samples for each distribution
---    sample_negative_binomial(k, p, shape=(2)) = [[ 15.,  50.],
---                                                 [ 16.,  12.]]
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\random\multisample_op.cc:L289
--- This function support variable length of positional input.
---
---
---@param k any @Symbol | Limits of unsuccessful experiments.
---@param shape any @Shape(tuple), optional, default=None | Shape to be sampled from each random distribution.
---@param dtype any @{'None', 'float16', 'float32', 'float64'},optional, default='None' | DType of the output in case this can't be inferred. Defaults to float32 if not defined (dtype=None).
---@param p any @Symbol | Failure probabilities in each experiment.
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._sample_negative_binomial(k, p, shape, dtype, name, attr, out, kwargs)
end

--- Concurrent sampling from multiple
--- normal distributions with parameters *mu* (mean) and *sigma* (standard deviation).
--- 
--- The parameters of the distributions are provided as input arrays.
--- Let *[s]* be the shape of the input arrays, *n* be the dimension of *[s]*, *[t]*
--- be the shape specified as the parameter of the operator, and *m* be the dimension
--- of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape *[s]x[t]*.
--- 
--- For any valid *n*-dimensional index *i* with respect to the input arrays, *output[i]*
--- will be an *m*-dimensional array that holds randomly drawn samples from the distribution
--- which is parameterized by the input values at index *i*. If the shape parameter of the
--- operator is not set, then one sample will be drawn per distribution and the output array
--- has the same shape as the input arrays.
--- 
--- ### Examples
--- 
---    mu = [ 0.0, 2.5 ]
---    sigma = [ 1.0, 3.7 ]
--- 
---    // Draw a single sample for each distribution
---    sample_normal(mu, sigma) = [-0.56410581,  0.95934606]
--- 
---    // Draw a vector containing two samples for each distribution
---    sample_normal(mu, sigma, shape=(2)) = [[-0.56410581,  0.2928229 ],
---                                           [ 0.95934606,  4.48287058]]
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\random\multisample_op.cc:L279
--- This function support variable length of positional input.
---
---
---@param mu any @Symbol | Means of the distributions.
---@param shape any @Shape(tuple), optional, default=None | Shape to be sampled from each random distribution.
---@param dtype any @{'None', 'float16', 'float32', 'float64'},optional, default='None' | DType of the output in case this can't be inferred. Defaults to float32 if not defined (dtype=None).
---@param sigma any @Symbol | Standard deviations of the distributions.
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._sample_normal(mu, sigma, shape, dtype, name, attr, out, kwargs)
end

--- Concurrent sampling from multiple
--- Poisson distributions with parameters lambda (rate).
--- 
--- The parameters of the distributions are provided as an input array.
--- Let *[s]* be the shape of the input array, *n* be the dimension of *[s]*, *[t]*
--- be the shape specified as the parameter of the operator, and *m* be the dimension
--- of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape *[s]x[t]*.
--- 
--- For any valid *n*-dimensional index *i* with respect to the input array, *output[i]*
--- will be an *m*-dimensional array that holds randomly drawn samples from the distribution
--- which is parameterized by the input value at index *i*. If the shape parameter of the
--- operator is not set, then one sample will be drawn per distribution and the output array
--- has the same shape as the input array.
--- 
--- Samples will always be returned as a floating point data type.
--- 
--- ### Examples
--- 
---    lam = [ 1.0, 8.5 ]
--- 
---    // Draw a single sample for each distribution
---    sample_poisson(lam) = [  0.,  13.]
--- 
---    // Draw a vector containing two samples for each distribution
---    sample_poisson(lam, shape=(2)) = [[  0.,   4.],
---                                      [ 13.,   8.]]
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\random\multisample_op.cc:L286
--- This function support variable length of positional input.
---
---
---@param lam any @Symbol | Lambda (rate) parameters of the distributions.
---@param shape any @Shape(tuple), optional, default=None | Shape to be sampled from each random distribution.
---@param dtype any @{'None', 'float16', 'float32', 'float64'},optional, default='None' | DType of the output in case this can't be inferred. Defaults to float32 if not defined (dtype=None).
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._sample_poisson(lam, shape, dtype, name, attr, out, kwargs)
end

--- Concurrent sampling from multiple
--- uniform distributions on the intervals given by *[low,high)*.
--- 
--- The parameters of the distributions are provided as input arrays.
--- Let *[s]* be the shape of the input arrays, *n* be the dimension of *[s]*, *[t]*
--- be the shape specified as the parameter of the operator, and *m* be the dimension
--- of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape *[s]x[t]*.
--- 
--- For any valid *n*-dimensional index *i* with respect to the input arrays, *output[i]*
--- will be an *m*-dimensional array that holds randomly drawn samples from the distribution
--- which is parameterized by the input values at index *i*. If the shape parameter of the
--- operator is not set, then one sample will be drawn per distribution and the output array
--- has the same shape as the input arrays.
--- 
--- ### Examples
--- 
---    low = [ 0.0, 2.5 ]
---    high = [ 1.0, 3.7 ]
--- 
---    // Draw a single sample for each distribution
---    sample_uniform(low, high) = [ 0.40451524,  3.18687344]
--- 
---    // Draw a vector containing two samples for each distribution
---    sample_uniform(low, high, shape=(2)) = [[ 0.40451524,  0.18017688],
---                                            [ 3.18687344,  3.68352246]]
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\random\multisample_op.cc:L277
--- This function support variable length of positional input.
---
---
---@param low any @Symbol | Lower bounds of the distributions.
---@param shape any @Shape(tuple), optional, default=None | Shape to be sampled from each random distribution.
---@param dtype any @{'None', 'float16', 'float32', 'float64'},optional, default='None' | DType of the output in case this can't be inferred. Defaults to float32 if not defined (dtype=None).
---@param high any @Symbol | Upper bounds of the distributions.
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._sample_uniform(low, high, shape, dtype, name, attr, out, kwargs)
end

--- Draw random samples from an an approximately log-uniform
--- or Zipfian distribution without replacement.
--- 
--- This operation takes a 2-D shape `(batch_size, num_sampled)`,
--- and randomly generates *num_sampled* samples from the range of integers [0, range_max)
--- for each instance in the batch.
--- 
--- The elements in each instance are drawn without replacement from the base distribution.
--- The base distribution for this operator is an approximately log-uniform or Zipfian distribution:
--- 
---   P(class) = (log(class + 2) - log(class + 1)) / log(range_max + 1)
--- 
--- Additionaly, it also returns the number of trials used to obtain `num_sampled` samples for
--- each instance in the batch.
--- 
--- ### Example
--- 
---    samples, trials = _sample_unique_zipfian(750000, shape=(4, 8192))
---    unique(samples[0]) = 8192
---    unique(samples[3]) = 8192
---    trials[0] = 16435
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\random\unique_sample_op.cc:L66
--- This function support variable length of positional input.
---
---
---@param range_max any @int, required | The number of possible classes.
---@param shape any @Shape(tuple), optional, default=None | 2-D shape of the output, where shape[0] is the batch size, and shape[1] is the number of candidates to sample for each batch.
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._sample_unique_zipfian(range_max, shape, name, attr, out, kwargs)
end

--- Divides arguments element-wise.  If the left-hand-side input is 'row_sparse', then
--- only the values which exist in the left-hand sparse array are computed.  The 'missing' values
--- are ignored.
--- 
--- The storage type of ``_scatter_elemwise_div`` output depends on storage types of inputs
--- 
--- - _scatter_elemwise_div(row_sparse, row_sparse) = row_sparse
--- - _scatter_elemwise_div(row_sparse, dense) = row_sparse
--- - _scatter_elemwise_div(row_sparse, csr) = row_sparse
--- - otherwise, ``_scatter_elemwise_div`` behaves exactly like elemwise_div and generates output
--- with default storage
--- 
--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._scatter_elemwise_div(lhs, rhs, name, attr, out, kwargs)
end

--- Subtracts a scalar to a tensor element-wise.  If the left-hand-side input is
--- 'row_sparse' or 'csr', then only the values which exist in the left-hand sparse array are computed.
--- The 'missing' values are ignored.
--- 
--- The storage type of ``_scatter_minus_scalar`` output depends on storage types of inputs
--- 
--- - _scatter_minus_scalar(row_sparse, scalar) = row_sparse
--- - _scatter_minus_scalar(csr, scalar) = csr
--- - otherwise, ``_scatter_minus_scalar`` behaves exactly like _minus_scalar and generates output
--- with default storage
--- 
--- 
--- This function support variable length of positional input.
---
---
---@param data any @Symbol | source input
---@param scalar any @float | scalar input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._scatter_minus_scalar(data, scalar, name, attr, out, kwargs)
end

--- Adds a scalar to a tensor element-wise.  If the left-hand-side input is
--- 'row_sparse' or 'csr', then only the values which exist in the left-hand sparse array are computed.
--- The 'missing' values are ignored.
--- 
--- The storage type of ``_scatter_plus_scalar`` output depends on storage types of inputs
--- 
--- - _scatter_plus_scalar(row_sparse, scalar) = row_sparse
--- - _scatter_plus_scalar(csr, scalar) = csr
--- - otherwise, ``_scatter_plus_scalar`` behaves exactly like _plus_scalar and generates output
--- with default storage
--- 
--- 
--- This function support variable length of positional input.
---
---
---@param data any @Symbol | source input
---@param scalar any @float | scalar input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._scatter_plus_scalar(data, scalar, name, attr, out, kwargs)
end

--- This operator has the same functionality as scatter_nd
--- except that it does not reset the elements not indexed by the input
--- index `NDArray` in the input data `NDArray`. output should be explicitly
--- given and be the same as lhs.
--- 
--- ### Note: This operator is for internal use only.
--- 
--- ### Examples
--- 
---   data = [2, 3, 0]
---   indices = [[1, 1, 0], [0, 1, 0]]
---   out = [[1, 1], [1, 1]]
---   _scatter_set_nd(lhs=out, rhs=data, indices=indices, out=out)
---   out = [[0, 1], [2, 3]]
--- 
--- 
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | source input
---@param rhs any @Symbol | value to assign
---@param indices any @Symbol | indices
---@param shape any @Shape(tuple), required | Shape of output.
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._scatter_set_nd(lhs, rhs, indices, shape, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---
---@param src any @real_t | Source input to the function.
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._set_value(src, name, attr, out, kwargs)
end

--- _sg_mkldnn_conv
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\subgraph\mkldnn\mkldnn_conv.cc:L770
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._sg_mkldnn_conv(name, attr, out, kwargs)
end

--- _sg_mkldnn_fully_connected
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\subgraph\mkldnn\mkldnn_fc.cc:L410
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._sg_mkldnn_fully_connected(name, attr, out, kwargs)
end

--- Randomly shuffle the elements.
--- 
--- This shuffles the array along the first axis.
--- The order of the elements in each subarray does not change.
--- For example, if a 2D array is given, the order of the rows randomly changes,
--- but the order of the elements in each row does not change.
--- 
--- This function support variable length of positional input.
---
---
---@param data any @Symbol | Data to be shuffled.
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._shuffle(data, name, attr, out, kwargs)
end

--- Assign the rhs to a cropped subset of lhs.
--- 
--- Requirements
--- ------------
--- - output should be explicitly given and be the same as lhs.
--- - lhs and rhs are of the same data type, and on the same device.
--- 
--- 
--- From:C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\matrix_op.cc:537
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | Source input
---@param rhs any @Symbol | value to assign
---@param begin any @Shape(tuple), required | starting indices for the slice operation, supports negative indices.
---@param end_ any @Shape(tuple), required | ending indices for the slice operation, supports negative indices.
---@param step any @Shape(tuple), optional, default=[] | step for the slice operation, supports negative values.
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._slice_assign(lhs, rhs, begin, end_, step, name, attr, out, kwargs)
end

--- (Assign the scalar to a cropped subset of the input.
--- 
--- Requirements
--- ------------
--- - output should be explicitly given and be the same as input
--- )
--- 
--- From:C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\matrix_op.cc:562
--- This function support variable length of positional input.
---
---
---@param data any @Symbol | Source input
---@param scalar any @double, optional, default=0 | The scalar value for assignment.
---@param begin any @Shape(tuple), required | starting indices for the slice operation, supports negative indices.
---@param end_ any @Shape(tuple), required | ending indices for the slice operation, supports negative indices.
---@param step any @Shape(tuple), optional, default=[] | step for the slice operation, supports negative values.
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._slice_assign_scalar(data, scalar, begin, end_, step, name, attr, out, kwargs)
end

--- Splits an array along a particular axis into multiple sub-arrays.
--- 
--- ### Example
--- 
---    x  = [[[ 1.]
---           [ 2.]]
---          [[ 3.]
---           [ 4.]]
---          [[ 5.]
---           [ 6.]]]
---    x.shape = (3, 2, 1)
--- 
---    y = split_v2(x, axis=1, indices_or_sections=2) // a list of 2 arrays with shape (3, 1, 1)
---    y = [[[ 1.]]
---         [[ 3.]]
---         [[ 5.]]]
--- 
---        [[[ 2.]]
---         [[ 4.]]
---         [[ 6.]]]
--- 
---    y[0].shape = (3, 1, 1)
--- 
---    z = split_v2(x, axis=0, indices_or_sections=3) // a list of 3 arrays with shape (1, 2, 1)
---    z = [[[ 1.]
---          [ 2.]]]
--- 
---        [[[ 3.]
---          [ 4.]]]
--- 
---        [[[ 5.]
---          [ 6.]]]
--- 
---    z[0].shape = (1, 2, 1)
--- 
---    w = split_v2(x, axis=0, indices_or_sections=(1,)) // a list of 2 arrays with shape [(1, 2, 1), (2, 2, 1)]
---    w = [[[ 1.]
---          [ 2.]]]
--- 
---        [[[3.]
---          [4.]]
--- 
---         [[5.]
---          [6.]]]
--- 
---   w[0].shape = (1, 2, 1)
---   w[1].shape = (2, 2, 1)
--- 
--- `squeeze_axis=True` removes the axis with length 1 from the shapes of the output arrays.
--- **Note** that setting `squeeze_axis` to ``1`` removes axis with length 1 only
--- along the `axis` which it is split.
--- Also `squeeze_axis` can be set to true only if ``input.shape[axis] == indices_or_sections``.
--- 
--- ### Example
--- 
---    z = split_v2(x, axis=0, indices_or_sections=3, squeeze_axis=1) // a list of 3 arrays with shape (2, 1)
---    z = [[ 1.]
---         [ 2.]]
--- 
---        [[ 3.]
---         [ 4.]]
--- 
---        [[ 5.]
---         [ 6.]]
---    z[0].shape = (2, 1)
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\matrix_op.cc:L1192
--- This function support variable length of positional input.
---
---
---@param data any @Symbol | The input
---@param indices any @Shape(tuple), required | Indices of splits. The elements should denote the boundaries of at which split is performed along the `axis`.
---@param axis any @int, optional, default='1' | Axis along which to split.
---@param squeeze_axis any @boolean, optional, default=0 | If true, Removes the axis with length 1 from the shapes of the output arrays. **Note** that setting `squeeze_axis` to ``true`` removes axis with length 1 only along the `axis` which it is split. Also `squeeze_axis` can be set to ``true`` only if ``input.shape[axis] == num_outputs``.
---@param sections any @int, optional, default='0' | Number of sections if equally splitted. Default to 0 which means split by indices.
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._split_v2(data, indices, axis, squeeze_axis, sections, name, attr, out, kwargs)
end

--- 
--- This function support variable length of positional input.
---
---

---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._split_v2_backward(name, attr, out, kwargs)
end

--- Computes the square sum of array elements over a given axis
--- for row-sparse matrix. This is a temporary solution for fusing ops square and
--- sum together for row-sparse matrix to save memory for storing gradients.
--- It will become deprecated once the functionality of fusing operators is finished
--- in the future.
--- 
--- ### Example
--- 
---   dns = mx.nd.array([[0, 0], [1, 2], [0, 0], [3, 4], [0, 0]])
---   rsp = dns.tostype('row_sparse')
---   sum = mx.nd._internal._square_sum(rsp, axis=1)
---   sum = [0, 5, 0, 25, 0]
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\square_sum.cc:L63
--- This function support variable length of positional input.
---
---
---@param data any @Symbol | The input
---@param axis any @Shape or None, optional, default=None | The axis or axes along which to perform the reduction.      The default, `axis=()`, will compute over all elements into a      scalar array with shape `(1,)`.      If `axis` is int, a reduction is performed on a particular axis.      If `axis` is a tuple of ints, a reduction is performed on all the axes      specified in the tuple.      If `exclude` is true, reduction will be performed on the axes that are      NOT in axis instead.      Negative values means indexing from right to left.
---@param keepdims any @boolean, optional, default=0 | If this is set to `True`, the reduced axes are left in the result as dimension with size one.
---@param exclude any @boolean, optional, default=0 | Whether to perform reduction on axis that are NOT in axis instead.
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._square_sum(data, axis, keepdims, exclude, name, attr, out, kwargs)
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
--- This function support variable length of positional input.
---
---
---@param lhs any @Symbol | first input
---@param rhs any @Symbol | second input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._sub(lhs, rhs, name, attr, out, kwargs)
end

--- Converts an array of flat indices into a batch of index arrays. The operator follows numpy conventions so a single multi index is given by a column of the output matrix. The leading dimension may be left unspecified by using -1 as placeholder.  
--- 
--- ### Examples
--- 
---    A = [22,41,37]
---    unravel(A, shape=(7,6)) = [[3,6,6],[4,5,1]]
---    unravel(A, shape=(-1,6)) = [[3,6,6],[4,5,1]]
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\tensor\ravel.cc:L67
--- This function support variable length of positional input.
---
---
---@param data any @Symbol | Array of flat indices
---@param shape any @Shape(tuple), optional, default=None | Shape of the array into which the multi-indices apply.
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._unravel_index(data, shape, name, attr, out, kwargs)
end

--- Run a while loop over with user-defined condition and computation
--- 
--- From:C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\control_flow.cc:1151
--- This function support variable length of positional input.
---
---
---@param cond any @Symbol | Input graph for the loop condition.
---@param func any @Symbol | Input graph for the loop body.
---@param data any @Symbol[] | The input arrays that include data arrays and states.
---@param num_outputs any @int, required | The number of outputs of the subgraph.
---@param num_out_data any @int, required | The number of outputs from the function body.
---@param max_iterations any @int, required | Maximum number of iterations.
---@param cond_input_locs any @, required | The locations of cond's inputs in the given inputs.
---@param func_input_locs any @, required | The locations of func's inputs in the given inputs.
---@param func_var_locs any @, required | The locations of loop_vars among func's inputs.
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._while_loop(...)
end

--- fill target with zeros
--- This function support variable length of positional input.
---
---
---@param shape any @Shape(tuple), optional, default=[] | The shape of the output
---@param ctx any @string, optional, default='' | Context of output, in format [cpu|gpu|cpu_pinned](n).Only used for imperative calls.
---@param dtype any @{'float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8'},optional, default='float32' | Target data type.
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._zeros(shape, ctx, dtype, name, attr, out, kwargs)
end

--- fill target with zeros without default dtype
--- This function support variable length of positional input.
---
---
---@param shape any @Shape(tuple), optional, default=None | The shape of the output
---@param ctx any @string, optional, default='' | Context of output, in format [cpu|gpu|cpu_pinned](n).Only used for imperative calls.
---@param dtype any @int, optional, default='-1' | Target data type.
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M._zeros_without_dtype(shape, ctx, dtype, name, attr, out, kwargs)
end


return M