-- File content is auto-generated. Do not modify.

local NDArrayBase = require('mx.ndarray._internal').NDArrayBase
local _imperative_invoke = require('mx.ndarray._internal')._imperative_invoke
local _Null = require('mx.base')._Null
---@class mx.ndarray.gen_random
local M = {}

--- Draw random samples from an exponential distribution.
--- 
--- Samples are distributed according to an exponential distribution parametrized by *lambda* (rate).
--- 
--- ### Example
--- 
---    exponential(lam=4, shape=(2,2)) = [[ 0.0097189 ,  0.08999364],
---                                       [ 0.04146638,  0.31715935]]
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\random\sample_op.cc:L137
---
---
---@param lam any @float, optional, default=1 | Lambda parameter (rate) of the exponential distribution.
---@param shape any @Shape(tuple), optional, default=None | Shape of the output.
---@param ctx any @string, optional, default='' | Context of output, in format [cpu|gpu|cpu_pinned](n). Only used for imperative calls.
---@param dtype any @{'None', 'float16', 'float32', 'float64'},optional, default='None' | DType of the output in case this can't be inferred. Defaults to float32 if not defined (dtype=None).
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.exponential(lam, shape, ctx, dtype, out, name, kwargs)
end

--- Draw random samples from an exponential distribution according to the input array shape.
--- 
--- Samples are distributed according to an exponential distribution parametrized by *lambda* (rate).
--- 
--- ### Example
--- 
---    exponential(lam=4, data=ones(2,2)) = [[ 0.0097189 ,  0.08999364],
---                                          [ 0.04146638,  0.31715935]]
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\random\sample_op.cc:L242
---
---
---@param lam any @float, optional, default=1 | Lambda parameter (rate) of the exponential distribution.
---@param data any @NDArray | The input
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.exponential_like(data, lam, out, name, kwargs)
end

--- Draw random samples from a gamma distribution.
--- 
--- Samples are distributed according to a gamma distribution parametrized by *alpha* (shape) and *beta* (scale).
--- 
--- ### Example
--- 
---    gamma(alpha=9, beta=0.5, shape=(2,2)) = [[ 7.10486984,  3.37695289],
---                                             [ 3.91697288,  3.65933681]]
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\random\sample_op.cc:L125
---
---
---@param alpha any @float, optional, default=1 | Alpha parameter (shape) of the gamma distribution.
---@param beta any @float, optional, default=1 | Beta parameter (scale) of the gamma distribution.
---@param shape any @Shape(tuple), optional, default=None | Shape of the output.
---@param ctx any @string, optional, default='' | Context of output, in format [cpu|gpu|cpu_pinned](n). Only used for imperative calls.
---@param dtype any @{'None', 'float16', 'float32', 'float64'},optional, default='None' | DType of the output in case this can't be inferred. Defaults to float32 if not defined (dtype=None).
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.gamma(alpha, beta, shape, ctx, dtype, out, name, kwargs)
end

--- Draw random samples from a gamma distribution according to the input array shape.
--- 
--- Samples are distributed according to a gamma distribution parametrized by *alpha* (shape) and *beta* (scale).
--- 
--- ### Example
--- 
---    gamma(alpha=9, beta=0.5, data=ones(2,2)) = [[ 7.10486984,  3.37695289],
---                                                [ 3.91697288,  3.65933681]]
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\random\sample_op.cc:L231
---
---
---@param alpha any @float, optional, default=1 | Alpha parameter (shape) of the gamma distribution.
---@param beta any @float, optional, default=1 | Beta parameter (scale) of the gamma distribution.
---@param data any @NDArray | The input
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.gamma_like(data, alpha, beta, out, name, kwargs)
end

--- Draw random samples from a generalized negative binomial distribution.
--- 
--- Samples are distributed according to a generalized negative binomial distribution parametrized by
--- *mu* (mean) and *alpha* (dispersion). *alpha* is defined as *1/k* where *k* is the failure limit of the
--- number of unsuccessful experiments (generalized to real numbers).
--- Samples will always be returned as a floating point data type.
--- 
--- ### Example
--- 
---    generalized_negative_binomial(mu=2.0, alpha=0.3, shape=(2,2)) = [[ 2.,  1.],
---                                                                     [ 6.,  4.]]
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\random\sample_op.cc:L179
---
---
---@param mu any @float, optional, default=1 | Mean of the negative binomial distribution.
---@param alpha any @float, optional, default=1 | Alpha (dispersion) parameter of the negative binomial distribution.
---@param shape any @Shape(tuple), optional, default=None | Shape of the output.
---@param ctx any @string, optional, default='' | Context of output, in format [cpu|gpu|cpu_pinned](n). Only used for imperative calls.
---@param dtype any @{'None', 'float16', 'float32', 'float64'},optional, default='None' | DType of the output in case this can't be inferred. Defaults to float32 if not defined (dtype=None).
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.generalized_negative_binomial(mu, alpha, shape, ctx, dtype, out, name, kwargs)
end

--- Draw random samples from a generalized negative binomial distribution according to the
--- input array shape.
--- 
--- Samples are distributed according to a generalized negative binomial distribution parametrized by
--- *mu* (mean) and *alpha* (dispersion). *alpha* is defined as *1/k* where *k* is the failure limit of the
--- number of unsuccessful experiments (generalized to real numbers).
--- Samples will always be returned as a floating point data type.
--- 
--- ### Example
--- 
---    generalized_negative_binomial(mu=2.0, alpha=0.3, data=ones(2,2)) = [[ 2.,  1.],
---                                                                        [ 6.,  4.]]
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\random\sample_op.cc:L283
---
---
---@param mu any @float, optional, default=1 | Mean of the negative binomial distribution.
---@param alpha any @float, optional, default=1 | Alpha (dispersion) parameter of the negative binomial distribution.
---@param data any @NDArray | The input
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.generalized_negative_binomial_like(data, mu, alpha, out, name, kwargs)
end

--- Draw random samples from a negative binomial distribution.
--- 
--- Samples are distributed according to a negative binomial distribution parametrized by
--- *k* (limit of unsuccessful experiments) and *p* (failure probability in each experiment).
--- Samples will always be returned as a floating point data type.
--- 
--- ### Example
--- 
---    negative_binomial(k=3, p=0.4, shape=(2,2)) = [[ 4.,  7.],
---                                                  [ 2.,  5.]]
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\random\sample_op.cc:L164
---
---
---@param k any @int, optional, default='1' | Limit of unsuccessful experiments.
---@param p any @float, optional, default=1 | Failure probability in each experiment.
---@param shape any @Shape(tuple), optional, default=None | Shape of the output.
---@param ctx any @string, optional, default='' | Context of output, in format [cpu|gpu|cpu_pinned](n). Only used for imperative calls.
---@param dtype any @{'None', 'float16', 'float32', 'float64'},optional, default='None' | DType of the output in case this can't be inferred. Defaults to float32 if not defined (dtype=None).
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.negative_binomial(k, p, shape, ctx, dtype, out, name, kwargs)
end

--- Draw random samples from a negative binomial distribution according to the input array shape.
--- 
--- Samples are distributed according to a negative binomial distribution parametrized by
--- *k* (limit of unsuccessful experiments) and *p* (failure probability in each experiment).
--- Samples will always be returned as a floating point data type.
--- 
--- ### Example
--- 
---    negative_binomial(k=3, p=0.4, data=ones(2,2)) = [[ 4.,  7.],
---                                                     [ 2.,  5.]]
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\random\sample_op.cc:L267
---
---
---@param k any @int, optional, default='1' | Limit of unsuccessful experiments.
---@param p any @float, optional, default=1 | Failure probability in each experiment.
---@param data any @NDArray | The input
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.negative_binomial_like(data, k, p, out, name, kwargs)
end

--- Draw random samples from a normal (Gaussian) distribution.
--- 
--- ### Note: The existing alias ``normal`` is deprecated.
--- 
--- Samples are distributed according to a normal distribution parametrized by *loc* (mean) and *scale*
--- (standard deviation).
--- 
--- ### Example
--- 
---    normal(loc=0, scale=1, shape=(2,2)) = [[ 1.89171135, -1.16881478],
---                                           [-1.23474145,  1.55807114]]
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\random\sample_op.cc:L113
---
---
---@param loc any @float, optional, default=0 | Mean of the distribution.
---@param scale any @float, optional, default=1 | Standard deviation of the distribution.
---@param shape any @Shape(tuple), optional, default=None | Shape of the output.
---@param ctx any @string, optional, default='' | Context of output, in format [cpu|gpu|cpu_pinned](n). Only used for imperative calls.
---@param dtype any @{'None', 'float16', 'float32', 'float64'},optional, default='None' | DType of the output in case this can't be inferred. Defaults to float32 if not defined (dtype=None).
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.normal(loc, scale, shape, ctx, dtype, out, name, kwargs)
end

--- Draw random samples from a normal (Gaussian) distribution according to the input array shape.
--- 
--- Samples are distributed according to a normal distribution parametrized by *loc* (mean) and *scale*
--- (standard deviation).
--- 
--- ### Example
--- 
---    normal(loc=0, scale=1, data=ones(2,2)) = [[ 1.89171135, -1.16881478],
---                                              [-1.23474145,  1.55807114]]
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\random\sample_op.cc:L220
---
---
---@param loc any @float, optional, default=0 | Mean of the distribution.
---@param scale any @float, optional, default=1 | Standard deviation of the distribution.
---@param data any @NDArray | The input
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.normal_like(data, loc, scale, out, name, kwargs)
end

--- Draw random samples from a Poisson distribution.
--- 
--- Samples are distributed according to a Poisson distribution parametrized by *lambda* (rate).
--- Samples will always be returned as a floating point data type.
--- 
--- ### Example
--- 
---    poisson(lam=4, shape=(2,2)) = [[ 5.,  2.],
---                                   [ 4.,  6.]]
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\random\sample_op.cc:L150
---
---
---@param lam any @float, optional, default=1 | Lambda parameter (rate) of the Poisson distribution.
---@param shape any @Shape(tuple), optional, default=None | Shape of the output.
---@param ctx any @string, optional, default='' | Context of output, in format [cpu|gpu|cpu_pinned](n). Only used for imperative calls.
---@param dtype any @{'None', 'float16', 'float32', 'float64'},optional, default='None' | DType of the output in case this can't be inferred. Defaults to float32 if not defined (dtype=None).
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.poisson(lam, shape, ctx, dtype, out, name, kwargs)
end

--- Draw random samples from a Poisson distribution according to the input array shape.
--- 
--- Samples are distributed according to a Poisson distribution parametrized by *lambda* (rate).
--- Samples will always be returned as a floating point data type.
--- 
--- ### Example
--- 
---    poisson(lam=4, data=ones(2,2)) = [[ 5.,  2.],
---                                      [ 4.,  6.]]
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\random\sample_op.cc:L254
---
---
---@param lam any @float, optional, default=1 | Lambda parameter (rate) of the Poisson distribution.
---@param data any @NDArray | The input
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.poisson_like(data, lam, out, name, kwargs)
end

--- Draw random samples from a discrete uniform distribution.
--- 
--- Samples are uniformly distributed over the half-open interval *[low, high)*
--- (includes *low*, but excludes *high*).
--- 
--- ### Example
--- 
---    randint(low=0, high=5, shape=(2,2)) = [[ 0,  2],
---                                           [ 3,  1]]
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\random\sample_op.cc:L193
---
---
---@param low any @, required | Lower bound of the distribution.
---@param high any @, required | Upper bound of the distribution.
---@param shape any @Shape(tuple), optional, default=None | Shape of the output.
---@param ctx any @string, optional, default='' | Context of output, in format [cpu|gpu|cpu_pinned](n). Only used for imperative calls.
---@param dtype any @{'None', 'int32', 'int64'},optional, default='None' | DType of the output in case this can't be inferred. Defaults to int32 if not defined (dtype=None).
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.randint(low, high, shape, ctx, dtype, out, name, kwargs)
end

--- Draw random samples from a uniform distribution.
--- 
--- ### Note: The existing alias ``uniform`` is deprecated.
--- 
--- Samples are uniformly distributed over the half-open interval *[low, high)*
--- (includes *low*, but excludes *high*).
--- 
--- ### Example
--- 
---    uniform(low=0, high=1, shape=(2,2)) = [[ 0.60276335,  0.85794562],
---                                           [ 0.54488319,  0.84725171]]
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\random\sample_op.cc:L96
---
---
---@param low any @float, optional, default=0 | Lower bound of the distribution.
---@param high any @float, optional, default=1 | Upper bound of the distribution.
---@param shape any @Shape(tuple), optional, default=None | Shape of the output.
---@param ctx any @string, optional, default='' | Context of output, in format [cpu|gpu|cpu_pinned](n). Only used for imperative calls.
---@param dtype any @{'None', 'float16', 'float32', 'float64'},optional, default='None' | DType of the output in case this can't be inferred. Defaults to float32 if not defined (dtype=None).
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.uniform(low, high, shape, ctx, dtype, out, name, kwargs)
end

--- Draw random samples from a uniform distribution according to the input array shape.
--- 
--- Samples are uniformly distributed over the half-open interval *[low, high)*
--- (includes *low*, but excludes *high*).
--- 
--- ### Example
--- 
---    uniform(low=0, high=1, data=ones(2,2)) = [[ 0.60276335,  0.85794562],
---                                              [ 0.54488319,  0.84725171]]
--- 
--- 
--- 
--- ### Defined in C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\random\sample_op.cc:L208
---
---
---@param low any @float, optional, default=0 | Lower bound of the distribution.
---@param high any @float, optional, default=1 | Upper bound of the distribution.
---@param data any @NDArray | The input
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
function M.uniform_like(data, low, high, out, name, kwargs)
end


return M