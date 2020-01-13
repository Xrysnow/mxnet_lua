-- File content is auto-generated. Do not modify.

local SymbolBase = require('mx._ctypes.symbol').SymbolBase
local _symbol_creator = require('mx._ctypes.symbol')._symbol_creator
local NameManager = require('mx.name').NameManager
local AttrScope = require('mx.attribute').AttrScope
local _Null = require('mx.base')._Null
---@class mx.symbol.gen_random
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
--- This function support variable length of positional input.
---
---
---@param lam any @float, optional, default=1 | Lambda parameter (rate) of the exponential distribution.
---@param shape any @Shape(tuple), optional, default=None | Shape of the output.
---@param ctx any @string, optional, default='' | Context of output, in format [cpu|gpu|cpu_pinned](n). Only used for imperative calls.
---@param dtype any @{'None', 'float16', 'float32', 'float64'},optional, default='None' | DType of the output in case this can't be inferred. Defaults to float32 if not defined (dtype=None).
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M.exponential(lam, shape, ctx, dtype, name, attr, out, kwargs)
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
--- This function support variable length of positional input.
---
---
---@param lam any @float, optional, default=1 | Lambda parameter (rate) of the exponential distribution.
---@param data any @Symbol | The input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M.exponential_like(data, lam, name, attr, out, kwargs)
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
--- This function support variable length of positional input.
---
---
---@param alpha any @float, optional, default=1 | Alpha parameter (shape) of the gamma distribution.
---@param beta any @float, optional, default=1 | Beta parameter (scale) of the gamma distribution.
---@param shape any @Shape(tuple), optional, default=None | Shape of the output.
---@param ctx any @string, optional, default='' | Context of output, in format [cpu|gpu|cpu_pinned](n). Only used for imperative calls.
---@param dtype any @{'None', 'float16', 'float32', 'float64'},optional, default='None' | DType of the output in case this can't be inferred. Defaults to float32 if not defined (dtype=None).
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M.gamma(alpha, beta, shape, ctx, dtype, name, attr, out, kwargs)
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
--- This function support variable length of positional input.
---
---
---@param alpha any @float, optional, default=1 | Alpha parameter (shape) of the gamma distribution.
---@param beta any @float, optional, default=1 | Beta parameter (scale) of the gamma distribution.
---@param data any @Symbol | The input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M.gamma_like(data, alpha, beta, name, attr, out, kwargs)
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
--- This function support variable length of positional input.
---
---
---@param mu any @float, optional, default=1 | Mean of the negative binomial distribution.
---@param alpha any @float, optional, default=1 | Alpha (dispersion) parameter of the negative binomial distribution.
---@param shape any @Shape(tuple), optional, default=None | Shape of the output.
---@param ctx any @string, optional, default='' | Context of output, in format [cpu|gpu|cpu_pinned](n). Only used for imperative calls.
---@param dtype any @{'None', 'float16', 'float32', 'float64'},optional, default='None' | DType of the output in case this can't be inferred. Defaults to float32 if not defined (dtype=None).
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M.generalized_negative_binomial(mu, alpha, shape, ctx, dtype, name, attr, out, kwargs)
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
--- This function support variable length of positional input.
---
---
---@param mu any @float, optional, default=1 | Mean of the negative binomial distribution.
---@param alpha any @float, optional, default=1 | Alpha (dispersion) parameter of the negative binomial distribution.
---@param data any @Symbol | The input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M.generalized_negative_binomial_like(data, mu, alpha, name, attr, out, kwargs)
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
--- This function support variable length of positional input.
---
---
---@param k any @int, optional, default='1' | Limit of unsuccessful experiments.
---@param p any @float, optional, default=1 | Failure probability in each experiment.
---@param shape any @Shape(tuple), optional, default=None | Shape of the output.
---@param ctx any @string, optional, default='' | Context of output, in format [cpu|gpu|cpu_pinned](n). Only used for imperative calls.
---@param dtype any @{'None', 'float16', 'float32', 'float64'},optional, default='None' | DType of the output in case this can't be inferred. Defaults to float32 if not defined (dtype=None).
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M.negative_binomial(k, p, shape, ctx, dtype, name, attr, out, kwargs)
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
--- This function support variable length of positional input.
---
---
---@param k any @int, optional, default='1' | Limit of unsuccessful experiments.
---@param p any @float, optional, default=1 | Failure probability in each experiment.
---@param data any @Symbol | The input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M.negative_binomial_like(data, k, p, name, attr, out, kwargs)
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
--- This function support variable length of positional input.
---
---
---@param loc any @float, optional, default=0 | Mean of the distribution.
---@param scale any @float, optional, default=1 | Standard deviation of the distribution.
---@param shape any @Shape(tuple), optional, default=None | Shape of the output.
---@param ctx any @string, optional, default='' | Context of output, in format [cpu|gpu|cpu_pinned](n). Only used for imperative calls.
---@param dtype any @{'None', 'float16', 'float32', 'float64'},optional, default='None' | DType of the output in case this can't be inferred. Defaults to float32 if not defined (dtype=None).
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M.normal(loc, scale, shape, ctx, dtype, name, attr, out, kwargs)
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
--- This function support variable length of positional input.
---
---
---@param loc any @float, optional, default=0 | Mean of the distribution.
---@param scale any @float, optional, default=1 | Standard deviation of the distribution.
---@param data any @Symbol | The input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M.normal_like(data, loc, scale, name, attr, out, kwargs)
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
--- This function support variable length of positional input.
---
---
---@param lam any @float, optional, default=1 | Lambda parameter (rate) of the Poisson distribution.
---@param shape any @Shape(tuple), optional, default=None | Shape of the output.
---@param ctx any @string, optional, default='' | Context of output, in format [cpu|gpu|cpu_pinned](n). Only used for imperative calls.
---@param dtype any @{'None', 'float16', 'float32', 'float64'},optional, default='None' | DType of the output in case this can't be inferred. Defaults to float32 if not defined (dtype=None).
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M.poisson(lam, shape, ctx, dtype, name, attr, out, kwargs)
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
--- This function support variable length of positional input.
---
---
---@param lam any @float, optional, default=1 | Lambda parameter (rate) of the Poisson distribution.
---@param data any @Symbol | The input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M.poisson_like(data, lam, name, attr, out, kwargs)
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
--- This function support variable length of positional input.
---
---
---@param low any @, required | Lower bound of the distribution.
---@param high any @, required | Upper bound of the distribution.
---@param shape any @Shape(tuple), optional, default=None | Shape of the output.
---@param ctx any @string, optional, default='' | Context of output, in format [cpu|gpu|cpu_pinned](n). Only used for imperative calls.
---@param dtype any @{'None', 'int32', 'int64'},optional, default='None' | DType of the output in case this can't be inferred. Defaults to int32 if not defined (dtype=None).
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M.randint(low, high, shape, ctx, dtype, name, attr, out, kwargs)
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
--- This function support variable length of positional input.
---
---
---@param low any @float, optional, default=0 | Lower bound of the distribution.
---@param high any @float, optional, default=1 | Upper bound of the distribution.
---@param shape any @Shape(tuple), optional, default=None | Shape of the output.
---@param ctx any @string, optional, default='' | Context of output, in format [cpu|gpu|cpu_pinned](n). Only used for imperative calls.
---@param dtype any @{'None', 'float16', 'float32', 'float64'},optional, default='None' | DType of the output in case this can't be inferred. Defaults to float32 if not defined (dtype=None).
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M.uniform(low, high, shape, ctx, dtype, name, attr, out, kwargs)
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
--- This function support variable length of positional input.
---
---
---@param low any @float, optional, default=0 | Lower bound of the distribution.
---@param high any @float, optional, default=1 | Upper bound of the distribution.
---@param data any @Symbol | The input
---@param name string @optional | Name of the resulting symbol.
---@return mx.symbol.Symbol @The result symbol.
function M.uniform_like(data, low, high, name, attr, out, kwargs)
end


return M