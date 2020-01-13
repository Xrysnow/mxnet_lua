---@class mx.ndarray.ramdom:mx.ndarray.gen_random
local M = {}
--

local _Null = require('mx.base')._Null
local current_context = require('mx.context').current_context
local _internal = require('mx.ndarray._internal')
local NDArray = require('mx.ndarray.ndarray').NDArray

---@return mx.ndarray.NDArray
function M._random_helper(random, sampler, params, shape, dtype, ctx, out, kwargs)
    if isinstance(params[1], NDArray) then
        for i = 2, #params do
            assert(isinstance(params[i], NDArray),
                   'Distribution parameters must all have the same type')
        end
        table.append(params, { shape, dtype, out, _Null })
        return sampler(arg_make(params, kwargs))
    elseif type(params[1]) == 'number' then
        if ctx == nil then
            ctx = current_context()
        end
        if shape == _Null and out == nil then
            shape = 1
        end
        for i = 2, #params do
            assert(type(params[i]) == 'number',
                   'Distribution parameters must all have the same type')
        end
        table.append(params, { shape, dtype, ctx, out, _Null })--TODO: check
        return random(arg_make(params, kwargs))
    end
    raise('ValueError', 'Distribution parameters must be either NDArray or numbers')
end

---
function M.uniform(low, high, shape, dtype, ctx, out, kwargs)
    low, high, shape, dtype = default(low, 0, high, 1, shape, _Null, dtype, _Null)
    return M._random_helper(
            _internal._random_uniform, _internal._sample_uniform,
            { low, high }, shape, dtype, ctx, out, kwargs)
end

---
function M.normal(loc, scale, shape, dtype, ctx, out, kwargs)
    loc, scale, shape, dtype = default(loc, 0, scale, 1, shape, _Null, dtype, _Null)
    return M._random_helper(
            _internal._random_normal, _internal._sample_normal,
            { loc, scale }, shape, dtype, ctx, out, kwargs)
end

---
function M.randn(...)
    local shape, kwargs = arg_kw(...)
    local loc = arg_pop(kwargs, 'loc', 0)
    local scale = arg_pop(kwargs, 'scale', 1)
    local dtype = arg_pop(kwargs, 'dtype', _Null)
    local ctx = arg_pop(kwargs, 'ctx', nil)
    local out = arg_pop(kwargs, 'out', nil)
    return M._random_helper(
            _internal._random_normal, _internal._sample_normal,
            { loc, scale }, shape, dtype, ctx, out, kwargs)
end

---
function M.poisson(lam, shape, dtype, ctx, out, kwargs)
    lam, shape, dtype = default(lam, 1, shape, _Null, dtype, _Null)
    return M._random_helper(
            _internal._random_poisson, _internal._sample_poisson,
            { lam }, shape, dtype, ctx, out, kwargs)
end

---
function M.exponential(scale, shape, dtype, ctx, out, kwargs)
    scale, shape, dtype = default(scale, 1, shape, _Null, dtype, _Null)
    return M._random_helper(
            _internal._random_exponential, _internal._sample_exponential,
            { 1 / scale }, shape, dtype, ctx, out, kwargs)
end

---
function M.gamma(alpha, beta, shape, dtype, ctx, out, kwargs)
    alpha, beta, shape, dtype = default(alpha, 1, beta, 1, shape, _Null, dtype, _Null)
    return M._random_helper(
            _internal._random_gamma, _internal._sample_gamma,
            { alpha, beta }, shape, dtype, ctx, out, kwargs)
end

---
function M.negative_binomial(k, p, shape, dtype, ctx, out, kwargs)
    k, p, shape, dtype = default(k, 1, p, 1, shape, _Null, dtype, _Null)
    return M._random_helper(
            _internal._random_negative_binomial, _internal._sample_negative_binomial,
            { k, p }, shape, dtype, ctx, out, kwargs)
end

---
function M.generalized_negative_binomial(mu, alpha, shape, dtype, ctx, out, kwargs)
    mu, alpha, shape, dtype = default(mu, 1, alpha, 1, shape, _Null, dtype, _Null)
    return M._random_helper(
            _internal._random_generalized_negative_binomial, _internal._sample_generalized_negative_binomial,
            { mu, alpha }, shape, dtype, ctx, out, kwargs)
end

---
function M.multinomial(data, shape, get_prob, out, dtype, kwargs)
    shape, get_prob, dtype = default(shape, _Null, get_prob, false, dtype, 'int32')
    return _internal._sample_multinomial(data, shape, get_prob, dtype, out, nil, kwargs)
end

---
function M.shuffle(data, kwargs)
    return _internal._shuffle(data, nil, nil, kwargs)
end

---
function M.randint(low, high, shape, dtype, ctx, out, kwargs)
    shape, dtype = default(shape, _Null, dtype, _Null)
    return M._random_helper(
            _internal._random_randint, nil,
            { low, high }, shape, dtype, ctx, out, kwargs)
end

return M
