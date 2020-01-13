---@class mx.symbol.ramdom:mx.symbol.gen_random
local M = {}

local _Null = require('mx.base')._Null
local _internal = require('mx.symbol._internal')
local Symbol = require('mx.symbol.symbol').Symbol

function M._random_helper(random, sampler, params, shape, dtype, kwargs)
    if type(params[1]) == 'number' then
        for i = 2, #params do
            assert(type(params[i]) == 'number', 'Distribution parameters must all have the same type')
        end
        table.append(params, { shape, nil, dtype, nil, nil })--TODO: check
        return random(arg_make(params, kwargs))
    elseif isinstance(params[1], Symbol) then
        for i = 2, #params do
            assert(isinstance(params[i], Symbol), 'Distribution parameters must all have the same type')
        end
        table.append(params, { shape, dtype, nil, nil, nil })
        return sampler(arg_make(params, kwargs))
    end
    raise('ValueError', 'Distribution parameters must be either Symbol or numbers')
end

---
function M.uniform(low, high, shape, dtype, kwargs)
    low, high, shape, dtype = default(low, 0, high, 1, shape, _Null, dtype, _Null)
    return M._random_helper(_internal._random_uniform, _internal._sample_uniform,
                            { low, high }, shape, dtype, kwargs)
end

---
function M.normal(loc, scale, shape, dtype, kwargs)
    loc, scale, shape, dtype = default(loc, 0, scale, 1, shape, _Null, dtype, _Null)
    return M._random_helper(
            _internal._random_normal, _internal._sample_normal,
            { loc, scale }, shape, dtype, kwargs)
end

---
function M.poisson(lam, shape, dtype, kwargs)
    lam, shape, dtype = default(lam, 1, shape, _Null, dtype, _Null)
    return M._random_helper(
            _internal._random_poisson, _internal._sample_poisson,
            { lam }, shape, dtype, kwargs)
end

---
function M.exponential(scale, shape, dtype, kwargs)
    scale, shape, dtype = default(scale, 1, shape, _Null, dtype, _Null)
    return M._random_helper(
            _internal._random_exponential, _internal._sample_exponential,
            { 1 / scale }, shape, dtype, kwargs)
end

---
function M.gamma(alpha, beta, shape, dtype, kwargs)
    alpha, beta, shape, dtype = default(alpha, 1, beta, 1, shape, _Null, dtype, _Null)
    return M._random_helper(
            _internal._random_gamma, _internal._sample_gamma,
            { alpha, beta }, shape, dtype, kwargs)
end

---
function M.negative_binomial(k, p, shape, dtype, kwargs)
    k, p, shape, dtype = default(k, 1, p, 1, shape, _Null, dtype, _Null)
    return M._random_helper(
            _internal._random_negative_binomial, _internal._sample_negative_binomial,
            { k, p }, shape, dtype, kwargs)
end

---
function M.generalized_negative_binomial(mu, alpha, shape, dtype, kwargs)
    mu, alpha, shape, dtype = default(mu, 1, alpha, 1, shape, _Null, dtype, _Null)
    return M._random_helper(
            _internal._random_generalized_negative_binomial, _internal._sample_generalized_negative_binomial,
            { mu, alpha }, shape, dtype, kwargs)
end

---
function M.multinomial(data, shape, get_prob, out, dtype, kwargs)
    shape, get_prob, dtype = default(shape, _Null, get_prob, false, dtype, 'int32')
    return _internal._sample_multinomial(data, shape, get_prob, dtype, out, nil, kwargs)
end

---
function M.shuffle(data, kwargs)
    return _internal._shuffle(data, kwargs)
end

---
function M.randint(low, high, shape, dtype, kwargs)
    shape, dtype = default(shape, _Null, dtype, _Null)
    return M._random_helper(
            _internal._random_randint, nil,
            { low, high }, shape, dtype, kwargs)
end

return M
