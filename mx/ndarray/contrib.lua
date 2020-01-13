---@class mx.ndarray.contrib:mx.ndarray.gen_contrib
local M = {}
--
local current_context = require('mx.context').current_context
local uniform = require('mx.ndarray.random').uniform
local _as_list = require('mx.base')._as_list
local ndarray = require('mx.ndarray.ndarray')
local gen_contrib = require('mx.ndarray.gen_contrib')

function M.rand_zipfian(true_classes, num_sampled, range_max, ctx)
    if ctx == nil then
        ctx = current_context()
    end
    local log_range = math.log(range_max + 1)
    local rand = uniform(0, log_range, { num_sampled }, 'float64', ctx)
    -- make sure sampled_classes are in the range of [0, range_max)
    local sampled_classes = (rand:exp() - 1):astype('int64') % range_max
    local true_cls = true_classes:as_in_context(ctx):astype('float64')
    local expected_count_true = ((true_cls + 2.0) / (true_cls + 1.0)):log() / log_range * num_sampled
    -- cast sampled classes to fp64 to avoid interget division
    local sampled_cls_fp64 = sampled_classes:astype('float64')
    local expected_prob_sampled = ((sampled_cls_fp64 + 2.0) / (sampled_cls_fp64 + 1.0)):log() / log_range
    local expected_count_sampled = expected_prob_sampled * num_sampled
    return sampled_classes, expected_count_true, expected_count_sampled
end

function M._flatten(args, inout_str)
    if isinstance(args, ndarray.NDArray) then
        return { args }, 0
    end
    assert(islist(args), ('%s must be (nested) list of NDArray'):format(inout_str))
    local flat, fmts = {}, {}
    for _, i in ipairs(args) do
        local arg, fmt = M._flatten(i, inout_str)
        table.append(flat, arg)
        table.insert(fmts, fmt)
    end
    return flat, fmts
end

function M._regroup(args, fmt)
    if type(fmt) == 'number' then
        if fmt == 0 then
            return args[1], table.slice(args, 2)
        end
        return table.slice(args, 1, fmt), table.slice(args, fmt + 1)
    end
    assert(islist(args), 'output must be (nested) list of NDArray')
    local ret = {}
    for _, i in ipairs(fmt) do
        local res
        res, args = M._regroup(args, i)
        table.insert(ret, res)
    end
    return ret, args
end

function M.foreach(body, data, init_states)
    local function check_input(inputs, in_type, msg)
        local is_NDArray_or_list = true
        if islist(inputs) then
            for _, i in ipairs(inputs) do
                if not isinstance(i, in_type) then
                    is_NDArray_or_list = false
                    break
                end
            end
        else
            is_NDArray_or_list = isinstance(inputs, in_type)
        end
        assert(is_NDArray_or_list, msg)
    end
    local flatten, _ = M._flatten(data, 'foreach input')
    check_input(flatten, ndarray.NDArray,
                "data should be an NDArray or a nested list of NDArrays")
    flatten, _ = M._flatten(init_states, "foreach states")
    check_input(flatten, ndarray.NDArray,
                "init_states should be an NDArray or a nested list of NDArrays")
    local not_data_list = isinstance(data, ndarray.NDArray)
    local num_iters = not_data_list and data.shape[1] or data[1].shape[1]
    local states = init_states
    local outputs = {}
    local out_fmt
    for i = 0, num_iters - 1 do
        local eles
        if not_data_list then
            eles = data[i]
        else
            eles = {}
            for _, d in ipairs(data) do
                table.insert(eles, d[i])
            end
        end
        local outs
        outs, states = body(eles, states)
        outs, out_fmt = M._flatten(outs, "foreach output")
        table.insert(outputs, outs)
    end
    outputs = table.zip(unpack(outputs))
    local tmp_outputs = {}
    for _, out in ipairs(outputs) do
        table.insert(tmp_outputs, ndarray.op.stack(unpack(out)))
    end
    outputs = tmp_outputs
    outputs, _ = M. _regroup(outputs, out_fmt)
    return { outputs, states }
end

--TODO
function M.while_loop(cond, func, loop_vars, max_iterations)
    raise('NotImplementedError')
end

function M.cond(pred, then_func, else_func)
    local function _to_python_scalar(inputs, type_, name)
        -- Converts "inputs", possibly typed mxnet NDArray, a numpy ndarray, other python types,
        -- to the given type
        if inputs.asscalar then
            inputs = inputs:asscalar()
        end
        inputs = type_(inputs)
        return inputs
    end
    local branch = _to_python_scalar(pred, bool, "pred")
    if branch then
        return then_func()
    else
        return else_func()
    end
end

function M.isinf(data)
    return ndarray.equal(data:abs(), math.inf)
end

function M.isfinite(data)
    local is_data_not_nan = ndarray.equal(data, data)
    local is_data_not_infinite = ndarray.not_equal(data:abs(), math.inf)
    return ndarray.logical_and(is_data_not_infinite, is_data_not_nan)
end

function M.isnan(data)
    return ndarray.not_equal(data, data)
end

function M.adamw_update(weight, grad, mean, var, rescale_grad, lr, eta,
                        beta1, beta2, epsilon, wd, clip_gradient,
                        out, name, kwargs)
    beta1, beta2, epsilon, wd, clip_gradient = default(
            beta1, 0.9, beta2, 0.999, epsilon, 1e-8, wd, 0, clip_gradient, -1)
    if not isinstance(rescale_grad, ndarray.NDArray) then
        rescale_grad = ndarray.full({ 1, }, rescale_grad, weight.context)
    else
        rescale_grad = rescale_grad:as_in_context(weight.context)
    end
    return ndarray._internal._adamw_update(
            weight, grad, mean, var,
            rescale_grad, lr,
            beta1, beta2, epsilon,
            wd, eta, clip_gradient,
            out, name, kwargs)
end

function M.mp_adamw_update(weight, grad, mean, var, weight32, rescale_grad, lr, eta,
                           beta1, beta2, epsilon, wd, clip_gradient,
                           out, name, kwargs)
    beta1, beta2, epsilon, wd, clip_gradient = default(
            beta1, 0.9, beta2, 0.999, epsilon, 1e-8, wd, 0, clip_gradient, -1)
    if not isinstance(rescale_grad, ndarray.NDArray) then
        rescale_grad = ndarray.full({ 1, }, rescale_grad, weight.context)
    else
        rescale_grad = rescale_grad:as_in_context(weight.context)
    end
    return ndarray._internal._mp_adamw_update(
            weight, grad, mean, var, weight32,
            rescale_grad, lr,
            beta1, beta2, epsilon,
            wd, eta, clip_gradient,
            out, name, kwargs)
end

return M
