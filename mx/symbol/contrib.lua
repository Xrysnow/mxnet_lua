---@class mx.symbol.contrib:mx.symbol.gen_contrib
local M = {}
--
local ctypes = require('ctypes')
local uniform = require('mx.symbol.random').uniform
local gen = require('mx.symbol.gen_contrib')
local symbol = require('mx.symbol.symbol')
local Symbol = symbol.Symbol
local base = require('mx.base')
local _LIB, check_call, SymbolHandle, _as_list = base._LIB, base.check_call, base.SymbolHandle, base._as_list
local AttrScope = require('mx.attribute').AttrScope

function M.rand_zipfian(true_classes, num_sampled, range_max)
    assert(isinstance(true_classes, Symbol), 'unexpected type')
    local log_range = math.log(range_max + 1)
    local rand = uniform(0, log_range, { num_sampled }, 'float64')
    -- make sure sampled_classes are in the range of [0, range_max)
    local sampled_classes = (rand:exp() - 1):astype('int64') % range_max

    local true_cls = true_classes:astype('float64')
    local expected_count_true = ((true_cls + 2.0) / (true_cls + 1.0)):log() / log_range * num_sampled
    -- cast sampled classes to fp64 to avoid interget division
    local sampled_cls_fp64 = sampled_classes:astype('float64')
    local expected_prob_sampled = ((sampled_cls_fp64 + 2.0) / (sampled_cls_fp64 + 1.0)):log() / log_range
    local expected_count_sampled = expected_prob_sampled * num_sampled
    return sampled_classes, expected_count_true, expected_count_sampled
end

function M._flatten(args, inout_str)
    if isinstance(args, symbol.Symbol) then
        local length = len(args:list_outputs())
        if length <= 1 then
            length = 0
        end
        return { args }, length
    end
    assert(islist(args), ('%s must be (nested) list of Symbol'):format(inout_str))
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
    assert(islist(args), 'output must be (nested) list of Symbol')
    local ret = {}
    for _, i in ipairs(fmt) do
        local res
        res, args = M._regroup(args, i)
        table.insert(ret, res)
    end
    return ret, args
end

function M._get_sym_uniq_name(sym)
    return ('%s-%s'):format(sym.name, sym:attr('_value_index'))
end

function M._get_graph_inputs(subg)
    local num_handles = ctypes.c_int(0)
    local handles = ctypes.POINTER(SymbolHandle)()
    check_call(_LIB.MXSymbolGetInputSymbols(subg.handle, ctypes.byref(handles),
                                            ctypes.byref(num_handles)))
    local syms = {}
    for i = 1, num_handles.value do
        table.insert(syms, Symbol(handles[i - 1]))
    end
    return syms
end

function M._cut_subgraph(subg)
    local num_handles = ctypes.c_int(0)
    local handles = ctypes.POINTER(SymbolHandle)()
    check_call(_LIB.MXSymbolCutSubgraph(subg.handle, ctypes.byref(handles),
                                        ctypes.byref(num_handles)))
    local syms = {}
    for i = 1, num_handles.value do
        table.insert(syms, Symbol(handles[i - 1]))
    end
    return syms
end

function M._get_unique_subgraph_name(subgraph_name)
    local attrs = AttrScope._current.value._attr
    if attrs.__subgraph_name__ then
        subgraph_name = table.concat({ attrs.__subgraph_name__, '$', subgraph_name })
    end
    AttrScope._subgraph_names[subgraph_name] = AttrScope._subgraph_names[subgraph_name] + 1
    subgraph_name = subgraph_name .. ('%d'):format(AttrScope._subgraph_names[subgraph_name] - 1)
    return subgraph_name
end

function M._construct_subgraph(sym_out, sym_states, name)
    sym_out = _as_list(sym_out)
    sym_states = _as_list(sym_states)
    local all_outputs = {}
    table.append(all_outputs, sym_out)
    table.append(all_outputs, sym_states)
    local g = symbol.Group(all_outputs)

    local flat_out = {}
    local all_input_names = g:list_inputs()
    local output_names = table.map(sym_out, function(o)
        return o.name
    end)
    for _, o in ipairs(sym_out) do
        if all_input_names[o.name] or o:list_attr().__subgraph_name__ ~= name then
            table.insert(flat_out, require('mx.symbol.op').identity(o))
        else
            table.insert(flat_out, o)
        end
    end
    for _, s in ipairs(sym_states) do
        if all_input_names[s.name] or output_names[s.name] or s:list_attr().__subgraph_name__ ~= name then
            table.insert(flat_out, require('mx.symbol.op').identity(s))
        else
            table.insert(flat_out, s)
        end
    end
    return symbol.Group(flat_out)
end

function M._check_data(inputs, in_type, msg)
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

function M.foreach(body, data, init_states, name)
    raise('NotImplementedError')
end

function M.while_loop(cond, func, loop_vars, max_iterations, name)
    raise('NotImplementedError')
end

function M.cond(pred, then_func, else_func, name)
    raise('NotImplementedError')
end

function M.adamw_update(weight, grad, mean, var, rescale_grad, lr, eta,
                        beta1, beta2, epsilon, wd, clip_gradient,
                        out, name, kwargs)
    beta1, beta2, epsilon, wd, clip_gradient = default(
            beta1, 0.9, beta2, 0.999, epsilon, 1e-8, wd, 0, clip_gradient, -1)
    if not isinstance(rescale_grad, Symbol) then
        rescale_grad = symbol.full({ 1, }, rescale_grad)
    end
    return require('mx.symbol._internal')._adamw_update(
            weight, grad, mean, var,
            rescale_grad, lr,
            beta1, beta2, epsilon,
            wd, eta, clip_gradient,
            name, nil, out, kwargs)
end

function M.mp_adamw_update(weight, grad, mean, var, weight32, rescale_grad, lr, eta,
                           beta1, beta2, epsilon, wd, clip_gradient,
                           out, name, kwargs)
    beta1, beta2, epsilon, wd, clip_gradient = default(
            beta1, 0.9, beta2, 0.999, epsilon, 1e-8, wd, 0, clip_gradient, -1)
    if not isinstance(rescale_grad, Symbol) then
        rescale_grad = symbol.full({ 1, }, rescale_grad)
    end
    return require('mx.symbol._internal')._mp_adamw_update(
            weight, grad, mean, var, weight32,
            rescale_grad, lr,
            beta1, beta2, epsilon,
            wd, eta, clip_gradient,
            name, nil, out, kwargs)
end

return M
