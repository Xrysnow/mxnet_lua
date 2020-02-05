--
local M = {}

local logging = require('mx_py.logging')
local _nd = require('mx.ndarray.__init__')
local NDArray, zeros, clip, sqrt, cast, maximum, NDabs, array, multiply, sgd_update, sgd_mom_update, adam_update, rmsprop_update, rmspropalex_update, mp_sgd_update, mp_sgd_mom_update, square, ftrl_update, ftml_update, signsgd_update, signum_update, nag_mom_update, mp_nag_mom_update, multi_sgd_update, multi_sgd_mom_update, multi_mp_sgd_update, multi_mp_sgd_mom_update, sparse = _nd.NDArray, _nd.zeros, _nd.clip, _nd.sqrt, _nd.cast, _nd.maximum, _nd.abs, _nd.array, _nd.multiply, _nd.sgd_update, _nd.sgd_mom_update, _nd.adam_update, _nd.rmsprop_update, _nd.rmspropalex_update, _nd.mp_sgd_update, _nd.mp_sgd_mom_update, _nd.square, _nd.ftrl_update, _nd.ftml_update, _nd.signsgd_update, _nd.signum_update, _nd.nag_mom_update, _nd.mp_nag_mom_update, _nd.multi_sgd_update, _nd.multi_sgd_mom_update, _nd.multi_mp_sgd_update, _nd.multi_mp_sgd_mom_update, _nd.sparse
local normal = require('mx.random').normal

local function _flatten_list(nested_list)
    local ret = {}
    for _, sublist in ipairs(nested_list) do
        for _, item in ipairs(sublist) do
            table.insert(ret, item)
        end
    end
    return ret
end
M._flatten_list = _flatten_list

---@class mx.optimizer.Optimizer
local Optimizer = class('Optimizer')
M.Optimizer = Optimizer

function Optimizer:ctor(...)
    local args = { ... }
    local narg = select('#', ...)
    local rescale_grad, param_idx2name, wd,
    clip_gradient, learning_rate,
    lr_scheduler, sym, begin_num_update,
    multi_precision, param_dict
    if narg == 1 and type(args[1]) == 'table' then
        local kwargs = args[1]
        rescale_grad, param_idx2name, wd,
        clip_gradient, learning_rate,
        lr_scheduler, sym, begin_num_update,
        multi_precision, param_dict = kwargs.rescale_grad,
        kwargs.param_idx2name, kwargs.wd,
        clip_gradient, kwargs.learning_rate,
        lr_scheduler, kwargs.sym, kwargs.begin_num_update,
        multi_precision, kwargs.param_dict
    else
        rescale_grad, param_idx2name, wd,
        clip_gradient, learning_rate,
        lr_scheduler, sym, begin_num_update,
        multi_precision, param_dict = unpack(args, 1, 10)
    end
    rescale_grad, wd, learning_rate, begin_num_update, multi_precision = default(
            rescale_grad, 1, wd, 0, learning_rate, 0.01, begin_num_update, 0, multi_precision, false)
    self.rescale_grad = rescale_grad
    self.lr = learning_rate
    self.lr_scheduler = lr_scheduler
    if not isnone(lr_scheduler) then
        self.lr_scheduler.base_lr = learning_rate
    end
    self.wd = wd
    self.lr_mult = {}
    self.wd_mult = {}
    self.begin_num_update = begin_num_update
    self.num_update = begin_num_update
    self._all_index_update_counts = { [0] = {} }
    self._index_update_count = self._all_index_update_counts[0]
    self.clip_gradient = clip_gradient
    self.multi_precision = multi_precision
    self.aggregate_num = 0
    if isnone(param_idx2name) then
        param_idx2name = {}
    end
    assert(#param_idx2name == 0,
           'param_idx2name should be a dict of param indexes to names.')
    self.idx2name = table.clone(param_idx2name)
    if not isnone(sym) then
        self.sym_info = { sym:attr_dict(), sym:list_arguments() }
    else
        self.sym_info = {}
    end
    self.param_dict = isnone(param_dict) and {} or param_dict

    self:set_lr_mult({})
    self:set_wd_mult({})
end

function Optimizer.register(klass)
    assert(type(klass) == 'table')
    local name = getclassname(klass):lower()
    local t = string.split(name, '.')
    name = t[#t]
    if Optimizer.opt_registry[name] then
        logging.warning(
                'New optimizer %s.%s is overriding existing optimizer %s', name)
    end
    Optimizer.opt_registry[name] = klass
    return klass
end

function Optimizer.create_optimizer(name, ...)
    local name_ = name:lower()
    if Optimizer.opt_registry[name_] then
        return Optimizer.opt_registry[name_](...)
    else
        raise('ValueError', ('Cannot find optimizer %s'):format(name))
    end
end

--TODO

return M
