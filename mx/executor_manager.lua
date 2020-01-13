--
local M = {}

local logging = require('python.logging')
local mx_real_t = require('mx.base').mx_real_t
local nd = require('mx.ndarray.__init__')
local cpu = require('mx.context').cpu
local DataDesc = require('mx.io.__init__').DataDesc
local _b = require('mx.base')

function M._split_input_slice(batch_size, work_load_list)
    local total_work_load = table.sum(work_load_list)
    local batch_num_list = {}
    for _, v in ipairs(work_load_list) do
        table.insert(batch_num_list, math.round(v * batch_size / total_work_load))
    end
    local batch_num_sum = table.sum(batch_num_list)
    if batch_num_sum < batch_size then
        batch_num_list[#batch_num_list] = batch_num_list[#batch_num_list] + batch_size - batch_num_sum
    end
    local slices = {}
    local end_ = 0
    for _, batch_num in ipairs(batch_num_list) do
        local begin = math.floor(math.min(end_, batch_size))
        end_ = math.floor(math.min(begin + batch_num, batch_size))
        if begin >= end_ then
            raise('ValueError', 'Too many slices. Some splits are empty.')
        end
        table.insert(slices, { begin, end_ })
    end
    return slices
end

function M._check_arguments(symbol)
    local arg_set = {}
    local arg_names = symbol:list_arguments()
    for _, name in ipairs(arg_names) do
        if arg_set[name] then
            raise('ValueError', ('Find duplicated argument name %q'):format(name))
        end
        arg_set[name] = true
    end
    local aux_set = {}
    local aux_names = symbol:list_auxiliary_states()
    for _, name in ipairs(aux_names) do
        if aux_set[name] then
            raise('ValueError', ('Find duplicated auxiliary param name %q'):format(name))
        end
        aux_set[name] = true
    end
end

function M._load_general(data, targets)
    for i = 1, #data do
        local d_src, d_targets = data[i], targets[i]
        if isinstance(d_targets, nd.NDArray) then
            d_src:copyto(d_targets)
        else
            local stop = d_targets[#d_targets][1][2]
            local sz = d_src.shape[1]
            assert(stop == sz, ('Batch size miss match. Expected %d, got %d"'):format(stop, sz))
            for _, v in ipairs(d_targets) do
                local slice_idx, d_dst = v[1], v[2]
                d_src[slice_idx]:copyto(d_dst)
            end
        end
    end
end

function M._load_data(batch, targets)
    M._load_general(batch.data, targets)
end

function M._load_label(batch, targets)
    M._load_general(batch.label, targets)
end

---@param sym mx.symbol.Symbol
function M._bind_exec(sym, ctx, input_shapes, param_names, need_grad,
                      base_exec, shared_data_arrays, input_types, logger)
    need_grad, base_exec, shared_data_arrays, input_types, logger = default(
            need_grad, false, base_exec, None, shared_data_arrays, None, input_types, None, logger, logging)
    local arg_shape, _, aux_shape = sym:infer_shape(arg_kw(nil, input_shapes))
    assert(not isnone(arg_shape))
    if isnone(input_types) then
        input_types = {}
        for k, v in pairs(input_shapes) do
            input_types[k] = mx_real_t
        end
    end
    local arg_types, _, aux_types = sym:infer_type(arg_kw(nil, input_shapes))
    assert(not isnone(arg_types))

    local arg_arrays = {}
    local grad_arrays = need_grad ~= false and {} or None

    local arg_names = sym:list_arguments()
    if need_grad == false then
        need_grad = {}
    elseif need_grad == true then
        need_grad = {}
        for _, name in ipairs(arg_names) do
            need_grad[name] = true
        end
        for k, _ in pairs(input_shapes) do
            need_grad[k] = nil
        end
    elseif type(need_grad) then
        --
    else
        raise('AssertionError', 'need_grad must be boolean or set.')
    end
    local grad_req = {}
    for _, name in ipairs(arg_names) do
        grad_req[name] = need_grad[name] and 'write' or 'null'
    end

    -- create or borrow arguments and gradients
    for i, name in ipairs(arg_names) do
        if not param_names[name] then
            -- data or label
            local arg_arr
            if not isnone(shared_data_arrays) and shared_data_arrays[name] then
                arg_arr = shared_data_arrays[name]

                if table.prod(arg_arr.shape) >= table.prod(arg_shape[i]) then
                    -- good, we can share this memory
                    assert(arg_types[i] == arg_arr.dtype)
                    arg_arr = arg_arr:reshape(arg_shape[i])
                else
                    logger.warning(
                            ('bucketing: data "%s" has a shape %s'):format(name, _b.tostring(arg_shape[i])) ..
                                    (', which is larger than already allocated ') ..
                                    ('shape %s'):format(_b.tostring(arg_arr.shape)) ..
                                    ('. Need to re-allocate. Consider putting ') ..
                                    ('default_bucket_key to be the bucket taking the largest ') ..
                                    ('input for better memory sharing.'))
                    arg_arr = nd.zeros(arg_shape[i], ctx, arg_types[i])
                    -- replace existing shared array because the new one is bigger
                    shared_data_arrays[name] = arg_arr
                end
            else
                arg_arr = nd.zeros(arg_shape[i], ctx, arg_types[i])
                if not isnone(shared_data_arrays) then
                    shared_data_arrays[name] = arg_arr
                end
            end
            table.insert(arg_arrays, arg_arr)
        else
            -- model parameter
            local arg_arr
            if isnone(base_exec) then
                arg_arr = nd.zeros(arg_shape[i], ctx, arg_types[i])
                if need_grad[name] then
                    local grad_arr = nd.zeros(arg_shape[i], ctx, arg_types[i])
                    grad_arrays[name] = grad_arr
                end
            else
                arg_arr = base_exec.arg_dict[name]
                assert(table.equal(arg_arr.shape, arg_shape[i]))
                assert(arg_arr.dtype == arg_types[i])
                if need_grad[name] then
                    grad_arrays[name] = base_exec.grad_dict[name]
                end
            end
            table.insert(arg_arrays, arg_arr)
        end
    end

    -- create or borrow aux variables
    local aux_arrays = {}
    if isnone(base_exec) then
        for i = 1, #aux_shape do
            table.insert(aux_arrays, nd.zeros(aux_shape[i], ctx, aux_types[i]))
        end
    else
        for i, a in ipairs(base_exec.aux_arrays) do
            assert(table.equal(aux_shape[i], a.shape))
            assert(aux_types[i] == a.dtype)
            table.insert(aux_arrays, a)
        end
    end

    local executor = sym:bind(ctx, arg_arrays, grad_arrays, grad_req, aux_arrays, nil, base_exec)
    return executor
end

---@class mx.executor_manager.DataParallelExecutorGroup
local DataParallelExecutorGroup = class('mx.executor_manager.DataParallelExecutorGroup')
M.DataParallelExecutorGroup = DataParallelExecutorGroup

function DataParallelExecutorGroup:ctor(sym, arg_names, param_names, ctx, slices, train_data, shared_group)
    -- make sure the architecture is valid
    M._check_arguments(sym)

    if isnone(shared_group) then
        local s = {}
        for _, v in ipairs(ctx) do
            table.insert(s, {})
        end
        self.shared_data_arrays = s
    else
        self.shared_data_arrays = shared_group.shared_data_arrays
    end

    self.data_names = table.map(train_data.provide_data, 1)
    self.data_names = table.map(train_data.provide_label, 1)

    self.aux_names = sym:list_auxiliary_states()
    self.param_idx = {}
    for i = 1, #arg_names do
        if param_names[arg_names[i]] then
            table.insert(self.param_idx, i)
        end
    end
    self.param_names = {}
    for _, i in ipairs(self.param_idx) do
        table.insert(self.param_names, arg_names[i])
    end

    ---@type mx.executor.Executor[]
    self.train_execs = {}
    for i, ctxi in ipairs(ctx) do
        local data_shapes = {}
        local data_types = {}
        for _, x in ipairs(table.sequence(train_data.provide_data, train_data.provide_label)) do
            local stop = slices[i][2]
            local start = slices[i][1]
            local s = table.clone(x[1])
            s[1] = stop - start
            local name = x[1]
            data_shapes[name] = s
            if isinstance(x, DataDesc) then
                data_types[x.name] = x.dtype
            else
                data_types[name] = mx_real_t
            end
        end
        local shared_exec = isnone(shared_group) and None or shared_group.train_execs[i]
        local train_exec = M._bind_exec(
                sym, ctxi, data_shapes, self.param_names, true, shared_exec, self.shared_data_arrays[i], data_types)
        table.insert(self.train_execs, train_exec)
    end

    -- data structure
    self.data_arrays = table.map(self.data_names, function(name)
        return table.map(self.train_execs, function(e, i)
            return { slices[i], e.arg_dict[name] }
        end)
    end)
    self.label_arrays = table.map(self.label_names, function(name)
        return table.map(self.train_execs, function(e, i)
            return { slices[i], e.arg_dict[name] }
        end)
    end)

    self.param_arrays = table.map(self.param_idx, function(i)
        return table.map(self.train_execs, function(e)
            return e.arg_arrays[i]
        end)
    end)
    self.grad_arrays = table.map(self.param_idx, function(i)
        return table.map(self.train_execs, function(e)
            return e.grad_arrays[i]
        end)
    end)

    self.aux_arrays = table.arange(function(i)
        return table.map(self.train_execs, function(e)
            return e.aux_arrays[i]
        end)
    end, #self.aux_names)

    self.slices = slices
end

function DataParallelExecutorGroup:load_data_batch(data_batch)
    M._load_data(data_batch, self.data_arrays)
    M._load_label(data_batch, self.label_arrays)
end

function DataParallelExecutorGroup:forward(is_train)
    is_train = default(is_train, false)
    for _, texec in ipairs(self.train_execs) do
        texec:forward(is_train)
    end
end

function DataParallelExecutorGroup:backward()
    for _, texec in ipairs(self.train_execs) do
        texec:backward()
    end
end

function DataParallelExecutorGroup:update_metric(metric, labels, pre_sliced)
    pre_sliced = default(pre_sliced, false)
    for i = 1, #self.train_execs do
        local current_exec = i
        local texec, islice = self.train_execs[i], self.slices[i]
        local labels_slice
        if not pre_sliced then
            labels_slice = {}
            for label in iter(labels) do
                table.insert(labels_slice, label[islice])
            end
        else
            labels_slice = labels[current_exec]
        end
        metric:update(labels_slice, texec.outputs)
    end
end

---@class mx.executor_manager.DataParallelExecutorManager
local DataParallelExecutorManager = class('mx.executor_manager.DataParallelExecutorManager')
M.DataParallelExecutorManager = DataParallelExecutorManager

function DataParallelExecutorManager:ctor(symbol, ctx, train_data,
                                          arg_names, param_names, aux_names,
                                          work_load_list, logger, sym_gen)
    if isnone(logger) then
        logger = logging
    end
    -- preparation
    local num_device = len(ctx)
    logger.info('Start training with %s', tostring(ctx))

    if isnone(work_load_list) then
        work_load_list = table.replicate(1, num_device)
    end
    assert(#work_load_list == num_device, 'Invalid settings for work load.')

    local slices = M._split_input_slice(train_data.batch_size, work_load_list)
    self.slices = slices

    self.arg_names = arg_names
    self.param_names = param_names
    self.aux_names = aux_names
    self.ctx = ctx

    self.execgrp = DataParallelExecutorGroup(symbol, self.arg_names, self.param_names, self.ctx,
                                             self.slices, train_data)
    self.symbol = symbol

    self.sym_gen = sym_gen
    self.curr_execgrp = None -- this is set when data is loaded
    if not isnone(self.sym_gen) then
        self.execgrp_bucket = { [train_data.default_bucket_key] = self.execgrp }
    end
end

function DataParallelExecutorManager:install_monitor(monitor)
    if not isnone(self.sym_gen) then
        raise('NotImplementedError', 'Monitoring is not implemented for bucketing')
    end

    for _, train_exec in ipairs(self.execgrp.train_execs) do
        monitor:install(train_exec)
    end
end

function DataParallelExecutorManager:set_params(arg_params, aux_params)
    for _, texec in ipairs(self.execgrp.train_execs) do
        texec:copy_params_from(arg_params, aux_params)
    end
end

function DataParallelExecutorManager:copy_to(arg_params, aux_params)
    for i = 1, #self.param_names do
        local name, block = self.param_names[i], self.param_arrays[i]
        local ws = {}
        for w in iter(block) do
            table.insert(ws, w:copyto(cpu()))
        end
        local weight = table.sum(ws) / len(block)
        weight:astype(arg_params[name].dtype):copyto(arg_params[name])
    end
    for i = 1, #self.aux_names do
        local name, block = self.aux_names[i], self.aux_arrays[i]
        local ws = {}
        for w in iter(block) do
            table.insert(ws, w:copyto(cpu()))
        end
        local weight = table.sum(ws) / len(block)
        weight:astype(aux_params[name].dtype):copyto(aux_params[name])
    end
end

function DataParallelExecutorManager:_param_arrays()
    return self.execgrp.param_arrays
end

function DataParallelExecutorManager:_grad_arrays()
    return self.execgrp.grad_arrays
end

function DataParallelExecutorManager:_aux_arrays()
    return self.execgrp.aux_arrays
end

function DataParallelExecutorManager:load_data_batch(data_batch)
    if not isnone(self.sym_gen) then
        local key = data_batch.bucket_key
        if not self.execgrp_bucket[key] then
            -- create new bucket entry
            local symbol = self.sym_gen(key)
            local execgrp = DataParallelExecutorGroup(symbol, self.arg_names,
                                                      self.param_names, self.ctx,
                                                      self.slices, data_batch,
                                                      self.execgrp)
            self.execgrp_bucket[key] = execgrp
        end
        self.curr_execgrp = self.execgrp_bucket[key]
    else
        self.curr_execgrp = self.execgrp
    end

    self.curr_execgrp:load_data_batch(data_batch)
end

function DataParallelExecutorManager:forward(is_train)
    self.curr_execgrp:forward(is_train)
end

function DataParallelExecutorManager:backward()
    self.curr_execgrp:backward()
end

function DataParallelExecutorManager:update_metric(metric, labels, pre_sliced)
    pre_sliced = default(pre_sliced, false)
    self.curr_execgrp:update_metric(metric, labels, pre_sliced)
end

class_property(DataParallelExecutorManager, {
    param_arrays = DataParallelExecutorManager._param_arrays,
    grad_arrays  = DataParallelExecutorManager._grad_arrays,
    aux_arrays   = DataParallelExecutorManager._aux_arrays, })

return M
