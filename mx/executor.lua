--
local M = {}

local copy = require('python.copy')
local ctypes = require('ctypes')
local base = require('mx.base')
local _LIB, mx_uint, NDArrayHandle, ExecutorHandle, py_str, mx_int, check_call, c_handle_array, c_array_buf, c_str_array = base._LIB, base.mx_uint, base.NDArrayHandle, base.ExecutorHandle, base.py_str, base.mx_int, base.check_call, base.c_handle_array, base.c_array_buf, base.c_str_array
local __ndarray = require('mx.ndarray.__init__')
local NDArray = __ndarray.NDArray
local _ndarray_cls = __ndarray._ndarray_cls

local function _monitor_callback_wrapper(callback)
    return function(name, array, _)
        callback(name, array)
    end
end

---@class mx.executor.Executor
---@field arg_dict table<string, mx.ndarray.NDArray>
---@field grad_dict table<string, mx.ndarray.NDArray>
---@field aux_dict table<string, mx.ndarray.NDArray>
---@field output_dict table<string, mx.ndarray.NDArray>
local Executor = class('mx.executor.Executor')
M.Executor = Executor

function Executor:ctor(handle, symbol, ctx, grad_req, group2ctx)
    --if not isinstance(handle, ExecutorHandle) then
    if not ctypes.is_ctype(handle) then
        print(handle, type(handle), type(handle.value))
        raise('TypeError', 'Handle type error')
    end
    self.handle = handle
    self.arg_arrays = {}
    self.grad_arrays = {}
    self.aux_arrays = {}
    self.outputs = self:_get_outputs()
    self._symbol = copy.deepcopy(symbol)
    self._optimized_symbol = None
    self._arg_dict = None
    self._grad_dict = None
    self._aux_dict = None
    self._output_dict = None
    self._monitor_callback = None
    self._ctx = copy.deepcopy(ctx)
    self._grad_req = copy.deepcopy(grad_req)
    self._group2ctx = copy.deepcopy(group2ctx)
end

function Executor:dtor()
    check_call(_LIB.MXExecutorFree(self.handle))
end

function Executor._get_dict(names, ndarrays)
    local nset = {}
    for _, nm in ipairs(names) do
        if nset[nm] then
            raise('ValueError', 'Duplicate names detected: ' .. nm)
        end
        nset[nm] = true
    end
    local ret = {}
    for i = 1, #names do
        ret[names[i]] = ndarrays[i]
    end
    return ret
end

function Executor:_get_outputs()
    local out_size = mx_uint()
    local handles = ctypes.POINTER(NDArrayHandle)()
    check_call(_LIB.MXExecutorOutputs(self.handle,
                                      ctypes.byref(out_size), ctypes.byref(handles)))
    local num_output = out_size.value
    local outputs = {}
    for i = 1, num_output do
        table.insert(outputs, _ndarray_cls(NDArrayHandle(handles[i - 1])))
    end
    return outputs
end

function Executor:forward(is_train, kwargs)
    is_train = default(is_train, false)
    if kwargs and not table.empty(kwargs) then
        local arg_dict = self.arg_dict
        for name, array in pairs(kwargs) do
            if not isinstance(array, NDArray) then
                raise('ValueError', 'only accept keyword argument of NDArrays')
            end
            if not arg_dict[name] then
                raise('TypeError', 'Unknown argument ' .. name)
            end
            if not table.equal(arg_dict[name], array.shape) then
                raise('ValueError', 'shape not match')
            end
            arg_dict[name][{}] = array
        end
    end
    check_call(_LIB.MXExecutorForward(
            self.handle,
            ctypes.c_int(is_train)))
    self.outputs = self:_get_outputs()
    return self.outputs
end

function Executor:backward(out_grads, is_train)
    is_train = default(is_train, true)
    if isnone(out_grads) then
        out_grads = {}
    elseif isinstance(out_grads, NDArray) then
        out_grads = { out_grads }
    elseif #out_grads == 0 then
        -- is dict
        local tmp = {}
        for _, k in ipairs(self._symbol:list_outputs()) do
            table.insert(tmp, out_grads[k])
        end
        out_grads = tmp
    end
    for _, v in ipairs(out_grads) do
        if not isinstance(v, NDArray) then
            raise('TypeError', 'inputs must be NDArray')
        end
    end
    local arr = c_handle_array(out_grads)
    check_call(_LIB.MXExecutorBackwardEx(
            self.handle,
            mx_uint(len(out_grads)),
            arr,
            ctypes.c_int(is_train)))
end

function Executor:set_monitor_callback(callback, monitor_all)
    monitor_all = default(monitor_all, false)
    --cb_type = ctypes.CFUNCTYPE(None, ctypes.c_char_p, NDArrayHandle, ctypes.c_void_p)
    --self._monitor_callback = cb_type(_monitor_callback_wrapper(callback))
    self._monitor_callback = ffi.cast('void(*)(const char*, void*, void*)',
                                      _monitor_callback_wrapper(callback))
    check_call(_LIB.MXExecutorSetMonitorCallbackEX(
            self.handle,
            self._monitor_callback,
            nil,
            ctypes.c_int(monitor_all)))
end

function Executor:__arg_dict()
    if isnone(self._arg_dict) then
        self._arg_dict = Executor._get_dict(
                self._symbol:list_arguments(), self.arg_arrays)
    end
    return self._arg_dict
end

function Executor:__grad_dict()
    if isnone(self._grad_dict) then
        self._grad_dict = Executor._get_dict(
                self._symbol:list_arguments(), self.grad_arrays)
    end
    return self._grad_dict
end

function Executor:__aux_dict()
    if isnone(self._aux_dict) then
        self._aux_dict = Executor._get_dict(
                self._symbol:list_auxiliary_states(), self.aux_arrays)
    end
    return self._aux_dict
end

function Executor:__output_dict()
    if isnone(self._output_dict) then
        self._output_dict = Executor._get_dict(
                self._symbol:list_outputs(), self.outputs)
    end
    return self._output_dict
end

function Executor:copy_params_from(arg_params, aux_params, allow_extra_params)
    allow_extra_params = default(allow_extra_params, false)
    for name, array in pairs(arg_params) do
        if self.arg_dict[name] then
            local dst = self.arg_dict[name]
            array:astype(dst.dtype):copyto(dst)
        elseif not allow_extra_params then
            raise('ValueError', ('name %s is not in the arguments'):format(name))
        end
    end
    if isnone(aux_params) then
        return
    end
    for name, array in pairs(aux_params) do
        if self.aux_dict[name] then
            local dst = self.aux_dict[name]
            array:astype(dst.dtype):copyto(dst)
        elseif not allow_extra_params then
            raise('ValueError', ('name %s is not in the auxiliary states'):format(name))
        end
    end
end

function Executor:reshape(partial_shaping, allow_up_sizing, kwargs)
    partial_shaping, allow_up_sizing, kwargs = default(
            partial_shaping, false, allow_up_sizing, false, kwargs, {})
    local provided_arg_shape_data = {}  -- shape data
    -- argument shape index in sdata,
    -- e.g. [sdata[indptr[0]], sdata[indptr[1]]) is the shape of the first arg
    local provided_arg_shape_idx = { 0 }
    local provided_arg_shape_names = {}  -- provided argument names
    for k, v in pairs(kwargs) do
        if islist(v) then
            table.insret(provided_arg_shape_names, k)
            table.append(provided_arg_shape_data, v)
            table.insret(provided_arg_shape_idx, #provided_arg_shape_data)
        end
    end
    local ctx_map_keys = {}
    local ctx_map_dev_types = {}
    local ctx_map_dev_ids = {}
    if self._group2ctx then
        for key, val in pairs(self._group2ctx) do
            table.insret(ctx_map_keys, key)
            table.insret(ctx_map_dev_types, val.device_typeid)
            table.insret(ctx_map_dev_ids, val.device_id)
        end
    end
    local handle = ExecutorHandle()
    local shared_handle = self.handle

    local num_in_args = ctypes.c_uint()
    local in_arg_handles = ctypes.POINTER(NDArrayHandle)()
    local arg_grad_handles = ctypes.POINTER(NDArrayHandle)()
    local num_aux_states = ctypes.c_uint()
    local aux_state_handles = ctypes.POINTER(NDArrayHandle)()

    check_call(_LIB.MXExecutorReshapeEx(ctypes.c_int(partial_shaping),
                                        ctypes.c_int(allow_up_sizing),
                                        ctypes.c_int(self._ctx.device_typeid),
                                        ctypes.c_int(self._ctx.device_id),
                                        mx_uint(len(ctx_map_keys)),
                                        c_str_array(ctx_map_keys),
                                        c_array_buf(ctypes.c_int,
                                                    ctx_map_dev_types),
                                        c_array_buf(ctypes.c_int,
                                                    ctx_map_dev_ids),
                                        mx_uint(len(provided_arg_shape_names)),
                                        c_str_array(provided_arg_shape_names),
                                        c_array_buf(mx_int,
                                                    provided_arg_shape_data),
                                        c_array_buf(mx_uint,
                                                    provided_arg_shape_idx),
                                        ctypes.byref(num_in_args),
                                        ctypes.byref(in_arg_handles),
                                        ctypes.byref(arg_grad_handles),
                                        ctypes.byref(num_aux_states),
                                        ctypes.byref(aux_state_handles),
                                        shared_handle,
                                        ctypes.byref(handle)))

    local arg_arrays = table.arange(function(i)
        return _ndarray_cls(NDArrayHandle(in_arg_handles[i - 1]))
    end, num_in_args.value)
    local grad_arrays = table.arange(function(i)
        local hdl = arg_grad_handles[i]
        if ffi.isnullptr(hdl) then
            return None
        else
            return _ndarray_cls(hdl)
        end
    end, num_in_args.value)
    local aux_arrays = table.arange(function(i)
        return _ndarray_cls(NDArrayHandle(aux_state_handles[i - 1]))
    end, num_aux_states.value)

    local executor = Executor(handle, self._symbol, self._ctx, self._grad_req, self._group2ctx)
    executor.arg_arrays = arg_arrays
    executor.grad_arrays = grad_arrays
    executor.aux_arrays = aux_arrays
    return executor
end

function Executor:debug_str()
    local debug_str = ctypes.c_char_p()
    check_call(_LIB.MXExecutorPrint(
            self.handle, ctypes.byref(debug_str)))
    return py_str(debug_str.value)
end

local _field = {
    arg_dict    = Executor.__arg_dict,
    grad_dict   = Executor.__grad_dict,
    aux_dict    = Executor.__aux_dict,
    output_dict = Executor.__output_dict,
}

function Executor:__index(k)
    local tk = type(k)
    if tk == 'string' then
        if _field[k] then
            return _field[k](self)
        elseif not isnone(Executor[k]) then
            return Executor[k]
        end
    end
    return rawget(self, k)
end

return M
