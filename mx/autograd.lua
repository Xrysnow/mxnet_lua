--
local M = {}
local ctypes = require('ctypes')
local c_int, c_void_p, CFUNCTYPE, POINTER, cast = ctypes.c_int,
ctypes.c_void_p, ctypes.CFUNCTYPE, ctypes.POINTER, ctypes.cast

local base = require('mx.base')
local _LIB, check_call, string_types, mx_uint, NDArrayHandle,
c_array, c_handle_array, c_array_buf, MXCallbackList, SymbolHandle = base._LIB,
base.check_call, base.string_types, base.mx_uint, base.NDArrayHandle,
base.c_array, base.c_handle_array, base.c_array_buf, base.MXCallbackList, base.SymbolHandle

local ndarray = require('mx.ndarray.__init__')
local NDArray, _ndarray_cls, _GRAD_REQ_MAP = ndarray.NDArray, ndarray._ndarray_cls, ndarray._GRAD_REQ_MAP

function M.set_recording(is_recording)
    local prev = ctypes.int()
    check_call(_LIB.MXAutogradSetIsRecording(
            ctypes.c_int(is_recording), ctypes.byref(prev)))
    return bool(prev.value)
end

function M.set_training(train_mode)
    local prev = ctypes.int()
    check_call(_LIB.MXAutogradSetIsTraining(
            ctypes.c_int(train_mode), ctypes.byref(prev)))
    return bool(prev.value)
end

function M.is_recording()
    local curr = ctypes.c_bool()
    check_call(_LIB.MXAutogradIsRecording(ctypes.byref(curr)))
    return curr.value
end

function M.is_training()
    local curr = ctypes.c_bool()
    check_call(_LIB.MXAutogradIsTraining(ctypes.byref(curr)))
    return curr.value
end

---@class mx.autograd._RecordingStateScope
local _RecordingStateScope = class('mx.autograd._RecordingStateScope')
M._RecordingStateScope = _RecordingStateScope

function _RecordingStateScope:ctor(is_record, train_mode)
    self._enter_is_record = is_record
    self._enter_train_mode = train_mode
    self._prev_is_record = nil
    self._prev_train_mode = nil
end

function _RecordingStateScope:__enter()
    if not isnone(self._enter_is_record) then
        self._prev_is_record = M.set_recording(self._enter_is_record)
    end
    if not isnone(self._enter_train_mode) then
        self._prev_train_mode = M.set_training(self._enter_train_mode)
    end
end

function _RecordingStateScope:__exit()
    if not isnone(self._enter_is_record) and self._prev_is_record ~= self._enter_is_record then
        M.set_recording(self._prev_is_record)
    end
    if not isnone(self._enter_train_mode) and self._prev_train_mode ~= self._enter_train_mode then
        M.set_training(self._prev_train_mode)
    end
end

function M.record(train_mode)
    train_mode = default(train_mode, true)
    return _RecordingStateScope(true, train_mode)
end

function M.pause(train_mode)
    train_mode = default(train_mode, false)
    return _RecordingStateScope(false, train_mode)
end

function M.train_mode()
    return _RecordingStateScope(nil, true)
end

function M.predict_mode()
    return _RecordingStateScope(nil, false)
end

function M.mark_variables(variables, gradients, grad_reqs)
    grad_reqs = default(grad_reqs, 'write')
    if isinstance(variables, NDArray) then
        assert(isinstance(gradients, NDArray))
        variables = { variables }
        gradients = { gradients }
    end
    if type(grad_reqs) == 'string' then
        grad_reqs = table.replicate(_GRAD_REQ_MAP[grad_reqs], len(variables))
    else
        grad_reqs = table.map(grad_reqs, function(i)
            return _GRAD_REQ_MAP[i]
        end)
    end
    check_call(_LIB.MXAutogradMarkVariables(
            len(variables),
            c_handle_array(variables),
            c_array(mx_uint, grad_reqs),
            c_handle_array(gradients)))
end

function M._parse_head(heads, head_grads)
    if isinstance(heads, NDArray) then
        heads = { heads }
    end
    if isinstance(head_grads, NDArray) then
        head_grads = { head_grads }
    end
    local head_handles = c_handle_array(heads)
    local hgrad_handles
    if isnone(head_grads) then
        hgrad_handles = ctypes.c_void_p(0)
    else
        assert(len(heads) == len(head_grads),
               "heads and head_grads must be lists of the same length")
        hgrad_handles = {}
        for _, i in ipairs(head_grads) do
            if not isnone(i) then
                table.insert(hgrad_handles, i.handle)
            else
                table.insert(hgrad_handles, NDArrayHandle(0))
            end
        end
        hgrad_handles = c_array(NDArrayHandle, hgrad_handles)
    end
    return head_handles, hgrad_handles
end

function M.backward(heads, head_grads, retain_graph, train_mode)
    head_grads, retain_graph, train_mode = default(
            head_grads, None, retain_graph, false, train_mode, true)
    local head_handles, hgrad_handles = M._parse_head(heads, head_grads)
    check_call(_LIB.MXAutogradBackwardEx(
            len(head_handles),
            head_handles,
            hgrad_handles,
            0,
            ctypes.c_void_p(0),
            ctypes.c_int(retain_graph),
            ctypes.c_int(0),
            ctypes.c_int(train_mode),
            ctypes.c_void_p(0),
            ctypes.c_void_p(0)))
end

function M.grad(heads, variables, head_grads, retain_graph, create_graph, train_mode)
    head_grads, retain_graph, create_graph, train_mode = default(
            head_grads, None, retain_graph, None, create_graph, false, train_mode, true)
    local head_handles, hgrad_handles = M._parse_head(heads, head_grads)
    if isinstance(variables, NDArray) then
        variables = { variables }
    else
        assert(#variables > 0, 'variables cannot be an empty list.')
    end
    local var_handles = c_handle_array(variables)
    if isnone(retain_graph) then
        retain_graph = create_graph
    end
    local grad_vars = ctypes.POINTER(NDArrayHandle)()
    local grad_stypes = ctypes.POINTER(ctypes.c_int)()

    check_call(_LIB.MXAutogradBackwardEx(
            len(head_handles),
            head_handles,
            hgrad_handles,
            len(var_handles),
            var_handles,
            ctypes.c_int(retain_graph),
            ctypes.c_int(create_graph),
            ctypes.c_int(train_mode),
            ctypes.byref(grad_vars),
            ctypes.byref(grad_stypes)))

    local ret = {}
    for i = 0, #var_handles - 1 do
        table.insert(ret, _ndarray_cls(
                ctypes.cast(grad_vars[i], NDArrayHandle), nil, grad_stypes[i]))
    end
    if isinstance(variables, NDArray) then
        return ret[0]
    end
    return ret
end

function M.get_symbol(x)
    local hdl = SymbolHandle()
    check_call(_LIB.MXAutogradGetSymbol(x.handle, ctypes.byref(hdl)))
    return Symbol(hdl) --TODO
end

---@class mx.autograd.Function
local Function = class('mx.autograd.Function')
M.Function = Function
--TODO


return M
