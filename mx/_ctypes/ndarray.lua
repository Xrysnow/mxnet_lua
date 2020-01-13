local ctypes = require('ctypes')
local base = require('mx.base')
local _LIB = base._LIB
local c_str_array = base.c_str_array
local c_handle_array = base.c_handle_array
local NDArrayHandle = base.NDArrayHandle
local CachedOpHandle = base.CachedOpHandle
local check_call = base.check_call

--
local M = {}

---@class mx._ctypes.ndarray.NDArrayBase
local NDArrayBase = class('mx._ctypes.ndarray.NDArrayBase')
M.NDArrayBase = NDArrayBase

function NDArrayBase:ctor(handle, writable)
    writable = default(writable, true)
    if not isnone(handle) then
        --assert(isinstance(handle, NDArrayHandle))
        assert(type(handle) == 'cdata' or handle.value)
    end
    self.handle = handle
    self.writable = writable
end

function NDArrayBase:dtor()
    check_call(_LIB.MXNDArrayFree(self.handle))
end

local _ndarray_cls
M._ndarray_cls = _ndarray_cls

function M._set_ndarray_class(cls)
    _ndarray_cls = cls
end

function M._imperative_invoke(handle, ndargs, keys, vals, out)
    local original_output, num_output, output_vars
    if not isnone(out) then
        original_output = out
        if isinstance(out, NDArrayBase) then
            out = { out }
        end
        num_output = ctypes.c_int(len(out))
        output_vars = c_handle_array(out)
        output_vars = ctypes.cast(output_vars, ctypes.POINTER(NDArrayHandle))
    else
        original_output = None
        output_vars = ctypes.POINTER(NDArrayHandle)()
        num_output = ctypes.c_int(0)
    end
    -- return output stypes to avoid the c_api call for checking
    -- a handle's stype in _ndarray_cls
    local out_stypes = ctypes.POINTER(ctypes.c_int)()

    vals = table.map(vals, base.tostring)

    check_call(_LIB.MXImperativeInvokeEx(
            ctypes.c_void_p(handle),
            ctypes.c_int(len(ndargs)),
            c_handle_array(ndargs),
            ctypes.byref(num_output),
            ctypes.byref(output_vars),
            ctypes.c_int(len(keys)),
            c_str_array(keys),
            c_str_array(vals),
            ctypes.byref_const(out_stypes)))

    if not isnone(original_output) then
        return original_output
    end
    local ret = {}
    for i = 0, num_output.value - 1 do
        table.insert(ret, _ndarray_cls(ctypes.cast(output_vars[i], NDArrayHandle),
                                       out_stypes[i]))
    end
    if num_output.value == 1 then
        return ret[1]
    else
        return ret
    end
end

---@class mx._ctypes.ndarray.CachedOp
local CachedOp = class('CachedOp')
M.CachedOp = CachedOp

function CachedOp:ctor(sym, flags)
    flags = default(flags, {})
    self.handle = CachedOpHandle()
    local keys, vals = {}, {}
    for _, v in ipairs(flags) do
        table.insert(keys, v[1])
        table.insert(vals, base.tostring(v[2]))
    end
    check_call(_LIB.MXCreateCachedOpEx(
            sym.handle,
            len(flags),
            c_str_array(keys),
            c_str_array(vals),
            ctypes.byref(self.handle)))
end

function CachedOp:dtor()
    check_call(_LIB.MXFreeCachedOp(self.handle))
end

function CachedOp:__call(...)
    local args, kwargs = arg_kw(...)
    local out = arg_pop(kwargs, 'out', None)
    local original_output, num_output, output_vars
    if not isnone(out) then
        original_output = out
        if isinstance(out, NDArrayBase) then
            out = { out }
        end
        num_output = ctypes.c_int(len(out))
        output_vars = c_handle_array(out)
        output_vars = ctypes.cast(output_vars, ctypes.POINTER(NDArrayHandle))
    else
        original_output = None
        output_vars = ctypes.POINTER(NDArrayHandle)()
        num_output = ctypes.c_int(0)
    end
    if not table.empty(kwargs) then
        raise('TypeError',
              "CachedOp.__call__ got unexpected keyword argument(s): ",
              table.concat(table.keys(kwargs), ', '))
    end
    -- return output stypes to avoid the c_api call for checking
    -- a handle's stype in _ndarray_cls
    local out_stypes = ctypes.POINTER(ctypes.c_int)()

    check_call(_LIB.MXInvokeCachedOpEx(
            self.handle,
            ctypes.c_int(len(args)),
            c_handle_array(args),
            ctypes.byref(num_output),
            ctypes.byref(output_vars),
            ctypes.byref(out_stypes)))

    if not isnone(original_output) then
        return original_output
    end
    local ret = {}
    for i = 0, num_output.value - 1 do
        table.insert(ret, _ndarray_cls(ctypes.cast(output_vars[i], NDArrayHandle),
                                       out_stypes[i]))
    end
    if num_output.value == 1 then
        return ret[1]
    else
        return ret
    end
end

return M
