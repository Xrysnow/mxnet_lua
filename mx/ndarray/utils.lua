--
local ctypes = require('ctypes')
local base = require('mx.base')
local _LIB, check_call, py_str, c_str, string_types, mx_uint, NDArrayHandle, c_array, c_handle_array, c_str_array = base._LIB, base.check_call, base.py_str, base.c_str, base.string_types, base.mx_uint, base.NDArrayHandle, base.c_array, base.c_handle_array, base.c_str_array
local nd = require('mx.ndarray.ndarray')
local NDArray, _array, _empty_ndarray, _zeros_ndarray = nd.NDArray, nd.array, nd.empty, nd.zeros
--from .sparse import zeros as _zeros_sparse_ndarray
--from .sparse import empty as _empty_sparse_ndarray
--from .sparse import array as _sparse_array
--from .sparse import _ndarray_cls
local _sparse = require('mx.ndarray.sparse')
local _zeros_sparse_ndarray, _empty_sparse_ndarray, _sparse_array, _ndarray_cls = _sparse.zeros, _sparse.empty, _sparse.array, _sparse._ndarray_cls

--
local M = {}

--- Return a new array of given shape and type, filled with zeros.
---@return mx.ndarray.NDArray
function M.zeros(shape, ctx, dtype, stype, kwargs)
    if stype == nil or stype == 'default' then
        return _zeros_ndarray(shape, ctx, dtype, kwargs)
    else
        return _zeros_sparse_ndarray(stype, shape, ctx, dtype, kwargs)
    end
end

--- Returns a new array of given shape and type, without initializing entries.
---@return mx.ndarray.NDArray
function M.empty(shape, ctx, dtype, stype)
    if stype == nil or stype == 'default' then
        return _empty_ndarray(shape, ctx, dtype)
    else
        return _empty_sparse_ndarray(stype, shape, ctx, dtype)
    end
end

--- Creates an array from any object exposing the array interface.
---@return mx.ndarray.NDArray
function M.array(source_array, ctx, dtype)
    if isinstance(source_array, NDArray) and source_array.stype ~= 'default' then
        return _sparse_array(source_array, ctx, dtype)
    else
        return _array(source_array, ctx, dtype)
    end
end

--- Loads an array from file.
function M.load(fname)
    if type(fname) ~= 'string' then
        raise('TypeError', 'fname required to be a string')
    end
    local out_size = mx_uint()
    local out_name_size = mx_uint()
    local handles = ctypes.POINTER(NDArrayHandle)()
    local names = ctypes.POINTER(ctypes.c_char_p)()
    check_call(_LIB.MXNDArrayLoad(c_str(fname),
                                  ctypes.byref(out_size),
                                  ctypes.byref(handles),
                                  ctypes.byref(out_name_size),
                                  ctypes.byref(names)))
    if out_name_size.value == 0 then
        local ret = {}
        for i = 0, out_size.value - 1 do
            table.insert(ret, _ndarray_cls(NDArrayHandle(handles[i])))
        end
        return ret
    else
        assert(out_name_size.value == out_size.value)
        local ret = {}
        for i = 0, out_size.value - 1 do
            ret[py_str(names[i])] = _ndarray_cls(NDArrayHandle(handles[i]))
        end
        return ret
    end
end

--- Loads an array dictionary or list from a buffer
function M.load_frombuffer(buf)
    local out_size = mx_uint()
    local out_name_size = mx_uint()
    local handles = ctypes.POINTER(NDArrayHandle)()
    local names = ctypes.POINTER(ctypes.c_char_p)()
    check_call(_LIB.MXNDArrayLoadFromBuffer(buf,
                                            mx_uint(len(buf)),
                                            ctypes.byref(out_size),
                                            ctypes.byref(handles),
                                            ctypes.byref(out_name_size),
                                            ctypes.byref(names)))
    if out_name_size.value == 0 then
        local ret = {}
        for i = 0, out_size.value - 1 do
            table.insert(ret, _ndarray_cls(NDArrayHandle(handles[i])))
        end
        return ret
    else
        assert(out_name_size.value == out_size.value)
        local ret = {}
        for i = 0, out_size.value - 1 do
            ret[py_str(names[i])] = _ndarray_cls(NDArrayHandle(handles[i]))
        end
        return ret
    end
end

--- Saves a list of arrays or a dict of str->array to file.
function M.save(fname, data)
    local handles, keys
    if isinstance(data, NDArray) then
        data = { data }
    end
    assert(type(data) == 'table')
    if #data == 0 then
        local str_keys = table.keys(data)
        local nd_vals = table.values(data)
        keys = c_str_array(str_keys)
        handles = c_handle_array(nd_vals)
    else
        keys = nil
        handles = c_handle_array(data)
    end
    check_call(_LIB.MXNDArraySave(c_str(fname),
                                  mx_uint(len(handles)),
                                  handles,
                                  keys))
end

return M
