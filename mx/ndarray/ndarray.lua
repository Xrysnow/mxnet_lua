--
local ctypes = require('ctypes')
local base = require('mx.base')
local _LIB, numeric_types, integer_types,
c_str, c_array, c_array_buf, c_handle_array,
mx_real_t, mx_uint, NDArrayHandle, check_call, DLPackHandle,
mx_int, ctypes2buffer = base._LIB, base.numeric_types, base.integer_types,
base.c_str, base.c_array, base.c_array_buf, base.c_handle_array,
base.mx_real_t, base.mx_uint, base.NDArrayHandle, base.check_call, base.DLPackHandle,
base.mx_int, base.ctypes2buffer

local context = require('mx.context')
local Context, current_context = context.Context, context.current_context

local _internal = require('mx.ndarray._internal')
local op = require('mx.ndarray.op')
local NDArrayBase = _internal.NDArrayBase

---@class mx.ndarray.ndarray
local M = {}
M.op = op
M.NDArrayBase = NDArrayBase
--

M._STORAGE_TYPE_UNDEFINED = -1
M._STORAGE_TYPE_DEFAULT = 0
M._STORAGE_TYPE_ROW_SPARSE = 1
M._STORAGE_TYPE_CSR = 2

M._DTYPE_NP_TO_MX = {
    None    = -1,
    float32 = 0,
    float64 = 1,
    float16 = 2,
    uint8   = 3,
    int32   = 4,
    int8    = 5,
    int64   = 6,
}

M._DTYPE_MX_TO_NP = {}
for k, v in pairs(M._DTYPE_NP_TO_MX) do
    M._DTYPE_MX_TO_NP[v] = k
end

local function check_dtype(dtype)
    if not dtype or not M._DTYPE_NP_TO_MX[dtype] then
        raise('ValueError', 'invalid dtype: ' .. tostring(dtype))
    end
    return dtype
end

M._STORAGE_TYPE_STR_TO_ID = {
    undefined  = M._STORAGE_TYPE_UNDEFINED,
    default    = M._STORAGE_TYPE_DEFAULT,
    row_sparse = M._STORAGE_TYPE_ROW_SPARSE,
    csr        = M._STORAGE_TYPE_CSR,
}

M._STORAGE_TYPE_ID_TO_STR = {}
for k, v in pairs(M._STORAGE_TYPE_STR_TO_ID) do
    M._STORAGE_TYPE_ID_TO_STR[v] = k
end

M._GRAD_REQ_MAP = {
    null  = 0,
    write = 1,
    add   = 3,
}

-- Return code for dispatching indexing function call
M._NDARRAY_UNSUPPORTED_INDEXING = -1
M._NDARRAY_BASIC_INDEXING = 0
M._NDARRAY_ADVANCED_INDEXING = 1

local _DTYPE_TO_CTYPE = {
    None    = 'void',
    float32 = 'float',
    float64 = 'double',
    float16 = 'uint16_t',
    uint8   = 'uint8_t',
    int32   = 'int32_t',
    int8    = 'int8_t',
    int64   = 'int64_t',
}

function M._new_empty_handle()
    local hdl = NDArrayHandle()
    check_call(_LIB.MXNDArrayCreateNone(ctypes.byref(hdl)))
    return hdl
end

function M._new_alloc_handle(shape, ctx, delay_alloc, dtype)
    dtype = check_dtype(default(dtype, 'float32'))
    local hdl = NDArrayHandle()
    check_call(_LIB.MXNDArrayCreateEx(
            c_array(mx_uint, shape),
            mx_uint(len(shape)),
            ctypes.c_int(ctx.device_typeid),
            ctypes.c_int(ctx.device_id),
            ctypes.c_int(delay_alloc),
            ctypes.c_int(M._DTYPE_NP_TO_MX[dtype]),
            ctypes.byref(hdl)))
    return hdl
end

function M._new_from_shared_mem(shared_pid, shared_id, shape, dtype)
    dtype = check_dtype(dtype)
    local hdl = NDArrayHandle()
    check_call(_LIB.MXNDArrayCreateFromSharedMemEx(
            ctypes.c_int(shared_pid),
            ctypes.c_int(shared_id),
            c_array(mx_int, shape),
            mx_int(len(shape)),
            ctypes.c_int(M._DTYPE_NP_TO_MX[dtype]),
            ctypes.byref(hdl)))
    return hdl
end

--- Wait for all async operations to finish in MXNet.
--- This function is used for benchmarking only.
--- note: If your mxnet code throws an exception, then waitall can cause performance impact.
function M.waitall()
    check_call(_LIB.MXNDArrayWaitAll())
end

function M._storage_type(handle)
    local storage_type = ctypes.c_int(0)
    check_call(_LIB.MXNDArrayGetStorageType(handle, ctypes.byref(storage_type)))
    return storage_type.value
end

--

---@class mx.ndarray.NDArray:mx._ctypes.ndarray.NDArrayBase
---@alias mx.NDArray
---@field ndim number
---@field shape number[]
---@field size number
---@field context mx.Context
---@field dtype string
---@field stype string
---@field T mx.ndarray.NDArray
---@field _fresh_grad number
---@field grad mx.ndarray.NDArray
local NDArray = class('mx.ndarray.NDArray', NDArrayBase)
M.NDArray = NDArray

function NDArray:__tostring()
    local shape_info = table.concat(table.map(self.shape, function(x)
        return ('%d'):format(x)
    end), 'x')
    local info = string.format('<%s %s @%s>', 'NDArray', shape_info, self.context)
    if self.size <= 100 and self.ndim <= 2 then
        local t = self:astable()
        if self.ndim == 1 then
            info = ('%s\n%s'):format(base.tostring(t), info)
        else
            local s = table.concat(table.map(t, base.tostring), ',\n ')
            info = ('[%s]\n%s'):format(s, info)
        end
    end
    return info
end

function NDArray:_to_shared_mem()
    local shared_pid = ctypes.c_int()
    local shared_id = ctypes.c_int()
    check_call(_LIB.MXNDArrayGetSharedMemHandle(
            self.handle, ctypes.byref(shared_pid), ctypes.byref(shared_id)))
    return shared_pid.value, shared_id.value, self.shape, self.dtype
end

function NDArray:__add(other)
    return M.add(self, other)
end

function NDArray:iadd(other)
    if not self.writable then
        raise('ValueError', 'trying to add to a readonly NDArray')
    end
    if isinstance(other, NDArray) then
        return op.broadcast_add(self, other, self)
    elseif type(other) == 'number' then
        return _internal._plus_scalar(self, other, self)
    else
        raise('TypeError', 'type not supported')
    end
end

function NDArray:__sub(other)
    return M.subtract(self, other)
end

function NDArray:isub(other)
    if not self.writable then
        raise('ValueError', 'trying to subtract from a readonly NDArray')
    end
    if isinstance(other, NDArray) then
        return op.broadcast_sub(self, other, self)
    elseif type(other) == 'number' then
        return _internal._minus_scalar(self, other, self)
    else
        raise('TypeError', 'type not supported')
    end
end

function NDArray:__unm()
    return _internal._mul_scalar(self, -1.0)
end

function NDArray:__mul(other)
    return M.multiply(self, other)
end

function NDArray:imul(other)
    if not self.writable then
        raise('ValueError', 'trying to multiply to a readonly NDArray')
    end
    if isinstance(other, NDArray) then
        return op.broadcast_mul(self, other, self)
    elseif type(other) == 'number' then
        return _internal._mul_scalar(self, other, self)
    else
        raise('TypeError', 'type not supported')
    end
end

function NDArray:__div(other)
    return M.divide(self, other)
end

function NDArray:idiv(other)
    if not self.writable then
        raise('ValueError', 'trying to divide from a readonly NDArray')
    end
    if isinstance(other, NDArray) then
        return op.broadcast_div(self, other, self)
    elseif type(other) == 'number' then
        return _internal._div_scalar(self, other, self)
    else
        raise('TypeError', 'type not supported')
    end
end

function NDArray:__mod(other)
    return M.modulo(self, other)
end

function NDArray:imod(other)
    if not self.writable then
        raise('ValueError', 'trying to take modulo from a readonly NDArray')
    end
    if isinstance(other, NDArray) then
        return op.broadcast_mod(self, other, self)
    elseif type(other) == 'number' then
        return _internal._mod_scalar(self, other, self)
    else
        raise('TypeError', 'type not supported')
    end
end

function NDArray:__pow(other)
    return M.power(self, other)
end
--[[
function NDArray:__eq(other)
    return M.equal(self, other)
end

function NDArray:__lt(other)
    return M.lesser(self, other)
end

function NDArray:__le(other)
    return M.lesser_equal(self, other)
end
]]
function NDArray:__len()
    return self.shape[1]
end

function NDArray:__getstate()
    local handle = self.handle
    local this = { handle = None }
    if not isnone(handle) then
        local length = ctypes.c_size_t()
        local cptr = ctypes.POINTER(ctypes.c_char)()
        check_call(_LIB.MXNDArraySaveRawBytes(self.handle,
                                              ctypes.byref(length),
                                              ctypes.byref(cptr)))
        this['handle'] = ctypes2buffer(cptr, length.value)
    end
    return this
end

function NDArray:__setstate(state)
    local handle = state['handle']
    if not isnone(handle) then
        local buf = handle
        handle = NDArrayHandle()
        local ptr = (ctypes.c_char * len(buf)).from_buffer(buf)
        local length = ctypes.c_size_t(len(buf))
        check_call(_LIB.MXNDArrayLoadFromRawBytes(ptr, length, ctypes.byref(handle)))
        self.handle = handle
    else
        self.handle = None
    end
end

function NDArray:__setitem(key, value)
    local indexing_dispatch_code = M._get_indexing_dispatch_code(key)
    if indexing_dispatch_code == M._NDARRAY_BASIC_INDEXING then
        self:_set_nd_basic_indexing(key, value)
    elseif indexing_dispatch_code == M._NDARRAY_ADVANCED_INDEXING then
        self:_set_nd_advanced_indexing(key, value)
    else
        raise('ValueError', 'Indexing NDArray not supported')
    end
end

function NDArray:__getitem(key)
    local indexing_dispatch_code = M._get_indexing_dispatch_code(key)
    if indexing_dispatch_code == M._NDARRAY_BASIC_INDEXING then
        return self:_get_nd_basic_indexing(key)
    elseif indexing_dispatch_code == M._NDARRAY_ADVANCED_INDEXING then
        return self:_get_nd_advanced_indexing(key)
    else
        raise('ValueError', 'Indexing NDArray not supported')
    end
end

function NDArray:_get_index_nd(key)
    -- TODO
    raise('NotImplementedError')
end

function NDArray:_prepare_value_nd(value, vshape)
    local value_nd
    local vt = type(value)
    if vt == 'number' then
        value_nd = M.full(vshape, value, self.context, self.dtype)
    elseif isinstance(value, NDArray) then
        value_nd = value:as_in_context(self.context)
        if value_nd.dtype ~= self.dtype then
            value_nd = value_nd:astype(self.dtype)
        end
    elseif vt == 'cdata' or vt == 'table' then
        value_nd = M.array(value, self.context, self.dtype)
    else
        -- not supported
        --value_nd = M.array(value, self.context, self.dtype)
        raise('TypeError')
    end
    if not table.equal(value_nd.shape, vshape) then
        value_nd = value_nd:broadcast_to(vshape)
    end
    return value_nd
end

local function is_empty_table(t)
    return type(t) == 'table' and #t == 0
end

function NDArray:_set_nd_basic_indexing(key, value)
    local shape = self.shape
    if type(key) == 'number' then
        if key < 0 then
            key = key + shape[1]
        end
        if key < 0 or key > shape[1] then
            raise('IndexError', 'index is out of bounds for axis 0')
        end
        key = { { key, key + 1 } }
    end
    -- [:] -> {} / {{}}
    if is_empty_table(key) or (#key == 1 and is_empty_table(key[1])) then
        -- assign value to self
        if isinstance(value, NDArray) then
            if value.handle ~= self.handle then
                if not table.equal(value.shape, shape) then
                    value = value:broadcast_to(shape)
                end
                value:copyto(self)
            end
        elseif type(value) == 'number' then
            _internal._full(shape, self.context, self.dtype, value, self)
        else
            -- value might be a table or cdata
            local value_nd = self:_prepare_value_nd(value, shape)
            value_nd:copyto(self)
        end
        return
    end
    local begin, end_, steps, oshape, vshape = {}, {}, {}, {}, {}
    local count = table.maxn(key)
    for i = 1, count do
        local slice_i = key[i]
        local dim_size = 1
        if type(slice_i) == 'table' or isnone(slice_i) then
            if isnone(slice_i) then
                slice_i = {}
            end
            if slice_i[3] == 0 then
                raise('ValueError', 'slice step cannot be zero')
            end
            local s1, s2, s3 = default_none(unpack(slice_i, 1, 3))
            table.insert(begin, s1)
            table.insert(end_, s2)
            table.insert(steps, s3)
            local start, stop, step = M._get_index_range(
                    s1, s2, shape[i], s3)
            dim_size = M._get_dim_size(start, stop, step)
            table.insert(vshape, dim_size)
        elseif type(slice_i) == 'number' then
            table.insert(begin, slice_i)
            table.insert(end_, slice_i ~= -1 and slice_i + 1 or self.shape[i])
            table.insert(steps, 1)
        else
            raise('ValueError', 'basic indexing not support')
        end
        table.insert(oshape, dim_size)
    end
    for i = count + 1, #shape do
        table.insert(oshape, shape[i])
        table.insert(vshape, shape[i])
    end
    -- if key contains all integers, vshape should be (1,)
    if #vshape == 0 then
        vshape = { 1 }
    end

    if type(value) == 'number' then
        _internal._slice_assign_scalar(self, value, begin, end_, steps, self)
    else
        local value_nd = self:_prepare_value_nd(value, vshape)
        if not table.equal(vshape, oshape) then
            value_nd = value_nd:reshape(oshape)
        end
        _internal._slice_assign(self, value_nd, begin, end_, steps, self)
    end
end

function NDArray:_set_nd_advanced_indexing(key, value)
    raise('NotImplementedError')
end

function NDArray:_get_nd_basic_indexing(key)
    local shape = self.shape
    if type(key) == 'number' then
        if key > shape[1] - 1 then
            raise('IndexError', 'index is out of bounds for axis 0')
        end
        return self:_at(key)
    elseif #key == 0 then
        return self
    elseif #key == 1 then
        local k = key[1]
        if type(k) == 'number' then
            if k > shape[1] - 1 then
                raise('IndexError', 'index is out of bounds for axis 0')
            end
            return self:_at(k)
        else
            local start, stop, step = default_none(k[1], k[2], k[3])
            if step and step ~= 1 then
                if step == 0 then
                    raise('ValueError', 'slice step cannot be zero')
                end
                return op.slice(self, { start }, { stop }, { step })
            elseif start or stop then
                return self:_slice(start, stop)
            else
                return self
            end
        end
    end
    local begin, end_, step, kept_axes = {}, {}, {}, {}
    local count = table.maxn(key)
    for i = 1, count do
        local slice_i = key[i]
        if type(slice_i) == 'number' then
            table.insert(begin, slice_i)
            table.insert(end_, slice_i ~= -1 and slice_i + 1 or self.shape[i])
            table.insert(step, 1)
        elseif type(slice_i) == 'table' or isnone(slice_i) then
            if isnone(slice_i) then
                slice_i = {}
            end
            if slice_i[3] == 0 then
                raise('ValueError', 'slice step cannot be zero')
            end
            local s1, s2, s3 = default_none(unpack(slice_i, 1, 3))
            table.insert(begin, s1)
            table.insert(end_, s2)
            table.insert(step, s3)
            table.insert(kept_axes, i)
        else
            raise('ValueError', 'basic_indexing not support')
        end
    end
    for i = count + 1, #shape do
        table.insert(kept_axes, i)
    end
    local sliced_nd = op.slice(self, begin, end_, step)
    if table.equal(kept_axes, shape) then
        return sliced_nd
    end
    -- squeeze sliced_shape to remove the axes indexed by integers
    local oshape = {}
    local sliced_shape = sliced_nd.shape
    for _, axis in ipairs(kept_axes) do
        table.insert(oshape, sliced_shape[axis])
    end
    -- if key is a tuple of integers, still need to keep 1 dim
    -- while in Numpy, the output will become an value instead of an ndarray
    if #oshape == 0 then
        oshape = { 1 }
    end
    assert(table.reduce(oshape, 'mul') == table.reduce(sliced_shape, 'mul'),
           ('out shape %s has different size than sliced shape %s'):format(
                   base.tostring(oshape), base.tostring(sliced_shape)))
    return sliced_nd:reshape(oshape)
end

function NDArray:_get_nd_advanced_indexing(key)
    raise('NotImplementedError')
end

--- Performs a synchronized copy from the `source_array` to the current array.
--- This is called through ``x[:] = source_array``, where the `source_array`
--- is a cdata or array-like table.
--- This function blocks until all the pending read/write operations with respect
--- to the current `NDArray` are finished and carry out the copy operation to the
--- current NDArray.
function NDArray:_sync_copyfrom(source_array)
    if type(source_array) == 'table' then
        local shape
        source_array, shape = M._table_to_cdata(source_array, self.dtype)
        if not table.equal(shape, self.shape) then
            raise('ValueError', ('Shape inconsistent: expected %s vs got %s'):format(
                    table.concat(shape, ', '), table.concat(self.shape, ', ')))
        end
    end
    assert(type(source_array) == 'cdata')
    assert(ffi.sizeof(source_array) == ffi.sizeof(_DTYPE_TO_CTYPE[self.dtype]) * self.size)
    check_call(_LIB.MXNDArraySyncCopyFromCPU(
            self.handle,
            ffi.cast('void*', source_array),
            ctypes.c_size_t(self.size)))
end

--- Returns a sliced NDArray that shares memory with the current one.
--- This is called through ``x[start:stop]``.
function NDArray:_slice(start, stop)
    local handle = NDArrayHandle()
    start, stop = M._get_index_range(start, stop, self.shape[0])
    check_call(_LIB.MXNDArraySlice(
            self.handle, mx_uint(start), mx_uint(stop), ctypes.byref(handle)))
    return NDArray(handle, self.writable)
end

--- Returns a view of the array sliced at `idx` in the first dim.
--- This is called through ``x[idx]``.
function NDArray:_at(idx)
    local handle = NDArrayHandle()
    if idx < 0 then
        local length = self.shape[1]
        idx = idx + length
        if idx < 0 then
            raise('IndexError', 'index is out of bounds for axis 0')
        end
    end
    check_call(_LIB.MXNDArrayAt(
            self.handle, mx_uint(idx), ctypes.byref(handle)))
    return NDArray(handle, self.writable)
end

--- Returns a **view** of this array with a new shape without altering any data.
function NDArray:reshape(...)
    local shape, kwargs = arg_kw(...)
    if #shape == 1 and type(shape[1]) == 'table' then
        shape = shape[1]
    elseif #shape == 0 then
        shape = kwargs['shape']
        assert(shape, 'Shape must be provided.')
    end
    local reverse = default(kwargs['reverse'], false)
    local handle = NDArrayHandle()

    -- Actual reshape
    check_call(_LIB.MXNDArrayReshape64(self.handle,
                                       len(shape),
                                       c_array(ctypes.c_int64, shape),
                                       reverse,
                                       ctypes.byref(handle)))
    return NDArray(handle, self.writable)
end

--

local function __register()
    NDArray.reshape_like = op.reshape_like
    NDArray.zeros_like = op.zeros_like
    NDArray.ones_like = op.ones_like
    NDArray.broadcast_axes = op.broadcast_axes
    NDArray.repeat_ = op.repeat_
    NDArray.pad = op.pad
    NDArray.swapaxes = op.swapaxes
    NDArray.split = op.split
    NDArray.split_v2 = M.split_v2
    NDArray.slice = op.slice
    NDArray.slice_axis = op.slice_axis
    NDArray.slice_like = op.slice_like
    NDArray.take = op.take
    NDArray.one_hot = op.one_hot
    NDArray.pick = op.pick
    NDArray.sort = op.sort
    NDArray.topk = op.topk
    NDArray.argsort = op.argsort
    NDArray.argmax = op.argmax
    NDArray.argmax_channel = op.argmax_channel
    NDArray.argmin = op.argmin
    NDArray.clip = op.clip
    NDArray.abs = op.abs
    NDArray.sign = op.sign
    NDArray.flatten = op.flatten
    NDArray.shape_array = op.shape_array
    NDArray.size_array = op.size_array
    NDArray.expand_dims = op.expand_dims
    NDArray.tile = op.tile
    NDArray.transpose = op.transpose
    NDArray.flip = op.flip
    NDArray.depth_to_space = op.depth_to_space
    NDArray.space_to_depth = op.space_to_depth

    NDArray.sum = op.sum
    NDArray.nansum = op.nansum
    NDArray.prod = op.prod
    NDArray.nanprod = op.nanprod
    NDArray.mean = op.mean
    NDArray.max = op.max
    NDArray.min = op.min
    NDArray.norm = op.norm
    NDArray.round = op.round
    NDArray.rint = op.rint
    NDArray.fix = op.fix
    NDArray.floor = op.floor
    NDArray.ceil = op.ceil
    NDArray.trunc = op.trunc
    NDArray.sin = op.sin
    NDArray.cos = op.cos
    NDArray.tan = op.tan
    NDArray.arcsin = op.arcsin
    NDArray.arccos = op.arccos
    NDArray.arctan = op.arctan
    NDArray.degrees = op.degrees
    NDArray.radians = op.radians
    NDArray.sinh = op.sinh
    NDArray.cosh = op.cosh
    NDArray.tanh = op.tanh
    NDArray.arcsinh = op.arcsinh
    NDArray.arccosh = op.arccosh
    NDArray.arctanh = op.arctanh
    NDArray.exp = op.exp
    NDArray.expm1 = op.expm1
    NDArray.log = op.log
    NDArray.log10 = op.log10
    NDArray.log2 = op.log2
    NDArray.log1p = op.log1p
    NDArray.sqrt = op.sqrt
    NDArray.rsqrt = op.rsqrt
    NDArray.cbrt = op.cbrt
    NDArray.rcbrt = op.rcbrt
    NDArray.square = op.square
    NDArray.reciprocal = op.reciprocal
    NDArray.relu = op.relu
    NDArray.sigmoid = op.sigmoid
    NDArray.softmax = op.softmax
    NDArray.log_softmax = op.log_softmax
    NDArray.softmin = op.softmin
    NDArray.squeeze = op.squeeze
end

function NDArray:diag(k, kwargs)
    return op.diag(self, default(k, 0), kwargs)
end

NDArray.__register = __register

--

--- Broadcasts the input array to a new shape.
---
--- Broadcasting is only allowed on axes with size 1. The new shape cannot change
--- the number of dimensions.
--- For example, you could broadcast from shape (2, 1) to (2, 3), but not from
--- shape (2, 3) to (2, 3, 3).
function NDArray:broadcast_to(shape)
    local cur_shape = self.shape
    local err_str = string.format(
            'operands could not be broadcast together with remapped shapes [original->remapped]: %s and requested shape %s',
            table.concat(cur_shape, ', '), table.concat(shape, ', '))
    if #shape < #cur_shape then
        raise('ValueError', err_str)
    end
    local cur_shape_ = {}
    for i = 1, #shape - #cur_shape do
        table.insert(cur_shape_, 1)
    end
    table.append(cur_shape_, cur_shape)
    cur_shape = cur_shape_
    local broadcasting_axes = {}
    for i = 1, #shape do
        if cur_shape[i] ~= shape[i] then
            table.insert(broadcasting_axes, i)
        end
    end
    for _, v in ipairs(broadcasting_axes) do
        if cur_shape[v] ~= 1 then
            raise('ValueError', err_str)
        end
    end
    if not table.equal(cur_shape, self.shape) then
        return op.broadcast_to(self:reshape(cur_shape), shape)
    else
        return op.broadcast_to(self, shape)
    end
end

--- Broadcasts the input array to the shape of other.
function NDArray:broadcast_like(other)
    return self:broadcast_to(other.shape)
end

--- Waits until all previous write operations on the current array are finished.
---
--- This method guarantees that all previous write operations that pushed
--- into the backend engine for execution are actually finished.
function NDArray:wait_to_read()
    check_call(_LIB.MXNDArrayWaitToRead(self.handle))
end

--- Returns the number of dimensions of this array
function NDArray:_ndim()
    return #self.shape
end

--- Tuple of array dimensions.
function NDArray:_shape()
    local ndim = mx_int()
    local pdata = ctypes.POINTER(mx_int)()
    check_call(_LIB.MXNDArrayGetShapeEx(
            self.handle, ctypes.byref(ndim), ctypes.byref_const(pdata)))
    if ndim.value == -1 then
        return None
    else
        local ret = {}
        for i = 0, ndim.value - 1 do
            table.insert(ret, pdata[i])
        end
        return ret
    end
end

--- Number of elements in the array.
--- Equivalent to the product of the array's dimensions.
function NDArray:_size()
    local size = 1
    for _, v in ipairs(self.shape) do
        size = size * v
    end
    return size
end

--- Device context of the array.
function NDArray:_context()
    local dev_typeid = ctypes.c_int()
    local dev_id = ctypes.c_int()
    check_call(_LIB.MXNDArrayGetContext(
            self.handle, ctypes.byref(dev_typeid), ctypes.byref(dev_id)))
    return Context(Context.devtype2str[dev_typeid.value], dev_id.value)
end

--- Data-type of the array's elements.
function NDArray:_dtype()
    local mx_dtype = ctypes.c_int()
    check_call(_LIB.MXNDArrayGetDType(
            self.handle, ctypes.byref(mx_dtype)))
    return M._DTYPE_MX_TO_NP[mx_dtype.value]
end

--- Storage-type of the array.
function NDArray:_stype()
    return M._STORAGE_TYPE_ID_TO_STR[M._storage_type(self.handle)]
end

--- Returns a copy of the array with axes transposed.
---
--- Equivalent to ``mx.nd.transpose(self)`` except that
--- self is returned if ``self.ndim < 2``.
---
--- Unlike ``numpy.ndarray.T``, this function returns a copy
--- rather than a view of the array unless ``self.ndim < 2``.
function NDArray:_T()
    if #self.shape < 2 then
        return self
    end
    return op.transpose(self)
end

--- Whether this array's corresponding gradient array
--- (registered via `autograd.mark_variables`) has been
--- updated by `autograd.backward` since last reset.
---
--- `_fresh_grad` need to be manually set to False
--- after consuming gradient (usually after updating this
--- array).
function NDArray:__fresh_grad()
    local out = ctypes.c_int()
    check_call(_LIB.MXNDArrayGetGradState(self.handle, ctypes.byref(out)))
    return out.value
end

function NDArray:__fresh_grad_set(state)
    check_call(_LIB.MXNDArraySetGradState(self.handle, ctypes.c_int(state)))
end

function NDArray:asnumpy()
    return self:ascdata()
end

--- Returns a cdata object with value copied from this array.
---@return ffi.cdata
function NDArray:ascdata()
    local data = ffi.new(string.format('%s[%d]', _DTYPE_TO_CTYPE[self.dtype], self.size))
    check_call(_LIB.MXNDArraySyncCopyToCPU(
            self.handle,
            ffi.cast('void*', data),
            ctypes.c_size_t(self.size)))
    return data
end

function NDArray:astable()
    local sz = self.size
    if sz > 65536 or self.ndim >= 3 then
        raise('ValueError', 'NDArray is too big')
    end
    local cdata = self:ascdata()
    if self.ndim == 1 then
        local ret = {}
        for i = 0, sz - 1 do
            table.insert(ret, tonumber(cdata[i]))
        end
        return ret
    else
        local shape = self.shape
        local s1, s2 = shape[1], shape[2]
        local ret = {}
        for i = 0, s1 - 1 do
            local a = {}
            for j = 0, s2 - 1 do
                table.insert(a, tonumber(cdata[i * s2 + j]))
            end
            table.insert(ret, a)
        end
        return ret
    end
end

--- Returns a scalar whose value is copied from this array.
---
--- This function is equivalent to ``self.asnumpy()[0]``. This NDArray must have shape (1,).
function NDArray:asscalar()
    if not table.equal(self.shape, { 1 }) then
        raise('ValueError', 'The current array is not a scalar')
    end
    return self.asnumpy()[0]
end

--- Returns a copy of the array after casting to a specified type.
function NDArray:astype(dtype, copy)
    copy = default(copy, true)
    dtype = check_dtype(dtype)
    if not copy and dtype == self.dtype then
        return self
    end
    local ret = M.empty(self.shape, self.context, dtype)
    self:copyto(ret)
    return ret
end

--- Copies the value of this array to another array.
---
--- If ``other`` is a ``NDArray`` object, then ``other.shape`` and
--- ``self.shape`` should be the same. This function copies the value from
--- ``self`` to ``other``.
---
--- If ``other`` is a context, a new ``NDArray`` will be first created on
--- the target context, and the value of ``self`` is copied.
function NDArray:copyto(other)
    if isinstance(other, NDArray) then
        if other.handle == self.handle then
            -- warning: You are attempting to copy an array to itself
            return false
        end
        return _internal._copyto(self, other)
    elseif isinstance(other, Context) then
        local hret = NDArray(M._new_alloc_handle(self.shape, other, true, self.dtype))
        return _internal._copyto(self, hret)
    else
        raise('TypeError', 'copyto not support')
    end
end

--- Makes a copy of this ``NDArray``, keeping the same context.
function NDArray:copy()
    return self:copyto(self.context)
end

--- Returns an array on the target device with the same value as this array.
---
--- If the target context is the same as ``self.context``, then ``self`` is
--- returned.  Otherwise, a copy is made.
function NDArray:as_in_context(context_)
    if self.context == context_ then
        return self
    end
    return self:copyto(context_)
end

--- Attach a gradient buffer to this NDArray, so that `backward`
--- can compute gradient with respect to it.
function NDArray:attach_grad(grad_req, stype)
    grad_req, stype = default(grad_req, 'write', stype, None)
    local _zeros = require('mx.ndarray.__init__').zeros
    local grad
    if not isnone(stype) then
        grad = _zeros(self.shape, None, None, stype)
    else
        grad = op.zeros_like(self)
    end
    grad_req = M._GRAD_REQ_MAP[grad_req]
    check_call(_LIB.MXAutogradMarkVariables(
            1, ctypes.pointer(self.handle),
            ctypes.pointer(mx_uint(grad_req)),
            ctypes.pointer(grad.handle)))
end

--- Returns gradient buffer attached to this NDArray.
function NDArray:_grad()
    local _ndarray_cls = require('mx.ndarray.__init__')._ndarray_cls
    local hdl = NDArrayHandle()
    check_call(_LIB.MXNDArrayGetGrad(self.handle, ctypes.byref(hdl)))
    if hdl.value == 0 then
        return None
    end
    return _ndarray_cls(hdl)
end

--- Returns a new NDArray, detached from the current graph.
function NDArray:detach()
    local _ndarray_cls = require('mx.ndarray.__init__')._ndarray_cls
    local hdl = NDArrayHandle()
    check_call(_LIB.MXNDArrayDetach(self.handle, ctypes.byref(hdl)))
    return _ndarray_cls(hdl)
end

--- Compute the gradients of this NDArray w.r.t variables.
function NDArray:backward(out_grad, retain_graph, train_mode)
    out_grad, retain_graph, train_mode = default(
            out_grad, None, retain_graph, false, train_mode, true)
    local ograd_handles
    if isnone(out_grad) then
        ograd_handles = { NDArrayHandle(0) }
    else
        ograd_handles = { out_grad.handle }
    end
    check_call(_LIB.MXAutogradBackwardEx(
            1, c_handle_array({ self }),
            c_array(NDArrayHandle, ograd_handles),
            0,
            ctypes.c_void_p(0),
            ctypes.c_int(retain_graph),
            ctypes.c_int(0),
            ctypes.c_int(train_mode),
            ctypes.c_void_p(0),
            ctypes.c_void_p(0)))
end

--- Return a copy of the array with chosen storage type.
function NDArray:tostype(stype)
    return op.cast_storage(self, stype)
end

--- Returns a reference view of NDArray that represents as DLManagedTensor until
--- all previous write operations on the current array are finished.
function NDArray:to_dlpack_for_read()
    return M.to_dlpack_for_read(self)
end

--- Returns a reference view of NDArray that represents as DLManagedTensor until
--- all previous read/write operations on the current array are finished.
function NDArray:to_dlpack_for_write()
    return M.to_dlpack_for_write(self)
end

--

local _NDArray_field = {
    ndim        = NDArray._ndim,
    shape       = NDArray._shape,
    size        = NDArray._size,
    context     = NDArray._context,
    dtype       = NDArray._dtype,
    stype       = NDArray._stype,
    T           = NDArray._T,
    _fresh_grad = NDArray.__fresh_grad,
    grad        = NDArray._grad,
}

function NDArray:__index(k)
    local tk = type(k)
    if tk == 'string' then
        if _NDArray_field[k] then
            return _NDArray_field[k](self)
        elseif not isnone(NDArray[k]) then
            return NDArray[k]
        else
            return rawget(self, k)
        end
    else
        return self:__getitem(k)
    end
end

function NDArray:__newindex(k, v)
    local tk = type(k)
    if tk == 'string' then
        rawset(self, k, v)
    else
        self:__setitem(k, v)
    end
end

--

--TODO
--- Returns a dispatch code for calling basic or advanced indexing functions.
function M._get_indexing_dispatch_code(key)
    if isinstance(key, NDArray) then
        return M._NDARRAY_ADVANCED_INDEXING
    elseif type(key) == 'table' or type(key) == 'number' then
        return M._NDARRAY_BASIC_INDEXING
    else
        return M._NDARRAY_UNSUPPORTED_INDEXING
    end
end

--- Given start, stop, step and array length, return
--- absolute values of start, stop, and step for generating index range.
--- The returned values have been compensated by adding length if they
--- are less than zero for all the cases but slice(None, None, -1).
--- Note that the returned value of stop is not necessarily >= 0, since
--- absolute stop is -1 in the case of slice(None, None, -1).
function M._get_index_range(start, stop, length, step)
    step = default(step, 1)
    if step == 0 then
        raise('ValueError', 'step size cannot be zero')
    end
    if length < 0 then
        raise('ValueError', 'array length cannot be less than zero')
    end
    if isnone(start) then
        if step > 0 then
            start = 0
        else
            start = length - 1
        end
    elseif start < 0 then
        start = start + length
        if start < 0 then
            raise('IndexError', 'Slicing start exceeds limit')
        end
    elseif start >= length then
        raise('IndexError', 'Slicing start exceeds limit')
    end
    if isnone(stop) then
        if step > 0 then
            stop = length
        else
            -- this supports case such as ::-1
            -- stop = -1 here refers to the element before index 0,
            -- instead of the last element in the array
            stop = -1
        end
    elseif stop < 0 then
        stop = stop + length
        if stop < 0 then
            raise('IndexError', 'Slicing stop exceeds limit')
        end
    elseif stop > length then
        raise('IndexError', 'Slicing stop exceeds limit')
    end
    return start, stop, step
end

--- Given data and index shapes, get the output `NDArray` shape.
--- This basically implements the infer shape logic of op gather_nd.
function M._get_oshape_of_gather_nd_op(dshape, ishape)
    assert(#dshape > 0 and #ishape > 0)
    local oshape = {}
    for i = 2, #ishape do
        table.insert(oshape, ishape[i])
    end
    if ishape[1] < #dshape then
        for i = ishape[1], #dshape do
            table.insert(oshape, dshape[i])
        end
    end
    return oshape
end

--- Given start, stop, and stop, calculate the number of elements
--- of this slice.
function M._get_dim_size(start, stop, step)
    assert(step ~= 0)
    local dim_size
    if step > 0 then
        assert(start < stop)
        dim_size = math.floor((stop - start - 1) / step) + 1
    else
        assert(stop < start)
        dim_size = math.floor((start - stop - 1) / (-step)) + 1
    end
    return dim_size
end

--- Given two shapes that are not identical, find the shape
--- that both input shapes can broadcast to.
function M._get_broadcast_shape(shape1, shape2)
    if table.equal(shape1, shape2) then
        return shape1
    end
    local shape
    local length1, length2 = #shape1, #shape2
    if length1 > length2 then
        shape = table.clone(shape1)
    else
        shape = table.clone(shape2)
    end
    local i = math.max(length1, length2)
    for a, b in zipairs(table.reverse(shape1), table.reverse(shape2)) do
        if a ~= 1 and b ~= 1 and a ~= b then
            raise('ValueError', 'shape is not broadcastable')
        end
        shape[i] = math.max(a, b)
        i = i - 1
    end
    return shape
end

--- One-hot encoding indices into matrix out.
--- note: `onehot_encode` is deprecated. Use `one_hot` instead.
--function M.onehot_encode(indices, out)
--return _internal._onehot_encode(indices, out, out)
--end

--- Returns a new array filled with all ones, with the given shape and type.
function M.ones(shape, ctx, dtype, kwargs)
    if isnone(ctx) then
        ctx = current_context()
    end
    dtype = check_dtype(default(dtype, 'float32'))
    return _internal._ones(shape, ctx, dtype, None, None, kwargs)
end

--- Returns a new array of given shape and type, filled with the given value `val`.
function M.full(shape, val, ctx, dtype, out)
    if isnone(out) then
        out = M.empty(shape, ctx, dtype)
    end
    out[{}] = val
    return out
end

local function _table_next_index(state, idx, shape)
    local n = #state
    for i = #shape - 1, 1, -1 do
        local limit = shape[i]
        local cur = idx[i]
        if cur + 1 <= limit then
            idx[i] = cur + 1
            state[i + 1] = state[i][cur + 1]
            break
        else
            idx[i] = 1
            state[i + 1] = None
        end
    end
    if #state < n then
        for i = #state + 1, n do
            state[i] = state[i - 1][1]
        end
    end
end

local function _table_shape(t, csize)
    local shape = {}
    local ty = type(t)
    if ty == 'number' then
        return
    elseif ty == 'table' then
        table.insert(shape, #t)
        local s = _table_shape(t[1], csize)
        if not s then
            return shape
        end
        for i = 2, #t do
            if not table.equal(s, _table_shape(t[i], csize)) then
                error('input is not an array like table')
            end
        end
        for _, v in ipairs(s) do
            table.insert(shape, v)
        end
        return shape
    elseif ty == 'cdata' then
        table.insert(shape, ffi.sizeof(t) / csize)
        return shape
    else
        error('input is not an array like table')
    end
end

local function _table_to_cdata(t, ctype)
    assert(type(t) == 'table' and type(ctype) == 'string')
    if #t == 0 then
        return ffi.new(('%s[0]'):format(ctype)), { 0 }
    end
    local shape = _table_shape(t, ffi.sizeof(ctype))
    local idx = { 1 }
    local sz = 1
    for _, v in ipairs(shape) do
        sz = sz * v
        table.insert(idx, 1)
    end
    local cdata = ffi.new(('%s[%d]'):format(ctype, sz))
    if #shape == 1 then
        -- number array
        for i = 1, shape[1] do
            cdata[i - 1] = t[i]
        end
    else
        local cur = { t }
        local last = cur[#cur]
        local shape_last = shape[#shape]
        local index = 0
        for i = 1, #shape - 1 do
            table.insert(cur, last[idx[i]])
            last = cur[#cur]
        end
        while true do
            last = cur[#cur]
            local last_t = type(last)
            for i = 1, shape_last do
                if index >= sz then
                    error('input is not an array like table')
                end
                local val = last_t == 'cdata' and last[i - 1] or last[i]
                assert(val, 'input is not an array like table')
                cdata[index] = val
                index = index + 1
            end
            if index >= sz then
                break
            end
            _table_next_index(cur, idx, shape)
        end
    end
    return cdata, shape
end
M._table_to_cdata = _table_to_cdata

--- Creates an array from any object exposing the array interface.
function M.array(source_array, ctx, dtype)
    local shape, is_cdata
    if isinstance(source_array, NDArray) then
        dtype = source_array.dtype
        dtype = check_dtype(default(dtype, source_array.dtype))
        shape = source_array.shape
    else
        -- table or cdata
        dtype = check_dtype(default(dtype, 'float32'))
        local ctype = _DTYPE_TO_CTYPE[dtype]
        if type(source_array) == 'table' then
            source_array, shape = _table_to_cdata(source_array, ctype)
        end
        assert(type(source_array) == 'cdata')
        shape = shape or { ffi.sizeof(source_array) / ffi.sizeof(ctype) }
        is_cdata = true
    end
    local arr = M.empty(shape, ctx, dtype)
    if is_cdata then
        arr:_sync_copyfrom(source_array)
    else
        arr[{}] = source_array
    end
    return arr
end

local function normalize_axis_index(axis, ndim)
    local axis_ = axis
    if axis < 0 then
        axis = axis + ndim
    end
    if axis < 0 or axis >= ndim then
        raise('AxisError', ('axis %d is out of bounds for array of dimension %d'):format(axis_, ndim))
    end
    return axis
end
local function normalize_axis_tuple(axis, ndim)
    if type(axis) == 'number' then
        axis = { axis }
    end
    local ret = {}
    for _, ax in ipairs(axis) do
        table.insert(ret, normalize_axis_index(ax, ndim))
    end
    return ret
end

---
function M.moveaxis(tensor, source, destination)
    source = normalize_axis_tuple(source, tensor.ndim)
    destination = normalize_axis_tuple(destination, tensor.ndim)
    if #source ~= #destination then
        raise('ValueError', '`source` and `destination` arguments must have the same number of elements')
    end
    local order = {}
    for i = 0, tensor.ndim - 1 do
        if not table.has_value(source, i) then
            table.insert(order, i)
        end
    end
    local t = table.zip(destination, source)
    table.sort(t, function(a, b)
        return a[1] < b[1]
    end)
    for _, v in ipairs(t) do
        table.insert(order, v[1] + 1, v[2])
    end
    return op.transpose(tensor, order)
end

--- Returns evenly spaced values within a given interval.
---
--- Values are generated within the half-open interval [`start`, `stop`). In other
--- words, the interval includes `start` but excludes `stop`. The function is
--- similar to the built-in Python function `range` and to `numpy.arange`,
--- but returns an `NDArray`.
function M.arange(start, stop, step, repeat_, infer_range, ctx, dtype)
    -- infer_range is deprecated
    step, repeat_ = default(step, 1, repeat_, 1)
    if isnone(ctx) then
        ctx = current_context()
    end
    return _internal._arange(start, stop, step, repeat_, false, tostring(ctx), dtype)
end

--- Return evenly spaced numbers within a specified interval.
---
--- Values are generated within the half-open interval [`start`, `stop`) or
--- closed interval [start, stop] depending on whether `endpoint` is True or
--- False. The function is similar to `numpy.linspace`, but returns an `NDArray`.
function M.linspace(start, stop, num, endpoint, ctx, dtype)
    endpoint = default(endpoint, true)
    dtype = check_dtype(default(dtype, 'float32'))
    if isnone(ctx) then
        ctx = current_context()
    end
    return _internal._linspace(start, stop, None, None, None, endpoint, tostring(ctx), dtype, None, None, { num = num })
end

--- Helper function for element-wise operation.
--- The function will perform numpy-like broadcasting if needed and call different functions.
---@return mx.ndarray.NDArray|number
function M._ufunc_helper(lhs, rhs, fn_array, fn_scalar, lfn_scalar, rfn_scalar)
    if type(lhs) == 'number' then
        if type(rhs) == 'number' then
            return fn_scalar(lhs, rhs)
        else
            if isnone(rfn_scalar) then
                -- commutative function
                return lfn_scalar(rhs, lhs)
            else
                return rfn_scalar(rhs, lhs)
            end
        end
    elseif type(rhs) == 'number' then
        return lfn_scalar(lhs, rhs)
    elseif isinstance(rhs, NDArray) then
        return fn_array(lhs, rhs)
    else
        raise('TypeError', 'type not supported')
    end
end
local _ufunc_helper = M._ufunc_helper

local operator = {}
setmetatable(operator, { __index = function(t, k)
    operator = require('mx.operator')
    return operator[k]
end })

--- Returns element-wise sum of the input arrays with broadcasting.
---
--- Equivalent to ``lhs + rhs``, ``mx.nd.broadcast_add(lhs, rhs)`` and
--- ``mx.nd.broadcast_plus(lhs, rhs)``.
---
--- .. note::
---
---    If the corresponding dimensions of two arrays have the same size or one of them has size 1,
---    then the arrays are broadcastable to a common shape
function M.add(lhs, rhs)
    return _ufunc_helper(
            lhs,
            rhs,
            op.broadcast_add,
            operator.add,
            _internal._plus_scalar,
            nil)
end

---
function M.subtract(lhs, rhs)
    return _ufunc_helper(
            lhs,
            rhs,
            op.broadcast_sub,
            operator.sub,
            _internal._minus_scalar,
            _internal._rminus_scalar)
end

---
function M.multiply(lhs, rhs)
    return _ufunc_helper(
            lhs,
            rhs,
            op.broadcast_mul,
            operator.mul,
            _internal._mul_scalar,
            nil)
end

---
function M.divide(lhs, rhs)
    return _ufunc_helper(
            lhs,
            rhs,
            op.broadcast_div,
            operator.truediv,
            _internal._div_scalar,
            _internal._rdiv_scalar)
end

---
function M.modulo(lhs, rhs)
    return _ufunc_helper(
            lhs,
            rhs,
            op.broadcast_mod,
            operator.mod,
            _internal._mod_scalar,
            _internal._rmod_scalar)
end

---
function M.power(base, exp)
    return _ufunc_helper(
            base,
            exp,
            op.broadcast_power,
            operator.pow,
            _internal._power_scalar,
            _internal._rpower_scalar)
end

---
function M.maximum(lhs, rhs)
    return _ufunc_helper(
            lhs,
            rhs,
            op.broadcast_maximum,
            function(x, y)
                return x > y and x or y
            end,
            _internal._maximum_scalar,
            nil)
end

---
function M.minimum(lhs, rhs)
    return _ufunc_helper(
            lhs,
            rhs,
            op.broadcast_minimum,
            function(x, y)
                return x < y and x or y
            end,
            _internal._minimum_scalar,
            nil)
end

---
function M.equal(lhs, rhs)
    return _ufunc_helper(
            lhs,
            rhs,
            op.broadcast_equal,
            function(x, y)
                return x == y and 1 or 0
            end,
            _internal._equal_scalar,
            nil)
end

---
function M.not_equal(lhs, rhs)
    return _ufunc_helper(
            lhs,
            rhs,
            op.broadcast_not_equal,
            function(x, y)
                return x ~= y and 1 or 0
            end,
            _internal._not_equal_scalar,
            nil)
end

---
function M.greater(lhs, rhs)
    return _ufunc_helper(
            lhs,
            rhs,
            op.broadcast_greater,
            function(x, y)
                return x > y and 1 or 0
            end,
            _internal._greater_scalar,
            _internal._lesser_scalar)
end

---
function M.greater_equal(lhs, rhs)
    return _ufunc_helper(
            lhs,
            rhs,
            op.broadcast_greater_equal,
            function(x, y)
                return x >= y and 1 or 0
            end,
            _internal._greater_equal_scalar,
            _internal._lesser_equal_scalar)
end

---
function M.lesser(lhs, rhs)
    return _ufunc_helper(
            lhs,
            rhs,
            op.broadcast_lesser,
            function(x, y)
                return x < y and 1 or 0
            end,
            _internal._lesser_scalar,
            _internal._greater_scalar)
end

---
function M.lesser_equal(lhs, rhs)
    return _ufunc_helper(
            lhs,
            rhs,
            op.broadcast_lesser_equal,
            function(x, y)
                return x <= y and 1 or 0
            end,
            _internal._lesser_equal_scalar,
            _internal._greater_equal_scalar)
end

---
function M.logical_and(lhs, rhs)
    return _ufunc_helper(
            lhs,
            rhs,
            op.broadcast_logical_and,
            function(x, y)
                x, y = bool(x), bool(y)
                return (x and y) and 1 or 0
            end,
            _internal._logical_and_scalar,
            nil)
end

---
function M.logical_or(lhs, rhs)
    return _ufunc_helper(
            lhs,
            rhs,
            op.broadcast_logical_or,
            function(x, y)
                x, y = bool(x), bool(y)
                return (x or y) and 1 or 0
            end,
            _internal._logical_or_scalar,
            nil)
end

---
function M.logical_xor(lhs, rhs)
    return _ufunc_helper(
            lhs,
            rhs,
            op.broadcast_logical_xor,
            function(x, y)
                x, y = bool(x), bool(y)
                return ((x or y) and (not x or not y)) and 1 or 0
            end,
            _internal._logical_xor_scalar,
            nil)
end

---
function M.true_divide(lhs, rhs)
    return M.divide(lhs, rhs)
end

--- DEPRECATED, use ``concat`` instead
--function M.concatenate(arrays, axis, always_copy)
--end

--- DEPRECATED, use mx.img instead
--function M.imdecode(str_img, clip_rect, out, index, channels, mean)
--end

--- Returns a new array filled with all zeros, with the given shape and type.
function M.zeros(shape, ctx, dtype, kwargs)
    if isnone(ctx) then
        ctx = current_context()
    end
    dtype = check_dtype(default(dtype, 'float32'))
    return _internal._zeros(shape, ctx, dtype, None, None, kwargs)
end

--- Return a 2-D array with ones on the diagonal and zeros elsewhere.
function M.eye(N, M_, k, ctx, dtype, kwargs)
    if isnone(ctx) then
        ctx = current_context()
    end
    dtype = check_dtype(default(dtype, 'float32'))
    return _internal._eye(N, M_, k, ctx, dtype, None, None, kwargs)
end

--- Returns a new array of given shape and type, without initializing entries.
function M.empty(shape, ctx, dtype)
    if type(shape) == 'number' then
        shape = { shape }
    end
    if isnone(ctx) then
        ctx = current_context()
    end
    dtype = check_dtype(default(dtype, 'float32'))
    return NDArray(M._new_alloc_handle(shape, ctx, false, dtype))
end

--- Compute the histogram of the input data.
function M.histogram(a, bins, range)
    if isinstance(bins, NDArray) then
        return _internal._histogram(a, bins)
    elseif type(bins) == 'number' then
        assert(range)
        return _internal._histogram(a, None, bins, range)
    end
    raise('ValueError', 'bins argument should be either an integer or an NDArray')
end

---
function M.split_v2(ary, indices_or_sections, axis, squeeze_axis)
    axis, squeeze_axis = default(axis, 0, squeeze_axis, false)
    local indices = {}
    local axis_size = ary.shape[axis]
    if type(indices_or_sections) == 'number' then
        local sections = indices_or_sections
        if axis_size % sections ~= 0 then
            raise('ValueError', 'array split does not result in an equal division')
        end
        local section_size = math.floor(axis_size / sections)
        for i = 0, sections - 1 do
            table.insert(indices, i * section_size)
        end
    elseif type(indices_or_sections) == 'table' then
        indices = table.clone(indices_or_sections)
        table.insert(indices, 1, 0)
    else
        raise('ValueError', 'indices_or_sections must either int or array of ints')
    end
    return _internal._split_v2(ary, indices, axis, squeeze_axis)
end

--

--[[
local PyCapsuleDestructor = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
local _c_str_dltensor = c_str('dltensor')
local _c_str_used_dltensor = c_str('used_dltensor')

local function _dlpack_deleter(pycapsule)
end

local _c_dlpack_deleter = PyCapsuleDestructor(_dlpack_deleter)

function M.to_dlpack_for_read(data)
end

function M.to_dlpack_for_write(data)
end

function M.from_dlpack(dlpack)
end

local DLContext
local DLDataType
local DLTensor
local DLManagedTensor

local DeleterFunc = ctypes.CFUNCTYPE(None, ctypes.POINTER(DLManagedTensor))

function M.dl_managed_tensor_deleter(dl_managed_tensor_handle)
end

function M.from_numpy(ndarray, zero_copy)
end
]]

return M
