---@class mx.ndarray.sparse:mx.ndarray.gen_sparse
local M = {}

local base = require('mx.base')
local _LIB, c_array_buf, mx_real_t, integer_types, mx_uint, NDArrayHandle, check_call = base._LIB, base.c_array_buf, base.mx_real_t, base.integer_types, base.mx_uint, base.NDArrayHandle, base.check_call

local Context, current_context = require('mx.context').Context, require('mx.context').current_context
local _internal = require('mx.ndarray._internal')
local op = require('mx.ndarray.op')

local ndarray = require('mx.ndarray.ndarray')
local NDArray, _storage_type, _DTYPE_NP_TO_MX, _DTYPE_MX_TO_NP, _STORAGE_TYPE_STR_TO_ID, _STORAGE_TYPE_ROW_SPARSE, _STORAGE_TYPE_CSR, _STORAGE_TYPE_UNDEFINED, _STORAGE_TYPE_DEFAULT = ndarray.NDArray, ndarray._storage_type, ndarray._DTYPE_NP_TO_MX, ndarray._DTYPE_MX_TO_NP, ndarray._STORAGE_TYPE_STR_TO_ID, ndarray._STORAGE_TYPE_ROW_SPARSE, ndarray._STORAGE_TYPE_CSR, ndarray._STORAGE_TYPE_UNDEFINED, ndarray._STORAGE_TYPE_DEFAULT
local _zeros_ndarray = ndarray.zeros
local _array = ndarray.array
local _ufunc_helper = ndarray._ufunc_helper

local _set_ndarray_class = require('mx.ndarray._internal')._set_ndarray_class

--

local _STORAGE_AUX_TYPES = {
    row_sparse = { 'int64' },
    csr        = { 'int64', 'int64' },
}
M._STORAGE_AUX_TYPES = _STORAGE_AUX_TYPES

function M._new_alloc_handle(stype, shape, ctx, delay_alloc, dtype, aux_types, aux_shapes)
    raise('NotImplementedError')
end

---@class mx.ndarray.sparse.BaseSparseNDArray:mx.ndarray.NDArray
local BaseSparseNDArray = class('mx.ndarray.sparse.BaseSparseNDArray', NDArray)
M.BaseSparseNDArray = BaseSparseNDArray

function BaseSparseNDArray:ctor(handle, writable)
    raise('NotImplementedError')
end

---@class mx.ndarray.sparse.CSRNDArray:mx.ndarray.sparse.BaseSparseNDArray
local CSRNDArray = class('mx.ndarray.sparse.CSRNDArray', BaseSparseNDArray)
M.CSRNDArray = CSRNDArray

---@class mx.ndarray.sparse.RowSparseNDArray:mx.ndarray.sparse.BaseSparseNDArray
local RowSparseNDArray = class('mx.ndarray.sparse.RowSparseNDArray', BaseSparseNDArray)
M.RowSparseNDArray = RowSparseNDArray

function M._prepare_src_array(source_array, dtype)
    raise('NotImplementedError')
end

function M._prepare_default_dtype(src_array, dtype)
    if dtype == nil then
        if isinstance(src_array, NDArray) then
            dtype = src_array.dtype
        else
            dtype = mx_real_t
        end
    end
    return dtype
end

function M._check_shape(s1, s2)
    if s1 and s2 and not table.equal(s1, s2) then
        raise('ValueError', 'Shape mismatch detected.')
    end
end

function M.csr_matrix(arg1, shape, ctx, dtype)
    raise('NotImplementedError')
end

function M._csr_matrix_from_definition(data, indices, indptr, shape, ctx, dtype, indices_type, indptr_type)
    raise('NotImplementedError')
end

function M.row_sparse_array(arg1, shape, ctx, dtype)
    raise('NotImplementedError')
end

function M._row_sparse_ndarray_from_definition(data, indices, shape, ctx, dtype, indices_type)
    raise('NotImplementedError')
end

function M._ndarray_cls(handle, writable, stype)
    writable, stype = default(writable, true, stype, _STORAGE_TYPE_UNDEFINED)
    if stype == _STORAGE_TYPE_UNDEFINED then
        stype = _storage_type(handle)
    end
    if stype == _STORAGE_TYPE_DEFAULT then
        return NDArray(handle, writable)
    elseif stype == _STORAGE_TYPE_CSR then
        return CSRNDArray(handle, writable)
    elseif stype == _STORAGE_TYPE_ROW_SPARSE then
        return RowSparseNDArray(handle, writable)
    else
        raise('Exception', 'unknown storage type')
    end
end

_set_ndarray_class(M._ndarray_cls)

local operator = {}
setmetatable(operator, { __index = function(t, k)
    operator = require('mx.operator')
    return operator[k]
end })

function M.add(lhs, rhs)
    if isinstance(lhs, NDArray) and isinstance(rhs, NDArray) and table.equal(lhs.shape, rhs.shape) then
        return _ufunc_helper(
                lhs,
                rhs,
                op.elemwise_add,
                operator.add,
                _internal._plus_scalar,
                nil)
    end
    return _ufunc_helper(
            lhs,
            rhs,
            op.broadcast_add,
            operator.add,
            _internal._plus_scalar,
            nil)
end

function M.subtract(lhs, rhs)
    if isinstance(lhs, NDArray) and isinstance(rhs, NDArray) and table.equal(lhs.shape, rhs.shape) then
        return _ufunc_helper(
                lhs,
                rhs,
                op.elemwise_sub,
                operator.sub,
                _internal._minus_scalar,
                nil)
    end
    return _ufunc_helper(
            lhs,
            rhs,
            op.broadcast_sub,
            operator.sub,
            _internal._minus_scalar,
            nil)
end

function M.subtract(lhs, rhs)
    if isinstance(lhs, NDArray) and isinstance(rhs, NDArray) and table.equal(lhs.shape, rhs.shape) then
        return _ufunc_helper(
                lhs,
                rhs,
                op.elemwise_mul,
                operator.mul,
                _internal._mul_scalar,
                nil)
    end
    return _ufunc_helper(
            lhs,
            rhs,
            op.broadcast_mul,
            operator.mul,
            _internal._mul_scalar,
            nil)
end

function M.divide(lhs, rhs)
    if isinstance(lhs, NDArray) and isinstance(rhs, NDArray) and table.equal(lhs.shape, rhs.shape) then
        return _ufunc_helper(
                lhs,
                rhs,
                op.elemwise_div,
                operator.truediv,
                _internal._div_scalar,
                nil)
    end
    return _ufunc_helper(
            lhs,
            rhs,
            op.broadcast_div,
            operator.truediv,
            _internal._div_scalar,
            nil)
end

function M.zeros(stype, shape, ctx, dtype, kwargs)
    if stype == 'default' then
        return _zeros_ndarray(shape, ctx, dtype, kwargs)
    end
    if ctx == nil then
        ctx = current_context()
    end
    dtype = dtype or mx_real_t
    local aux_types
    if _STORAGE_AUX_TYPES[stype] then
        aux_types = _STORAGE_AUX_TYPES[stype]
    else
        raise('ValueError', 'unknown storage type')
    end
    local out = M._ndarray_cls(M._new_alloc_handle(stype, shape, ctx, true, dtype, aux_types))
    return _internal._zeros(shape, ctx, dtype, out, nil, kwargs)
end

function M.empty(stype, shape, ctx, dtype)
    if type(shape) == 'number' then
        shape = { shape }
    end
    if ctx == nil then
        ctx = current_context()
    end
    dtype = dtype or mx_real_t
    assert(stype)
    if _STORAGE_AUX_TYPES[stype] then
        return M.zeros(stype, shape, ctx, dtype)
    else
        raise('ValueError', 'unknown storage type')
    end
end

function M.array(source_array, ctx, dtype)
    if ctx == nil then
        ctx = current_context()
    end
    if isinstance(source_array, NDArray) then
        assert(source_array.stype ~= 'default',
               'Please use `tostype` to create RowSparseNDArray or CSRNDArray from an NDArray')
        dtype = M._prepare_default_dtype(source_array, dtype)
        local arr
        if source_array.dtype ~= dtype and source_array.context ~= ctx then
            arr = M.empty(source_array.stype, source_array.shape, nil, dtype)
            arr[{}] = source_array
            arr = arr:as_in_context(ctx)
        else
            arr = M.empty(source_array.stype, source_array.shape, ctx, dtype)
            arr[{}] = source_array
        end
        return arr
    else
        raise('ValueError', 'Unexpected source_array type')
    end
end

return M
