---@class mx.io.utils
local M = {}

local sparse = require('mx.ndarray.sparse')
local CSRNDArray = sparse.CSRNDArray
local sparse_array = sparse.array

local ndarray = require('mx.ndarray.ndarray')
local NDArray, array = ndarray.NDArray, ndarray.array

--- Convert data into canonical form.
--- { {key1, arr1}, {key2, arr2}, ... }
function M._init_data(data, allow_empty, default_name)
    assert(data or allow_empty)
    if data == nil then
        data = {}
    end
    if isinstance(data, NDArray) then
        data = { data }
    end
    if not allow_empty then
        assert(len(data) > 0)
    end
    if len(data) == 1 then
        data = { { default_name, data[1] } }
    else
        local dat = {}
        for i = 1, len(data) do
            table.insert(dat, { ('_%d_%s'):format(i - 1, default_name), data[i] })
        end
        data = dat
    end
    for i = 1, len(data) do
        local v = data[i][2]
        if not isinstance(v, NDArray) then
            data[i][2] = array(v)
        end
    end
    return data
end

--- Return True if ``data`` has instance of ``dtype``.
--- This function is called after _init_data.
--- ``data`` is a list of (str, NDArray)
function M._has_instance(data, dtype)
    for _, item in ipairs(data) do
        local arr = item[2]
        if isinstance(arr, dtype) then
            return true
        end
    end
    return false
end

function M._getdata_by_idx(data, idx)
    local shuffle_data = {}
    for _, d in ipairs(data) do
        local k, v = d[1], d[2]
        if isinstance(v, CSRNDArray) then
            --table.insert(shuffle_data, { k, sparse_array(v:asscipy()[idx], v.context) })
            table.insert(shuffle_data, { k, sparse_array(v[idx], v.context) })
        else
            --table.insert(shuffle_data, { k, array(v:asnumpy()[idx], v.context) })
            table.insert(shuffle_data, { k, array(v[idx], v.context) })
        end
    end
    return shuffle_data
end

return M
