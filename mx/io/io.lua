---@class mx.io.io
local M = {}

local ctypes = require('ctypes')
local namedtuple = require('mx_py.collections').namedtuple

local _b = require('mx.base')
local _LIB, c_str_array, mx_uint, py_str, DataIterHandle, NDArrayHandle, mx_real_t, check_call, _build_param_doc = _b._LIB, _b.c_str_array, _b.mx_uint, _b.py_str, _b.DataIterHandle, _b.NDArrayHandle, _b.mx_real_t, _b.check_call, _b.build_param_doc

local _nd = require('mx.ndarray.__init__')
local NDArray, _ndarray_cls, array, concat = _nd.NDArray, _nd._ndarray_cls, _nd.array, _nd.concat
local CSRNDArray = require('mx.ndarray.sparse').CSRNDArray

local _utils = require('mx.io.utils')
local _init_data, _has_instance, _getdata_by_idx = _utils._init_data, _utils._has_instance, _utils._getdata_by_idx

local _DataDesc = namedtuple('DataDesc', { 'name', 'shape' })
---@class mx.io.io.DataDesc
local DataDesc = class('mx.io.io.DataDesc', _DataDesc)
M.DataDesc = DataDesc

function DataDesc:ctor(name, shape, dtype, layout)
    dtype, layout = default(dtype, mx_real_t, layout, 'NCHW')
    _DataDesc.ctor(self, name, shape)
    self.dtype = dtype
    self.layout = layout
end

function DataDesc:__tostring()
    return ('DataDesc[%s,%s,%s,%s]'):format(self.name, _b.tostring(self.shape), self.dtype, self.layout)
end

function DataDesc.get_batch_axis(layout)
    if isnone(layout) then
        return 0
    end
    return layout:find('N') - 1
end

function DataDesc.get_list(shapes, types)
    if not isnone(types) then
        local type_dict = {}
        for _, v in ipairs(types) do
            type_dict[v[1]] = v[2]
        end
        local ret = {}
        for _, x in ipairs(shapes) do
            table.insert(ret, DataDesc(x[1], x[2], type_dict[x[1]]))
        end
        return ret
    else
        local ret = {}
        for _, x in ipairs(shapes) do
            table.insert(ret, DataDesc(x[1], x[2]))
        end
        return ret
    end
end

---@class mx.io.io.DataBatch
local DataBatch = class('mx.io.io.DataBatch')
M.DataBatch = DataBatch

function DataBatch:ctor(data, label, pad, index,
                        bucket_key, provide_data, provide_label)
    if not isnone(data) then
        assert(islist(data), 'Data must be list of NDArrays')
    end
    if not isnone(label) then
        assert(islist(label), 'Label must be list of NDArrays')
    end
    self.data = data
    self.label = label
    self.pad = pad
    self.index = index

    self.bucket_key = bucket_key
    self.provide_data = provide_data
    self.provide_label = provide_label
end

function DataBatch:__tostring()
    local data_shapes = table.map(self.data, function(d)
        return d.shape
    end)
    local label_shapes = None
    if bool(self.label) then
        label_shapes = table.map(self.label, function(l)
            return l.shape
        end)
    end
    return ('%s: data shapes: %s label shapes: %s'):format(
            getclassname(self), _b.tostring(data_shapes), _b.tostring(label_shapes))
end

---@class mx.io.io.DataIter
local DataIter = class('mx.io.io.DataIter')
M.DataIter = DataIter

function DataIter:ctor(batch_size)
    batch_size = default(batch_size, 0)
    self.batch_size = batch_size
end

function DataIter:__iter()
    return self
end

function DataIter:reset()
end

function DataIter:next()
    if self:iter_next() then
        return DataBatch(self:getdata(), self:getlabel(), self:getpad(), self:getindex())
    else
        return nil
    end
end

function DataIter:__next()
    return self:next()
end

function DataIter:__call()
    return self:__next()
end

function DataIter:iter_next()
end

function DataIter:getdata()
end

function DataIter:getlabel()
end

function DataIter:getindex()
end

function DataIter:getpad()
end

---@class mx.io.io.ResizeIter:mx.io.io.DataIter
local ResizeIter = class('mx.io.io.ResizeIter', DataIter)
M.ResizeIter = ResizeIter

function ResizeIter:ctor(data_iter, size, reset_internal)
    reset_internal = default(reset_internal, true)
    DataIter.ctor(self)
    self.data_iter = data_iter
    self.size = size
    self.reset_internal = reset_internal
    self.cur = 0
    self.current_batch = None

    self.provide_data = data_iter.provide_data
    self.provide_label = data_iter.provide_label
    self.batch_size = data_iter.batch_size

    if data_iter.default_bucket_key ~= nil then
        self.default_bucket_key = data_iter.default_bucket_key
    end
end

function ResizeIter:reset()
    self.cur = 0
    if self.reset_internal then
        self.data_iter:reset()
    end
end

function ResizeIter:iter_next()
    if self.cur == self.size then
        return false
    end
    self.current_batch = self.data_iter:next()
    if self.current_batch == nil then
        self.data_iter:reset()
        self.current_batch = self.data_iter:next()
    end
    self.cur = self.cur + 1
    return true
end

function ResizeIter:getdata()
    return self.current_batch.data
end

function ResizeIter:getlabel()
    return self.current_batch.label
end

function ResizeIter:getindex()
    return self.current_batch.index
end

function ResizeIter:getpad()
    return self.current_batch.pad
end

---@class mx.io.io.PrefetchingIter:mx.io.io.DataIter
local PrefetchingIter = class('mx.io.io.PrefetchingIter', DataIter)
M.PrefetchingIter = PrefetchingIter

function PrefetchingIter:ctor()
    raise('NotImplementedError')
end

---@class mx.io.io.NDArrayIter:mx.io.io.DataIter
local NDArrayIter = class('mx.io.io.NDArrayIter', DataIter)
M.NDArrayIter = NDArrayIter

function NDArrayIter:ctor(data, label, batch_size, shuffle,
                          last_batch_handle, data_name,
                          label_name)
    label, batch_size, shuffle,
    last_batch_handle, data_name,
    label_name = default(label, None, batch_size, 1, shuffle, false,
                         last_batch_handle, 'pad', data_name, 'data',
                         label_name, 'softmax_label')

    DataIter.ctor(self, batch_size)

    self.data = _init_data(data, false, data_name)
    self.label = _init_data(label, true, label_name)

    if (_has_instance(self.data, CSRNDArray) or
            _has_instance(self.label, CSRNDArray)) and
            last_batch_handle ~= 'discard' then
        raise('NotImplementedError', '`NDArrayIter` only supports ``CSRNDArray`` with `last_batch_handle` set to `discard`.')
    end
    --TODO
    self.idx = require('mx.ndarray.ndarray').arange(self.data[1][2].shape[1])
    self.shuffle = shuffle
    self.last_batch_handle = last_batch_handle
    self.batch_size = batch_size
    self.cursor = -self.batch_size
    self.num_data = self.idx.shape[1]
    -- shuffle
    self:reset()

    local data_list = {}
    for _, x in ipairs(self.data) do
        table.insert(data_list, x[2])
    end
    for _, x in ipairs(self.label) do
        table.insert(data_list, x[2])
    end
    self.data_list = data_list
    self.num_source = len(self.data_list)
    -- used for 'roll_over'
    self._cache_data = None
    self._cache_label = None
end

function NDArrayIter:_provide_data()
    local ret = {}
    for _, x in ipairs(self.data) do
        local k, v = x[1], x[2]
        local shape = table.clone(v.shape)
        shape[1] = self.batch_size
        table.insert(ret, DataDesc(k, shape, v.dtype))
    end
    return ret
end

function NDArrayIter:_provide_label()
    local ret = {}
    for _, x in ipairs(self.label) do
        local k, v = x[1], x[2]
        local shape = table.clone(v.shape)
        shape[1] = self.batch_size
        table.insert(ret, DataDesc(k, shape, v.dtype))
    end
    return ret
end

function NDArrayIter:hard_reset()
    if self.shuffle then
        self:_shuffle_data()
    end
    self.cursor = -self.batch_size
    self._cache_data = None
    self._cache_label = None
end

function NDArrayIter:reset()
    if self.shuffle then
        self:_shuffle_data()
    end
    -- the range below indicate the last batch
    if self.last_batch_handle == 'roll_over' and
            self.num_data - self.batch_size < self.cursor and
            self.cursor < self.num_data then
        -- (self.cursor - self.num_data) represents the data we have for the last batch
        self.cursor = self.cursor - self.num_data - self.batch_size
    else
        self.cursor = -self.batch_size
    end
end

function NDArrayIter:iter_next()
    self.cursor = self.cursor + self.batch_size
    return self.cursor < self.num_data
end

function NDArrayIter:next()
    if not self:iter_next() then
        return nil
    end
    local data = self:getdata()
    local label = self:getlabel()
    -- iter should stop when last batch is not complete
    if data[1].shape[1] ~= self.batch_size then
        -- in this case, cache it for next epoch
        self._cache_data = data
        self._cache_label = label
        return nil
    end
    return DataBatch(data, label, self:getpad(), None)
end

function NDArrayIter:_getdata(data_source, start, end_)
    assert(not isnone(start) or not isnone(end_), 'should at least specify start or end')
    start = default(start, 0)
    if isnone(end_) then
        if not isnone(data_source) then
            end_ = data_source[1][2].shape[1]
        else
            end_ = 0
        end
    end
    local ret = {}
    for _, x in ipairs(data_source) do
        local a = x[2]
        if isinstance(a, NDArray) then
            table.insert(ret, a[{ start, end_ }])
        else
            -- h5py (only supports indices in increasing order)
            raise('NotImplementedError')
        end
    end
    return ret
end

function NDArrayIter:_concat(first_data, second_data)
    assert(len(first_data) == len(second_data),
           'data source should contain the same size')
    if #first_data ~= 0 and #second_data ~= 0 then
        return table.arange(function(x)
            return concat(
                    arg_make({ first_data[x], second_data[x] }, { dim = 0 }))
        end, #first_data)
    elseif #first_data == 0 and #second_data == 0 then
        return {}
    else
        return table.arange(function(x)
            if first_data then
                return first_data[1]
            else
                return second_data[1]
            end
        end, #first_data)
    end
end

function NDArrayIter:_batchify(data_source)
    assert(self.cursor < self.num_data, 'DataIter needs reset.')
    -- first batch of next epoch with 'roll_over'
    if self.last_batch_handle == 'roll_over' and
            -self.batch_size < self.cursor and
            self.cursor < 0 then
        assert(not isnone(self._cache_data) or not isnone(self._cache_label),
               'next epoch should have cached data')
        local cache_data = isnone(self._cache_data) and self._cache_label or self._cache_data
        local second_data = self:_getdata(data_source, None, self.cursor + self.batch_size)
        if not isnone(self._cache_data) then
            self._cache_data = None
        else
            self._cache_label = None
        end
        return self:_concat(cache_data, second_data)
    elseif self.last_batch_handle == 'pad' and
            self.cursor + self.batch_size > self.num_data then
        local pad = self.batch_size - self.num_data + self.cursor
        local first_data = self:_getdata(data_source, self.cursor)
        local second_data = self:_getdata(data_source, None, pad)
        return self:_concat(first_data, second_data)
    else
        local end_idx
        if self.cursor + self.batch_size < self.num_data then
            end_idx = self.cursor + self.batch_size
        else
            end_idx = self.num_data
        end
        return self:_getdata(data_source, self.cursor, end_idx)
    end
end

function NDArrayIter:getdata()
    return self:_batchify(self.data)
end

function NDArrayIter:getlabel()
    return self:_batchify(self.label)
end

function NDArrayIter:getpad()
    if self.last_batch_handle == 'pad' and
            self.cursor + self.batch_size > self.num_data then
        return self.cursor + self.batch_size - self.num_data
    elseif self.last_batch_handle == 'roll_over' and
            -self.batch_size < self.cursor < 0 then
        return -self.cursor
    else
        return 0
    end
end

function NDArrayIter:_shuffle_data()
    self.idx = require('mx.ndarray.random').shuffle(self.idx)
    self.data = _getdata_by_idx(self.data, self.idx)
    self.label = _getdata_by_idx(self.label, self.idx)
end

class_property(NDArrayIter, {
    provide_data  = NDArrayIter._provide_data,
    provide_label = NDArrayIter._provide_label, })

---@class mx.io.io.MXDataIter:mx.io.io.DataIter
local MXDataIter = class('mx.io.io.MXDataIter', DataIter)
M.MXDataIter = MXDataIter

function MXDataIter:ctor(handle, data_name, label_name)
    DataIter.ctor(self)
    self.handle = handle
    -- debug option, used to test the speed with io effect eliminated
    self._debug_skip_load = false

    -- load the first batch to get shape information
    self.first_batch = None
    self.first_batch = self:next()
    local data = self.first_batch.data[1]
    local label = self.first_batch.label[1]

    -- properties
    self.provide_data = { DataDesc(data_name, data.shape, data.dtype) }
    self.provide_label = { DataDesc(label_name, label.shape, label.dtype) }
    self.batch_size = data.shape[1]
end

function MXDataIter:dtor()
    check_call(_LIB.MXDataIterFree(self.handle))
end

function MXDataIter:debug_skip_load()
    self._debug_skip_load = true
    print('Set debug_skip_load to be true, will simply return first batch')
end

function MXDataIter:reset()
    self._debug_at_begin = true
    self.first_batch = None
    check_call(_LIB.MXDataIterBeforeFirst(self.handle))
end

function MXDataIter:next()
    if self._debug_skip_load and not self._debug_at_begin then
        return DataBatch({ self:getdata() }, { self:getlabel() }, self:getpad(), self:getindex())
    end
    if not isnone(self.first_batch) then
        local batch = self.first_batch
        self.first_batch = None
        return batch
    end
    self._debug_at_begin = false
    local next_res = ctypes.c_int(0)
    check_call(_LIB.MXDataIterNext(self.handle, ctypes.byref(next_res)))
    if next_res.value ~= 0 then
        return DataBatch({ self:getdata() }, { self:getlabel() }, self:getpad(), self:getindex())
    else
        return nil
    end
end

function MXDataIter:iter_next()
    if not isnone(self.first_batch) then
        return true
    end
    local next_res = ctypes.c_int(0)
    check_call(_LIB.MXDataIterNext(self.handle, ctypes.byref(next_res)))
    return next_res.value ~= 0
end

function MXDataIter:getdata()
    local hdl = NDArrayHandle()
    check_call(_LIB.MXDataIterGetData(self.handle, ctypes.byref(hdl)))
    return _ndarray_cls(hdl, false)
end

function MXDataIter:getlabel()
    local hdl = NDArrayHandle()
    check_call(_LIB.MXDataIterGetLabel(self.handle, ctypes.byref(hdl)))
    return _ndarray_cls(hdl, false)
end

function MXDataIter:getindex()
    local index_size = ctypes.c_uint64(0)
    local index_data = ctypes.POINTER(ctypes.c_uint64)()
    check_call(_LIB.MXDataIterGetIndex(self.handle,
                                       ctypes.byref(index_data),
                                       ctypes.byref(index_size)))
    if index_size.value ~= 0 then
        -- NDArray only support int64
        local cdata = ffi.new(('int64_t[%d]'):format(index_size.value))
        ffi.copy(cdata, index_data.value, index_size.value * ffi.sizeof('int64_t'))
        return _nd.array(cdata, nil, 'int64')
    else
        return None
    end
end

function MXDataIter:getpad()
    local pad = ctypes.c_int(0)
    check_call(_LIB.MXDataIterGetPadNum(self.handle, ctypes.byref(pad)))
    return pad.value
end

function M._make_io_iterator(handle)
    local name = ctypes.c_char_p()
    local desc = ctypes.c_char_p()
    local num_args = mx_uint()
    local arg_names = ctypes.POINTER(ctypes.c_char_p)()
    local arg_types = ctypes.POINTER(ctypes.c_char_p)()
    local arg_descs = ctypes.POINTER(ctypes.c_char_p)()

    check_call(_LIB.MXDataIterGetIterInfo(
            handle, ctypes.byref(name), ctypes.byref(desc),
            ctypes.byref(num_args),
            ctypes.byref(arg_names),
            ctypes.byref(arg_types),
            ctypes.byref(arg_descs)))
    local iter_name = py_str(name.value)

    local narg = num_args.value
    local param_str = _build_param_doc(
            table.arange(function(i)
                return py_str(arg_names[i - 1])
            end, narg),
            table.arange(function(i)
                return py_str(arg_types[i - 1])
            end, narg),
            table.arange(function(i)
                return py_str(arg_descs[i - 1])
            end, narg)
    )
    local doc_str = '%s\n\n' ..
            '%s\n' ..
            '---@return mx.io.io.MXDataIter @The result iterator.'
    doc_str = doc_str:format(ffi.string(desc.value), param_str)

    local function creator(kwargs)
        kwargs = default(kwargs, {})
        local param_keys = {}
        local param_vals = {}
        for k, v in pairs(kwargs) do
            table.insert(param_keys, k)
            table.insert(param_vals, _b.tostring(v))
        end
        -- create atomic symbol
        param_keys = c_str_array(param_keys)
        param_vals = c_str_array(param_vals)
        local iter_handle = DataIterHandle()
        check_call(_LIB.MXDataIterCreateIter(
                handle,
                mx_uint(len(param_keys)),
                param_keys, param_vals,
                ctypes.byref(iter_handle)))
        return MXDataIter(iter_handle, kwargs.data_name, kwargs.label_name)
    end
    return creator, iter_name, doc_str
end

function M._init_io_module()
    local plist = ctypes.POINTER(ctypes.c_void_p)()
    local size = ctypes.c_uint()
    check_call(_LIB.MXListDataIters(ctypes.byref(size), ctypes.byref(plist)))
    for i = 0, size.value - 1 do
        local hdl = plist[i]
        local dataiter, iter_name, doc_str = M._make_io_iterator(hdl)
        M[iter_name] = dataiter
    end
end

print('start register mx.io')
M._init_io_module()
print('finish register mx.io')

return M
