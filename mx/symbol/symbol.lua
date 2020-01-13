---@class mx.symbol.symbol
local M = {}

local ctypes = require('ctypes')
local AttrScope = require('mx.attribute').AttrScope
local base = require('mx.base')
local _LIB, numeric_types, c_array, c_array_buf, c_str, c_str_array, c_handle_array
, mx_uint, py_str, string_types, integer_types, mx_int
, NDArrayHandle, ExecutorHandle, SymbolHandle
, check_call, MXNetError, NotImplementedForSymbol = base._LIB, base.numeric_types, base.c_array, base.c_array_buf, base.c_str, base.c_str_array, base.c_handle_array
, base.mx_uint, base.py_str, base.string_types, base.integer_types, base.mx_int
, base.NDArrayHandle, base.ExecutorHandle, base.SymbolHandle
, base.check_call, base.MXNetError, base.NotImplementedForSymbol

local Context, current_context = require('mx.context').Context, require('mx.context').current_context
local __nd = require('mx.ndarray.__init__')
local NDArray, _DTYPE_NP_TO_MX, _DTYPE_MX_TO_NP, _GRAD_REQ_MAP, _ndarray_cls = __nd.NDArray, __nd._DTYPE_NP_TO_MX, __nd._DTYPE_MX_TO_NP, __nd._GRAD_REQ_MAP, __nd._ndarray_cls
local _STORAGE_TYPE_STR_TO_ID = require('mx.ndarray.ndarray')._STORAGE_TYPE_STR_TO_ID

local Executor = require('mx.executor').Executor
local _internal = require('mx.symbol._internal')
local op = require('mx.symbol.op')
local SymbolBase, _set_symbol_class = _internal.SymbolBase, _internal._set_symbol_class
local is_np_shape = require('mx.util').is_np_shape

---@class mx.symbol.Symbol:mx._ctypes.symbol.SymbolBase
---@field name string
local Symbol = class('mx.symbol.Symbol', SymbolBase)
M.Symbol = Symbol

function Symbol:__tostring()
    local name = self.name
    if isnone(name) then
        name = {}
        for i in self:__iter() do
            table.insert(name, i.name)
        end
        name = table.concat(name, ', ')
        return ('<%s group [%s]>'):format('Symbol', name)
    else
        return ('<%s %s>'):format('Symbol', name)
    end
end

function Symbol:__iter()
    local sz = #self
    local function f(t, i)
        i = i + 1
        if i >= sz then
            return
        end
        return self:__getitem(i)
    end
    return f, self, -1
end

local function _op_helper(lhs, rhs, f1, f2, f3)
    if isinstance(lhs, Symbol) then
        if isinstance(rhs, Symbol) then
            return f1(lhs, rhs)
        elseif type(rhs) == 'number' then
            return f2(lhs, rhs)
        else
            raise('TypeError', ('type %s not supported'):format(gettypename(rhs)))
        end
    else
        if type(lhs) == 'number' then
            return f3(rhs, lhs)
        else
            raise('TypeError', ('type %s not supported'):format(gettypename(lhs)))
        end
    end
end

function Symbol:__add(other)
    return _op_helper(self, other, _internal._Plus, _internal._PlusScalar, Symbol.__add)
end

function Symbol:__bool()
    raise('NotImplementedForSymbol')
end

function Symbol:iadd(other)
    raise('NotImplementedForSymbol')
end

function Symbol:__sub(other)
    return _op_helper(self, other, _internal._Minus, _internal._MinusScalar, _internal._RMinusScalar)
end

function Symbol:isub(other)
    raise('NotImplementedForSymbol')
end

function Symbol:__mul(other)
    return _op_helper(self, other, _internal._Mul, _internal._MulScalar, Symbol.__mul)
end

function Symbol:imul(other)
    raise('NotImplementedForSymbol')
end

function Symbol:__div(other)
    return _op_helper(self, other, _internal._Div, _internal._DivScalar, _internal._RDivScalar)
end

function Symbol:idiv(other)
    raise('NotImplementedForSymbol')
end

function Symbol:__mod(other)
    return _op_helper(self, other, _internal._Mod, _internal._ModScalar, _internal._RModScalar)
end

function Symbol:__pow(other)
    return _op_helper(self, other, _internal._Power, _internal._PowerScalar, _internal._RPowerScalar)
end

function Symbol:__unm()
    return self:__mul(-1.0)
end

function Symbol:__copy()
    return self:__deepcopy()
end

function Symbol:__deepcopy()
    local handle = SymbolHandle()
    check_call(_LIB.MXSymbolCopy(self.handle,
                                 ctypes.byref(handle)))
    return Symbol(handle)
end

local function _cmp_helper(lhs, rhs, f1, f2)
    if isinstance(rhs, Symbol) then
        return f1(lhs, rhs)
    elseif type(rhs) == 'number' then
        return f2(lhs, rhs)
    else
        raise('TypeError', ('type %s not supported'):format(gettypename(rhs)))
    end
end

function Symbol:eq(other)
    return _cmp_helper(self, other, _internal._equal, _internal._equal_scalar)
end

function Symbol:ne(other)
    return _cmp_helper(self, other, _internal._not_equal, _internal._not_equal_scalar)
end

function Symbol:gt(other)
    return _cmp_helper(self, other, _internal._greater, _internal._greater_scalar)
end

function Symbol:ge(other)
    return _cmp_helper(self, other, _internal._greater_equal, _internal._greater_equal_scalar)
end

function Symbol:lt(other)
    return _cmp_helper(self, other, _internal._lesser, _internal._lesser_scalar)
end

function Symbol:le(other)
    return _cmp_helper(self, other, _internal._lesser_equal, _internal._lesser_equal_scalar)
end

function Symbol:__getstate()
    if not isnone(self.handle) then
        return { handle = self:tojson() }
    else
        return { handle = None }
    end
end

function Symbol:__setstate(state)
    local handle = state.handle
    if not isnone(handle) then
        local json_str = handle
        handle = SymbolHandle()
        check_call(_LIB.MXSymbolCreateFromJSON(c_str(json_str), ctypes.byref(handle)))
        self.handle = handle
    else
        self.handle = None
    end
end

function Symbol:__call(...)
    local s = self:__copy()
    s:_compose(...)
    return s
end

function Symbol:_compose(...)
    local args, kwargs = arg_kw(...)
    local name = arg_pop(kwargs, 'name', None)
    if name then
        name = c_str(name)
    end
    if #args ~= 0 and not table.empty(kwargs) then
        raise('TypeError', 'compose only accept input Symbols either as positional or keyword arguments, not both')
    end
    for i, arg in ipairs(args) do
        if not isinstance(arg, Symbol) then
            raise('TypeError', ('Compose expect `Symbol` as arguments, but got `%s` at %d'):format(gettypename(arg, i)))
        end
    end
    for k, val in pairs(kwargs) do
        if not isinstance(val, Symbol) then
            raise('TypeError', ('Compose expect `Symbol` as arguments, but got `%s` at `%s`'):format(gettypename(val), k))
        end
    end
    local num_args = len(args) + table.len(kwargs)
    local keys
    if not table.empty(kwargs) then
        keys = c_str_array(table.keys(kwargs))
        args = c_handle_array(table.values(kwargs))
    else
        keys = None
        args = c_handle_array(table.values(kwargs))
    end
    check_call(_LIB.NNSymbolCompose(
            self.handle, name, num_args, keys, args))
end

function Symbol:__getitem(index)
    local output_count = #self
    if islist(index) then
        local start, stop, step = index[1] or 0, index[2] or output_count, index[3] or 1
        local function f(i)
            return self[i]
        end
        return M.Group(table.arange(f, start, stop - 1, step))
    end
    if type(index) == 'string' then
        local output_names = self:list_outputs()
        local idx
        for i, name in ipairs(output_names) do
            if name == index then
                if idx then
                    raise('ValueError', 'There are multiple outputs with name ' .. index)
                end
                idx = i - 1
            end
        end
        if not idx then
            raise('ValueError', 'Cannot find output that matches name ' .. index)
        end
        index = idx
    end
    if type(index) ~= 'number' then
        raise('TypeError', 'Symbol only support integer index to fetch i-th output')
    end
    if index >= output_count then
        raise('IndexError')
    end
    local handle = SymbolHandle()
    check_call(_LIB.MXSymbolGetOutput(
            self.handle, mx_uint(index), ctypes.byref(handle)))
    return Symbol(handle)
end

function Symbol:_name()
    local ret = ctypes.c_char_p()
    local success = ctypes.c_int()
    check_call(_LIB.MXSymbolGetName(
            self.handle, ctypes.byref(ret), ctypes.byref(success)))
    if success.value ~= 0 then
        return py_str(ret.value)
    else
        return None
    end
end

function Symbol:attr(key)
    local ret = ctypes.c_char_p()
    local success = ctypes.c_int()
    check_call(_LIB.MXSymbolGetAttr(
            self.handle, ctypes.byref(ret), ctypes.byref(success)))
    if success.value ~= 0 then
        return py_str(ret.value)
    else
        return None
    end
end

function Symbol:list_attr(recursive)
    recursive = default(recursive, false)
    if recursive then
        raise('DeprecationWarning', 'Please use attr_dict instead')
    end
    local size = mx_uint()
    local pairs = ctypes.POINTER(ctypes.c_char_p)()
    check_call(_LIB.MXSymbolListAttrShallow(self.handle, ctypes.byref(size), ctypes.byref(pairs)))
    local ret = {}
    for i = 0, size.value - 1 do
        ret[py_str(pairs[i * 2])] = py_str(pairs[i * 2 + 1])
    end
    return ret
end

function Symbol:attr_dict()
    local size = mx_uint()
    local pairs = ctypes.POINTER(ctypes.c_char_p)()
    check_call(_LIB.MXSymbolListAttr(self.handle, ctypes.byref(size), ctypes.byref(pairs)))
    local ret = {}
    for i = 0, size.value - 1 do
        local name, key = unpack(py_str(pairs[i * 2]):split('$'))
        local val = py_str(pairs[i * 2 + 1])
        if not ret[name] then
            ret[name] = {}
        end
        ret[name][key] = val
    end
    return ret
end

function Symbol:_set_attr(kwargs)
    for key, value in pairs(kwargs) do
        if type(value) ~= 'string' then
            raise('ValueError', 'Set Attr only accepts string values')
        end
        check_call(_LIB.MXSymbolSetAttr(
                self.handle, c_str(key), c_str(value)))
    end
end

function Symbol:get_internals()
    local handle = SymbolHandle()
    check_call(_LIB.MXSymbolGetInternals(
            self.handle, ctypes.byref(handle)))
    return Symbol(handle)
end

function Symbol:get_children()
    local handle = SymbolHandle()
    check_call(_LIB.MXSymbolGetChildren(
            self.handle, ctypes.byref(handle)))
    local ret = Symbol(handle)
    if #ret:list_outputs() == 0 then
        return None
    end
    return ret
end

local function _list_helper(hdl, f)
    local size = ctypes.c_uint()
    local sarr = ctypes.POINTER(ctypes.c_char_p)()
    check_call(f(
            hdl, ctypes.byref(size), ctypes.byref(sarr)))
    return table.arange(function(i)
        return py_str(sarr[i - 1])
    end, size.value)
end

function Symbol:list_arguments()
    return _list_helper(self.handle, _LIB.MXSymbolListArguments)
end

function Symbol:list_outputs()
    return _list_helper(self.handle, _LIB.MXSymbolListOutputs)
end

function Symbol:__len()
    local output_count = mx_uint()
    check_call(_LIB.MXSymbolGetNumOutputs(self.handle, ctypes.byref(output_count)))
    return tonumber(output_count.value)
end

function Symbol:list_auxiliary_states()
    return _list_helper(self.handle, _LIB.MXSymbolListAuxiliaryStates)
end

function Symbol:list_inputs()
    local size = ctypes.c_uint()
    local sarr = ctypes.POINTER(ctypes.c_char_p)()
    check_call(_LIB.NNSymbolListInputNames(
            self.handle, 0, ctypes.byref(size), ctypes.byref(sarr)))
    return table.arange(function(i)
        return py_str(sarr[i - 1])
    end, size.value)
end

function Symbol:infer_type(...)
    local try = function(...)
        local res = self:_infer_type_impl(false, ...)
        if isnone(res[2]) then
            local arg_types = self:_infer_type_impl(true, ...)[1]
            local arg_names = self:list_arguments()
            local unknowns = {}
            for i = 1, #arg_names do
                local name, dtype = arg_names[i], arg_types[i]
                if not dtype then
                    if #unknowns > 10 then
                        table.insert(unknowns, '...')
                        break
                    end
                    table.insert(unknowns, ('%s: %s'):format(name, tostring(dtype)))
                end
            end
            print('Cannot decide type for the following arguments.',
                  'Consider providing them as input:\n\t',
                  table.concat(unknowns, '\n\t'))
        end
        return res
    end
    local ok, ret = pcall(try, ...)
    if ok then
        return unpack(ret)
    else
        local args, kwargs = arg_kw(...)
        print("infer_type error. Arguments:")
        for i, arg in ipairs(args) do
            print(('  #%d: %s'):format(i, tostring(arg)))
        end
        for k, v in pairs(kwargs) do
            print(('  %s: %s'):format(k, tostring(v)))
        end
        raise()
    end
end

function Symbol:infer_type_partial(...)
    return unpack(self:_infer_type_impl(true, ...))
end

function Symbol:_infer_type_impl(partial, ...)
    local args, kwargs = arg_kw(...)
    if #args ~= 0 and not table.empty(kwargs) then
        raise('ValueError', 'Can only specify known argument types either by positional or kwargs way.')
    end
    local sdata = {}
    local keys
    if #args ~= 0 then
        keys = c_array(ctypes.c_char_p, {})
        for _, s in ipairs(args) do
            if not isnone(s) then
                s = _DTYPE_NP_TO_MX[s]
                if not s then
                    raise('TypeError')
                end
                table.insert(sdata, s)
            else
                table.insert(sdata, -1)
            end
        end
    else
        local str_keys = {}
        for k, v in pairs(kwargs) do
            if _DTYPE_NP_TO_MX[v] then
                table.insert(str_keys, k)
                table.insert(sdata, _DTYPE_NP_TO_MX[v])
            end
        end
        keys = c_str_array(str_keys)
    end
    local arg_type_size = mx_uint()
    local arg_type_data = ctypes.POINTER(ctypes.c_int)()
    local out_type_size = mx_uint()
    local out_type_data = ctypes.POINTER(ctypes.c_int)()
    local aux_type_size = mx_uint()
    local aux_type_data = ctypes.POINTER(ctypes.c_int)()
    local complete = ctypes.c_int()
    local infer_func = partial and _LIB.MXSymbolInferTypePartial or _LIB.MXSymbolInferType
    check_call(infer_func(
            self.handle,
            mx_uint(len(sdata)),
            keys,
            c_array_buf(ctypes.c_int, sdata),
            ctypes.byref(arg_type_size),
            ctypes.byref(arg_type_data),
            ctypes.byref(out_type_size),
            ctypes.byref(out_type_data),
            ctypes.byref(aux_type_size),
            ctypes.byref(aux_type_data),
            ctypes.byref(complete)))
    if complete.value ~= 0 then
        local arg_types = table.arange(function(i)
            return _DTYPE_MX_TO_NP[tonumber(arg_type_data[i - 1])]
        end, arg_type_size.value)
        local out_types = table.arange(function(i)
            return _DTYPE_MX_TO_NP[tonumber(out_type_data[i - 1])]
        end, out_type_size.value)
        local aux_types = table.arange(function(i)
            return _DTYPE_MX_TO_NP[tonumber(aux_type_data[i - 1])]
        end, aux_type_size.value)
        return { arg_types, out_types, aux_types }
    else
        return { None, None, None }
    end
end

function Symbol:infer_shape(...)
    local try = function(...)
        local res = self:_infer_shape_impl(false, ...)
        if isnone(res[2]) then
            local arg_shapes = self:_infer_shape_impl(true, ...)[1]
            local arg_names = self:list_arguments()
            local unknowns = {}
            for i = 1, #arg_names do
                local name, shape = arg_names[i], arg_shapes[i]
                local shape_is_none = isnone(shape)
                if not shape_is_none then
                    if is_np_shape() then
                        shape_is_none = table.has_value(shape, -1)
                    else
                        shape_is_none = table.has_value(shape, 0)
                    end
                end
                if shape_is_none then
                    if #unknowns > 10 then
                        table.insert(unknowns, '...')
                        break
                    end
                    table.insert(unknowns, ('%s: %s'):format(name, base.tostring(shape)))
                end
            end
            print('Cannot decide shape for the following arguments ',
                  '(0s in shape means unknown dimensions).',
                  'Consider providing them as input:\n\t',
                  table.concat(unknowns, '\n\t'))
        end
        return res
    end
    local ok, ret = pcall(try, ...)
    if ok then
        return unpack(ret)
    else
        local args, kwargs = arg_kw(...)
        print("infer_shape error. Arguments:")
        for i, arg in ipairs(args) do
            print(('  #%d: %s'):format(i, tostring(arg)))
        end
        for k, v in pairs(kwargs) do
            print(('  %s: %s'):format(k, tostring(v)))
        end
        raise()
    end
end

function Symbol:infer_shape_partial(...)
    return unpack(self:_infer_shape_impl(true, ...))
end

function Symbol:_infer_shape_impl(partial, ...)
    local args, kwargs = arg_kw(...)
    if #args ~= 0 and not table.empty(kwargs) then
        raise('ValueError', 'Can only specify known argument types either by positional or kwargs way.')
    end
    local sdata = {}
    local indptr = { 0 }
    local keys
    if #args ~= 0 then
        keys = c_array(ctypes.c_char_p, {})
        for _, s in ipairs(args) do
            if not isnone(s) then
                if not islist(s) then
                    raise('TypeError', 'Arguments need to be shapes')
                end
                table.append(sdata, s)
            end
            table.insert(indptr, #sdata)
        end
    else
        local str_keys = {}
        for k, v in pairs(kwargs) do
            if not islist(v) then
                raise('TypeError', 'Arguments need to be shapes')
            end
            table.insert(str_keys, k)
            table.append(sdata, v)
            table.insert(indptr, #sdata)
        end
        keys = c_str_array(str_keys)
    end
    local arg_shape_size = mx_uint()
    local arg_shape_ndim = ctypes.POINTER(mx_int)()
    local arg_shape_data = ctypes.POINTER(ctypes.POINTER(mx_int))()
    local out_shape_size = mx_uint()
    local out_shape_ndim = ctypes.POINTER(mx_int)()
    local out_shape_data = ctypes.POINTER(ctypes.POINTER(mx_int))()
    local aux_shape_size = mx_uint()
    local aux_shape_ndim = ctypes.POINTER(mx_int)()
    local aux_shape_data = ctypes.POINTER(ctypes.POINTER(mx_int))()
    local complete = ctypes.c_int()
    local infer_func = partial and _LIB.MXSymbolInferShapePartialEx or _LIB.MXSymbolInferShapeEx
    check_call(infer_func(
            self.handle,
            mx_uint(#indptr - 1),
            keys,
            c_array_buf(mx_uint, indptr),
            c_array_buf(mx_int, sdata),
            ctypes.byref(arg_shape_size),
            ctypes.byref(arg_shape_ndim),
            ctypes.byref(arg_shape_data),
            ctypes.byref(out_shape_size),
            ctypes.byref(out_shape_ndim),
            ctypes.byref(out_shape_data),
            ctypes.byref(aux_shape_size),
            ctypes.byref(aux_shape_ndim),
            ctypes.byref(aux_shape_data),
            ctypes.byref(complete)))
    if complete.value ~= 0 then
        local function make(a1, a2, a3)
            return table.arange(function(i)
                if a2[i - 1] >= 0 then
                    return table.arange(function(j)
                        return a1[i - 1][j - 1]
                    end, a2[i - 1])
                else
                    return None
                end
            end, a3)
        end
        local arg_shapes = make(arg_shape_data, arg_shape_ndim, arg_shape_size.value)
        local out_shapes = make(out_shape_data, out_shape_ndim, out_shape_size.value)
        local aux_shapes = make(aux_shape_data, aux_shape_ndim, aux_shape_size.value)
        return { arg_shapes, out_shapes, aux_shapes }
    else
        return { None, None, None }
    end
end

function Symbol:debug_str()
    local debug_str = ctypes.c_char_p()
    check_call(_LIB.MXSymbolPrint(
            self.handle, ctypes.byref(debug_str)))
    return py_str(debug_str.value)
end

function Symbol:save(fname, remove_amp_cast)
    remove_amp_cast = default(remove_amp_cast, true)
    if type(fname) ~= 'string' then
        raise('TypeError', 'fname need to be string')
    end
    if remove_amp_cast then
        local handle = SymbolHandle()
        check_call(_LIB.MXSymbolRemoveAmpCast(self.handle, ctypes.byref(handle)))
        check_call(_LIB.MXSymbolSaveToFile(handle, c_str(fname)))
    else
        check_call(_LIB.MXSymbolSaveToFile(self.handle, c_str(fname)))
    end
end

function Symbol:tojson()
    local json_str = ctypes.c_char_p()
    check_call(_LIB.MXSymbolSaveToJSON(self.handle, ctypes.byref(json_str)))
    return py_str(json_str.value)
end

function Symbol._get_ndarray_inputs(arg_key, args, arg_names, allow_missing)
    local arg_handles = {}
    local arg_arrays = {}
    if #args > 0 then
        if #args ~= #arg_names then
            raise('ValueError', ('Length of %s does not match the number of arguments'):format(arg_key))
        end
        for _, narr in ipairs(args) do
            if isnone(narr) and allow_missing then
                table.insert(arg_handles, None)
            elseif not isinstance(narr, NDArray) then
                raise('TypeError', 'Only accept list of NDArrays or dict of str to NDArray')
            else
                table.insert(arg_handles, narr.handle)
            end
        end
        arg_arrays = args
    else
        for _, name in ipairs(arg_names) do
            if args[name] then
                local narr = args[name]
                if not isinstance(narr, NDArray) then
                    raise('TypeError', 'Only accept list of NDArrays or dict of str to NDArray')
                end
                table.insert(arg_handles, narr.handle)
                table.insert(arg_arrays, narr.handle)
            else
                if allow_missing then
                    table.insert(arg_handles, None)
                    table.insert(arg_arrays, None)
                else
                    raise('ValueError', ('key %q is missing in %q'):format(name, arg_key))
                end
            end
        end
    end
    return c_array(NDArrayHandle, arg_handles), arg_arrays
end

function Symbol:_gen_atomic_symbol()
    local handle = SymbolHandle()
    check_call(_LIB.MXGenAtomicSymbolFromSymbol(self.handle, ctypes.byref(handle)))
    return Symbol(handle)
end

function Symbol:simple_bind(ctx, grad_req, type_dict, stype_dict,
                            group2ctx, shared_arg_names, shared_exec,
                            shared_buffer, kwargs)
    grad_req, kwargs = default(grad_req, 'write', kwargs, {})

    -- data types
    local num_provided_arg_types = 0
    local provided_arg_type_names = ctypes.POINTER(ctypes.c_char_p)()  -- provided type argument names
    local provided_arg_type_data = ctypes.POINTER(mx_uint)()  -- provided types
    if not isnone(type_dict) then
        provided_arg_type_names = {}
        provided_arg_type_data = {}
        for k, v in pairs(type_dict) do
            if _DTYPE_NP_TO_MX[v] then
                table.insert(provided_arg_type_names, k)
                table.insert(provided_arg_type_data, _DTYPE_NP_TO_MX[v])
            end
        end
        num_provided_arg_types = mx_uint(#provided_arg_type_names)
        provided_arg_type_names = c_str_array(provided_arg_type_names)
        provided_arg_type_data = c_array_buf(ctypes.c_int, provided_arg_type_data)
    end

    -- storage types
    local num_provided_arg_stypes = 0
    -- provided storage type argument names
    local provided_arg_stype_names = ctypes.POINTER(ctypes.c_char_p)()
    local provided_arg_stype_data = ctypes.POINTER(mx_uint)()  -- provided storage types
    if not isnone(stype_dict) then
        provided_arg_stype_names = {}
        provided_arg_stype_data = {}
        for k, v in pairs(stype_dict) do
            if _STORAGE_TYPE_STR_TO_ID[v] then
                table.insert(provided_arg_stype_names, k)
                table.insert(provided_arg_stype_data, _STORAGE_TYPE_STR_TO_ID[v])
            end
        end
        num_provided_arg_stypes = mx_uint(#provided_arg_stype_names)
        provided_arg_stype_names = c_str_array(provided_arg_stype_names)
        provided_arg_stype_data = c_array_buf(ctypes.c_int, provided_arg_stype_data)
    end

    local provided_arg_shape_data = {}  -- shape data
    -- argument shape index in sdata,
    -- e.g. [sdata[indptr[0]], sdata[indptr[1]]) is the shape of the first arg
    local provided_arg_shape_idx = { 0 }
    local provided_arg_shape_names = {}  -- provided argument names
    for k, v in pairs(kwargs) do
        -- if k not in listed_arguments and k not in listed_aux_states:
        --   raise ValueError('arg name %s is not valid', k)
        if islist(v) then
            table.insert(provided_arg_shape_names, k)
            table.append(provided_arg_shape_data, v)
            table.insert(provided_arg_shape_idx, #provided_arg_shape_data)
        end
    end

    local provided_req_type_list_len = 0
    local provided_grad_req_types = ctypes.POINTER(ctypes.c_char_p)()
    local provided_grad_req_names = ctypes.POINTER(ctypes.c_char_p)()
    if not isnone(grad_req) then
        if type(grad_req) == 'string' then
            -- use provided_req_type_list_len = 0 to indicate this situation
            provided_req_type_list_len = 0
            provided_grad_req_types = { grad_req }
        elseif #grad_req > 0 then
            provided_grad_req_types = grad_req
            provided_req_type_list_len = #grad_req
        else
            if table.empty(grad_req) then
                raise('RuntimeError', 'grad_req in simple_bind cannot be empty')
            end
            provided_grad_req_names = {}
            provided_grad_req_types = {}
            for k, v in pairs(grad_req) do
                table.insert(provided_grad_req_names, k)
                table.append(provided_grad_req_types, v)
            end
            provided_grad_req_names = c_str_array(provided_grad_req_names)
            provided_req_type_list_len = #provided_grad_req_types
        end
        provided_grad_req_types = c_str_array(provided_grad_req_types)
    end

    local num_ctx_map_keys = mx_uint(0)
    local ctx_map_keys = ctypes.POINTER(ctypes.c_char_p)()
    local ctx_map_dev_types = ctypes.POINTER(ctypes.c_int)()
    local ctx_map_dev_ids = ctypes.POINTER(ctypes.c_int)()
    if not isnone(group2ctx) then
        ctx_map_keys = {}
        ctx_map_dev_types = {}
        ctx_map_dev_ids = {}
        for k, v in pairs(group2ctx) do
            table.insert(ctx_map_keys, k)
            table.insert(ctx_map_dev_types, v.device_typeid)
            table.insert(ctx_map_dev_ids, v.device_id)
        end
        num_ctx_map_keys = mx_uint(#ctx_map_keys)
        ctx_map_keys = c_str_array(ctx_map_keys)
        ctx_map_dev_types = c_array(ctypes.c_int, ctx_map_dev_types)
        ctx_map_dev_ids = c_array(ctypes.c_int, ctx_map_dev_ids)
    end

    -- prepare param names
    local shared_arg_name_list = {}
    if not isnone(shared_arg_names) then
        if not islist(shared_arg_names) then
            raise('ValueError', 'shared_arg_names in simple_bind must be a list or None')
        end
        shared_arg_name_list = shared_arg_names
    end

    -- prepare shared_buffer
    local shared_buffer_len, shared_buffer_names, shared_buffer_handles
    if isnone(shared_buffer) then
        shared_buffer_len = ctypes.c_int(-1)
        shared_buffer_names = ctypes.POINTER(ctypes.c_char_p)()
        shared_buffer_handles = ctypes.POINTER(NDArrayHandle)()
    else
        if type(shared_buffer) ~= 'table' then
            raise('ValueError', 'shared_buffer in simple_bind must be dict or None')
        end
        local buffer_names, buffer_arrays = table.kv(shared_buffer)
        for _, v in ipairs(buffer_arrays) do
            assert(v.stype == 'default',
                   'shared_buffer is expected to only contain NDArrays with default storage')
        end
        shared_buffer_names = c_str_array(buffer_names)
        shared_buffer_len = ctypes.c_int(#buffer_arrays)
        shared_buffer_handles = c_handle_array(buffer_arrays)
    end
    local updated_shared_buffer_names = ctypes.POINTER(ctypes.c_char_p)()
    local updated_shared_buffer_handles = ctypes.POINTER(NDArrayHandle)()

    -- prepare shared_exec_handle
    local shared_exec_handle = isnone(shared_exec) and ExecutorHandle() or shared_exec.handle

    -- prepare current executor handle
    local exe_handle = ExecutorHandle()

    -- prepare current executor's in_args, arg_grads, and aux_states
    local num_in_args = ctypes.c_uint()
    local in_arg_handles = ctypes.POINTER(NDArrayHandle)()
    local arg_grad_handles = ctypes.POINTER(NDArrayHandle)()
    local num_aux_states = ctypes.c_uint()
    local aux_state_handles = ctypes.POINTER(NDArrayHandle)()

    local try = function()
        check_call(_LIB.MXExecutorSimpleBindEx(
                self.handle,
                ctypes.c_int(ctx.device_typeid),
                ctypes.c_int(ctx.device_id),
                num_ctx_map_keys,
                ctx_map_keys,
                ctx_map_dev_types,
                ctx_map_dev_ids,
                mx_uint(provided_req_type_list_len),
                provided_grad_req_names,
                provided_grad_req_types,
                mx_uint(#provided_arg_shape_names),
                c_str_array(provided_arg_shape_names),
                c_array_buf(mx_int,
                            provided_arg_shape_data),
                c_array_buf(mx_uint,
                            provided_arg_shape_idx),
                num_provided_arg_types,
                provided_arg_type_names,
                provided_arg_type_data,
                num_provided_arg_stypes,
                provided_arg_stype_names,
                provided_arg_stype_data,
                mx_uint(len(shared_arg_name_list)),
                c_str_array(shared_arg_name_list),
                ctypes.byref(shared_buffer_len),
                shared_buffer_names,
                shared_buffer_handles,
                ctypes.byref(updated_shared_buffer_names),
                ctypes.byref(updated_shared_buffer_handles),
                ctypes.byref(num_in_args),
                ctypes.byref(in_arg_handles),
                ctypes.byref(arg_grad_handles),
                ctypes.byref(num_aux_states),
                ctypes.byref(aux_state_handles),
                shared_exec_handle,
                ctypes.byref(exe_handle)))
    end
    local ok, ret = pcall(try)
    if not ok then
        local error_msg = ret .. "\nsimple_bind error. Arguments:\n"
        for k, v in pairs(kwargs) do
            error_msg = error_msg .. ('%s: %s\n'):format(k, tostring(v))
        end
        raise('RuntimeError', error_msg)
    end

    -- update shared_buffer
    if not isnone(shared_buffer) then
        for i = 1, shared_buffer_len.value do
            local k = py_str(updated_shared_buffer_names[i - 1])
            local v = NDArray(NDArrayHandle(updated_shared_buffer_handles[i - 1]))
            shared_buffer[k] = v
        end
    end

    -- create in_args, arg_grads, and aux_states for the current executor
    local arg_arrays = table.arange(function(i)
        return _ndarray_cls(NDArrayHandle(in_arg_handles[i - 1]))
    end, num_in_args.value)
    local grad_arrays = table.arange(function(i)
        local hdl = arg_grad_handles[i - 1]
        if ffi.isnullptr(hdl) then
            return None
        else
            return _ndarray_cls(NDArrayHandle(hdl))
        end
    end, num_in_args.value)
    local aux_arrays = table.arange(function(i)
        return _ndarray_cls(NDArrayHandle(aux_state_handles[i - 1]))
    end, num_aux_states.value)

    local executor = Executor(exe_handle, self, ctx, grad_req, group2ctx)
    executor.arg_arrays = arg_arrays
    executor.grad_arrays = grad_arrays
    executor.aux_arrays = aux_arrays
    return executor
end

function Symbol:bind(ctx, args, args_grad, grad_req, aux_states, group2ctx, shared_exec)
    grad_req = default(grad_req, 'write')

    if not isinstance(ctx, Context) then
        raise('TypeError', 'Context type error')
    end

    local listed_arguments = self:list_arguments()
    local args_handle, args_grad_handle
    args_handle, args = self:_get_ndarray_inputs('args', args, listed_arguments, false)
    -- setup args gradient
    if isnone(args_grad) then
        args_grad_handle = c_array(NDArrayHandle, table.replicate(None, #args))
    else
        args_grad_handle, args_grad = self:_get_ndarray_inputs(
                'args_grad', args_grad, listed_arguments, true)
    end

    if isnone(aux_states) then
        aux_states = {}
    end
    local aux_args_handle
    aux_args_handle, aux_states = self:_get_ndarray_inputs(
            'aux_states', aux_states, self:list_auxiliary_states(), false)

    -- setup requirements
    local reqs_array
    if type(grad_req) == 'string' then
        if not _GRAD_REQ_MAP[grad_req] then
            raise('ValueError', 'invalid grad_req: ' .. grad_req)
        end
        reqs_array = c_array_buf(mx_uint,
                                 table.replicate(_GRAD_REQ_MAP[grad_req], len(listed_arguments)))
    elseif #grad_req > 0 then
        reqs_array = c_array_buf(mx_uint,
                                 table.map(grad_req, function(v)
                                     return _GRAD_REQ_MAP[v]
                                 end))
    else
        local req_array = {}
        for _, name in ipairs(listed_arguments) do
            if grad_req[name] then
                table.insert(req_array, _GRAD_REQ_MAP[grad_req[name]])
            else
                table.insert(req_array, 0)
            end
        end
        reqs_array = c_array_buf(mx_uint, req_array)
    end

    local ctx_map_keys = {}
    local ctx_map_dev_types = {}
    local ctx_map_dev_ids = {}

    if group2ctx then
        for k, v in pairs(group2ctx) do
            table.insert(ctx_map_keys, k)
            table.insert(ctx_map_dev_types, v.device_typeid)
            table.insert(ctx_map_dev_ids, v.device_id)
        end
    end

    local handle = ExecutorHandle()
    local shared_handle = isnone(shared_exec) and ExecutorHandle() or shared_exec.handle
    check_call(_LIB.MXExecutorBindEX(self.handle,
                                     ctypes.c_int(ctx.device_typeid),
                                     ctypes.c_int(ctx.device_id),
                                     mx_uint(len(ctx_map_keys)),
                                     c_str_array(ctx_map_keys),
                                     c_array_buf(ctypes.c_int, ctx_map_dev_types),
                                     c_array_buf(ctypes.c_int, ctx_map_dev_ids),
                                     mx_uint(len(args)),
                                     args_handle,
                                     args_grad_handle,
                                     reqs_array,
                                     mx_uint(len(aux_states)),
                                     aux_args_handle,
                                     shared_handle,
                                     ctypes.byref(handle)))
    local executor = Executor(handle, self, ctx, grad_req, group2ctx)
    executor.arg_arrays = args
    executor.grad_arrays = args_grad
    executor.aux_arrays = aux_states
    return executor
end

function Symbol:gradient(wrt)
    local handle = SymbolHandle()
    local c_wrt = c_str_array(wrt)
    check_call(_LIB.MXSymbolGrad(self.handle,
                                 mx_uint(#wrt),
                                 c_wrt,
                                 ctypes.byref(handle)))
    return Symbol(handle)
end

function Symbol:eval(ctx, kwargs)
    if isnone(ctx) then
        ctx = current_context()
    end
    return self:bind(ctx, kwargs):forward()
end

local function __register()
    Symbol.reshape = op.reshape
    Symbol.reshape_like = op.reshape_like
    Symbol.astype = op.cast
    Symbol.zeros_like = op.zeros_like
    Symbol.ones_like = op.ones_like
    Symbol.broadcast_axes = op.broadcast_axes
    Symbol.repeat_ = op.repeat_
    Symbol.pad = op.pad
    Symbol.swapaxes = op.swapaxes
    Symbol.split = op.split
    Symbol.split_v2 = M.split_v2
    Symbol.slice = op.slice
    Symbol.slice_axis = op.slice_axis
    Symbol.slice_like = op.slice_like
    Symbol.take = op.take
    Symbol.one_hot = op.one_hot
    Symbol.pick = op.pick
    Symbol.sort = op.sort
    Symbol.topk = op.topk
    Symbol.argsort = op.argsort
    Symbol.argmax = op.argmax
    Symbol.argmax_channel = op.argmax_channel
    Symbol.argmin = op.argmin
    Symbol.clip = op.clip
    Symbol.abs = op.abs
    Symbol.sign = op.sign
    Symbol.flatten = op.flatten
    Symbol.shape_array = op.shape_array
    Symbol.size_array = op.size_array
    Symbol.expand_dims = op.expand_dims
    Symbol.broadcast_to = op.broadcast_to
    Symbol.broadcast_like = op.broadcast_like
    Symbol.tile = op.tile
    Symbol.transpose = op.transpose
    Symbol.flip = op.flip
    Symbol.depth_to_space = op.depth_to_space
    Symbol.space_to_depth = op.space_to_depth

    Symbol.sum = op.sum
    Symbol.nansum = op.nansum
    Symbol.prod = op.prod
    Symbol.nanprod = op.nanprod
    Symbol.mean = op.mean
    Symbol.max = op.max
    Symbol.min = op.min
    Symbol.norm = op.norm
    Symbol.round = op.round
    Symbol.rint = op.rint
    Symbol.fix = op.fix
    Symbol.floor = op.floor
    Symbol.ceil = op.ceil
    Symbol.trunc = op.trunc
    Symbol.sin = op.sin
    Symbol.cos = op.cos
    Symbol.tan = op.tan
    Symbol.arcsin = op.arcsin
    Symbol.arccos = op.arccos
    Symbol.arctan = op.arctan
    Symbol.degrees = op.degrees
    Symbol.radians = op.radians
    Symbol.sinh = op.sinh
    Symbol.cosh = op.cosh
    Symbol.tanh = op.tanh
    Symbol.arcsinh = op.arcsinh
    Symbol.arccosh = op.arccosh
    Symbol.arctanh = op.arctanh
    Symbol.exp = op.exp
    Symbol.expm1 = op.expm1
    Symbol.log = op.log
    Symbol.log10 = op.log10
    Symbol.log2 = op.log2
    Symbol.log1p = op.log1p
    Symbol.sqrt = op.sqrt
    Symbol.rsqrt = op.rsqrt
    Symbol.cbrt = op.cbrt
    Symbol.rcbrt = op.rcbrt
    Symbol.square = op.square
    Symbol.reciprocal = op.reciprocal
    Symbol.relu = op.relu
    Symbol.sigmoid = op.sigmoid
    Symbol.softmax = op.softmax
    Symbol.log_softmax = op.log_softmax
    Symbol.softmin = op.softmin
    Symbol.squeeze = op.squeeze
end

function Symbol:diag(k, kwargs)
    return op.diag(self, default(k, 0), kwargs)
end

Symbol.__register = __register

function Symbol:get_backend_symbol(backend)
    local out = SymbolHandle()
    check_call(_LIB.MXGenBackendSubgraph(self.handle, c_str(backend), ctypes.byref(out)))
    return Symbol(out)
end

function Symbol:wait_to_read()
    raise('NotImplementedForSymbol')
end
function Symbol:asnumpy()
    raise('NotImplementedForSymbol')
end
function Symbol:asscalar()
    raise('NotImplementedForSymbol')
end
function Symbol:copy()
    raise('NotImplementedForSymbol')
end
function Symbol:as_in_context()
    raise('NotImplementedForSymbol')
end
function Symbol:detach()
    raise('NotImplementedForSymbol')
end
function Symbol:backward()
    raise('NotImplementedForSymbol')
end

--

local _Symbol_field = {
    name = Symbol._name,
}

function Symbol:__index(k)
    local tk = type(k)
    if tk == 'string' then
        if _Symbol_field[k] then
            return _Symbol_field[k](self)
        elseif not isnone(Symbol[k]) then
            return Symbol[k]
        end
    end
    return self:__getitem(k)
end

--

function M.var(name, attr, shape, lr_mult, wd_mult, dtype,
               init, stype, kwargs)
    kwargs = default(kwargs, {})
    if type(name) ~= 'string' then
        raise('TypeError', 'Expect a string for variable `name`')
    end
    local handle = SymbolHandle()
    check_call(_LIB.MXSymbolCreateVariable(c_str(name), ctypes.byref(handle)))
    local ret = Symbol(handle)
    if not AttrScope._current.value then
        AttrScope._current.value = AttrScope()
    end
    attr = AttrScope._current.value:get(attr)
    if isnone(attr) then
        attr = {}
    end
    local str = base.tostring
    if not isnone(shape) then
        attr['__shape__'] = str(shape)
    end
    if not isnone(lr_mult) then
        attr['__lr_mult__'] = str(lr_mult)
    end
    if not isnone(wd_mult) then
        attr['__wd_mult__'] = str(wd_mult)
    end
    if not isnone(dtype) then
        attr['__dtype__'] = str(_DTYPE_NP_TO_MX[dtype])
    end
    if not isnone(init) then
        if type(init) ~= 'string' then
            init = init:dumps()
        end
        attr['__init__'] = init
    end
    if not isnone(stype) then
        attr['__storage_type__'] = str(_STORAGE_TYPE_STR_TO_ID[stype])
    end
    for k, v in pairs(kwargs) do
        if k:starts_with('__') and k:ends_with('__') then
            attr[k] = str(v)
        else
            raise('ValueError', ('Attribute name=%s is not supported.'):format(k))
        end
    end
    ret:_set_attr(attr)
    return ret
end

-- for back compatibility
M.Variable = M.var

function M.Group(symbols)
    local function check(v)
        return not isinstance(v, Symbol)
    end
    if not symbols or table.any(symbols, check) then
        raise('TypeError', 'Expected a list of symbols as input')
    end
    local handle = SymbolHandle()
    check_call(_LIB.MXSymbolCreateGroup(
            mx_uint(len(symbols)),
            c_handle_array(symbols), ctypes.byref(handle)))
    return Symbol(handle)
end

function M.load(fname)
    if type(fname) ~= 'string' then
        raise('TypeError', 'fname need to be string')
    end
    local handle = SymbolHandle()
    check_call(_LIB.MXSymbolCreateFromFile(c_str(fname), ctypes.byref(handle)))
    return Symbol(handle)
end

function M.load_json(json_str)
    if type(json_str) ~= 'string' then
        raise('TypeError', 'json_str need to be string')
    end
    local handle = SymbolHandle()
    check_call(_LIB.MXSymbolCreateFromJSON(c_str(json_str), ctypes.byref(handle)))
    return Symbol(handle)
end

local function _sym_helper(lhs, rhs, f1, f2, f3, f4)
    local a1 = isinstance(lhs, Symbol)
    local a2 = isinstance(rhs, Symbol)
    local b1 = not a1 and type(lhs) == 'number'
    local b2 = not a2 and type(rhs) == 'number'
    if a1 and a2 then
        return f1(lhs, rhs)
    elseif a1 and b2 then
        return f2(lhs, rhs)
    elseif b1 and a2 then
        return f3(rhs, lhs)
    elseif b1 and b2 then
        return f4(lhs, rhs)
    else
        raise('TypeError', 'type not supported')
    end
end

function M.power(base_, exp)
    return _sym_helper(
            base_, exp, _internal._Power, _internal._PowerScalar, _internal._RPowerScalar,
            function(a, b)
                return a ^ b
            end)
end
M.pow = M.power

function M.maximum(left, right)
    return _sym_helper(
            left, right, _internal._Maximum, _internal._MaximumScalar, _internal._MaximumScalar,
            function(a, b)
                if a > b then
                    return a
                end
                return b
            end)
end

function M.minimum(left, right)
    return _sym_helper(
            left, right, _internal._Minimum, _internal._MinimumScalar, _internal._MinimumScalar,
            function(a, b)
                if a < b then
                    return a
                end
                return b
            end)
end

function M.hypot(left, right)
    return _sym_helper(
            left, right, _internal._Hypot, _internal._HypotScalar, _internal._HypotScalar,
            function(a, b)
                return math.sqrt(a * a + b * b)
            end)
end

local function check_dtype(dtype)
    if isnone(dtype) then
        dtype = 'float32'
    end
    return dtype
end

function M.eye(N, M_, k, dtype, kwargs)
    M_, k = default(M_, 0, k, 0)
    dtype = check_dtype(dtype)
    return _internal._eye(N, M_, k, nil, dtype, nil, nil, nil, kwargs)
end

function M.zeros(shape, dtype, kwargs)
    dtype = check_dtype(dtype)
    return _internal._zeros(shape, nil, dtype, nil, nil, nil, kwargs)
end

function M.ones(shape, dtype, kwargs)
    dtype = check_dtype(dtype)
    return _internal._ones(shape, nil, dtype, nil, nil, nil, kwargs)
end

function M.full(shape, val, dtype, kwargs)
    dtype = check_dtype(dtype)
    return _internal._full(shape, nil, dtype, val, nil, nil, nil, kwargs)
end

function M.arange(start, stop, step, repeat_, infer_range, name, dtype)
    step, repeat_, infer_range = default(step, 1, repeat_, 1, infer_range, false)
    dtype = check_dtype(dtype)
    return _internal._arange(start, stop, step, repeat_, infer_range, nil, dtype, name)
end

function M.linspace(start, stop, num, endpoint, name, dtype)
    endpoint = default(endpoint, true)
    dtype = check_dtype(dtype)
    return _internal._linspace(start, stop, nil, nil, nil, nil, dtype, name, nil, nil, {
        num  = num,
        stop = stop,
    })
end

function M.histogram(a, bins, range, kwargs)
    if isinstance(bins, Symbol) then
        return _internal._histogram(a, bins, nil, nil, nil, nil, nil, kwargs)
    elseif type(bins) == 'number' then
        if isnone(range) then
            raise('ValueError', 'null range is not supported in symbol mode')
        end
        return _internal._histogram(a, nil, bins, range, nil, nil, nil, kwargs)
    end
    raise('ValueError', 'bins argument should be either an integer or an NDArray')
end

function M.split_v2(ary, indices_or_sections, axis, squeeze_axis)
    axis, squeeze_axis = default(axis, 0, squeeze_axis, false)
    local indices = {}
    local sections = 0
    if type(indices_or_sections) == 'number' then
        sections = indices_or_sections
    elseif islist(indices_or_sections) then
        indices = table.sequence({ 0 }, indices_or_sections)
    else
        raise('ValueError', 'indices_or_sections must either int or tuple of ints')
    end
    return _internal._split_v2(ary, indices, axis, squeeze_axis, sections)
end

_set_symbol_class(Symbol)

return M
