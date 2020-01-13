--
local ctypes = require('ctypes')
local _build_doc = require('mx.ndarray_doc')._build_doc
local base = require('mx.base')
local mx_uint, check_call, _LIB, py_str, _init_op_module, _Null = base.mx_uint, base.check_call, base._LIB, base.py_str, base._init_op_module, base._Null
local append = table.insert

--
local M = {}

---@language Lua
local code_arr1 = [[

    local ndargs = {}
    for _, v in ipairs(arr_name) do
        assert(isinstance(v, NDArrayBase), 'Positional arguments must have NDArray type')
        table.insert(ndargs, v)
    end]]
---@language Lua
local code_arr2 = [[

    -- check dtype
    --if kwargs[dtype_name] then
    --end]]
---@language Lua
local code_arr3 = [[

    local _ = arg_pop(kwargs, 'name', None)
    local out = arg_pop(kwargs, 'out', None)
    local keys, vals = table.kv(kwargs)]]
---@language Lua
local code_1 = [[

    local ndargs = {}
    local keys, vals = table.kv(kwargs)]]
---@language Lua
local code_2 = [[

    if not isnone(_name_) then
        assert(isinstance(_name_, NDArrayBase), 'Argument _name_ must have NDArray type')
        table.insert(ndargs, _name_)
    end]]
---@language Lua
local code_3 = [[

    if _name_ ~= _Null then
        table.insert(keys, '_name_key_')
        table.insert(vals, _name_)
    end]]
---@language Lua
local code_null_def = [[

    for _, v in ipairs(_null_def_) do
        kwargs[v] = default(kwargs[v], _Null)
    end]]

function M._generate_ndarray_function_code(handle, name, func_name, signature_only, no_doc)
    signature_only = default(signature_only, false)
    local real_name = ctypes.c_char_p()
    local desc = ctypes.c_char_p()
    local num_args = mx_uint()
    local arg_names = ctypes.POINTER(ctypes.c_char_p)()
    local arg_types = ctypes.POINTER(ctypes.c_char_p)()
    local arg_descs = ctypes.POINTER(ctypes.c_char_p)()
    local key_var_num_args = ctypes.c_char_p()
    local ret_type = ctypes.c_char_p()
    check_call(_LIB.MXSymbolGetAtomicSymbolInfo(
            handle, ctypes.byref(real_name), ctypes.byref(desc),
            ctypes.byref(num_args),
            ctypes.byref(arg_names),
            ctypes.byref(arg_types),
            ctypes.byref(arg_descs),
            ctypes.byref(key_var_num_args),
            ctypes.byref(ret_type)))
    local narg = tonumber(num_args.value)
    local arg_names_, arg_types_, arg_descs_ = {}, {}, {}
    for i = 0, narg - 1 do
        append(arg_names_, py_str(arg_names[i]))
        append(arg_types_, py_str(arg_types[i]))
        append(arg_descs_, py_str(arg_descs[i]))
    end
    arg_names, arg_types, arg_descs = arg_names_, arg_types_, arg_descs_
    key_var_num_args = py_str(key_var_num_args.value)
    if ffi.isnullptr(ret_type.value) then
        ret_type = ''
    else
        ret_type = py_str(ret_type.value)
    end
    local doc_str = _build_doc(name,
                               py_str(desc.value),
                               arg_names,
                               arg_types,
                               arg_descs,
                               key_var_num_args,
                               ret_type)
    local dtype_name
    local arr_name
    local ndsignature = {}
    local signature = {}
    local ndarg_names = {}
    local kwarg_names = {}
    for i = 1, narg do
        local name, atype = arg_names[i], arg_types[i]
        if name == 'dtype' then
            dtype_name = name
            append(signature, name)-- name=_Null
        elseif string.starts_with(atype, 'NDArray') or string.starts_with(atype, 'Symbol') then
            assert(not arr_name,
                   'Op can only have one argument with variable size and it must be the last argument.')
            if string.ends_with(atype, '[]') then
                append(ndsignature, name)-- *name
                arr_name = name
            else
                append(ndsignature, name)-- name=None
                append(ndarg_names, name)
            end
        else
            append(signature, name)-- name=_Null
            append(kwarg_names, name)
        end
    end
    local null_def = table.clone(signature)
    append(signature, 'out')
    append(signature, 'name')
    append(signature, 'kwargs')
    signature = table.sequence(ndsignature, signature)

    local code = {}
    if arr_name then
        append(code, ('function M.%s(...)'):format(func_name))-- *arr_name, **kwargs
        if not signature_only then
            append(code, (('\n    local arr_name, kwargs = arg_kw(...)'):gsub('arr_name', arr_name)))
            append(code, (code_arr1:gsub('arr_name', arr_name)))
            if dtype_name ~= nil then
                append(code, (code_arr2:gsub('dtype_name', dtype_name)))
            end
            append(code, code_arr3)
        end
    else
        for i = 1, #signature do
            if string.is_keyword(signature[i]) then
                signature[i] = signature[i] .. '_'
            end
        end
        append(code, ('function M.%s(%s)'):format(func_name, table.concat(signature, ', ')))
        if not signature_only then
            append(code, '\n    kwargs = kwargs or {}')
            append(code, code_1)
            -- NDArray args
            for _, _name_ in ipairs(ndarg_names) do
                if string.is_keyword(_name_) then
                    _name_ = _name_ .. '_'
                end
                append(code, (code_2:gsub('_name_', _name_)))
            end
            -- kwargs
            for _, _name_ in ipairs(kwarg_names) do
                local _name_key_ = _name_
                if string.is_keyword(_name_) then
                    _name_ = _name_ .. '_'
                end
                append(code, (code_3:gsub('_name_key_', _name_key_):gsub('_name_', _name_)))
            end
            -- dtype
            if dtype_name ~= nil then
                local _name_key_ = dtype_name
                if string.is_keyword(dtype_name) then
                    dtype_name = dtype_name .. '_'
                end
                append(code, (code_3:gsub('_name_key_', _name_key_):gsub('_name_', dtype_name)))
            end
        end
    end
    if not signature_only then
        local addr = ffi.cast('uint64_t', handle.value)
        append(code, ('\n    return _imperative_invoke(%s, ndargs, keys, vals, out)\nend\n'):format(tostring(addr)))
    else
        append(code, '\nend\n')
        -- only write doc when signature only
        if not no_doc then
            table.insert(code, 1, doc_str)
        end
    end
    if not signature_only and #null_def > 0 then
        for i = 1, #null_def do
            if string.is_keyword(null_def[i]) then
                null_def[i] = null_def[i] .. '_'
            end
        end
        if arr_name then
            for i = 1, #null_def do
                null_def[i] = ('%q'):format(null_def[i])
            end
            local null_def_t = ('{ %s }'):format(table.concat(null_def, ', '))
            local null_def_str = code_null_def:gsub('_null_def_', null_def_t)
            table.insert(code, 3, null_def_str)
        else
            local null_def_str = ('\n    %s = default(%s)'):format(
                    table.concat(null_def, ', '),
                    table.concat(null_def, ', _Null, ') .. ', _Null')
            table.insert(code, 3, null_def_str)
        end
    end
    return table.concat(code, ''), doc_str
end

function M._make_ndarray_function(handle, name, func_name)
    local code, doc_str = M._generate_ndarray_function_code(handle, name, func_name)
    local header = {
        [[local NDArrayBase = require('mx.ndarray._internal').NDArrayBase]],
        [[local _imperative_invoke = require('mx.ndarray._internal')._imperative_invoke]],
        [[local _Null = require('mx.base')._Null]],
        [[local M = {}]],
    }
    local footer = { [[return M]], '' }
    code = string.format(
            '%s\n%s\n%s',
            table.concat(header, '\n'),
            code,
            table.concat(footer, '\n'))
    local env = setmetatable({}, { __index = _G })
    local f, msg = load(code, func_name, nil, env)
    if f == nil then
        raise(msg)
    end
    return f()[func_name]
end

print('start register mx.ndarray')
_init_op_module('mx', 'ndarray', M._make_ndarray_function)
print('finish register mx.ndarray')
-- update
require('mx.ndarray.ndarray').NDArray.__register()

return M
