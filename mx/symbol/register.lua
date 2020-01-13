--
local ctypes = require('ctypes')
local _build_doc = require('mx.symbol_doc')._build_doc
local base = require('mx.base')
local mx_uint, check_call, _LIB, py_str, _init_op_module, _Null = base.mx_uint, base.check_call, base._LIB, base.py_str, base._init_op_module, base._Null
local append = table.insert

--
local M = {}

---@language Lua
local code_arr1 = [[

    local sym_args = {}
    for _, v in ipairs(arr_name) do
        assert(isinstance(v, SymbolBase), 'Positional arguments must be Symbol instances')
        table.insert(sym_args, v)
    end]]
---@language Lua
local code_arr2 = [[

    -- check dtype
    --if kwargs[dtype_name] then
    --end]]
---@language Lua
local code_arr3 = [[

    local attr = arg_pop(kwargs, 'attr', None)
    if not AttrScope._current.value then
        AttrScope._current.value = AttrScope()
    end
    table.append(kwargs, AttrScope._current.value:get(attr))
    local name = arg_pop(kwargs, 'name', None)
    if not NameManager._current.value then
        NameManager._current.value = NameManager()
    end
    name = NameManager._current.value:get(name, '_func_name_')
    local _ = arg_pop(kwargs, 'out', None)
    local keys, vals, sym_kwargs = {}, {}, {}
    for k, v in pairs(kwargs) do
        if isinstance(v, SymbolBase) then
            sym_kwargs[k] = v
        else
            table.insert(keys, k)
            table.insert(vals, v)
        end
    end]]
---@language Lua
local code_arr4 = [[

    if not kwargs['_key_var_num_args_'] then
        table.insert(keys, '_key_var_num_args_')
        table.insert(vals, #sym_args + table.len(sym_kwargs))
    end]]

---@language Lua
local code_arr5 = [[

    return _symbol_creator(_handle_, sym_args, sym_kwargs, keys, vals, name)]]

---@language Lua
local code_1 = [[

    if not AttrScope._current.value then
        AttrScope._current.value = AttrScope()
    end
    table.append(kwargs, AttrScope._current.value:get(attr))
    local sym_kwargs = {}
    local _keys, _vals = {}, {}
    for k, v in pairs(kwargs) do
        if isinstance(v, SymbolBase) then
            sym_kwargs[k] = v
        else
            table.insert(_keys, k)
            table.insert(_vals, v)
        end
    end]]
---@language Lua
local code_2 = [[

    if not isnone(_name_) then
        assert(isinstance(_name_, SymbolBase), 'Argument _name_ must be Symbol instances')
        sym_kwargs['_name_'] = _name_
    end]]
---@language Lua
local code_3 = [[

    if _name_ ~= _Null then
        table.insert(_keys, '_namekey_')
        table.insert(_vals, _name_)
    end]]
---@language Lua
local code_4 = [[

    if _name_ ~= _Null then
        table.insert(_keys, '_namekey_')
        table.insert(_vals, _name_)
    end]]
---@language Lua
local code_5 = [[

    if not NameManager._current.value then
        NameManager._current.value = NameManager()
    end
    name = NameManager._current.value:get(name, '_func_name_')
    return _symbol_creator(_handle_, None, sym_kwargs, _keys, _vals, name)]]
---@language Lua
local code_null_def = [[

    for _, v in ipairs(_null_def_) do
        kwargs[v] = default(kwargs[v], _Null)
    end]]

function M._generate_symbol_function_code(handle, name, func_name, signature_only, no_doc)
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
    append(signature, 'name')
    append(signature, 'attr')
    append(signature, 'out')
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
            append(code, (code_arr3:gsub('_func_name_', func_name:lower())))
            if key_var_num_args ~= '' then
                append(code, (code_arr4:gsub('_key_var_num_args_', key_var_num_args)))
            end
            local addr = ffi.cast('uint64_t', handle.value)
            append(code, (code_arr5:gsub('_handle_', tostring(addr))))
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
                append(code, (code_3:gsub('_namekey_', _name_key_):gsub('_name_', _name_)))
            end
            -- dtype
            if dtype_name ~= nil then
                local _name_key_ = dtype_name
                if string.is_keyword(dtype_name) then
                    dtype_name = dtype_name .. '_'
                end
                append(code, (code_4:gsub('_namekey_', _name_key_):gsub('_name_', dtype_name)))
            end
            local addr = ffi.cast('uint64_t', handle.value)
            append(code, (code_5:gsub('_func_name_', func_name):gsub('_handle_', tostring(addr))))
        end
    end
    append(code, '\nend\n')
    if signature_only then
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

function M._make_symbol_function(handle, name, func_name)
    local code, doc_str = M._generate_symbol_function_code(handle, name, func_name)
    local header = {
        [[local SymbolBase = require('mx._ctypes.symbol').SymbolBase]],
        [[local _symbol_creator = require('mx._ctypes.symbol')._symbol_creator]],
        [[local NameManager = require('mx.name').NameManager]],
        [[local AttrScope = require('mx.attribute').AttrScope]],
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

print('start register mx.symbol')
_init_op_module('mx', 'symbol', M._make_symbol_function)
print('finish register mx.symbol')
-- update
require('mx.symbol.symbol').Symbol.__register()

return M
