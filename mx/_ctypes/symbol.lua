local ctypes = require('ctypes')
local base = require('mx.base')
local _LIB = base._LIB
local c_str_array = base.c_str_array
local c_handle_array = base.c_handle_array
local c_str = base.c_str
local mx_uint = base.mx_uint
local SymbolHandle = base.SymbolHandle
local check_call = base.check_call

--
local M = {}

local _symbol_cls

---@class mx._ctypes.symbol.SymbolBase
local SymbolBase = class('mx._ctypes.symbol.SymbolBase')
M.SymbolBase = SymbolBase

function SymbolBase:ctor(handle)
    self.handle = handle
end

function SymbolBase:dtor()
    check_call(_LIB.NNSymbolFree(self.handle))
end

function SymbolBase:_compose(...)
    local args, kwargs = arg_kw(...)
    local name = arg_pop(kwargs, 'name', None)
    if name then
        name = c_str(name)
    end
    if #args ~= 0 and not table.empty(kwargs) then
        raise('TypeError', 'compose only accept input Symbols either as positional or keyword arguments, not both')
    end
    for _, arg in ipairs(args) do
        if not isinstance(arg, SymbolBase) then
            raise('TypeError', 'Compose expect `Symbol` as arguments')
        end
    end
    for _, val in ipairs(table.values(kwargs)) do
        if not isinstance(val, SymbolBase) then
            raise('TypeError', 'Compose expect `Symbol` as arguments')
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

function SymbolBase:_set_attr(kwargs)
    local keys = c_str_array(table.keys(kwargs))
    local vals = c_str_array(table.map(table.values(kwargs), base.tostring))
    local num_args = mx_uint(table.len(kwargs))
    check_call(_LIB.MXSymbolSetAttrs(
            self.handle, num_args, keys, vals))
end

function SymbolBase:_set_handle(handle)
    self.handle = handle
end

function M._set_symbol_class(cls)
    _symbol_cls = cls
end

function M._symbol_creator(handle, args, kwargs, keys, vals, name)
    args, kwargs = default(args, {}, kwargs, {})
    local sym_handle = SymbolHandle()
    check_call(_LIB.MXSymbolCreateAtomicSymbol(
            ctypes.c_void_p(handle),
            mx_uint(len(keys)),
            c_str_array(keys),
            c_str_array(table.map(vals, base.tostring)),
            ctypes.byref(sym_handle)))

    if not table.empty(args) and not table.empty(kwargs) then
        raise('TypeError', 'Operators with variable length input can only accept input Symbols either as positional or keyword arguments, not both')
    end
    local s = _symbol_cls(sym_handle)
    if not table.empty(args) then
        s:_compose(arg_make(args, { name = name }))
    elseif not table.empty(kwargs) then
        kwargs.name = name
        s:_compose(kw(kwargs))
    else
        s:_compose(kw { name = name })
    end
    return s
end

return M
