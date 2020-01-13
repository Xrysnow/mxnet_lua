---@class mx.symbol._internal:mx.symbol.gen__internal
local M = {}

local sym = require('mx._ctypes.symbol')
M.SymbolBase, M._set_symbol_class, M._symbol_creator = sym.SymbolBase, sym._set_symbol_class, sym._symbol_creator

M.AttrScope = require('mx.attribute').AttrScope
M._Null = require('mx.base')._Null
M.NameManager = require('mx.name').NameManager

--table.merge(M, require('mx.symbol.gen__internal'))

return M
