---@class mx.ndarray._internal:mx.ndarray.gen__internal
local M = {}

local nd = require('mx._ctypes.ndarray')
M.NDArrayBase = nd.NDArrayBase
M.CachedOp = nd.CachedOp
M._set_ndarray_class = nd._set_ndarray_class
M._imperative_invoke = nd._imperative_invoke

M._Null = require('mx.base')._Null

return M
