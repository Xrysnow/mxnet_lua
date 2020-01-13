---@class mx.ndarray:mx.ndarray.ndarray
local M = {}

local _internal, contrib, linalg, op, random, sparse, utils, image, ndarray = require('mx.ndarray._internal'), require('mx.ndarray.contrib'), require('mx.ndarray.linalg'), require('mx.ndarray.op'), require('mx.ndarray.random'), require('mx.ndarray.sparse'), require('mx.ndarray.utils'), require('mx.ndarray.image'), require('mx.ndarray.ndarray')
M._internal, M.contrib, M.linalg, M.op, M.random, M.sparse, M.utils, M.image, M.ndarray = _internal, contrib, linalg, op, random, sparse, utils, image, ndarray

M.register = require('mx.ndarray.register')

for k, v in pairs(require('mx.ndarray.op')) do
    M[k] = v
end

for k, v in pairs(require('mx.ndarray.ndarray')) do
    M[k] = v
end
M.NDArray = require('mx.ndarray.ndarray').NDArray

M.load, M.load_frombuffer, M.save, M.zeros, M.empty, M.array = utils.load, utils.load_frombuffer, utils.save, utils.zeros, utils.empty, utils.array
M._ndarray_cls = sparse._ndarray_cls
M._GRAD_REQ_MAP, M._DTYPE_MX_TO_NP, M._DTYPE_NP_TO_MX, M._new_empty_handle = ndarray._GRAD_REQ_MAP, ndarray._DTYPE_MX_TO_NP, ndarray._DTYPE_NP_TO_MX, ndarray._new_empty_handle

return M
