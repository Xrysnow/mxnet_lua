---@class mx.symbol:mx.symbol.symbol
local M = {}

M._internal, M.contrib, M.linalg, M.op, M.random, M.sparse, M.image, M.symbol = require('mx.symbol._internal'),
require('mx.symbol.contrib'), require('mx.symbol.linalg'), require('mx.symbol.op'), require('mx.symbol.random'),
require('mx.symbol.sparse'), require('mx.symbol.image'), require('mx.symbol.symbol')

table.merge(M, require('mx.symbol.gen_op'))

M.register = require('mx.symbol.register')

table.merge(M, require('mx.symbol.op'))
table.merge(M, require('mx.symbol.symbol'))

return M
