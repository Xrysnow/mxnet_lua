--
local M = {}

local context = require('mx.context')
M.Context, M.current_context, M.cpu, M.gpu, M.cpu_pinned = context.Context, context.current_context, context.cpu, context.gpu, context.cpu_pinned
M.engine = require('mx.engine')
M.MXNetError = require('mx.base').MXNetError
local util = require('mx.util')
M.is_np_shape, M.set_np_shape, M.np_shape, M.use_np_shape = util.is_np_shape, util.set_np_shape, util.np_shape, util.use_np_shape
M.base = require('mx.base')
--M.contrib = require('mx.contrib.__init__')
M.ndarray = require('mx.ndarray.__init__')
M.nd = require('mx.ndarray.__init__')
M.base = require('mx.base')
M.name = require('mx.name')

M.symbol = require('mx.symbol.__init__')
M.sym = require('mx.symbol.__init__')
M.symbol_doc = require('mx.symbol_doc')

M.io = require('mx.io.__init__')
--M.recordio = require('mx.recordio')
M.operator = require('mx.operator')
M.random = require('mx.random')
M.rnd = require('mx.random')
--M.optimizer = require('mx.optimizer.__init__')
--M.model = require('mx.model')
--M.metric = require('mx.metric')
--M.notebook = require('mx.notebook.__init__')
M.initializer = require('mx.initializer')
M.init = require('mx.initializer')
--M.visualization = require('mx.visualization')
--M.viz = require('mx.visualization')
--M.callback = require('mx.callback')
M.lr_scheduler = require('mx.lr_scheduler')
--M.kv = require('mx.kvstore')
--M.rtc = require('mx.rtc')
--M.AttrScope = require('mx.attribute').AttrScope

--M.monitor = require('mx.monitor')
--M.mon = require('mx.monitor')

--M.torch = require('mx.torch')
--M.th = require('mx.torch')

--M.profiler = require('mx.profiler')
--M.log = require('mx.log')

--M.module = require('mx.module.__init__')
--M.mod = require('mx.module.__init__')

--M.image = require('mx.image.__init__')
--M.img = require('mx.image.__init__')

--M.test_utils = require('mx.test_utils')

--M.rnn = require('mx.rnn.__init__')

--M.gluon = require('mx.gluon.__init__')

--M.kvstore_server = require('mx.kvstore_server')

return M
