--
local ctypes = require('ctypes')
local _LIB, check_call = require('mx.base')._LIB, require('mx.base').check_call
local Context = require('mx.context').Context

--
local M = {}

--- Seeds the random number generators in MXNet.
---
--- This affects the behavior of modules in MXNet that uses random number generators,
--- like the dropout operator and `NDArray`'s random sampling operators.
function M.seed(seed_state, ctx)
    ctx = ctx or 'all'
    assert(type(seed_state) == 'number')
    seed_state = ctypes.c_int(seed_state)
    if ctx == 'all' then
        check_call(_LIB.MXRandomSeed(seed_state))
    else
        ctx = Context(ctx)
        check_call(_LIB.MXRandomSeedContext(seed_state, ctx.device_typeid, ctx.device_id))
    end
end

return M
