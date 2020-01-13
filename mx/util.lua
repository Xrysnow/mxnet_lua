--
local M = {}

local ctypes = require('ctypes')
local base = require('mx.base')
local _LIB, check_call = base._LIB, base.check_call

function M.makedirs(d)
    os.execute(([[mkdir "%s"]]):format(d))
end

function M.get_gpu_count()
    local size = ctypes.c_int()
    check_call(_LIB.MXGetGPUCount(ctypes.byref(size)))
    return size.value
end

function M.get_gpu_memory(gpu_dev_id)
    local free_mem = ctypes.c_uint64(0)
    local total_mem = ctypes.c_uint64(0)
    check_call(_LIB.MXGetGPUMemoryInformation64(gpu_dev_id, ctypes.byref(free_mem), ctypes.byref(total_mem)))
    return free_mem.value, total_mem.value
end

function M.set_np_shape(active)
    local prev = ctypes.c_int()
    check_call(_LIB.MXSetIsNumpyShape(ctypes.c_int(active), ctypes.byref(prev)))
    return bool(prev.value)
end

function M.is_np_shape()
    local curr = ctypes.c_bool()
    check_call(_LIB.MXIsNumpyShape(ctypes.byref(curr)))
    return curr.value
end

---@class mx.util._NumpyShapeScope
local _NumpyShapeScope = class('mx.util._NumpyShapeScope')
M._NumpyShapeScope = _NumpyShapeScope

function _NumpyShapeScope:ctor(is_np_shape)
    self._enter_is_np_shape = is_np_shape
    self._prev_is_np_shape = None
end

function _NumpyShapeScope:__enter()
    if not isnone(self._enter_is_np_shape) then
        self._prev_is_np_shape = M.set_np_shape(self._enter_is_np_shape)
    end
end

function _NumpyShapeScope:__exit()
    if not isnone(self._enter_is_np_shape) and self._prev_is_np_shape ~= self._enter_is_np_shape then
        M.set_np_shape(self._prev_is_np_shape)
    end
end

function M.np_shape(active)
    active = default(active, true)
    return _NumpyShapeScope(active)
end

function M.use_np_shape(func)
    return function(...)
        local args = { ... }
        local narg = select('#', ...)
        return with(M.np_shape(true), function()
            return func(unpack(args, narg))
        end)
    end
end

return M
