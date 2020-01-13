--
local ctypes = require('ctypes')
local base = require('mx.base')
local _LIB, check_call = base._LIB, base.check_call

--
local M = {}

---@class mx.Context
local Context = class('mx.Context')
M.Context = Context

Context._default_ctx = {}
Context.devtype2str = { [1] = 'cpu', [2] = 'gpu', [3] = 'cpu_pinned', [5] = 'cpu_shared' }
Context.devstr2type = { cpu = 1, gpu = 2, cpu_pinned = 3, cpu_shared = 5 }

function Context:ctor(device_type, device_id)
    device_id = device_id or 0
    if isinstance(device_type, Context) then
        self.device_typeid = device_type.device_typeid
        self.device_id = device_type.device_id
    else
        self.device_typeid = assert(Context.devstr2type[device_type])
        self.device_id = device_id
    end
    self._old_ctx = nil
end

--- Returns the device type of current context.
function Context:device_type()
    return Context.devtype2str[self.device_typeid]
end

function Context:__eq(other)
    return isinstance(other, Context) and
            self.device_typeid == other.device_typeid and
            self.device_id == other.device_id
end

function Context:__tostring()
    return ('%s(%d)'):format(self:device_type(), self.device_id)
end

function Context:__enter()
    if not Context._default_ctx.value then
        Context._default_ctx.value = Context('cpu', 0)
    end
    self._old_ctx = Context._default_ctx.value
    Context._default_ctx.value = self
    return self
end

function Context:__exit()
    Context._default_ctx.value = self._old_ctx
end

--- Empties the memory cache for the current contexts device.
--- MXNet utilizes a memory pool to avoid excessive allocations.
--- Calling empty_cache will empty the memory pool of the contexts
--- device. This will only free the memory of the unreferenced data.
function Context:empty_cache()
    local dev_type = ctypes.c_int(self.device_typeid)
    local dev_id = ctypes.c_int(self.device_id)
    check_call(_LIB.MXStorageEmptyCache(dev_type, dev_id))
end

-- initialize the default context in Context
Context._default_ctx.value = Context('cpu', 0)

function M.cpu(device_id)
    device_id = device_id or 0
    return Context('cpu', device_id)
end

function M.cpu_pinned(device_id)
    device_id = device_id or 0
    return Context('cpu_pinned', device_id)
end

function M.gpu(device_id)
    device_id = device_id or 0
    return Context('gpu', device_id)
end

function M.num_gpus()
    local count = ctypes.c_int()
    check_call(_LIB.MXGetGPUCount(ctypes.byref(count)))
    return count.value
end

function M.gpu_memory_info(device_id)
    device_id = device_id or 0
    local free = ctypes.c_uint64()
    local total = ctypes.c_uint64()
    local dev_id = ctypes.c_int(device_id)
    check_call(_LIB.MXGetGPUMemoryInformation64(dev_id, ctypes.byref(free), ctypes.byref(total)))
    return free.value, total.value
end

function M.current_context()
    if not Context._default_ctx.value then
        Context._default_ctx.value = Context('cpu', 0)
    end
    return Context._default_ctx.value
end

return M
