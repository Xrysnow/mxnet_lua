--
local ctypes = require('ctypes')
local base = require('mx.base')
local _LIB, check_call = base._LIB, base.check_call

--
local M = {}

--- Set size limit on bulk execution.
---
--- Bulk execution bundles many operators to run together.
--- This can improve performance when running a lot of small
--- operators sequentially.
function M.set_bulk_size(size)
    local prev = ctypes.c_int()
    check_call(_LIB.MXEngineSetBulkSize(
            ctypes.c_int(size), ctypes.byref(prev)))
    return prev.value
end

local _BulkScope = class('mx.engine._BulkScope')
function _BulkScope:ctor(size)
    self._size = size
    self._old_size = nil
end
function _BulkScope:__enter()
    self._old_size = M.set_bulk_size(self._size)
    return self
end
function _BulkScope:__exit()
    M.set_bulk_size(self._old_size)
end

--- Bulk execution bundles many operators to run together.
--- This can improve performance when running a lot of small
--- operators sequentially.
---
--- Returns a scope for managing bulk size
function M.bulk(size)
    return _BulkScope(size)
end

return M
