--
local M = {}

---@class mx.name.NameManager
local NameManager = class('mx.name.NameManager')
M.NameManager = NameManager

NameManager._current = {}

function NameManager:ctor()
    self._counter = {}
    self._old_manager = nil
end

function NameManager:get(name, hint)
    if name then
        return name
    end
    if not self._counter[hint] then
        self._counter[hint] = 0
    end
    name = ('%s%d'):format(hint, self._counter[hint])
    self._counter[hint] = self._counter[hint] + 1
    return name
end

function NameManager:__enter()
    if not NameManager._current.value then
        NameManager._current.value = NameManager()
    end
    self._old_manager = NameManager._current.value
    NameManager._current.value = self
    return self
end

function NameManager:__exit()
    assert(self._old_manager)
    NameManager._current.value = self._old_manager
end

---@class mx.name.Prefix:mx.name.NameManager
local Prefix = class('mx.name.Prefix', NameManager)
M.Prefix = Prefix

function Prefix:ctor(prefix)
    self.super.ctor(self)
    self._prefix = prefix
end

function Prefix:get(name, hint)
    name = NameManager.get(self, name, hint)
    return self._prefix .. name
end

-- initialize the default name manager
NameManager._current.value = NameManager()

return M
