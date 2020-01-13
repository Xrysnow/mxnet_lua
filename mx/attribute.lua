--
local M = {}

---@class mx.attribute.AttrScope
local AttrScope = class('mx.attribute.AttrScope')
M.AttrScope = AttrScope

function AttrScope:ctor(kwargs)
    kwargs = default(kwargs, {})
    self._old_scope = nil
    for _, value in ipairs(table.values(kwargs)) do
        if not type(value) == 'string' then
            raise('ValueError', 'Attributes need to be string')
        end
    end
    self._attr = kwargs
end

function AttrScope:get(attr)
    if self._attr then
        local ret = table.clone(self._attr)
        if attr then
            for k, v in pairs(attr) do
                ret[k] = v
            end
        end
        return ret
    else
        return attr or {}
    end
end

function AttrScope:__enter()
    if not AttrScope._current.value then
        AttrScope._current.value = AttrScope()
    end
    self._old_scope = AttrScope._current.value
    local attr = table.clone(AttrScope._current.value._attr)
    for k, v in pairs(self._attr) do
        attr[k] = v
    end
    self._attr = attr
    AttrScope._current.value = self
    return self
end

function AttrScope:__exit()
    assert(self._old_scope)
    AttrScope._current.value = self._old_scope
end

AttrScope._current = { value = AttrScope() }

return M
