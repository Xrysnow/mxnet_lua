---@class list
---@field append fun(object:any)
local M = class('list', {})

function M:ctor(...)
end

function M:__index(k)
    if k == 'append' then
        return function(o)
            table.insert(self, o)
        end
    end
end

return M
