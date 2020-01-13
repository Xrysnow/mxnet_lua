--
local M = {}

function M.namedtuple(typename, field_names, verbose, rename)
    if type(field_names) == 'string' then
        field_names = string.split(field_names, ',')
    end
    assert(type(field_names) == 'table')
    field_names = table.map(field_names, tostring)
    typename = tostring(typename)
    if rename then
        --
    end
    for _, name in ipairs(table.sequence({ typename }, field_names)) do
        if type(name) ~= 'string' then
            raise('TypeError')
        end
        --
        if string.is_keyword(name) then
            raise('ValueError')
        end
    end
    local seen = {}
    for _, name in ipairs(field_names) do
        if string.starts_with(name, '_') and not rename then
            raise('ValueError', ('Field names cannot start with an underscore: %s'):format(name))
        end
        if seen[name] then
            raise('ValueError', ('Encountered duplicate field name: %s'):format(name))
        end
        seen[name] = true
    end

    local nfield = #field_names
    local cls = class(typename)
    function cls:ctor(...)
        local args = { ... }
        local narg = select('#', ...)
        if nfield ~= 1 and narg == 1 and #args[1] == 0 then
            local kwa = args[1]
            for _, v in ipairs(field_names) do
                self[v] = kwa[v]
            end
        else
            assert(nfield == narg, 'invalid positional arguments')
            for i = 1, nfield do
                self[field_names[i]] = args[i]
            end
        end
    end
    function cls:__tostring()
        local s = {}
        for _, name in ipairs(field_names) do
            table.insert(s, ('%s=%s'):format(name, tostring(self[name])))
        end
        s = table.concat(s, ', ')
        return s
    end
    function cls:__index(k)
        local t = type(k)
        if t == 'number' then
            if 1 <= k and k <= nfield then
                return rawget(self, field_names[k])
            end
        elseif t == 'string' then
            if cls[k] then
                return cls[k]
            end
        end
        return rawget(self, k)
    end

    return cls
end

return M
