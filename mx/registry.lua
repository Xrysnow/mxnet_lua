--
local M = {}

local json = require('util.json')

local _REGISTRY = {}
M._REGISTRY = _REGISTRY

function M.get_registry(base_class)
    if not _REGISTRY[base_class] then
        _REGISTRY[base_class] = {}
    end
    return table.clone(_REGISTRY[base_class])
end

function M.get_register_func(base_class, nickname)
    if not _REGISTRY[base_class] then
        _REGISTRY[base_class] = {}
    end
    local registry = _REGISTRY[base_class]
    local function register(klass, name)
        assert(issubclass(klass, base_class),
               'Can only register subclass')
        if isnone(name) then
            name = getclassname(klass)
        end
        name = name:lower()
        if registry[name] then
            print('registry override')
        end
        registry[name] = klass
        return klass
    end
    return register
end

function M.get_alias_func(base_class, nickname)
    local register = M.get_register_func(base_class, nickname)
    local function alias(...)
        local aliases = { ... }
        local function reg(klass)
            for _, name in ipairs(aliases) do
                register(klass, name)
            end
            return klass
        end
        return reg
    end
    return alias
end

function M.get_create_func(base_class, nickname)
    if not _REGISTRY[base_class] then
        _REGISTRY[base_class] = {}
    end
    local registry = _REGISTRY[base_class]
    local function create(...)
        local args, kwargs = arg_kw(...)
        local name
        if not table.empty(args) then
            name = args[1]
            table.remove(args, 1)
        else
            name = arg_pop(kwargs, nickname)
        end
        if isinstance(name, base_class) then
            assert(#args == 0 and table.empty(kwargs),
                   ('%s is already an instance. Additional arguments are invalid'):format(nickname))
            return name
        end
        if type(name) == 'table' then
            return create(arg_make(nil, name))
        end
        assert(type(name) == 'string', ('%s must be of string type'):format(nickname))

        if string.starts_with(name, '[') then
            assert(table.empty(args) and table.empty(kwargs))
            local j = json.decode(name)
            name, kwargs = j[1], j[2]
            return create(arg_make({ name }, kwargs))
        elseif string.starts_with(name, '{') then
            assert(table.empty(args) and table.empty(kwargs))
            kwargs = json.decode(name)
            return create(arg_make(nil, kwargs))
        end

        name = name:lower()
        assert(registry[name],
               ('%s is not registered. Please register with %s.register first'):format(name, nickname))
        return registry[name](...)
    end
    return create
end

return M
