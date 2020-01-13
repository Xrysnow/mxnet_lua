--

require('python.__init__')
require('util.functions')
require('util.stringify')
require('util.ffi_')
require('util.string')
ffi = ffi or require('ffi')

function default(val, def, ...)
    local t = { ... }
    local n = select('#', ...)
    assert(n % 2 == 0)
    local first
    if isnone(val) then
        first = def
    else
        first = val
    end
    if n == 0 then
        return first
    else
        local ret = { first }
        local n_half = n / 2
        for i = 1, n_half do
            if isnone(t[i * 2 - 1]) then
                table.insert(ret, t[i * 2])
            else
                table.insert(ret, t[i * 2 - 1])
            end
        end
        return unpack(ret, 1, n_half + 1)
    end
end

function default_none(...)
    local t = { ... }
    local n = select('#', ...)
    if n == 0 then
        return
    end
    for i = 1, n do
        if t[i] == nil then
            t[i] = None
        end
    end
    return unpack(t, 1, n)
end

function arg_pop(kwargs, k, def)
    local ret = default(kwargs[k], def)
    kwargs[k] = nil
    return ret
end

function arg_kw(...)
    local args = { ... }
    local n = select('#', ...)
    local kwargs = args[n]
    if type(kwargs) == 'table' and kwargs['.kwarg'] then
        args[n] = nil
        kwargs['.kwarg'] = nil
        return args, kwargs
    else
        return args, {}
    end
end

function arg_make(args, kwargs)
    local arg = {}
    local narg = 0
    if args then
        narg = table.maxn(args)
        for i = 1, narg do
            table.insert(arg, args[i])
        end
    end
    if kwargs then
        kwargs['.kwarg'] = true
        narg = narg + 1
        table.insert(arg, kwargs)
    end
    return unpack(arg, 1, narg)
end

function kw(t)
    t['.kwarg'] = true
    return t
end

function zipairs(t1, t2)
    local i = 0
    return function()
        i = i + 1
        return t1[i], t2[i]
    end
end

function islist(t)
    return type(t) == 'table' and getmetatable(t) == nil
end

function isnone(v)
    return v == nil or v == None
end

function iprint(t)
    for _, v in ipairs(t) do
        print(v)
    end
end

function tprint(t)
    print(stringify(t))
end
