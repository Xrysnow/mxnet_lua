--
local M = {}
require('util.deepcopy')

local copyfunc = {}
local _deepcopy_dispatch = {}

local function deepcopy_cls(x, memo, cls, cname)
    local copier = _deepcopy_dispatch[cname]
    if copier then
        return copier(x, memo)
    end
    copier = cls.__deepcopy
    if copier then
        return copier(x, memo)
    end
    local rv
    if cls.__reduce_ex then
        rv = cls.__reduce_ex(x, 4)
    elseif cls.__reduce then
        rv = cls.__reduce(x)
    end
    if rv then
        return M._reconstruct(x, rv, true, memo)
    end
    -- default
    --local ok, ret = pcall(cls)
    --if not ok then
    --    return table.deepcopy(x, memo, copyfunc)
    --end
    --for k, v in pairs(x) do
    --    k = M.deepcopy(k, memo)
    --    v = M.deepcopy(v, memo)
    --    ret[k] = v
    --end
    --return ret

    local y = setmetatable({}, getmetatable(x))
    y['.class'] = cls
    for k, v in pairs(x) do
        if k ~= '.class' then
            k = M.deepcopy(k, memo)
            v = M.deepcopy(v, memo)
            y[k] = v
        end
    end
    return y
end

for k, v in pairs(table.deepcopy_copyfunc_list) do
    copyfunc[k] = v
end
copyfunc['function'] = table.deepcopy_copyfunc_list._plainolddata

local rawget = rawget
local rawset = rawset
local next = next
local getmetatable = debug and debug.getmetatable or getmetatable
local setmetatable = debug and debug.setmetatable or setmetatable
function copyfunc.table(stack, orig, copy, state, arg1, arg2, arg3, arg4)
    local orig_prevkey, grabkey = nil, false
    if state == nil then
        copy = stack[orig]
        if copy ~= nil then
            return copy, true
        else
            local orig_meta = getmetatable(orig)
            if orig_meta ~= nil then
                local cls = orig['.class']
                local cname = orig['.classname']
                if cls and cname then
                    copy = deepcopy_cls(orig, stack, cls, cname)
                    stack[orig] = copy
                    return copy, true
                else
                    copy = {}
                    stack[orig] = copy
                    if not stack.metatable_immutable then
                        stack:_recurse(orig_meta)
                        return copy, 'metatable'
                    else
                        setmetatable(copy, orig_meta)
                    end
                end
            else
                copy = {}
                stack[orig] = copy
            end
        end
        orig_prevkey = nil
        grabkey = true
    elseif state == 'metatable' then
        local copy_meta = arg2
        stack:_pop(2)
        if copy_meta ~= nil then
            setmetatable(copy, copy_meta)
        end
        orig_prevkey = nil
        grabkey = true
    elseif state == 'key' then
        local orig_key = arg1
        local copy_key = arg2
        if copy_key ~= nil then
            local orig_value = rawget(orig, orig_key)
            stack:_recurse(orig_value)
            return copy, 'value'
        else
            stack:_pop(2)
            orig_prevkey = orig_key
            grabkey = true
        end
    elseif state == 'value' then
        local orig_key = arg1
        local copy_key = arg2
        local copy_value = arg4
        stack:_pop(4)
        if copy_value ~= nil then
            rawset(copy, copy_key, copy_value)
        end
        orig_prevkey = orig_key
        grabkey = true
    end
    if grabkey then
        local orig_key, orig_value = next(orig, orig_prevkey)
        if orig_key ~= nil then
            stack:_recurse(orig_key)
            return copy, 'key'
        else
            return copy, true
        end
    end
    return
end

function M.deepcopy(x, memo, _nil)
    local ty = type(x)
    if ty ~= 'table' then
        return x
    end
    memo, _nil = default(memo, {}, _nil, {})
    local y = memo[x] or _nil
    if y ~= _nil then
        return y
    end
    return table.deepcopy(x, memo, copyfunc)
end

function M._reconstruct(x, info, deep, memo)
    if type(info) == 'string' then
        return x
    end
    memo = memo or {}
    local deepcopy = M.deepcopy
    local callable, args, state, listiter, dictiter = info[1], info[2], info[3], info[4], info[5]
    if deep then
        args = deepcopy(args, memo)
    end
    local y = callable(unpack(args))
    memo[x] = y
    if state then
        if deep then
            state = deepcopy(state, memo)
        end
        if y.__setstate then
            y:__setstate(state)
        else
            local slotstate
            if islist(state) and #state == 2 then
                state, slotstate = state[1], state[2]
            end
            if state then
                for k, v in pairs(state) do
                    y[k] = v
                end
            end
            if slotstate then
                for k, v in pairs(slotstate) do
                    y[k] = v
                end
            end
        end
    end
    if listiter then
        for item in listiter do
            if deep then
                item = M.deepcopy(item, memo)
            end
            table.insert(y, item)
        end
    end
    if dictiter then
        for key, value in dictiter do
            if deep then
                key = deepcopy(key, memo)
                value = deepcopy(value, memo)
            end
            y[key] = value
        end
    end
    return y
end

return M
