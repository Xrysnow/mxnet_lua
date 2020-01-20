--[[

Copyright (c) 2014-2017 Chukong Technologies Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

]]

function printLog(tag, fmt, ...)
    local t = {
        "[",
        string.upper(tostring(tag)),
        "] ",
        string.format(tostring(fmt), ...)
    }
    print(table.concat(t))
end

function printError(fmt, ...)
    printLog("ERR", fmt, ...)
    print(debug.traceback("", 2))
end

function printInfo(fmt, ...)
    if type(DEBUG) ~= "number" or DEBUG < 2 then
        return
    end
    printLog("INFO", fmt, ...)
end

local function dump_value_(v)
    if type(v) == "string" then
        v = "\"" .. v .. "\""
    end
    return tostring(v)
end

function dump(value, description, nesting)
    if type(nesting) ~= "number" then
        nesting = 3
    end

    local lookupTable = {}
    local result = {}

    local traceback = string.split(debug.traceback("", 2), "\n")
    print("dump from: " .. string.trim(traceback[3]))

    local function dump_(value, description, indent, nest, keylen)
        description = description or "<var>"
        local spc = ""
        if type(keylen) == "number" then
            spc = string.rep(" ", keylen - string.len(dump_value_(description)))
        end
        if type(value) ~= "table" then
            result[#result + 1] = string.format("%s%s%s = %s", indent, dump_value_(description), spc, dump_value_(value))
        elseif lookupTable[tostring(value)] then
            result[#result + 1] = string.format("%s%s%s = *REF*", indent, dump_value_(description), spc)
        else
            lookupTable[tostring(value)] = true
            if nest > nesting then
                result[#result + 1] = string.format("%s%s = *MAX NESTING*", indent, dump_value_(description))
            else
                result[#result + 1] = string.format("%s%s = {", indent, dump_value_(description))
                local indent2 = indent .. "    "
                local keys = {}
                local keylen = 0
                local values = {}
                for k, v in pairs(value) do
                    keys[#keys + 1] = k
                    local vk = dump_value_(k)
                    local vkl = string.len(vk)
                    if vkl > keylen then
                        keylen = vkl
                    end
                    values[k] = v
                end
                table.sort(keys, function(a, b)
                    if type(a) == "number" and type(b) == "number" then
                        return a < b
                    else
                        return tostring(a) < tostring(b)
                    end
                end)
                for i, k in ipairs(keys) do
                    dump_(values[k], k, indent2, nest + 1, keylen)
                end
                result[#result + 1] = string.format("%s}", indent)
            end
        end
    end
    dump_(value, description, "- ", 1)

    for i, line in ipairs(result) do
        print(line)
    end
end

function printf(fmt, ...)
    print(string.format(tostring(fmt), ...))
end

function checknumber(value, base)
    return tonumber(value, base) or 0
end

function checkint(value)
    return math.round(checknumber(value))
end

function checkbool(value)
    return (value ~= nil and value ~= false)
end

function checktable(value)
    if type(value) ~= "table" then
        value = {}
    end
    return value
end

function isset(hashtable, key)
    local t = type(hashtable)
    return (t == "table" or t == "userdata") and hashtable[key] ~= nil
end

local setmetatableindex_
setmetatableindex_ = function(t, index)
    if type(t) == "userdata" then
        local peer = tolua.getpeer(t)
        if not peer then
            peer = {}
            tolua.setpeer(t, peer)
        end
        setmetatableindex_(peer, index)
    else
        local mt = getmetatable(t)
        if not mt then
            mt = {}
        end
        if not mt.__index then
            mt.__index = index
            setmetatable(t, mt)
        elseif mt.__index ~= index then
            setmetatableindex_(mt, index)
        end
    end
end
setmetatableindex = setmetatableindex_

function clone(object)
    local lookup_table = {}
    local function _copy(object)
        if type(object) ~= "table" then
            return object
        elseif lookup_table[object] then
            return lookup_table[object]
        end
        local newObject = {}
        lookup_table[object] = newObject
        for key, value in pairs(object) do
            newObject[_copy(key)] = _copy(value)
        end
        return setmetatable(newObject, getmetatable(object))
    end
    return _copy(object)
end

local ffi = require('ffi')
local function dtor_proxy(ins, dtor)
    if dtor then
        ins['.dtor_proxy'] = ffi.gc(
                ffi.new('int32_t[0]'),
                function()
                    dtor(ins)
                end
        )
    end
end

function class(classname, ...)
    local cls = { __cname = classname }

    local supers = { ... }
    for _, super in ipairs(supers) do
        local superType = type(super)
        assert(superType == "nil" or superType == "table" or superType == "function",
               string.format("class() - create class \"%s\" with invalid super class type \"%s\"",
                             classname, superType))

        if superType == "function" then
            assert(cls.__create == nil,
                   string.format("class() - create class \"%s\" with more than one creating function",
                                 classname));
            -- if super is function, set it to __create
            cls.__create = super
        elseif superType == "table" then
            if super[".isclass"] then
                -- super is native class
                assert(cls.__create == nil,
                       string.format("class() - create class \"%s\" with more than one creating function or native class",
                                     classname));
                cls.__create = function()
                    return super:create()
                end
                --if not cls.super then
                --    cls.super = super
                --end
            else
                -- super is pure lua class
                cls.__supers = cls.__supers or {}
                cls.__supers[#cls.__supers + 1] = super
                if not cls.super then
                    -- set first super pure lua class as class.super
                    cls.super = super
                end
            end
        else
            error(string.format("class() - create class \"%s\" with invalid super type",
                                classname), 0)
        end
    end

    cls.__index = cls
    local __call = function(_, ...)
        return cls.new(...)
    end

    if not cls.__supers or #cls.__supers == 1 then
        setmetatable(cls, { __index = cls.super, __call = __call })
    else
        setmetatable(cls, {
            __index = function(_, key)
                local supers = cls.__supers
                for i = 1, #supers do
                    local super = supers[i]
                    if super[key] then
                        return super[key]
                    end
                end
            end,
            __call  = __call })
    end

    if not cls.ctor then
        -- add default constructor
        cls.ctor = function()
        end
    end
    local meta_method
    cls.new = function(...)
        local instance
        if cls.__create then
            instance = cls.__create(...)
        else
            instance = {}
        end
        setmetatableindex(instance, cls)

        instance['.class'] = cls
        instance['.classname'] = classname
        -- for super is native class
        if not instance.super then
            instance.super = {}
            setmetatable(instance.super, {
                __index = function(_, k)
                    local log = cls[k]
                    cls[k] = nil
                    local ret = instance[k]
                    cls[k] = log
                    return ret
                end
            })
        end
        local mt = getmetatable(instance)
        -- set once
        if not meta_method then
            meta_method = {}
            for _, v in ipairs(
                    { '__add', '__sub', '__mul', '__div', '__mod', '__pow', '__unm',
                      '__concat', '__len', '__eq', '__lt', '__le',
                      '__index', '__newindex', '__call', '__tostring', '__tonumber' }) do
                meta_method[v] = instance[v]
            end
            --local new_index = meta_method.__index
            --local old_index = rawget(mt, '__index')
            --if new_index and new_index ~= old_index then
            --    if type(old_index) == 'function' then
            --        meta_method.__index = function(t, k)
            --            return new_index(t, k) or old_index(t, k)
            --        end
            --    else
            --        meta_method.__index = function(t, k)
            --            return new_index(t, k) or old_index[k]
            --        end
            --    end
            --end
        end
        for k, v in pairs(meta_method) do
            rawset(mt, k, v)
        end
        mt.__supers = cls.__supers
        mt.__cname = cls.__cname
        dtor_proxy(instance, cls.dtor)
        instance:ctor(...)
        return instance
    end
    cls.create = function(_, ...)
        return cls.new(...)
    end

    return cls
end

function class_property(cls, getters)
    function cls:__index(k)
        local tk = type(k)
        if tk == 'string' then
            if getters[k] then
                return getters[k](self)
            elseif not isnone(cls[k]) then
                return cls[k]
            end
        end
        return rawget(self, k)
    end
end

---@return string
function getclassname(obj)
    local t = type(obj)
    if t ~= "table" and t ~= "userdata" then
        return
    end
    local ret
    ret = ret or obj['.classname']
    ret = ret or obj.__cname
    local mt
    if t == "userdata" then
        mt = getmetatable(tolua.getpeer(obj))
    else
        mt = getmetatable(obj)
    end
    if not mt then
        return ret
    end
    ret = ret or rawget(mt, '.classname')
    ret = ret or rawget(mt, '__cname')
    local index = rawget(mt, '__index')
    if index then
        ret = ret or rawget(index, '.classname')
        ret = ret or rawget(index, '__cname')
    end
    return ret
end

function gettypename(obj)
    return getclassname(obj) or type(obj)
end

local iskindof_
iskindof_ = function(cls, name)
    local __index = rawget(cls, "__index")
    local ti = type(__index)
    local __cname
    if ti == "table" then
        __cname = __cname or rawget(__index, "__cname")
    end
    __cname = __cname or cls.__cname
    if __cname == name then
        return true
    end
    local __supers
    if ti == 'table' then
        __supers = __supers or rawget(__index, "__supers")
    end
    __supers = __supers or cls.__supers
    if not __supers then
        --print('no supers')
        --print(stringify(cls))
        --print(stringify(getmetatable(cls)))
        return false
    end
    for _, super in ipairs(__supers) do
        if iskindof_(super, name) then
            return true
        end
    end
    --print('no kind')
    return false
end

function iskindof(obj, classname)
    local t = type(obj)
    if t ~= "table" and t ~= "userdata" then
        return false
    end

    local mt
    if t == "userdata" then
        if tolua.iskindof(obj, classname) then
            return true
        end
        mt = getmetatable(tolua.getpeer(obj))
    else
        if obj['.classname'] == classname then
            return true
        end
        mt = getmetatable(obj)
    end
    if mt then
        return iskindof_(mt, classname)
    end
    return false
end

function isinstance(obj, cls)
    if cls == 0 then
        return type(obj) == 'number'
    elseif cls == '' then
        return type(obj) == 'string'
    end
    return iskindof(obj, getclassname(cls))
end

function issubclass(cls, base)
    if type(cls) ~= 'table' or type(base) ~= 'table' then
        return false
    end
    if islist(base) then
        for _, v in ipairs(base) do
            if issubclass(cls, v) then
                return true
            end
        end
        return false
    end
    if cls == base then
        return true
    end
    local supers = cls.__supers
    if not supers then
        return false
    end
    for _, v in ipairs(supers) do
        if issubclass(v, base) then
            return true
        end
    end
    return false
end

function import_(moduleName, currentModuleName)
    local currentModuleNameParts
    local moduleFullName = moduleName
    local offset = 1

    while true do
        if string.byte(moduleName, offset) ~= 46 then
            -- .
            moduleFullName = string.sub(moduleName, offset)
            if currentModuleNameParts and #currentModuleNameParts > 0 then
                moduleFullName = table.concat(currentModuleNameParts, ".") .. "." .. moduleFullName
            end
            break
        end
        offset = offset + 1

        if not currentModuleNameParts then
            if not currentModuleName then
                local n, v = debug.getlocal(3, 1)
                currentModuleName = v
            end

            currentModuleNameParts = string.split(currentModuleName, ".")
        end
        table.remove(currentModuleNameParts, #currentModuleNameParts)
    end

    return require(moduleFullName)
end

function handler(obj, method)
    return function(...)
        return method(obj, ...)
    end
end

function math.round(value)
    value = checknumber(value)
    return math.floor(value + 0.5)
end

local pi_div_180 = math.pi / 180
function math.angle2radian(angle)
    return angle * pi_div_180
end

function math.radian2angle(radian)
    return radian * 180 / math.pi
end

function io.exists(path)
    local file = io.open(path, "r")
    if file then
        io.close(file)
        return true
    end
    return false
end

function io.readfile(path)
    local file = io.open(path, "r")
    if file then
        local content = file:read("*a")
        io.close(file)
        return content
    end
    return nil
end

function io.writefile(path, content, mode)
    mode = mode or "w+b"
    local file = io.open(path, mode)
    if file then
        if file:write(content) == nil then
            return false
        end
        io.close(file)
        return true
    else
        return false
    end
end

function io.pathinfo(path)
    local pos = string.len(path)
    local extpos = pos + 1
    while pos > 0 do
        local b = string.byte(path, pos)
        if b == 46 then
            -- 46 = char "."
            extpos = pos
        elseif b == 47 then
            -- 47 = char "/"
            break
        end
        pos = pos - 1
    end

    local dirname = string.sub(path, 1, pos)
    local filename = string.sub(path, pos + 1)
    extpos = extpos - pos
    local basename = string.sub(filename, 1, extpos - 1)
    local extname = string.sub(filename, extpos)
    return {
        dirname  = dirname,
        filename = filename,
        basename = basename,
        extname  = extname
    }
end

function io.filesize(path)
    local size = false
    local file = io.open(path, "r")
    if file then
        local current = file:seek()
        size = file:seek("end")
        file:seek("set", current)
        io.close(file)
    end
    return size
end

local insert = table.insert

function table.len(t)
    local count = 0
    for _, _ in pairs(t) do
        count = count + 1
    end
    return count
end

function table.keys(hashtable)
    local keys = {}
    for k, v in pairs(hashtable) do
        keys[#keys + 1] = k
    end
    return keys
end

function table.values(hashtable)
    local values = {}
    for k, v in pairs(hashtable) do
        values[#values + 1] = v
    end
    return values
end

function table.kv(t)
    local key, val = {}, {}
    for k, v in pairs(t) do
        insert(key, k)
        insert(val, v)
    end
    return key, val
end

function table.merge(dest, src)
    for k, v in pairs(src) do
        dest[k] = v
    end
end

--function table.insertto(dest, src, begin)
--    begin = checkint(begin)
--    if begin <= 0 then
--        begin = #dest + 1
--    end
--
--    local len = #src
--    for i = 0, len - 1 do
--        dest[i + begin] = src[i + 1]
--    end
--end

function table.indexof(array, value, begin)
    for i = begin or 1, #array do
        if array[i] == value then
            return i
        end
    end
    return false
end

function table.keyof(hashtable, value)
    for k, v in pairs(hashtable) do
        if v == value then
            return k
        end
    end
    return nil
end

function table.removebyvalue(array, value, removeall)
    local c, i, max = 0, 1, #array
    while i <= max do
        if array[i] == value then
            table.remove(array, i)
            c = c + 1
            i = i - 1
            max = max - 1
            if not removeall then
                break
            end
        end
        i = i + 1
    end
    return c
end

function table.map(t, f)
    local ret = {}
    local ty = type(f)
    if ty == 'function' then
        for k, v in pairs(t) do
            ret[k] = f(v, k)
        end
    else
        -- index
        for k, v in pairs(t) do
            ret[k] = v[f]
        end
    end
    return ret
end

function table.reduce(t, f)
    local ret, first
    if f == 'add' then
        f = function(a, b)
            return a + b
        end
    elseif f == 'mul' then
        f = function(a, b)
            return a * b
        end
    end
    for _, v in pairs(t) do
        if not first then
            first = true
            ret = v
        else
            ret = f(ret, v)
        end
    end
    return ret
end

function table.filter(t, f)
    local ret = {}
    for k, v in pairs(t) do
        if f(v, k) then
            ret[k] = v
        end
    end
    return ret
end

function table.any(t, f)
    for k, v in pairs(t) do
        if f(v, k) then
            return true
        end
    end
    return false
end

function table.all(t, f)
    for k, v in pairs(t) do
        if not f(v, k) then
            return false
        end
    end
    return true
end

function table.empty(t)
    for _, _ in pairs(t) do
        return false
    end
    return true
end

function table.equal(t1, t2)
    for k, v in pairs(t1) do
        if v ~= t2[k] then
            return false
        end
    end
    for k, v in pairs(t2) do
        if v ~= t1[k] then
            return false
        end
    end
    return true
end

function table.append(dst, src)
    for _, v in ipairs(src) do
        insert(dst, v)
    end
end

function table.sequence(t1, t2, ...)
    local ret = {}
    for _, t in ipairs({ t1, t2, ... }) do
        for _, v in ipairs(t) do
            insert(ret, v)
        end
    end
    return ret
end

function table.replicate(o, n)
    local ret = {}
    for i = 1, n do
        insert(ret, o)
    end
    return ret
end

function table.clone(t)
    local ret = {}
    for k, v in pairs(t) do
        ret[k] = v
    end
    return ret
end

function table.zip(t1, t2, ...)
    local ret = {}
    local arg = { t1, t2, ... }
    for i = 1, #t1 do
        local e = {}
        for j = 1, #arg do
            insert(e, arg[j][i])
        end
        insert(ret, e)
    end
    return ret
end

function table.reverse(t)
    local ret = {}
    for i = #t, 1, -1 do
        insert(ret, t[i])
    end
    return ret
end

function table.has_value(t, value)
    for k, v in pairs(t) do
        if v == value then
            return true
        end
    end
    return false
end

function table.slice(t, start, stop, step)
    local sz = #t
    start = start or 1
    stop = stop or sz
    if start < 0 then
        start = start + sz + 1
    end
    if stop < 0 then
        stop = stop + sz + 1
    end
    assert(start >= 1 and stop >= 1 and start <= sz and stop <= sz)
    if start == stop then
        return { t[start] }
    end
    if not step then
        step = start < stop and 1 or -1
    end
    local ret = {}
    for i = start, stop, step do
        insert(ret, t[i])
    end
    return ret
end

function table.arange(f, start, stop, step)
    step = step or 1
    if stop == nil then
        stop = start
        start = 1
    end
    local ret = {}
    for i = start, stop, step do
        insert(ret, f(i))
    end
    return ret
end

function table.sum(t)
    if #t == 0 then
        return 0
    end
    local ret = t[1]
    for i = 2, #t do
        ret = ret + t[i]
    end
    return ret
end

function table.prod(t)
    if #t == 0 then
        return 0
    end
    local ret = t[1]
    for i = 2, #t do
        ret = ret * t[i]
    end
    return ret
end

--

function string.split(input, delimiter)
    input = tostring(input)
    delimiter = tostring(delimiter)
    if (delimiter == '') then
        return false
    end
    local pos, arr = 0, {}
    -- for each divider found
    for st, sp in function()
        return string.find(input, delimiter, pos, true)
    end do
        table.insert(arr, string.sub(input, pos, st - 1))
        pos = sp + 1
    end
    table.insert(arr, string.sub(input, pos))
    return arr
end

function string.ltrim(input)
    return string.gsub(input, "^[ \t\n\r]+", "")
end

function string.rtrim(input)
    return string.gsub(input, "[ \t\n\r]+$", "")
end

function string.trim(input)
    input = string.gsub(input, "^[ \t\n\r]+", "")
    return string.gsub(input, "[ \t\n\r]+$", "")
end

function string.ucfirst(input)
    return string.upper(string.sub(input, 1, 1)) .. string.sub(input, 2)
end

local function urlencodechar(char)
    return "%" .. string.format("%02X", string.byte(char))
end
function string.urlencode(input)
    -- convert line endings
    input = string.gsub(tostring(input), "\n", "\r\n")
    -- escape all characters but alphanumeric, '.' and '-'
    input = string.gsub(input, "([^%w%.%- ])", urlencodechar)
    -- convert spaces to "+" symbols
    return string.gsub(input, " ", "+")
end

function string.urldecode(input)
    input = string.gsub(input, "+", " ")
    input = string.gsub(input, "%%(%x%x)", function(h)
        return string.char(checknumber(h, 16))
    end)
    input = string.gsub(input, "\r\n", "\n")
    return input
end

function string.utf8len(input)
    local len = string.len(input)
    local left = len
    local cnt = 0
    local arr = { 0, 0xc0, 0xe0, 0xf0, 0xf8, 0xfc }
    while left ~= 0 do
        local tmp = string.byte(input, -left)
        local i = #arr
        while arr[i] do
            if tmp >= arr[i] then
                left = left - i
                break
            end
            i = i - 1
        end
        cnt = cnt + 1
    end
    return cnt
end

