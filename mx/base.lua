--
local M = {}
local ffi = require('ffi')
local ctypes = require('ctypes')
local c_str = ctypes.c_str

local function py_str(x)
    return ctypes.str(x)
end
M.py_str = py_str

function M.data_dir_default()
    local system = ffi.os
    if system == 'Windows' then
        return os.getenv('APPDATA') .. '\\mxnet'
    else
        return os.getenv('HOME') .. '/.mxnet'
    end
end

function M.data_dir()
    return os.getenv('MXNET_HOME') or M.data_dir_default()
end

--

---@class mx._NullType
local _NullType = class('mx._NullType', {})
function M:__tostring()
    return '_Null'
end
M._NullType = _NullType
M._Null = _NullType()
--
---@class mx.MXNetError
local MXNetError = class('MXNetError', {})
function MXNetError:ctor(msg)
    self._msg = msg or 'MXNetError'
end
function MXNetError:__tostring()
    return self._msg
end
M.MXNetError = MXNetError
--
---@class mx.NotImplementedForSymbol
local NotImplementedForSymbol = class('mx.NotImplementedForSymbol', {})
function NotImplementedForSymbol:ctor(function_, alias, args)
    self.function_ = function_
    self.alias = alias
    self.args = {}
    for _, v in ipairs(args) do
        table.insert(self.args, gettypename(v))
    end
end
function M:__tostring()
    local msg = string.format('Function %s', tostring(self.function_))
    if self.alias then
        msg = msg .. (' (namely operator "%s")'):format(tostring(self.alias))
    else
        msg = msg .. (' with arguments (%s)'):format(table.concat(self.args, ', '))
    end
    msg = msg .. ' is not supported for SparseNDArray and only available in NDArray.'
    return msg
end
M.NotImplementedForSymbol = NotImplementedForSymbol
--
---@class mx.MXCallbackList
local MXCallbackList = class('mx.MXCallbackList', {})
ffi.cdef [[
struct MXCallbackList_ {
    int num_callbacks;
    void** callbacks; // int callbacks[0]();
    void** contexts;
};
typedef struct MXCallbackList_ MXCallbackList;
]]
function MXCallbackList:__call()
    return ffi.new('MXCallbackList')
end
M.MXCallbackList = MXCallbackList
--

--_MXClassPropertyDescriptor
--_MXClassPropertyMetaClass
--with_metaclass
--classproperty

--

local function _load_lib()
    return require('api.c_api')
end

M.__version__ = '1.5.0'
local _LIB = _load_lib()
M._LIB = _LIB

M.mx_int = ctypes.c_int
M.mx_uint = ctypes.c_uint
M.mx_float = ctypes.c_float
M.mx_float_p = ctypes.POINTER(M.mx_float)
M.mx_real_t = 'float32'
M.NDArrayHandle = ctypes.c_void_p
M.FunctionHandle = ctypes.c_void_p
M.OpHandle = ctypes.c_void_p
M.CachedOpHandle = ctypes.c_void_p
M.SymbolHandle = ctypes.c_void_p
M.ExecutorHandle = ctypes.c_void_p
M.DataIterCreatorHandle = ctypes.c_void_p
M.DataIterHandle = ctypes.c_void_p
M.KVStoreHandle = ctypes.c_void_p
M.RecordIOHandle = ctypes.c_void_p
M.RtcHandle = ctypes.c_void_p
M.CudaModuleHandle = ctypes.c_void_p
M.CudaKernelHandle = ctypes.c_void_p
M.ProfileHandle = ctypes.c_void_p
M.DLPackHandle = ctypes.c_void_p

function M.check_call(ret)
    if ret ~= 0 then
        error(ffi.string(_LIB.MXGetLastError()))
    end
end
local check_call = M.check_call

--- Create ctypes char * from a Lua string.
function M.c_str(str)
    return ctypes.c_char_p(str)
end

--- Create ctypes const char ** from a list of Lua strings.
function M.c_str_array(strs)
    local arr = (ctypes.c_char_p * len(strs))()
    for i = 0, #strs - 1 do
        arr[i] = strs[i + 1]
    end
    return arr
end

--- Create ctypes array from a Lua array.
function M.c_array(ctype, values)
    local out = (ctype * len(values))()
    for i = 0, #values - 1 do
        local v = values[i + 1]
        if not isnone(v) then
            out[i] = v
        end
    end
    return out
end

--- Create ctypes array from a Lua buffer.
--- For primitive types, using the buffer created with array.array is faster
--- than a c_array call.
function M.c_array_buf(ctype, buf)
    local sz = len(buf)
    local ret = (ctype * sz)()
    if islist(buf) then
        for i = 0, sz - 1 do
            ret[i] = buf[i + 1]
        end
    else
        for i = 0, sz - 1 do
            ret[i] = buf[i]
        end
    end
    return ret
end

--- Create ctypes const void ** from a list of MXNet objects with handles.
function M.c_handle_array(objs)
    local arr = (ctypes.c_void_p * len(objs))()
    for i = 0, #objs - 1 do
        local hdl = objs[i + 1].handle
        if not isnone(hdl) then
            arr[i] = hdl.value
        end
    end
    return arr
end

--- Convert ctypes pointer to buffer type.
function M.ctypes2buffer(cptr, length)
    return cptr
end

function M.ctypes2numpy_shared(sptr, shape)
    raise('NotImplementedError')
end

function M.build_param_doc(arg_names, arg_types, arg_descs, remove_dup)
    remove_dup = default(remove_dup, true)
    local param_keys = {}
    local param_str = {}
    for i = 1, #arg_names do
        local key, type_info, desc = arg_names[i], arg_types[i], arg_descs[i]
        local continue = param_keys[key] and remove_dup
        continue = continue or key == 'num_args'
        if not continue then
            param_keys[key] = true
            if string.is_keyword(key) then
                key = key .. '_'
            end
            local ret = ('---@param %s any @%s'):format(key, type_info)
            if #desc ~= 0 then
                ret = ('%s | %s'):format(ret, desc:gsub('\n', ''))
            end
            table.insert(param_str, ret)
        end
    end
    local doc_str = '---\n' ..
            '%s'
    doc_str = doc_str:format(table.concat(param_str, '\n'))
    return doc_str
end

--

local function _notify_shutdown()
    M.check_call(_LIB.MXNotifyShutdown())
end
M['.gc_proxy'] = ffi.gc(ffi.new('int[1]'), function()
    _notify_shutdown()
end)

function M.add_fileline_to_docstring(module, incursive)
    --incursive = default(incursive, true)
    --raise('NotImplementedError')
end

--- A utility function that converts the argument to a list if it is not already.
function M._as_list(obj)
    if islist(obj) then
        return obj
    else
        return { obj }
    end
end

--
local _OP_NAME_PREFIX_LIST = { '_contrib_', '_linalg_', '_sparse_', '_image_', '_random_' }

---@param op_name string
local function _get_op_name_prefix(op_name)
    for i, prefix in ipairs(_OP_NAME_PREFIX_LIST) do
        if op_name:starts_with(prefix) then
            return prefix
        end
    end
    return ''
end

function M._init_op_module(root_namespace, module_name, make_op_func)
    local plist = ctypes.POINTER(ctypes.c_char_p)()
    local size = ctypes.c_uint()
    check_call(_LIB.MXListAllOpNames(ctypes.byref(size),
                                     ctypes.byref(plist)))
    local op_names = {}
    for i = 0, size.value - 1 do
        table.insert(op_names, py_str(plist[i]))
    end
    local module_op = string.format('%s.%s.op', root_namespace, module_name)
    module_op = require(module_op)
    local module_internal = string.format('%s.%s._internal', root_namespace, module_name)
    module_internal = require(module_internal)
    --
    local submodule_dict = {}
    for i, op_name_prefix in ipairs(_OP_NAME_PREFIX_LIST) do
        local s = string.format(
                '%s.%s.%s', root_namespace, module_name, op_name_prefix:sub(2, -2))
        submodule_dict[op_name_prefix] = require(s)
        --print(('submodule_dict[%s] = %s'):format(op_name_prefix, s))
    end
    --print('total op:', #op_names)
    for i, name in ipairs(op_names) do
        local hdl = M.OpHandle()
        check_call(_LIB.NNGetOpHandle(c_str(name), ctypes.byref(hdl)))
        local op_name_prefix = _get_op_name_prefix(name)
        local module_name_local = module_name
        local func_name, cur_module
        if len(op_name_prefix) > 0 then
            if op_name_prefix ~= '_random_' or name:ends_with('_like') then
                func_name = name:sub(len(op_name_prefix) + 1, -1)
                cur_module = submodule_dict[op_name_prefix]
                module_name_local = string.format(
                        '%s.%s.%s', root_namespace, module_name, op_name_prefix:sub(2, -2))
            else
                func_name = name
                cur_module = module_internal
            end
        elseif name:starts_with('_') then
            func_name = name
            cur_module = module_internal
        else
            func_name = name
            cur_module = module_op
        end
        if string.is_keyword(func_name) then
            func_name = func_name .. '_'
        end
        local function_ = make_op_func(hdl, name, func_name)
        --function_.__module__ = module_name_local
        --setattr(cur_module, function_.__name__, function_)
        --table.insert(cur_module.__all__, function_.__name__)
        --[[
                local mod_name
                if cur_module == module_internal then
                    mod_name = '_internal'
                elseif cur_module == module_op then
                    mod_name = 'op'
                else
                    mod_name = op_name_prefix:sub(2, -2)
                end
                if not cur_module[func_name] then
                    print(('new %s.%s'):format(mod_name, func_name))
                else
                    print(('reg %s.%s'):format(mod_name, func_name))
                end
        ]]
        cur_module[func_name] = function_

        --if op_name_prefix == '_contrib_' then
        --    hdl = M.OpHandle()
        --    check_call(_LIB.NNGetOpHandle(c_str(name), ctypes.byref(hdl)))
        --    func_name = name:sub(len(op_name_prefix) + 1, -1)
        --    function_ = make_op_func(hdl, name, func_name)
        --    function_.__module__ = contrib_module_name_old
        --    setattr(contrib_module_old, function_.__name__, function_)
        --    contrib_module_old.__all__.append(function_.__name__)
        --end
    end
end

function M._generate_op_module_signature(root_namespace, module_name, op_code_gen_func)
    local function get_module_file(module_name_)
        local path = string.filefolder(debug.getinfo(1).source)
        if path:sub(1, 1) == '@' then
            path = path:sub(2)
        end
        local module_path = module_name_:split('.')
        module_path[#module_path] = 'gen_' .. module_path[#module_path]
        local file_name = table.concat({ path, '..', unpack(module_path) }, '/') .. '.lua'
        print('write to file', file_name)
        local module_file = io.open(file_name, 'w')
        module_file:write('-- File content is auto-generated. Do not modify.\n')
        module_file:write('\n')
        local dependencies = {
            symbol  = {
                [[local SymbolBase = require('mx._ctypes.symbol').SymbolBase]],
                [[local _symbol_creator = require('mx._ctypes.symbol')._symbol_creator]],
                [[local NameManager = require('mx.name').NameManager]],
                [[local AttrScope = require('mx.attribute').AttrScope]],
                [[local _Null = require('mx.base')._Null]],
                ([[---@class %s]]):format(table.concat(module_path, '.')),
                [[local M = {}]],
            },
            ndarray = {
                [[local NDArrayBase = require('mx.ndarray._internal').NDArrayBase]],
                [[local _imperative_invoke = require('mx.ndarray._internal')._imperative_invoke]],
                [[local _Null = require('mx.base')._Null]],
                ([[---@class %s]]):format(table.concat(module_path, '.')),
                [[local M = {}]],
            }, }
        module_file:write(table.concat(dependencies[module_path[2]], '\n'))
        module_file:write('\n')
        return module_file
    end
    local function write_all_str(module_file, module_all_list)
        module_file:write('\n\n')
        local all_str = [[return M]]
        module_file:write(all_str)
    end
    local plist = ctypes.POINTER(ctypes.c_char_p)()
    local size = ctypes.c_uint()
    check_call(_LIB.MXListAllOpNames(ctypes.byref(size),
                                     ctypes.byref(plist)))
    local op_names = {}
    for i = 1, size.value do
        local op_name = M.py_str(plist[i - 1])
        table.insert(op_names, op_name)
    end
    local module_op_file = get_module_file(('%s.%s.op'):format(root_namespace, module_name))
    local module_op_all = {}
    local module_internal_file = get_module_file(('%s.%s._internal'):format(root_namespace, module_name))
    local module_internal_all = {}
    local submodule_dict = {}
    for _, op_name_prefix in ipairs(_OP_NAME_PREFIX_LIST) do
        submodule_dict[op_name_prefix] = {
            get_module_file(('%s.%s.%s'):format(root_namespace, module_name, op_name_prefix:sub(2, -2))),
            {} }
    end
    for _, name in ipairs(op_names) do
        local hdl = M.OpHandle()
        check_call(_LIB.NNGetOpHandle(c_str(name), ctypes.byref(hdl)))
        local op_name_prefix = _get_op_name_prefix(name)
        local func_name, cur_module_file, cur_module_all
        if #op_name_prefix > 0 then
            func_name = name:sub(#op_name_prefix + 1, -1)
            cur_module_file, cur_module_all = unpack(submodule_dict[op_name_prefix])
        elseif string.starts_with(name, '_') then
            func_name = name
            cur_module_file = module_internal_file
            cur_module_all = module_internal_all
        else
            func_name = name
            cur_module_file = module_op_file
            cur_module_all = module_op_all
        end
        if string.is_keyword(func_name) then
            func_name = func_name .. '_'
        end
        local code, _ = op_code_gen_func(hdl, name, func_name, true)
        cur_module_file:write('\n')
        cur_module_file:write(code)
        table.insert(cur_module_all, func_name)
    end
    for _, v in pairs(submodule_dict) do
        local submodule_f, submodule_all = unpack(v)
        write_all_str(submodule_f, submodule_all)
        submodule_f:close()
    end
    write_all_str(module_op_file, module_op_all)
    module_op_file:close()
    write_all_str(module_internal_file, module_internal_all)
    module_internal_file:close()
end

function M.generate_nd_signature()
    M._generate_op_module_signature('mx', 'ndarray', require('mx.ndarray.register')._generate_ndarray_function_code)
end

function M.generate_sym_signature()
    M._generate_op_module_signature('mx', 'symbol', require('mx.symbol.register')._generate_symbol_function_code)
end

local _tostring
_tostring = function(v)
    if islist(v) then
        if #v == 0 then
            return '[]'
        else
            local vv = {}
            for i = 1, #v do
                vv[i] = _tostring(v[i])
            end
            return ('[%s]'):format(table.concat(vv, ', '))
        end
    elseif type(v) == 'number' then
        return string.format('%q', v):sub(2, -2)
    elseif v == true then
        return 'True'
    elseif v == false then
        return 'False'
    else
        return tostring(v)
    end
end

---@return string
function M.tostring(v)
    return _tostring(v)
end

return M
