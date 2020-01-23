--
local M = {}

local namedtuple = require('mx_py.collections').namedtuple
local ctypes = require('ctypes')
local _b = require('mx.base')
local _LIB, RecordIOHandle, check_call, c_str = _b._LIB, _b.RecordIOHandle, _b.check_call, _b.c_str

---@class mx.recordio.MXRecordIO
local MXRecordIO = class('mx.recordio.MXRecordIO')
M.MXRecordIO = MXRecordIO

function MXRecordIO:ctor(uri, flag)
    self.uri = c_str(uri)
    self.handle = RecordIOHandle()
    self.flag = flag
    self.pid = None
    self.is_open = false
    self:open()
end

function MXRecordIO:open()
    if self.flag == "w" then
        check_call(_LIB.MXRecordIOWriterCreate(self.uri, ctypes.byref(self.handle)))
        self.writable = true
    elseif self.flag == "r" then
        check_call(_LIB.MXRecordIOWriterCreate(self.uri, ctypes.byref(self.handle)))
        self.writable = false
    else
        raise('ValueError', ('Invalid flag %s'):format(self.flag))
    end
    --self.pid = current_process().pid
    self.is_open = true
end

function MXRecordIO:dtor()
    self:close()
end

function MXRecordIO:__getstate()
    local is_open = self.is_open
    self:close()
    local d = {}
    for k, v in pairs(self) do
        d[k] = v
    end
    d.is_open = is_open
    d.handle = None
    d.uri = ffi.string(self.uri.value)
    return d
end

function MXRecordIO:__setstate(d)
    for k, v in pairs(d) do
        self[k] = v
    end
    local is_open = d.is_open
    self.is_open = false
    self.handle = RecordIOHandle()
    self.uri = c_str(self.uri)
    if is_open then
        self:open()
    end
end

function MXRecordIO:_check_pid(allow_reset)
    --if not self.pid == current_process().pid then
    --    if allow_reset then
    --        self:reset()
    --    else
    --        raise('RuntimeError', 'Forbidden operation in multiple processes')
    --    end
    --end
end

function MXRecordIO:close()
    if not self.is_open then
        return
    end
    if self.writable then
        check_call(_LIB.MXRecordIOWriterFree(self.handle))
    else
        check_call(_LIB.MXRecordIOReaderFree(self.handle))
    end
    self.is_open = false
    self.pid = None
end

function MXRecordIO:reset()
    self:close()
    self:open()
end

function MXRecordIO:write(buf)
    assert(self.writable)
    self:_check_pid(false)
    check_call(_LIB.MXRecordIOWriterWriteRecord(self.handle,
                                                ctypes.c_char_p(buf),
                                                ctypes.c_size_t(len(buf))))
end

function MXRecordIO:read()
    assert(not self.writable)
    -- trying to implicitly read from multiple processes is forbidden,
    -- there's no elegant way to handle unless lock is introduced
    self:_check_pid(false)
    local buf = ctypes.c_char_p()
    local size = ctypes.c_size_t()
    check_call(_LIB.MXRecordIOReaderReadRecord(self.handle,
                                               ctypes.byref(buf),
                                               ctypes.byref(size)))
    if not ffi.isnullptr(buf.value) then
        return ffi.string(buf.value, size.value)
    else
        return None
    end
end

---@class mx.recordio.MXIndexedRecordIO:mx.recordio.MXRecordIO
local MXIndexedRecordIO = class('mx.recordio.MXIndexedRecordIO', MXRecordIO)
M.MXIndexedRecordIO = MXIndexedRecordIO

local function int(x)
    return math.floor(tonumber(x))
end

function MXIndexedRecordIO:ctor(idx_path, uri, flag, key_type)
    self.idx_path = idx_path
    self.idx = {}
    self.keys = {}
    self.key_type = key_type or int
    self.fidx = nil
    MXRecordIO.ctor(self, uri, flag)
end

function MXIndexedRecordIO:open()
    MXRecordIO.open(self)
    self.idx = {}
    self.keys = {}
    self.fidx = io.open(self.idx_path, self.flag)
    if self.fidx == nil then
        raise('IOError', ("can't open file %q"):format(self.idx_path))
    end
    if not self.writable then
        local line = self.fidx:read('l')
        while line do
            line = string.split(string.trim(line), '\t')
            line = self.fidx:read('l')
            local key = self.key_type(line[0])
            self.idx[key] = int(line[1])
            table.insert(self.keys, key)
        end
    end
end

function MXIndexedRecordIO:close()
    if not self.is_open then
        return
    end
    MXRecordIO.close(self)
    self.fidx:close()
end

function MXIndexedRecordIO:__getstate()
    local d = MXRecordIO.__getstate(self)
    d.fidx = nil
    return d
end

function MXIndexedRecordIO:seek(idx)
    assert(not self.writable)
    self:_check_pid(true)
    local pos = ctypes.c_size_t(self.idx[idx])
    check_call(_LIB.MXRecordIOReaderSeek(self.handle, pos))
end

function MXIndexedRecordIO:tell()
    assert(self.writable)
    local pos = ctypes.c_size_t()
    check_call(_LIB.MXRecordIOWriterTell(self.handle, ctypes.byref(pos)))
    return pos.value
end

function MXIndexedRecordIO:read_idx(idx)
    self:seek(idx)
    return self:read()
end

function MXIndexedRecordIO:write_idx(idx, buf)
    local key = self:key_type(idx)
    local pos = self:tell()
    self:write(buf)
    self.fidx:write(('%s\t%d\n'):format(tostring(key), pos))
    self.idx[key] = pos
    table.insert(self.keys, key)
end

---@class mx.recordio.IRHeader
local IRHeader = namedtuple('HEADER', { 'flag', 'label', 'id', 'id2' })
M.IRHeader = IRHeader

local def = [[
struct IRHeader {
    uint32_t flag;
    float label;
    uint64_t id;
    uint64_t id2;
};
]]
ffi.cdef(def)
local _IR_SIZE = ffi.sizeof('struct IRHeader')

function M.pack(header, s)
    local flag = header.flag
    local label = header.label
    local id = header.id
    local id2 = header.id2
    if type(label) == 'number' then
        flag = 0
    else
        local lb = require('mxnet').nd.array(label, nil, 'float32')
        flag = lb.size
        label = 0
        local data = lb:get_data()
        local str = ffi.string(data, lb.size * 4)
        s = str .. s
    end
    local h = ffi.new('struct IRHeader')
    h.flag = flag
    h.label = label
    h.id = id
    h.id2 = id2
    local pack = ffi.string(h, _IR_SIZE)
    s = pack .. s
    return s
end

function M.unpack(s)
    local pack = s:sub(1, _IR_SIZE)
    local h = ffi.cast('struct IRHeader*', pack)
    local header = IRHeader(h.flag, h.label, h.id, h.id2)
    s = s:sub(_IR_SIZE + 1)
    if h.flag > 0 then
        local data = s:sub(1, h.flag * 4)
        local lb = require('mxnet').nd.empty({ h.flag }, nil, 'float32')
        lb:_sync_copyfrom(data)
        header.label = lb
        s = s:sub(h.flag * 4 + 1)
    end
    return header, s
end

function M.unpack_img(s, iscolor)
    iscolor = default(iscolor, -1)
    local header, ss = M.unpack(s)
    local data = require('mxnet').nd.empty({ #ss }, nil, 'uint8')
    data:_sync_copyfrom(ss)
    -- note: cv2.imdecode -> _cvimdecode
    local img = require('mx.ndarray._internal')._cvimdecode(data, iscolor)
    return header, img
end

function M.pack_img(header, img, quality, img_fmt)
    raise('NotImplementedError', 'need cv2.imencode')
    --quality, img_fmt = default(quality, 95, img_fmt, '.jpg')
    --img_fmt = img_fmt:upper()
    --local encode_params
    --if img_fmt == '.JPG' or img_fmt == '.JPEG' then
    --    encode_params = { cv2.IMWRITE_JPEG_QUALITY, quality }
    --elseif img_fmt == '.PNG' then
    --    encode_params = { cv2.IMWRITE_PNG_COMPRESSION, quality }
    --end
    --local ret, buf = cv2.imencode(img_fmt, img, encode_params)
    --assert(ret, 'failed to encode image')
    --return M.pack(header, buf.tostring())
end

return M
