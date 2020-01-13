local ffi = require('ffi')
local helper = require('util.helper')

ffi.cdef(helper.loadFileString('headers/c_api.h'))
ffi.cdef(helper.loadFileString('headers/c_api_nnvm.h'))
ffi.cdef(helper.loadFileString('headers/c_api_test.h'))
ffi.cdef(helper.loadFileString('headers/c_predict_api.h'))

local is_win = ffi.os == 'Windows'

print('start load libmxnet')

local lib
local last_path
if is_win then
    last_path = helper.getCurrentDirectory()
    helper.setCurrentDirectory(last_path .. '\\bin')
    lib = ffi.load('libmxnet.dll')
    helper.setCurrentDirectory(last_path)
else
    lib = ffi.load('libmxnet.so')
end

print('finish load libmxnet')

return lib
