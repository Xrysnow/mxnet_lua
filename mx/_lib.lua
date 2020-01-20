local ffi = require('ffi')
local helper = require('mx_util.helper')

ffi.cdef(helper.loadFileString('mx_headers/c_api.h'))
ffi.cdef(helper.loadFileString('mx_headers/c_api_nnvm.h'))
ffi.cdef(helper.loadFileString('mx_headers/c_api_test.h'))
--ffi.cdef(helper.loadFileString('mx_headers/c_predict_api.h'))

local is_win = ffi.os == 'Windows'

print('start load libmxnet')

local lib
if is_win then
    lib = ffi.load('libmxnet.dll')
else
    lib = ffi.load('libmxnet.so')
end

print('finish load libmxnet')

return lib
