--
local M = {}

M.typecodes = 'bBuhHiIlLqQfd'


---@class ArrayType
local ArrayType = class('ArrayType', {})

function ArrayType:ctor(typecode, initializer)
end

M.ArrayType = ArrayType

return M
