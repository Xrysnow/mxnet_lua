--
local M = {}

function M.info(msg, ...)
    print(msg:format(...))
end

function M.warning(msg, ...)
    print(msg:format(...))
end

return M
