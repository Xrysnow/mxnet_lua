--
local M = {}

local _build_param_doc = require('mx.base').build_param_doc

--local NDArrayDoc
--local ReshapeDoc
--local elemwise_addDoc
--local BroadcastToDoc
--local StackDoc
--local CustomDoc

function M._build_doc(func_name,
                      desc,
                      arg_names,
                      arg_types,
                      arg_desc,
                      key_var_num_args,
                      ret_type)
    local param_str = _build_param_doc(arg_names, arg_types, arg_desc)
    --if key_var_num_args then
    --    desc = desc .. '\nThis function support variable length of positional input.'
    --end
    local doc_str = [[%s
---
%s
---@param out any @NDArray, optional | The output NDArray to hold the result.
---@return mx.ndarray.NDArray|mx.ndarray.NDArray[] @The output of this function.
]]
    desc = string.gsub(desc, '\n  # ', '\n  -- ')
    desc = string.gsub(desc, '\n.. note:: ', '\n### Note: ')
    desc = string.gsub(desc, '\nExample:\n', '\n### Example\n')
    desc = string.gsub(desc, '\nExample::\n', '\n### Example\n')
    desc = string.gsub(desc, '\nExamples::\n', '\n### Examples\n')
    desc = string.gsub(desc, '\nReferences\n', '\n### References\n')
    desc = string.gsub(desc, '\nReferences:\n', '\n### References\n')
    desc = string.gsub(desc, '\nDefined in ', '\n### Defined in ')
    desc = string.gsub('--- ' .. desc, '\n', '\n--- ')
    --if param_str:sub(-1) == '\n' then
    --    param_str = param_str:sub(1, -2)
    --end
    doc_str = doc_str:format(desc, param_str)
    --local extra_doc
    --doc_str = doc_str .. extra_doc
    doc_str = doc_str:gsub('NDArray%-or%-Symbol', 'NDArray')
    return doc_str
end

return M
