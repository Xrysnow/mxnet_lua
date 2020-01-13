--

local M = {}
local _TYPEDEF = require('ctypes').typedef
local _ENUMDEF = require('ctypes').enumdef
local _CALL = require('ctypes').caller(require('lib'))
local _FUNCDEF = require('ctypes').addDef
-- header/c_api.h

--

---@brief manually define unsigned int
_TYPEDEF("mx_uint", "unsigned int")

--

---@brief manually define float
_TYPEDEF("mx_float", "float")

--

---@brief data type to store dim size
_TYPEDEF("dim_t", "int64_t")

--

---@brief handle to NDArray
_TYPEDEF("NDArrayHandle", "void *")

--

---@brief handle to a mxnet narray function that changes NDArray
_TYPEDEF("FunctionHandle", "const void *")

--

---@brief handle to a function that takes param and creates symbol
_TYPEDEF("AtomicSymbolCreator", "void *")

--

---@brief handle to cached operator
_TYPEDEF("CachedOpHandle", "void *")

--

---@brief handle to a symbol that can be bind as operator
_TYPEDEF("SymbolHandle", "void *")

--

---@brief handle to a AtomicSymbol
_TYPEDEF("AtomicSymbolHandle", "void *")

--

---@brief handle to an Executor
_TYPEDEF("ExecutorHandle", "void *")

--

---@brief handle a dataiter creator
_TYPEDEF("DataIterCreator", "void *")

--

---@brief handle to a DataIterator
_TYPEDEF("DataIterHandle", "void *")

--

---@brief handle to KVStore
_TYPEDEF("KVStoreHandle", "void *")

--

---@brief handle to RecordIO
_TYPEDEF("RecordIOHandle", "void *")

--

---@brief handle to MXRtc
_TYPEDEF("RtcHandle", "void *")

--

---@brief handle to rtc cuda module
_TYPEDEF("CudaModuleHandle", "void *")

--

---@brief handle to rtc cuda kernel
_TYPEDEF("CudaKernelHandle", "void *")

--

---@brief handle to a Profile object (domain, duration, counter, etc.)
_TYPEDEF("ProfileHandle", "void *")

--

---@brief handle to DLManagedTensor
_TYPEDEF("DLManagedTensorHandle", "void *")

--

---@brief handle to Context
_TYPEDEF("ContextHandle", "const void *")

--

---@brief handle to Engine FnProperty
_TYPEDEF("EngineFnPropertyHandle", "const void *")

--

---@brief handle to Engine VarHandle
_TYPEDEF("EngineVarHandle", "void *")

--

---@brief Engine asynchronous operation
_TYPEDEF("EngineAsyncFunc", "void*")

--

---@brief Engine synchronous operation
_TYPEDEF("EngineSyncFunc", "void*")

--

---@brief Callback to free the param for EngineAsyncFunc/EngineSyncFunc
_TYPEDEF("EngineFuncParamDeleter", "void*")

--

_TYPEDEF("ExecutorMonitorCallback", "void*")

--

_TYPEDEF("MXGenericCallback", "void*")

--

_ENUMDEF("CustomOpCallbacks")

--

_ENUMDEF("CustomOpPropCallbacks")

--

--- state
_TYPEDEF("CustomOpFBFunc", "void*")

--

--- state
_TYPEDEF("CustomOpDelFunc", "void*")

--

--- state
_TYPEDEF("CustomOpListFunc", "void*")

--

--- state
_TYPEDEF("CustomOpInferShapeFunc", "void*")

--

--- state
_TYPEDEF("CustomOpInferStorageTypeFunc", "void*")

--

--- state
_TYPEDEF("CustomOpBackwardInferStorageTypeFunc", "void*")

--

--- state
_TYPEDEF("CustomOpInferTypeFunc", "void*")

--

--- state
_TYPEDEF("CustomOpBwdDepFunc", "void*")

--

--- state
_TYPEDEF("CustomOpCreateFunc", "void*")

--

--- ret
_TYPEDEF("CustomOpPropCreator", "void*")

--

_ENUMDEF("CustomFunctionCallbacks")

--

--- state
_TYPEDEF("CustomFunctionBwdFunc", "void*")

--

--- state
_TYPEDEF("CustomFunctionDelFunc", "void*")

--

--- 
---@brief return str message of the last error
--- all function in this file will return 0 when success
--- and -1 when an error occured,
--- MXGetLastError can be called to retrieve the error
--- 
--- this function is threadsafe and can be called by different thread
---@return string @(const char *) error info
--- 
function M.MXGetLastError()
    return _CALL("MXGetLastError")
end
_FUNCDEF("MXGetLastError", {  }, "const char *")

--

--- 
---@brief Get list of features supported on the runtime
---@param libFeature number @(const struct LibFeature * *) pointer to array of LibFeature
---@param size number @(size_t *) of the array
---@return number @(int) 0 when success, -1 when failure happens.
--- 
function M.MXLibInfoFeatures(libFeature, size)
    return _CALL("MXLibInfoFeatures", libFeature, size)
end
_FUNCDEF("MXLibInfoFeatures", { "const struct LibFeature * *", "size_t *" }, "int")

--

--- 
---@brief Seed all global random number generators in mxnet.
---@param seed number @(int) the random number seed.
---@return number @(int) 0 when success, -1 when failure happens.
--- 
function M.MXRandomSeed(seed)
    return _CALL("MXRandomSeed", seed)
end
_FUNCDEF("MXRandomSeed", { "int" }, "int")

--

--- 
---@brief Seed the global random number generator of the given device.
---@param seed number @(int) the random number seed.
---@return number @(int) 0 when success, -1 when failure happens.
--- 
function M.MXRandomSeedContext(seed, dev_type, dev_id)
    return _CALL("MXRandomSeedContext", seed, dev_type, dev_id)
end
_FUNCDEF("MXRandomSeedContext", { "int", "int", "int" }, "int")

--

--- 
---@brief Notify the engine about a shutdown,
---  This can help engine to print less messages into display.
--- 
---  User do not have to call this function.
---@return number @(int) 0 when success, -1 when failure happens.
--- 
function M.MXNotifyShutdown()
    return _CALL("MXNotifyShutdown")
end
_FUNCDEF("MXNotifyShutdown", {  }, "int")

--

--- 
---@brief Set up configuration of profiler for the process passed as profile_process in keys
---@param num_params number @(int) Number of parameters
---@param keys number @(const char * const *) array of parameter keys
---@param vals number @(const char * const *) array of parameter values
---@param kvStoreHandle number @(KVStoreHandle) handle to kvstore
---@return number @(int) 0 when success, -1 when failure happens.
--- 
function M.MXSetProcessProfilerConfig(num_params, keys, vals, kvStoreHandle)
    return _CALL("MXSetProcessProfilerConfig", num_params, keys, vals, kvStoreHandle)
end
_FUNCDEF("MXSetProcessProfilerConfig", { "int", "const char * const *", "const char * const *", "KVStoreHandle" }, "int")

--

--- 
---@brief Set up configuration of profiler for worker/current process
---@param num_params number @(int) Number of parameters
---@param keys number @(const char * const *) array of parameter keys
---@param vals number @(const char * const *) array of parameter values
---@return number @(int) 0 when success, -1 when failure happens.
--- 
function M.MXSetProfilerConfig(num_params, keys, vals)
    return _CALL("MXSetProfilerConfig", num_params, keys, vals)
end
_FUNCDEF("MXSetProfilerConfig", { "int", "const char * const *", "const char * const *" }, "int")

--

--- 
---@brief Set up state of profiler for either worker or server process
---@param state number @(int) indicate the working state of profiler,
---  profiler not running when state == 0,
---  profiler running when state == 1
---@param profile_process number @(int) an int,
--- when 0 command is for worker/current process,
--- when 1 command is for server process
---@param kvStoreHandle number @(KVStoreHandle) handle to kvstore, needed for server process profiling
---@return number @(int) 0 when success, -1 when failure happens.
--- 
function M.MXSetProcessProfilerState(state, profile_process, kvStoreHandle)
    return _CALL("MXSetProcessProfilerState", state, profile_process, kvStoreHandle)
end
_FUNCDEF("MXSetProcessProfilerState", { "int", "int", "KVStoreHandle" }, "int")

--

--- 
---@brief Set up state of profiler for current process
---@param state number @(int) indicate the working state of profiler,
---  profiler not running when state == 0,
---  profiler running when state == 1
---@return number @(int) 0 when success, -1 when failure happens.
--- 
function M.MXSetProfilerState(state)
    return _CALL("MXSetProfilerState", state)
end
_FUNCDEF("MXSetProfilerState", { "int" }, "int")

--

--- 
---@brief Save profile and stop profiler
---@param finished number @(int) true if stat output should stop after this point
---@param profile_process number @(int) an int,
--- when 0 command is for worker/current process,
--- when 1 command is for server process
---@param kvStoreHandle number @(KVStoreHandle) handle to kvstore
---@return number @(int) 0 when success, -1 when failure happens.
--- 
function M.MXDumpProcessProfile(finished, profile_process, kvStoreHandle)
    return _CALL("MXDumpProcessProfile", finished, profile_process, kvStoreHandle)
end
_FUNCDEF("MXDumpProcessProfile", { "int", "int", "KVStoreHandle" }, "int")

--

--- 
---@brief Save profile and stop profiler for worker/current process
---@param finished number @(int) true if stat output should stop after this point
---@return number @(int) 0 when success, -1 when failure happens.
--- 
function M.MXDumpProfile(finished)
    return _CALL("MXDumpProfile", finished)
end
_FUNCDEF("MXDumpProfile", { "int" }, "int")

--

--- 
---@brief Deprecated, use MXAggregateProfileStatsPrintEx instead.
---@param out_str number @(const char * *) Will receive a pointer to the output string
---@param reset number @(int) Clear the aggregate stats after printing
---@return number @(int) 0 when success, -1 when failure happens.
---@note
--- 
function M.MXAggregateProfileStatsPrint(out_str, reset)
    return _CALL("MXAggregateProfileStatsPrint", out_str, reset)
end
_FUNCDEF("MXAggregateProfileStatsPrint", { "const char * *", "int" }, "int")

--

--- 
---@brief Print sorted aggregate stats to the a string
---        How aggregate stats are stored will not change
---@param out_str number @(const char * *) will receive a pointer to the output string
---@param reset number @(int) clear the aggregate stats after printing
---@param format number @(int) whether to return in tabular or json format
---@param sort_by number @(int) sort by avg, min, max, or count
---@param ascending number @(int) whether to sort ascendingly
---@return number @(int) 0 when success, -1 when failure happens.
---@note
--- 
function M.MXAggregateProfileStatsPrintEx(out_str, reset, format, sort_by, ascending)
    return _CALL("MXAggregateProfileStatsPrintEx", out_str, reset, format, sort_by, ascending)
end
_FUNCDEF("MXAggregateProfileStatsPrintEx", { "const char * *", "int", "int", "int", "int" }, "int")

--

--- 
---@brief Pause profiler tuning collection
---@param paused number @(int) If nonzero, profiling pauses. Otherwise, profiling resumes/continues
---@param profile_process number @(int) integer which denotes whether to process worker or server process
---@param kvStoreHandle number @(KVStoreHandle) handle to kvstore
---@return number @(int) 0 when success, -1 when failure happens.
---@note pausing and resuming is global and not recursive
--- 
function M.MXProcessProfilePause(paused, profile_process, kvStoreHandle)
    return _CALL("MXProcessProfilePause", paused, profile_process, kvStoreHandle)
end
_FUNCDEF("MXProcessProfilePause", { "int", "int", "KVStoreHandle" }, "int")

--

--- 
---@brief Pause profiler tuning collection for worker/current process
---@param paused number @(int) If nonzero, profiling pauses. Otherwise, profiling resumes/continues
---@return number @(int) 0 when success, -1 when failure happens.
---@note pausing and resuming is global and not recursive
--- 
function M.MXProfilePause(paused)
    return _CALL("MXProfilePause", paused)
end
_FUNCDEF("MXProfilePause", { "int" }, "int")

--

--- 
---@brief Create profiling domain
---@param domain string @(const char *) String representing the domain name to create
---@param out number @(ProfileHandle *) Return domain object
---@return number @(int) 0 when success, -1 when failure happens.
--- 
function M.MXProfileCreateDomain(domain, out)
    return _CALL("MXProfileCreateDomain", domain, out)
end
_FUNCDEF("MXProfileCreateDomain", { "const char *", "ProfileHandle *" }, "int")

--

--- 
---@brief Create profile task
---@param task_name string @(const char *) Name of the task
---@param domain number @(ProfileHandle) Domain of the task
---@param out number @(ProfileHandle *) Output handle
---@return number @(int) 0 when success, -1 when failure happens.
--- 
function M.MXProfileCreateTask(domain, task_name, out)
    return _CALL("MXProfileCreateTask", domain, task_name, out)
end
_FUNCDEF("MXProfileCreateTask", { "ProfileHandle", "const char *", "ProfileHandle *" }, "int")

--

--- 
---@brief Create profile frame
---@param frame_name string @(const char *) Name of the frame
---@param domain number @(ProfileHandle) Domain of the frame
---@param out number @(ProfileHandle *) Output handle
---@return number @(int) 0 when success, -1 when failure happens.
--- 
function M.MXProfileCreateFrame(domain, frame_name, out)
    return _CALL("MXProfileCreateFrame", domain, frame_name, out)
end
_FUNCDEF("MXProfileCreateFrame", { "ProfileHandle", "const char *", "ProfileHandle *" }, "int")

--

--- 
---@brief Create profile event
---@param event_name string @(const char *) Name of the event
---@param out number @(ProfileHandle *) Output handle
---@return number @(int) 0 when success, -1 when failure happens.
--- 
function M.MXProfileCreateEvent(event_name, out)
    return _CALL("MXProfileCreateEvent", event_name, out)
end
_FUNCDEF("MXProfileCreateEvent", { "const char *", "ProfileHandle *" }, "int")

--

--- 
---@brief Create profile counter
---@param counter_name string @(const char *) Name of the counter
---@param domain number @(ProfileHandle) Domain of the counter
---@param out number @(ProfileHandle *) Output handle
---@return number @(int) 0 when success, -1 when failure happens.
--- 
function M.MXProfileCreateCounter(domain, counter_name, out)
    return _CALL("MXProfileCreateCounter", domain, counter_name, out)
end
_FUNCDEF("MXProfileCreateCounter", { "ProfileHandle", "const char *", "ProfileHandle *" }, "int")

--

--- 
---@brief Destroy a frame
---@param frame_handle number @(ProfileHandle) Handle to frame to destroy
---@return number @(int) 0 when success, -1 when failure happens.
--- 
function M.MXProfileDestroyHandle(frame_handle)
    return _CALL("MXProfileDestroyHandle", frame_handle)
end
_FUNCDEF("MXProfileDestroyHandle", { "ProfileHandle" }, "int")

--

--- 
---@brief Start timing the duration of a profile duration object such as an event, task or frame
---@param duration_handle number @(ProfileHandle) handle to the duration object
---@return number @(int) 0 when success, -1 when failure happens.
--- 
function M.MXProfileDurationStart(duration_handle)
    return _CALL("MXProfileDurationStart", duration_handle)
end
_FUNCDEF("MXProfileDurationStart", { "ProfileHandle" }, "int")

--

--- 
---@brief Stop timing the duration of a profile duration object such as an event, task or frame
---@param duration_handle number @(ProfileHandle) handle to the duration object
---@return number @(int) 0 when success, -1 when failure happens.
--- 
function M.MXProfileDurationStop(duration_handle)
    return _CALL("MXProfileDurationStop", duration_handle)
end
_FUNCDEF("MXProfileDurationStop", { "ProfileHandle" }, "int")

--

--- 
---@brief Set a counter, given its handle
---@param counter_handle number @(ProfileHandle) Handle to counter to set
---@param value number @(uint64_t) Value to set the counter to (64-bit unsigned integer)
---@return number @(int) 0 when success, -1 when failure happens.
--- 
function M.MXProfileSetCounter(counter_handle, value)
    return _CALL("MXProfileSetCounter", counter_handle, value)
end
_FUNCDEF("MXProfileSetCounter", { "ProfileHandle", "uint64_t" }, "int")

--

--- 
---@brief Adjust a counter by the given amount, given its handle
---@param counter_handle number @(ProfileHandle) Handle to counter to adjust
---@param value number @(int64_t) Value to adjust the counter by (64-bit signed integer)
---@return number @(int) 0 when success, -1 when failure happens.
--- 
function M.MXProfileAdjustCounter(counter_handle, value)
    return _CALL("MXProfileAdjustCounter", counter_handle, value)
end
_FUNCDEF("MXProfileAdjustCounter", { "ProfileHandle", "int64_t" }, "int")

--

--- 
---@brief Mark a single instant in time
---@param domain number @(ProfileHandle) Domain of the marker
---@param instant_marker_name string @(const char *) Name of the marker
---@param scope string @(const char *) Scope of marker ('global', 'process', 'thread', 'task', 'marker')
---@return number @(int) 0 when success, -1 when failure happens.
--- 
function M.MXProfileSetMarker(domain, instant_marker_name, scope)
    return _CALL("MXProfileSetMarker", domain, instant_marker_name, scope)
end
_FUNCDEF("MXProfileSetMarker", { "ProfileHandle", "const char *", "const char *" }, "int")

--

--- 
---@brief Set the number of OMP threads to use
---@param thread_num number @(int) Number of OMP threads desired
---@return number @(int) 0 when success, -1 when failure happens.
--- 
function M.MXSetNumOMPThreads(thread_num)
    return _CALL("MXSetNumOMPThreads", thread_num)
end
_FUNCDEF("MXSetNumOMPThreads", { "int" }, "int")

--

--- 
---@brief set bulk execution limit
---@param bulk_size number @(int) new bulk_size
---@param prev_bulk_size number @(int *) previous bulk_size
--- 
function M.MXEngineSetBulkSize(bulk_size, prev_bulk_size)
    return _CALL("MXEngineSetBulkSize", bulk_size, prev_bulk_size)
end
_FUNCDEF("MXEngineSetBulkSize", { "int", "int *" }, "int")

--

--- 
---@brief Get the number of GPUs.
---@param out number @(int *) pointer to int that will hold the number of GPUs available.
---@return number @(int) 0 when success, -1 when failure happens.
--- 
function M.MXGetGPUCount(out)
    return _CALL("MXGetGPUCount", out)
end
_FUNCDEF("MXGetGPUCount", { "int *" }, "int")

--

--- 
---@brief get the free and total available memory on a GPU
---  Note: Deprecated, use MXGetGPUMemoryInformation64 instead.
---@param dev number @(int) the GPU number to query
---@param free_mem number @(int *) pointer to the integer holding free GPU memory
---@param total_mem number @(int *) pointer to the integer holding total GPU memory
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXGetGPUMemoryInformation(dev, free_mem, total_mem)
    return _CALL("MXGetGPUMemoryInformation", dev, free_mem, total_mem)
end
_FUNCDEF("MXGetGPUMemoryInformation", { "int", "int *", "int *" }, "int")

--

--- 
---@brief get the free and total available memory on a GPU
---@param dev number @(int) the GPU number to query
---@param free_mem number @(uint64_t *) pointer to the uint64_t holding free GPU memory
---@param total_mem number @(uint64_t *) pointer to the uint64_t holding total GPU memory
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXGetGPUMemoryInformation64(dev, free_mem, total_mem)
    return _CALL("MXGetGPUMemoryInformation64", dev, free_mem, total_mem)
end
_FUNCDEF("MXGetGPUMemoryInformation64", { "int", "uint64_t *", "uint64_t *" }, "int")

--

--- 
---@brief get the MXNet library version as an integer
---@param out number @(int *) pointer to the integer holding the version number
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXGetVersion(out)
    return _CALL("MXGetVersion", out)
end
_FUNCDEF("MXGetVersion", { "int *" }, "int")

--

--- 
---@brief create a NDArray handle that is not initialized
---  can be used to pass in as mutate variables
---  to hold the result of NDArray
---@param out number @(NDArrayHandle *) the returning handle
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXNDArrayCreateNone(out)
    return _CALL("MXNDArrayCreateNone", out)
end
_FUNCDEF("MXNDArrayCreateNone", { "NDArrayHandle *" }, "int")

--

--- 
---@brief create a NDArray with specified shape
---@param shape number @(const mx_uint *) the pointer to the shape
---@param ndim number @(mx_uint) the dimension of the shape
---@param dev_type number @(int) device type, specify device we want to take
---@param dev_id number @(int) the device id of the specific device
---@param delay_alloc number @(int) whether to delay allocation until
---    the narray is first mutated
---@param out number @(NDArrayHandle *) the returning handle
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXNDArrayCreate(shape, ndim, dev_type, dev_id, delay_alloc, out)
    return _CALL("MXNDArrayCreate", shape, ndim, dev_type, dev_id, delay_alloc, out)
end
_FUNCDEF("MXNDArrayCreate", { "const mx_uint *", "mx_uint", "int", "int", "int", "NDArrayHandle *" }, "int")

--

--- 
---@brief create a NDArray with specified shape and data type
---@param shape number @(const mx_uint *) the pointer to the shape
---@param ndim number @(mx_uint) the dimension of the shape
---@param dev_type number @(int) device type, specify device we want to take
---@param dev_id number @(int) the device id of the specific device
---@param delay_alloc number @(int) whether to delay allocation until
---    the narray is first mutated
---@param dtype number @(int) data type of created array
---@param out number @(NDArrayHandle *) the returning handle
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXNDArrayCreateEx(shape, ndim, dev_type, dev_id, delay_alloc, dtype, out)
    return _CALL("MXNDArrayCreateEx", shape, ndim, dev_type, dev_id, delay_alloc, dtype, out)
end
_FUNCDEF("MXNDArrayCreateEx", { "const mx_uint *", "mx_uint", "int", "int", "int", "int", "NDArrayHandle *" }, "int")

--

--- 
---@brief create an empty sparse NDArray with specified shape and data type
---@param storage_type number @(int) the storage type of the ndarray
---@param shape number @(const mx_uint *) the pointer to the shape
---@param ndim number @(mx_uint) the dimension of the shape
---@param dev_type number @(int) device type, specify device we want to take
---@param dev_id number @(int) the device id of the specific device
---@param delay_alloc number @(int) whether to delay allocation until
---        the narray is first mutated
---@param dtype number @(int) data type of created array
---@param num_aux number @(mx_uint) the number of aux data to support this ndarray
---@param aux_type number @(int *) data type of the aux data for the created array
---@param aux_ndims number @(mx_uint *) the dimension of the shapes of aux data
---@param aux_shape number @(const mx_uint *) the shapes of aux data
---@param out number @(NDArrayHandle *) the returning handle
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXNDArrayCreateSparseEx(storage_type, shape, ndim, dev_type, dev_id, delay_alloc, dtype, num_aux, aux_type, aux_ndims, aux_shape, out)
    return _CALL("MXNDArrayCreateSparseEx", storage_type, shape, ndim, dev_type, dev_id, delay_alloc, dtype, num_aux, aux_type, aux_ndims, aux_shape, out)
end
_FUNCDEF("MXNDArrayCreateSparseEx", { "int", "const mx_uint *", "mx_uint", "int", "int", "int", "int", "mx_uint", "int *", "mx_uint *", "const mx_uint *", "NDArrayHandle *" }, "int")

--

--- 
---@brief create a NDArray handle that is loaded from raw bytes.
---@param buf number @(const void *) the head of the raw bytes
---@param size number @(size_t) size of the raw bytes
---@param out number @(NDArrayHandle *) the returning handle
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXNDArrayLoadFromRawBytes(buf, size, out)
    return _CALL("MXNDArrayLoadFromRawBytes", buf, size, out)
end
_FUNCDEF("MXNDArrayLoadFromRawBytes", { "const void *", "size_t", "NDArrayHandle *" }, "int")

--

--- 
---@brief save the NDArray into raw bytes.
---@param handle number @(NDArrayHandle) the NDArray handle
---@param out_size number @(size_t *) size of the raw bytes
---@param out_buf number @(const char * *) the head of returning memory bytes.
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXNDArraySaveRawBytes(handle, out_size, out_buf)
    return _CALL("MXNDArraySaveRawBytes", handle, out_size, out_buf)
end
_FUNCDEF("MXNDArraySaveRawBytes", { "NDArrayHandle", "size_t *", "const char * *" }, "int")

--

--- 
---@brief Save list of narray into the file.
---@param fname string @(const char *) name of the file.
---@param num_args number @(mx_uint) number of arguments to save.
---@param args number @(NDArrayHandle *) the array of NDArrayHandles to be saved.
---@param keys number @(const char * *) the name of the NDArray, optional, can be NULL
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXNDArraySave(fname, num_args, args, keys)
    return _CALL("MXNDArraySave", fname, num_args, args, keys)
end
_FUNCDEF("MXNDArraySave", { "const char *", "mx_uint", "NDArrayHandle *", "const char * *" }, "int")

--

--- 
---@brief Load list of narray from the file.
---@param fname string @(const char *) name of the file.
---@param out_size number @(mx_uint *) number of narray loaded.
---@param out_arr number @(NDArrayHandle * *) head of the returning narray handles.
---@param out_name_size number @(mx_uint *) size of output name arrray.
---@param out_names number @(const char * * *) the names of returning NDArrays, can be NULL
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXNDArrayLoad(fname, out_size, out_arr, out_name_size, out_names)
    return _CALL("MXNDArrayLoad", fname, out_size, out_arr, out_name_size, out_names)
end
_FUNCDEF("MXNDArrayLoad", { "const char *", "mx_uint *", "NDArrayHandle * *", "mx_uint *", "const char * * *" }, "int")

--

--- 
---@brief Load list / dictionary of narrays from file content loaded into memory.
--- This will load a list of ndarrays in a similar
--- manner to MXNDArrayLoad, however, it loads from
--- buffer containing the contents of a file, rather than
--- from a specified file.
---@param ndarray_buffer number @(const void *) pointer to the start of the ndarray file content
---@param size number @(size_t) size of the file
---@param out_size number @(mx_uint *) number of narray loaded.
---@param out_arr number @(NDArrayHandle * *) head of the returning narray handles.
---@param out_name_size number @(mx_uint *) size of output name arrray.
---@param out_names number @(const char * * *) the names of returning NDArrays, can be NULL
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXNDArrayLoadFromBuffer(ndarray_buffer, size, out_size, out_arr, out_name_size, out_names)
    return _CALL("MXNDArrayLoadFromBuffer", ndarray_buffer, size, out_size, out_arr, out_name_size, out_names)
end
_FUNCDEF("MXNDArrayLoadFromBuffer", { "const void *", "size_t", "mx_uint *", "NDArrayHandle * *", "mx_uint *", "const char * * *" }, "int")

--

--- 
---@brief Perform a synchronize copy from a continugous CPU memory region.
--- 
---  This function will call WaitToWrite before the copy is performed.
---  This is useful to copy data from existing memory region that are
---  not wrapped by NDArray(thus dependency not being tracked).
--- 
---@param handle number @(NDArrayHandle) the NDArray handle
---@param data number @(const void *) the data source to copy from.
---@param size number @(size_t) the memory size we want to copy from.
--- 
function M.MXNDArraySyncCopyFromCPU(handle, data, size)
    return _CALL("MXNDArraySyncCopyFromCPU", handle, data, size)
end
_FUNCDEF("MXNDArraySyncCopyFromCPU", { "NDArrayHandle", "const void *", "size_t" }, "int")

--

--- 
---@brief Perform a synchronize copyto a continugous CPU memory region.
--- 
---  This function will call WaitToRead before the copy is performed.
---  This is useful to copy data from existing memory region that are
---  not wrapped by NDArray(thus dependency not being tracked).
--- 
---@param handle number @(NDArrayHandle) the NDArray handle
---@param data number @(void *) the data source to copy into.
---@param size number @(size_t) the memory size we want to copy into.
--- 
function M.MXNDArraySyncCopyToCPU(handle, data, size)
    return _CALL("MXNDArraySyncCopyToCPU", handle, data, size)
end
_FUNCDEF("MXNDArraySyncCopyToCPU", { "NDArrayHandle", "void *", "size_t" }, "int")

--

--- 
---@brief Copy src.data() to dst.data() if i = -1, else dst.aux_data(i) if i >= 0
--- This function blocks. Do not use it in performance critical code.
---@param handle_dst number @(NDArrayHandle) handle of a dst ndarray whose data/aux_data has been allocated
---@param handle_src number @(const NDArrayHandle) handle of a src ndarray which has default storage type
---@param i number @(const int) dst data blob indicator
--- 
function M.MXNDArraySyncCopyFromNDArray(handle_dst, handle_src, i)
    return _CALL("MXNDArraySyncCopyFromNDArray", handle_dst, handle_src, i)
end
_FUNCDEF("MXNDArraySyncCopyFromNDArray", { "NDArrayHandle", "const NDArrayHandle", "const int" }, "int")

--

--- 
---@brief check whether the NDArray format is valid
---@param full_check boolean @(const bool) if `True`, rigorous check, O(N) operations
---    Otherwise basic check, O(1) operations
--- 
function M.MXNDArraySyncCheckFormat(handle, full_check)
    return _CALL("MXNDArraySyncCheckFormat", handle, full_check)
end
_FUNCDEF("MXNDArraySyncCheckFormat", { "NDArrayHandle", "const bool" }, "int")

--

--- 
---@brief Wait until all the pending writes with respect NDArray are finished.
---  Always call this before read data out synchronizely.
---@param handle number @(NDArrayHandle) the NDArray handle
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXNDArrayWaitToRead(handle)
    return _CALL("MXNDArrayWaitToRead", handle)
end
_FUNCDEF("MXNDArrayWaitToRead", { "NDArrayHandle" }, "int")

--

--- 
---@brief Wait until all the pending read/write with respect NDArray are finished.
---  Always call this before write data into NDArray synchronizely.
---@param handle number @(NDArrayHandle) the NDArray handle
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXNDArrayWaitToWrite(handle)
    return _CALL("MXNDArrayWaitToWrite", handle)
end
_FUNCDEF("MXNDArrayWaitToWrite", { "NDArrayHandle" }, "int")

--

--- 
---@brief wait until all delayed operations in
---   the system is completed
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXNDArrayWaitAll()
    return _CALL("MXNDArrayWaitAll")
end
_FUNCDEF("MXNDArrayWaitAll", {  }, "int")

--

--- 
---@brief free the narray handle
---@param handle number @(NDArrayHandle) the handle to be freed
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXNDArrayFree(handle)
    return _CALL("MXNDArrayFree", handle)
end
_FUNCDEF("MXNDArrayFree", { "NDArrayHandle" }, "int")

--

--- 
---@brief Slice the NDArray along axis 0.
---@param handle number @(NDArrayHandle) the handle to the NDArray
---@param slice_begin number @(mx_uint) The beginning index of slice
---@param slice_end number @(mx_uint) The ending index of slice
---@param out number @(NDArrayHandle *) The NDArrayHandle of sliced NDArray
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXNDArraySlice(handle, slice_begin, slice_end, out)
    return _CALL("MXNDArraySlice", handle, slice_begin, slice_end, out)
end
_FUNCDEF("MXNDArraySlice", { "NDArrayHandle", "mx_uint", "mx_uint", "NDArrayHandle *" }, "int")

--

--- 
---@brief Index the NDArray along axis 0.
---@param handle number @(NDArrayHandle) the handle to the NDArray
---@param idx number @(mx_uint) the index
---@param out number @(NDArrayHandle *) The NDArrayHandle of output NDArray
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXNDArrayAt(handle, idx, out)
    return _CALL("MXNDArrayAt", handle, idx, out)
end
_FUNCDEF("MXNDArrayAt", { "NDArrayHandle", "mx_uint", "NDArrayHandle *" }, "int")

--

--- 
---@brief get the storage type of the array
--- 
function M.MXNDArrayGetStorageType(handle, out_storage_type)
    return _CALL("MXNDArrayGetStorageType", handle, out_storage_type)
end
_FUNCDEF("MXNDArrayGetStorageType", { "NDArrayHandle", "int *" }, "int")

--

--- 
---@brief Reshape the NDArray.
---@param handle number @(NDArrayHandle) the handle to the narray
---@param ndim number @(int) number of dimensions of new shape
---@param dims number @(int *) new shape
---@param out number @(NDArrayHandle *) the NDArrayHandle of reshaped NDArray
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXNDArrayReshape(handle, ndim, dims, out)
    return _CALL("MXNDArrayReshape", handle, ndim, dims, out)
end
_FUNCDEF("MXNDArrayReshape", { "NDArrayHandle", "int", "int *", "NDArrayHandle *" }, "int")

--

--- 
---@brief Reshape the NDArray.
---@param handle number @(NDArrayHandle) the handle to the narray
---@param ndim number @(int) number of dimensions of new shape
---@param dims number @(dim_t *) new shape
---@param out number @(NDArrayHandle *) the NDArrayHandle of reshaped NDArray
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXNDArrayReshape64(handle, ndim, dims, reverse, out)
    return _CALL("MXNDArrayReshape64", handle, ndim, dims, reverse, out)
end
_FUNCDEF("MXNDArrayReshape64", { "NDArrayHandle", "int", "dim_t *", "bool", "NDArrayHandle *" }, "int")

--

--- 
---@brief DEPRECATED. Use MXNDArrayGetShapeEx instead.
--- get the shape of the array
---@param handle number @(NDArrayHandle) the handle to the narray
---@param out_dim number @(mx_uint *) the output dimension
---@param out_pdata number @(const mx_uint * *) pointer holder to get data pointer of the shape
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXNDArrayGetShape(handle, out_dim, out_pdata)
    return _CALL("MXNDArrayGetShape", handle, out_dim, out_pdata)
end
_FUNCDEF("MXNDArrayGetShape", { "NDArrayHandle", "mx_uint *", "const mx_uint * *" }, "int")

--

--- 
---@brief get the shape of the array
---@param handle number @(NDArrayHandle) the handle to the narray
---@param out_dim number @(int *) the output dimension
---@param out_pdata number @(const int * *) pointer holder to get data pointer of the shape
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXNDArrayGetShapeEx(handle, out_dim, out_pdata)
    return _CALL("MXNDArrayGetShapeEx", handle, out_dim, out_pdata)
end
_FUNCDEF("MXNDArrayGetShapeEx", { "NDArrayHandle", "int *", "const int * *" }, "int")

--

--- 
---@brief get the content of the data in NDArray
---@param handle number @(NDArrayHandle) the handle to the ndarray
---@param out_pdata number @(void * *) pointer holder to get pointer of data
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXNDArrayGetData(handle, out_pdata)
    return _CALL("MXNDArrayGetData", handle, out_pdata)
end
_FUNCDEF("MXNDArrayGetData", { "NDArrayHandle", "void * *" }, "int")

--

--- 
---@brief Create a reference view of NDArray that
---  represents as DLManagedTensor
---  Notice: MXNet uses asynchronous execution. Please call MXNDArrayWaitToRead or
---          MXNDArrayWaitToWrite before calling MXNDArrayToDLPack.
---@param handle number @(NDArrayHandle) the handle to the ndarray
---@param out_dlpack number @(DLManagedTensorHandle *) pointer holder to get pointer of DLManagedTensor
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXNDArrayToDLPack(handle, out_dlpack)
    return _CALL("MXNDArrayToDLPack", handle, out_dlpack)
end
_FUNCDEF("MXNDArrayToDLPack", { "NDArrayHandle", "DLManagedTensorHandle *" }, "int")

--

--- 
---@brief DEPRECATED. Use MXNDArrayFromDLPackEx instead.
--- 
--- 
--- This allows us to create a NDArray using the memory
--- allocated by an external deep learning framework
--- that is DLPack compatible.
--- 
--- The memory is retained until the NDArray went out of scope.
--- 
---@param dlpack number @(DLManagedTensorHandle) the pointer of the input DLManagedTensor
---@param transient_handle whether the handle will be destructed before calling the deleter
---@param out_handle number @(NDArrayHandle *) pointer holder to get pointer of NDArray
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXNDArrayFromDLPack(dlpack, out_handle)
    return _CALL("MXNDArrayFromDLPack", dlpack, out_handle)
end
_FUNCDEF("MXNDArrayFromDLPack", { "DLManagedTensorHandle", "NDArrayHandle *" }, "int")

--

--- 
---@brief Create a NDArray backed by a dlpack tensor.
--- 
--- This allows us to create a NDArray using the memory
--- allocated by an external deep learning framework
--- that is DLPack compatible.
--- 
--- The memory is retained until the NDArray went out of scope.
--- 
---@param dlpack number @(DLManagedTensorHandle) the pointer of the input DLManagedTensor
---@param transient_handle boolean @(const bool) whether the handle will be destructed before calling the deleter
---@param out_handle number @(NDArrayHandle *) pointer holder to get pointer of NDArray
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXNDArrayFromDLPackEx(dlpack, transient_handle, out_handle)
    return _CALL("MXNDArrayFromDLPackEx", dlpack, transient_handle, out_handle)
end
_FUNCDEF("MXNDArrayFromDLPackEx", { "DLManagedTensorHandle", "const bool", "NDArrayHandle *" }, "int")

--

--- 
---@brief Delete a dlpack tensor
---@param dlpack number @(DLManagedTensorHandle) the pointer of the input DLManagedTensor
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXNDArrayCallDLPackDeleter(dlpack)
    return _CALL("MXNDArrayCallDLPackDeleter", dlpack)
end
_FUNCDEF("MXNDArrayCallDLPackDeleter", { "DLManagedTensorHandle" }, "int")

--

--- 
---@brief get the type of the data in NDArray
---@param handle number @(NDArrayHandle) the handle to the narray
---@param out_dtype number @(int *) pointer holder to get type of data
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXNDArrayGetDType(handle, out_dtype)
    return _CALL("MXNDArrayGetDType", handle, out_dtype)
end
_FUNCDEF("MXNDArrayGetDType", { "NDArrayHandle", "int *" }, "int")

--

--- 
---@brief get the type of the ith aux data in NDArray
---@param handle number @(NDArrayHandle) the handle to the narray
---@param i number @(mx_uint) the index of the aux data
---@param out_type number @(int *) pointer holder to get type of aux data
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXNDArrayGetAuxType(handle, i, out_type)
    return _CALL("MXNDArrayGetAuxType", handle, i, out_type)
end
_FUNCDEF("MXNDArrayGetAuxType", { "NDArrayHandle", "mx_uint", "int *" }, "int")

--

--- 
---@brief Get a deep copy of the ith aux data blob
--- in the form of an NDArray of default storage type.
--- This function blocks. Do not use it in performance critical code.
--- 
function M.MXNDArrayGetAuxNDArray(handle, i, out)
    return _CALL("MXNDArrayGetAuxNDArray", handle, i, out)
end
_FUNCDEF("MXNDArrayGetAuxNDArray", { "NDArrayHandle", "mx_uint", "NDArrayHandle *" }, "int")

--

--- 
---@brief Get a deep copy of the data blob
--- in the form of an NDArray of default storage type.
--- This function blocks. Do not use it in performance critical code.
--- 
function M.MXNDArrayGetDataNDArray(handle, out)
    return _CALL("MXNDArrayGetDataNDArray", handle, out)
end
_FUNCDEF("MXNDArrayGetDataNDArray", { "NDArrayHandle", "NDArrayHandle *" }, "int")

--

--- 
---@brief get the context of the NDArray
---@param handle number @(NDArrayHandle) the handle to the narray
---@param out_dev_type number @(int *) the output device type
---@param out_dev_id number @(int *) the output device id
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXNDArrayGetContext(handle, out_dev_type, out_dev_id)
    return _CALL("MXNDArrayGetContext", handle, out_dev_type, out_dev_id)
end
_FUNCDEF("MXNDArrayGetContext", { "NDArrayHandle", "int *", "int *" }, "int")

--

--- 
---@brief return gradient buffer attached to this NDArray
---@param handle number @(NDArrayHandle) NDArray handle
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXNDArrayGetGrad(handle, out)
    return _CALL("MXNDArrayGetGrad", handle, out)
end
_FUNCDEF("MXNDArrayGetGrad", { "NDArrayHandle", "NDArrayHandle *" }, "int")

--

--- 
---@brief detach and ndarray from computation graph by clearing entry_
---@param handle number @(NDArrayHandle) NDArray handle
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXNDArrayDetach(handle, out)
    return _CALL("MXNDArrayDetach", handle, out)
end
_FUNCDEF("MXNDArrayDetach", { "NDArrayHandle", "NDArrayHandle *" }, "int")

--

--- 
---@brief set the flag for gradient array state.
---@param handle number @(NDArrayHandle) NDArray handle
---@param state number @(int) the new state.
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXNDArraySetGradState(handle, state)
    return _CALL("MXNDArraySetGradState", handle, state)
end
_FUNCDEF("MXNDArraySetGradState", { "NDArrayHandle", "int" }, "int")

--

--- 
---@brief get the flag for gradient array state.
---@param handle number @(NDArrayHandle) NDArray handle
---@param out number @(int *) state.
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXNDArrayGetGradState(handle, out)
    return _CALL("MXNDArrayGetGradState", handle, out)
end
_FUNCDEF("MXNDArrayGetGradState", { "NDArrayHandle", "int *" }, "int")

--

--- 
---@brief list all the available functions handles
---   most user can use it to list all the needed functions
---@param out_size number @(mx_uint *) the size of returned array
---@param out_array number @(FunctionHandle * *) the output function array
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXListFunctions(out_size, out_array)
    return _CALL("MXListFunctions", out_size, out_array)
end
_FUNCDEF("MXListFunctions", { "mx_uint *", "FunctionHandle * *" }, "int")

--

--- 
---@brief get the function handle by name
---@param name string @(const char *) the name of the function
---@param out number @(FunctionHandle *) the corresponding function handle
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXGetFunction(name, out)
    return _CALL("MXGetFunction", name, out)
end
_FUNCDEF("MXGetFunction", { "const char *", "FunctionHandle *" }, "int")

--

--- 
---@brief Get the information of the function handle.
---@param fun number @(FunctionHandle) The function handle.
---@param name number @(const char * *) The returned name of the function.
---@param description number @(const char * *) The returned description of the function.
---@param num_args number @(mx_uint *) Number of arguments.
---@param arg_names number @(const char * * *) Name of the arguments.
---@param arg_type_infos number @(const char * * *) Type information about the arguments.
---@param arg_descriptions number @(const char * * *) Description information about the arguments.
---@param return_type number @(const char * *) Return type of the function.
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXFuncGetInfo(fun, name, description, num_args, arg_names, arg_type_infos, arg_descriptions, return_type)
    return _CALL("MXFuncGetInfo", fun, name, description, num_args, arg_names, arg_type_infos, arg_descriptions, return_type)
end
_FUNCDEF("MXFuncGetInfo", { "FunctionHandle", "const char * *", "const char * *", "mx_uint *", "const char * * *", "const char * * *", "const char * * *", "const char * *" }, "int")

--

--- 
---@brief get the argument requirements of the function
---@param fun number @(FunctionHandle) input function handle
---@param num_use_vars number @(mx_uint *) how many NDArrays to be passed in as used_vars
---@param num_scalars number @(mx_uint *) scalar variable is needed
---@param num_mutate_vars number @(mx_uint *) how many NDArrays to be passed in as mutate_vars
---@param type_mask number @(int *) the type mask of this function
---@return number @(int) 0 when success, -1 when failure happens
---@sa MXFuncInvoke
--- 
function M.MXFuncDescribe(fun, num_use_vars, num_scalars, num_mutate_vars, type_mask)
    return _CALL("MXFuncDescribe", fun, num_use_vars, num_scalars, num_mutate_vars, type_mask)
end
_FUNCDEF("MXFuncDescribe", { "FunctionHandle", "mx_uint *", "mx_uint *", "mx_uint *", "int *" }, "int")

--

--- 
---@brief invoke a function, the array size of passed in arguments
---   must match the values in the
---@param fun number @(FunctionHandle) the function
---@param use_vars number @(NDArrayHandle *) the normal arguments passed to function
---@param scalar_args number @(mx_float *) the scalar qarguments
---@param mutate_vars number @(NDArrayHandle *) the mutate arguments
---@return number @(int) 0 when success, -1 when failure happens
---@sa MXFuncDescribeArgs
--- 
function M.MXFuncInvoke(fun, use_vars, scalar_args, mutate_vars)
    return _CALL("MXFuncInvoke", fun, use_vars, scalar_args, mutate_vars)
end
_FUNCDEF("MXFuncInvoke", { "FunctionHandle", "NDArrayHandle *", "mx_float *", "NDArrayHandle *" }, "int")

--

--- 
---@brief invoke a function, the array size of passed in arguments
---   must match the values in the
---@param fun number @(FunctionHandle) the function
---@param use_vars number @(NDArrayHandle *) the normal arguments passed to function
---@param scalar_args number @(mx_float *) the scalar qarguments
---@param mutate_vars number @(NDArrayHandle *) the mutate arguments
---@param num_params number @(int) number of keyword parameters
---@param param_keys number @(char * *) keys for keyword parameters
---@param param_vals number @(char * *) values for keyword parameters
---@return number @(int) 0 when success, -1 when failure happens
---@sa MXFuncDescribeArgs
--- 
function M.MXFuncInvokeEx(fun, use_vars, scalar_args, mutate_vars, num_params, param_keys, param_vals)
    return _CALL("MXFuncInvokeEx", fun, use_vars, scalar_args, mutate_vars, num_params, param_keys, param_vals)
end
_FUNCDEF("MXFuncInvokeEx", { "FunctionHandle", "NDArrayHandle *", "mx_float *", "NDArrayHandle *", "int", "char * *", "char * *" }, "int")

--

--- 
---@brief invoke a nnvm op and imperative function
---@param creator number @(AtomicSymbolCreator) the op
---@param num_inputs number @(int) number of input NDArrays
---@param inputs number @(NDArrayHandle *) input NDArrays
---@param num_outputs number @(int *) number of output NDArrays
---@param outputs number @(NDArrayHandle * *) output NDArrays
---@param num_params number @(int) number of keyword parameters
---@param param_keys number @(const char * *) keys for keyword parameters
---@param param_vals number @(const char * *) values for keyword parameters
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXImperativeInvoke(creator, num_inputs, inputs, num_outputs, outputs, num_params, param_keys, param_vals)
    return _CALL("MXImperativeInvoke", creator, num_inputs, inputs, num_outputs, outputs, num_params, param_keys, param_vals)
end
_FUNCDEF("MXImperativeInvoke", { "AtomicSymbolCreator", "int", "NDArrayHandle *", "int *", "NDArrayHandle * *", "int", "const char * *", "const char * *" }, "int")

--

--- 
---@brief invoke a nnvm op and imperative function
---@param creator number @(AtomicSymbolCreator) the op
---@param num_inputs number @(int) number of input NDArrays
---@param inputs number @(NDArrayHandle *) input NDArrays
---@param num_outputs number @(int *) number of output NDArrays
---@param outputs number @(NDArrayHandle * *) output NDArrays
---@param num_params number @(int) number of keyword parameters
---@param param_keys number @(const char * *) keys for keyword parameters
---@param param_vals number @(const char * *) values for keyword parameters
---@param out_stypes number @(const int * *) output ndarrays' stypes
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXImperativeInvokeEx(creator, num_inputs, inputs, num_outputs, outputs, num_params, param_keys, param_vals, out_stypes)
    return _CALL("MXImperativeInvokeEx", creator, num_inputs, inputs, num_outputs, outputs, num_params, param_keys, param_vals, out_stypes)
end
_FUNCDEF("MXImperativeInvokeEx", { "AtomicSymbolCreator", "int", "NDArrayHandle *", "int *", "NDArrayHandle * *", "int", "const char * *", "const char * *", "const int * *" }, "int")

--

--- 
---@brief set whether to record operator for autograd
---@param is_recording number @(int) 1 when recording, 0 when not recording.
---@param prev number @(int *) returns the previous status before this set.
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXAutogradSetIsRecording(is_recording, prev)
    return _CALL("MXAutogradSetIsRecording", is_recording, prev)
end
_FUNCDEF("MXAutogradSetIsRecording", { "int", "int *" }, "int")

--

--- 
---@brief set whether to record operator for autograd
---@param is_training number @(int) 1 when training, 0 when testing
---@param prev number @(int *) returns the previous status before this set.
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXAutogradSetIsTraining(is_training, prev)
    return _CALL("MXAutogradSetIsTraining", is_training, prev)
end
_FUNCDEF("MXAutogradSetIsTraining", { "int", "int *" }, "int")

--

--- 
---@brief get whether autograd recording is on
---@param curr number @(bool *) returns the current status.
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXAutogradIsRecording(curr)
    return _CALL("MXAutogradIsRecording", curr)
end
_FUNCDEF("MXAutogradIsRecording", { "bool *" }, "int")

--

--- 
---@brief get whether training mode is on
---@param curr number @(bool *) returns the current status.
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXAutogradIsTraining(curr)
    return _CALL("MXAutogradIsTraining", curr)
end
_FUNCDEF("MXAutogradIsTraining", { "bool *" }, "int")

--

--- 
---@brief get whether numpy compatibility is on
---@param curr number @(bool *) returns the current status
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXIsNumpyShape(curr)
    return _CALL("MXIsNumpyShape", curr)
end
_FUNCDEF("MXIsNumpyShape", { "bool *" }, "int")

--

--- 
---@brief set numpy compatibility switch
---@param is_np_shape number @(int) 1 when numpy shape semantics is on, 0 when off
---@param prev number @(int *) returns the previous status before this set
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXSetIsNumpyShape(is_np_shape, prev)
    return _CALL("MXSetIsNumpyShape", is_np_shape, prev)
end
_FUNCDEF("MXSetIsNumpyShape", { "int", "int *" }, "int")

--

--- 
---@brief mark NDArrays as variables to compute gradient for autograd
---@param num_var number @(mx_uint) number of variable NDArrays
---@param var_handles number @(NDArrayHandle *) variable NDArrays
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXAutogradMarkVariables(num_var, var_handles, reqs_array, grad_handles)
    return _CALL("MXAutogradMarkVariables", num_var, var_handles, reqs_array, grad_handles)
end
_FUNCDEF("MXAutogradMarkVariables", { "mx_uint", "NDArrayHandle *", "mx_uint *", "NDArrayHandle *" }, "int")

--

--- 
---@brief compute the gradient of outputs w.r.t variabels
---@param num_output number @(mx_uint) number of output NDArray
---@param output_handles number @(NDArrayHandle *) output NDArrays
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXAutogradComputeGradient(num_output, output_handles)
    return _CALL("MXAutogradComputeGradient", num_output, output_handles)
end
_FUNCDEF("MXAutogradComputeGradient", { "mx_uint", "NDArrayHandle *" }, "int")

--

--- 
---@brief compute the gradient of outputs w.r.t variabels
---@param num_output number @(mx_uint) number of output NDArray
---@param output_handles number @(NDArrayHandle *) output NDArrays
---@param ograd_handles number @(NDArrayHandle *) head gradient for NDArrays
---@param retain_graph number @(int) whether to keep the graph after backward
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXAutogradBackward(num_output, output_handles, ograd_handles, retain_graph)
    return _CALL("MXAutogradBackward", num_output, output_handles, ograd_handles, retain_graph)
end
_FUNCDEF("MXAutogradBackward", { "mx_uint", "NDArrayHandle *", "NDArrayHandle *", "int" }, "int")

--

--- 
---@brief compute the gradient of outputs w.r.t variabels
---@param num_output number @(mx_uint) number of output NDArray
---@param output_handles number @(NDArrayHandle *) output NDArrays
---@param ograd_handles number @(NDArrayHandle *) head gradient for NDArrays
---@param num_variables number @(mx_uint) number of variables
---@param
---@param retain_graph number @(int) whether to keep the graph after backward
---@param is_train number @(int) whether to do backward for training or inference
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXAutogradBackwardEx(num_output, output_handles, ograd_handles, num_variables, var_handles, retain_graph, create_graph, is_train, grad_handles, grad_stypes)
    return _CALL("MXAutogradBackwardEx", num_output, output_handles, ograd_handles, num_variables, var_handles, retain_graph, create_graph, is_train, grad_handles, grad_stypes)
end
_FUNCDEF("MXAutogradBackwardEx", { "mx_uint", "NDArrayHandle *", "NDArrayHandle *", "mx_uint", "NDArrayHandle *", "int", "int", "int", "NDArrayHandle * *", "int * *" }, "int")

--

--- 
---@brief get the graph constructed by autograd.
---@param handle number @(NDArrayHandle) ndarray handle
---@param out number @(SymbolHandle *) output symbol handle
--- 
function M.MXAutogradGetSymbol(handle, out)
    return _CALL("MXAutogradGetSymbol", handle, out)
end
_FUNCDEF("MXAutogradGetSymbol", { "NDArrayHandle", "SymbolHandle *" }, "int")

--

--- 
---@brief create cached operator
--- 
function M.MXCreateCachedOp(handle, out)
    return _CALL("MXCreateCachedOp", handle, out)
end
_FUNCDEF("MXCreateCachedOp", { "SymbolHandle", "CachedOpHandle *" }, "int")

--

--- 
---@brief create cached operator
--- 
function M.MXCreateCachedOpEx(handle, num_flags, keys, vals, out)
    return _CALL("MXCreateCachedOpEx", handle, num_flags, keys, vals, out)
end
_FUNCDEF("MXCreateCachedOpEx", { "SymbolHandle", "int", "const char * *", "const char * *", "CachedOpHandle *" }, "int")

--

--- 
---@brief free cached operator
--- 
function M.MXFreeCachedOp(handle)
    return _CALL("MXFreeCachedOp", handle)
end
_FUNCDEF("MXFreeCachedOp", { "CachedOpHandle" }, "int")

--

--- 
---@brief invoke cached operator
--- 
function M.MXInvokeCachedOp(handle, num_inputs, inputs, num_outputs, outputs)
    return _CALL("MXInvokeCachedOp", handle, num_inputs, inputs, num_outputs, outputs)
end
_FUNCDEF("MXInvokeCachedOp", { "CachedOpHandle", "int", "NDArrayHandle *", "int *", "NDArrayHandle * *" }, "int")

--

--- 
---@brief invoke a cached op
---@param handle number @(CachedOpHandle) the handle to the cached op
---@param num_inputs number @(int) number of input NDArrays
---@param inputs number @(NDArrayHandle *) input NDArrays
---@param num_outputs number @(int *) number of output NDArrays
---@param outputs number @(NDArrayHandle * *) output NDArrays
---@param out_stypes number @(const int * *) output ndarrays' stypes
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXInvokeCachedOpEx(handle, num_inputs, inputs, num_outputs, outputs, out_stypes)
    return _CALL("MXInvokeCachedOpEx", handle, num_inputs, inputs, num_outputs, outputs, out_stypes)
end
_FUNCDEF("MXInvokeCachedOpEx", { "CachedOpHandle", "int", "NDArrayHandle *", "int *", "NDArrayHandle * *", "const int * *" }, "int")

--

--- 
---@brief list all the available operator names, include entries.
---@param out_size number @(mx_uint *) the size of returned array
---@param out_array number @(const char * * *) the output operator name array.
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXListAllOpNames(out_size, out_array)
    return _CALL("MXListAllOpNames", out_size, out_array)
end
_FUNCDEF("MXListAllOpNames", { "mx_uint *", "const char * * *" }, "int")

--

--- 
---@brief list all the available AtomicSymbolEntry
---@param out_size number @(mx_uint *) the size of returned array
---@param out_array number @(AtomicSymbolCreator * *) the output AtomicSymbolCreator array
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXSymbolListAtomicSymbolCreators(out_size, out_array)
    return _CALL("MXSymbolListAtomicSymbolCreators", out_size, out_array)
end
_FUNCDEF("MXSymbolListAtomicSymbolCreators", { "mx_uint *", "AtomicSymbolCreator * *" }, "int")

--

--- 
---@brief Get the name of an atomic symbol.
---@param creator number @(AtomicSymbolCreator) the AtomicSymbolCreator.
---@param name number @(const char * *) The returned name of the creator.
--- 
function M.MXSymbolGetAtomicSymbolName(creator, name)
    return _CALL("MXSymbolGetAtomicSymbolName", creator, name)
end
_FUNCDEF("MXSymbolGetAtomicSymbolName", { "AtomicSymbolCreator", "const char * *" }, "int")

--

--- 
---@brief Get the input symbols of the graph.
---@param sym number @(SymbolHandle) The graph.
---@param inputs number @(SymbolHandle * *) The input symbols of the graph.
---@param input_size number @(int *) the number of input symbols returned.
--- 
function M.MXSymbolGetInputSymbols(sym, inputs, input_size)
    return _CALL("MXSymbolGetInputSymbols", sym, inputs, input_size)
end
_FUNCDEF("MXSymbolGetInputSymbols", { "SymbolHandle", "SymbolHandle * *", "int *" }, "int")

--

--- 
---@brief Cut a subgraph whose nodes are marked with a subgraph attribute.
--- The input graph will be modified. A variable node will be created for each
--- edge that connects to nodes outside the subgraph. The outside nodes that
--- connect to the subgraph will be returned.
---@param sym number @(SymbolHandle) The graph.
---@param inputs number @(SymbolHandle * *) The nodes that connect to the subgraph.
---@param input_size number @(int *) The number of such nodes.
--- 
function M.MXSymbolCutSubgraph(sym, inputs, input_size)
    return _CALL("MXSymbolCutSubgraph", sym, inputs, input_size)
end
_FUNCDEF("MXSymbolCutSubgraph", { "SymbolHandle", "SymbolHandle * *", "int *" }, "int")

--

--- 
---@brief Get the detailed information about atomic symbol.
---@param creator number @(AtomicSymbolCreator) the AtomicSymbolCreator.
---@param name number @(const char * *) The returned name of the creator.
---@param description number @(const char * *) The returned description of the symbol.
---@param num_args number @(mx_uint *) Number of arguments.
---@param arg_names number @(const char * * *) Name of the arguments.
---@param arg_type_infos number @(const char * * *) Type informations about the arguments.
---@param arg_descriptions number @(const char * * *) Description information about the arguments.
---@param key_var_num_args number @(const char * *) The keyword argument for specifying variable number of arguments.
---            When this parameter has non-zero length, the function allows variable number
---            of positional arguments, and will need the caller to pass it in in
---            MXSymbolCreateAtomicSymbol,
---            With key = key_var_num_args, and value = number of positional arguments.
---@param return_type number @(const char * *) Return type of the function, can be Symbol or Symbol[]
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXSymbolGetAtomicSymbolInfo(creator, name, description, num_args, arg_names, arg_type_infos, arg_descriptions, key_var_num_args, return_type)
    return _CALL("MXSymbolGetAtomicSymbolInfo", creator, name, description, num_args, arg_names, arg_type_infos, arg_descriptions, key_var_num_args, return_type)
end
_FUNCDEF("MXSymbolGetAtomicSymbolInfo", { "AtomicSymbolCreator", "const char * *", "const char * *", "mx_uint *", "const char * * *", "const char * * *", "const char * * *", "const char * *", "const char * *" }, "int")

--

--- 
---@brief Create an AtomicSymbol.
---@param creator number @(AtomicSymbolCreator) the AtomicSymbolCreator
---@param num_param number @(mx_uint) the number of parameters
---@param keys number @(const char * *) the keys to the params
---@param vals number @(const char * *) the vals of the params
---@param out number @(SymbolHandle *) pointer to the created symbol handle
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXSymbolCreateAtomicSymbol(creator, num_param, keys, vals, out)
    return _CALL("MXSymbolCreateAtomicSymbol", creator, num_param, keys, vals, out)
end
_FUNCDEF("MXSymbolCreateAtomicSymbol", { "AtomicSymbolCreator", "mx_uint", "const char * *", "const char * *", "SymbolHandle *" }, "int")

--

--- 
---@brief Create a Variable Symbol.
---@param name string @(const char *) name of the variable
---@param out number @(SymbolHandle *) pointer to the created symbol handle
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXSymbolCreateVariable(name, out)
    return _CALL("MXSymbolCreateVariable", name, out)
end
_FUNCDEF("MXSymbolCreateVariable", { "const char *", "SymbolHandle *" }, "int")

--

--- 
---@brief Create a Symbol by grouping list of symbols together
---@param num_symbols number @(mx_uint) number of symbols to be grouped
---@param symbols number @(SymbolHandle *) array of symbol handles
---@param out number @(SymbolHandle *) pointer to the created symbol handle
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXSymbolCreateGroup(num_symbols, symbols, out)
    return _CALL("MXSymbolCreateGroup", num_symbols, symbols, out)
end
_FUNCDEF("MXSymbolCreateGroup", { "mx_uint", "SymbolHandle *", "SymbolHandle *" }, "int")

--

--- 
---@brief Load a symbol from a json file.
---@param fname string @(const char *) the file name.
---@param out number @(SymbolHandle *) the output symbol.
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXSymbolCreateFromFile(fname, out)
    return _CALL("MXSymbolCreateFromFile", fname, out)
end
_FUNCDEF("MXSymbolCreateFromFile", { "const char *", "SymbolHandle *" }, "int")

--

--- 
---@brief Load a symbol from a json string.
---@param json string @(const char *) the json string.
---@param out number @(SymbolHandle *) the output symbol.
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXSymbolCreateFromJSON(json, out)
    return _CALL("MXSymbolCreateFromJSON", json, out)
end
_FUNCDEF("MXSymbolCreateFromJSON", { "const char *", "SymbolHandle *" }, "int")

--

--- 
---@brief Remove the operators amp_cast and amp_multicast
---@param sym_handle number @(SymbolHandle) the input symbol.
---@param ret_sym_handle number @(SymbolHandle *) the output symbol.
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXSymbolRemoveAmpCast(sym_handle, ret_sym_handle)
    return _CALL("MXSymbolRemoveAmpCast", sym_handle, ret_sym_handle)
end
_FUNCDEF("MXSymbolRemoveAmpCast", { "SymbolHandle", "SymbolHandle *" }, "int")

--

--- 
---@brief Save a symbol into a json file.
---@param symbol number @(SymbolHandle) the input symbol.
---@param fname string @(const char *) the file name.
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXSymbolSaveToFile(symbol, fname)
    return _CALL("MXSymbolSaveToFile", symbol, fname)
end
_FUNCDEF("MXSymbolSaveToFile", { "SymbolHandle", "const char *" }, "int")

--

--- 
---@brief Save a symbol into a json string
---@param symbol number @(SymbolHandle) the input symbol.
---@param out_json number @(const char * *) output json string.
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXSymbolSaveToJSON(symbol, out_json)
    return _CALL("MXSymbolSaveToJSON", symbol, out_json)
end
_FUNCDEF("MXSymbolSaveToJSON", { "SymbolHandle", "const char * *" }, "int")

--

--- 
---@brief Free the symbol handle.
---@param symbol number @(SymbolHandle) the symbol
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXSymbolFree(symbol)
    return _CALL("MXSymbolFree", symbol)
end
_FUNCDEF("MXSymbolFree", { "SymbolHandle" }, "int")

--

--- 
---@brief Copy the symbol to another handle
---@param symbol number @(SymbolHandle) the source symbol
---@param out number @(SymbolHandle *) used to hold the result of copy
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXSymbolCopy(symbol, out)
    return _CALL("MXSymbolCopy", symbol, out)
end
_FUNCDEF("MXSymbolCopy", { "SymbolHandle", "SymbolHandle *" }, "int")

--

--- 
---@brief Print the content of symbol, used for debug.
---@param symbol number @(SymbolHandle) the symbol
---@param out_str number @(const char * *) pointer to hold the output string of the printing.
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXSymbolPrint(symbol, out_str)
    return _CALL("MXSymbolPrint", symbol, out_str)
end
_FUNCDEF("MXSymbolPrint", { "SymbolHandle", "const char * *" }, "int")

--

--- 
---@brief Get string name from symbol
---@param symbol number @(SymbolHandle) the source symbol
---@param out number @(const char * *) The result name.
---@param success number @(int *) Whether the result is contained in out.
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXSymbolGetName(symbol, out, success)
    return _CALL("MXSymbolGetName", symbol, out, success)
end
_FUNCDEF("MXSymbolGetName", { "SymbolHandle", "const char * *", "int *" }, "int")

--

--- 
---@brief Get string attribute from symbol
---@param symbol number @(SymbolHandle) the source symbol
---@param key string @(const char *) The key of the symbol.
---@param out number @(const char * *) The result attribute, can be NULL if the attribute do not exist.
---@param success number @(int *) Whether the result is contained in out.
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXSymbolGetAttr(symbol, key, out, success)
    return _CALL("MXSymbolGetAttr", symbol, key, out, success)
end
_FUNCDEF("MXSymbolGetAttr", { "SymbolHandle", "const char *", "const char * *", "int *" }, "int")

--

--- 
---@brief Set string attribute from symbol.
---  NOTE: Setting attribute to a symbol can affect the semantics(mutable/immutable) of symbolic graph.
--- 
---  Safe recommendaton: use  immutable graph
---  - Only allow set attributes during creation of new symbol as optional parameter
--- 
---  Mutable graph (be careful about the semantics):
---  - Allow set attr at any point.
---  - Mutating an attribute of some common node of two graphs can cause confusion from user.
--- 
---@param symbol number @(SymbolHandle) the source symbol
---@param key string @(const char *) The key of the symbol.
---@param value string @(const char *) The value to be saved.
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXSymbolSetAttr(symbol, key, value)
    return _CALL("MXSymbolSetAttr", symbol, key, value)
end
_FUNCDEF("MXSymbolSetAttr", { "SymbolHandle", "const char *", "const char *" }, "int")

--

--- 
---@brief Get all attributes from symbol, including all descendents.
---@param symbol number @(SymbolHandle) the source symbol
---@param out_size number @(mx_uint *) The number of output attributes
---@param out number @(const char * * *) 2*out_size strings representing key value pairs.
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXSymbolListAttr(symbol, out_size, out)
    return _CALL("MXSymbolListAttr", symbol, out_size, out)
end
_FUNCDEF("MXSymbolListAttr", { "SymbolHandle", "mx_uint *", "const char * * *" }, "int")

--

--- 
---@brief Get all attributes from symbol, excluding descendents.
---@param symbol number @(SymbolHandle) the source symbol
---@param out_size number @(mx_uint *) The number of output attributes
---@param out number @(const char * * *) 2*out_size strings representing key value pairs.
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXSymbolListAttrShallow(symbol, out_size, out)
    return _CALL("MXSymbolListAttrShallow", symbol, out_size, out)
end
_FUNCDEF("MXSymbolListAttrShallow", { "SymbolHandle", "mx_uint *", "const char * * *" }, "int")

--

--- 
---@brief List arguments in the symbol.
---@param symbol number @(SymbolHandle) the symbol
---@param out_size number @(mx_uint *) output size
---@param out_str_array number @(const char * * *) pointer to hold the output string array
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXSymbolListArguments(symbol, out_size, out_str_array)
    return _CALL("MXSymbolListArguments", symbol, out_size, out_str_array)
end
_FUNCDEF("MXSymbolListArguments", { "SymbolHandle", "mx_uint *", "const char * * *" }, "int")

--

--- 
---@brief List returns in the symbol.
---@param symbol number @(SymbolHandle) the symbol
---@param out_size number @(mx_uint *) output size
---@param out_str_array number @(const char * * *) pointer to hold the output string array
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXSymbolListOutputs(symbol, out_size, out_str_array)
    return _CALL("MXSymbolListOutputs", symbol, out_size, out_str_array)
end
_FUNCDEF("MXSymbolListOutputs", { "SymbolHandle", "mx_uint *", "const char * * *" }, "int")

--

--- 
---@brief Get number of outputs of the symbol.
---@param symbol number @(SymbolHandle) The symbol
---@param out_size number of outputs
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXSymbolGetNumOutputs(symbol, output_count)
    return _CALL("MXSymbolGetNumOutputs", symbol, output_count)
end
_FUNCDEF("MXSymbolGetNumOutputs", { "SymbolHandle", "mx_uint *" }, "int")

--

--- 
---@brief Get a symbol that contains all the internals.
---@param symbol number @(SymbolHandle) The symbol
---@param out number @(SymbolHandle *) The output symbol whose outputs are all the internals.
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXSymbolGetInternals(symbol, out)
    return _CALL("MXSymbolGetInternals", symbol, out)
end
_FUNCDEF("MXSymbolGetInternals", { "SymbolHandle", "SymbolHandle *" }, "int")

--

--- 
---@brief Get a symbol that contains only direct children.
---@param symbol number @(SymbolHandle) The symbol
---@param out number @(SymbolHandle *) The output symbol whose outputs are the direct children.
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXSymbolGetChildren(symbol, out)
    return _CALL("MXSymbolGetChildren", symbol, out)
end
_FUNCDEF("MXSymbolGetChildren", { "SymbolHandle", "SymbolHandle *" }, "int")

--

--- 
---@brief Get index-th outputs of the symbol.
---@param symbol number @(SymbolHandle) The symbol
---@param index number @(mx_uint) the Index of the output.
---@param out number @(SymbolHandle *) The output symbol whose outputs are the index-th symbol.
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXSymbolGetOutput(symbol, index, out)
    return _CALL("MXSymbolGetOutput", symbol, index, out)
end
_FUNCDEF("MXSymbolGetOutput", { "SymbolHandle", "mx_uint", "SymbolHandle *" }, "int")

--

--- 
---@brief List auxiliary states in the symbol.
---@param symbol number @(SymbolHandle) the symbol
---@param out_size number @(mx_uint *) output size
---@param out_str_array number @(const char * * *) pointer to hold the output string array
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXSymbolListAuxiliaryStates(symbol, out_size, out_str_array)
    return _CALL("MXSymbolListAuxiliaryStates", symbol, out_size, out_str_array)
end
_FUNCDEF("MXSymbolListAuxiliaryStates", { "SymbolHandle", "mx_uint *", "const char * * *" }, "int")

--

--- 
---@brief Compose the symbol on other symbols.
--- 
---  This function will change the sym hanlde.
---  To achieve function apply behavior, copy the symbol first
---  before apply.
--- 
---@param sym number @(SymbolHandle) the symbol to apply
---@param name string @(const char *) the name of symbol
---@param num_args number @(mx_uint) number of arguments
---@param keys number @(const char * *) the key of keyword args (optional)
---@param args number @(SymbolHandle *) arguments to sym
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXSymbolCompose(sym, name, num_args, keys, args)
    return _CALL("MXSymbolCompose", sym, name, num_args, keys, args)
end
_FUNCDEF("MXSymbolCompose", { "SymbolHandle", "const char *", "mx_uint", "const char * *", "SymbolHandle *" }, "int")

--

--- 
---@brief Get the gradient graph of the symbol
--- 
---@param sym number @(SymbolHandle) the symbol to get gradient
---@param num_wrt number @(mx_uint) number of arguments to get gradient
---@param wrt number @(const char * *) the name of the arguments to get gradient
---@param out number @(SymbolHandle *) the returned symbol that has gradient
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXSymbolGrad(sym, num_wrt, wrt, out)
    return _CALL("MXSymbolGrad", sym, num_wrt, wrt, out)
end
_FUNCDEF("MXSymbolGrad", { "SymbolHandle", "mx_uint", "const char * *", "SymbolHandle *" }, "int")

--

--- 
---@brief DEPRECATED. Use MXSymbolInferShapeEx instead.
--- infer shape of unknown input shapes given the known one.
---  The shapes are packed into a CSR matrix represented by arg_ind_ptr and arg_shape_data
---  The call will be treated as a kwargs call if key != nullptr or num_args==0, otherwise it is positional.
--- 
---@param sym number @(SymbolHandle) symbol handle
---@param num_args number @(mx_uint) numbe of input arguments.
---@param keys number @(const char * *) the key of keyword args (optional)
---@param arg_ind_ptr number @(const mx_uint *) the head pointer of the rows in CSR
---@param arg_shape_data number @(const mx_uint *) the content of the CSR
---@param in_shape_size number @(mx_uint *) sizeof the returning array of in_shapes
---@param in_shape_ndim number @(const mx_uint * *) returning array of shape dimensions of eachs input shape.
---@param in_shape_data number @(const mx_uint * * *) returning array of pointers to head of the input shape.
---@param out_shape_size number @(mx_uint *) sizeof the returning array of out_shapes
---@param out_shape_ndim number @(const mx_uint * *) returning array of shape dimensions of eachs input shape.
---@param out_shape_data number @(const mx_uint * * *) returning array of pointers to head of the input shape.
---@param aux_shape_size number @(mx_uint *) sizeof the returning array of aux_shapes
---@param aux_shape_ndim number @(const mx_uint * *) returning array of shape dimensions of eachs auxiliary shape.
---@param aux_shape_data number @(const mx_uint * * *) returning array of pointers to head of the auxiliary shape.
---@param complete number @(int *) whether infer shape completes or more information is needed.
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXSymbolInferShape(sym, num_args, keys, arg_ind_ptr, arg_shape_data, in_shape_size, in_shape_ndim, in_shape_data, out_shape_size, out_shape_ndim, out_shape_data, aux_shape_size, aux_shape_ndim, aux_shape_data, complete)
    return _CALL("MXSymbolInferShape", sym, num_args, keys, arg_ind_ptr, arg_shape_data, in_shape_size, in_shape_ndim, in_shape_data, out_shape_size, out_shape_ndim, out_shape_data, aux_shape_size, aux_shape_ndim, aux_shape_data, complete)
end
_FUNCDEF("MXSymbolInferShape", { "SymbolHandle", "mx_uint", "const char * *", "const mx_uint *", "const mx_uint *", "mx_uint *", "const mx_uint * *", "const mx_uint * * *", "mx_uint *", "const mx_uint * *", "const mx_uint * * *", "mx_uint *", "const mx_uint * *", "const mx_uint * * *", "int *" }, "int")

--

--- 
---@brief infer shape of unknown input shapes given the known one.
---  The shapes are packed into a CSR matrix represented by arg_ind_ptr and arg_shape_data
---  The call will be treated as a kwargs call if key != nullptr or num_args==0, otherwise it is positional.
--- 
---@param sym number @(SymbolHandle) symbol handle
---@param num_args number @(mx_uint) numbe of input arguments.
---@param keys number @(const char * *) the key of keyword args (optional)
---@param arg_ind_ptr number @(const mx_uint *) the head pointer of the rows in CSR
---@param arg_shape_data number @(const int *) the content of the CSR
---@param in_shape_size number @(mx_uint *) sizeof the returning array of in_shapes
---@param in_shape_ndim number @(const int * *) returning array of shape dimensions of eachs input shape.
---@param in_shape_data number @(const int * * *) returning array of pointers to head of the input shape.
---@param out_shape_size number @(mx_uint *) sizeof the returning array of out_shapes
---@param out_shape_ndim number @(const int * *) returning array of shape dimensions of eachs input shape.
---@param out_shape_data number @(const int * * *) returning array of pointers to head of the input shape.
---@param aux_shape_size number @(mx_uint *) sizeof the returning array of aux_shapes
---@param aux_shape_ndim number @(const int * *) returning array of shape dimensions of eachs auxiliary shape.
---@param aux_shape_data number @(const int * * *) returning array of pointers to head of the auxiliary shape.
---@param complete number @(int *) whether infer shape completes or more information is needed.
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXSymbolInferShapeEx(sym, num_args, keys, arg_ind_ptr, arg_shape_data, in_shape_size, in_shape_ndim, in_shape_data, out_shape_size, out_shape_ndim, out_shape_data, aux_shape_size, aux_shape_ndim, aux_shape_data, complete)
    return _CALL("MXSymbolInferShapeEx", sym, num_args, keys, arg_ind_ptr, arg_shape_data, in_shape_size, in_shape_ndim, in_shape_data, out_shape_size, out_shape_ndim, out_shape_data, aux_shape_size, aux_shape_ndim, aux_shape_data, complete)
end
_FUNCDEF("MXSymbolInferShapeEx", { "SymbolHandle", "mx_uint", "const char * *", "const mx_uint *", "const int *", "mx_uint *", "const int * *", "const int * * *", "mx_uint *", "const int * *", "const int * * *", "mx_uint *", "const int * *", "const int * * *", "int *" }, "int")

--

--- 
---@brief DEPRECATED. Use MXSymbolInferShapePartialEx instead.
--- partially infer shape of unknown input shapes given the known one.
--- 
---  Return partially inferred results if not all shapes could be inferred.
---  The shapes are packed into a CSR matrix represented by arg_ind_ptr and arg_shape_data
---  The call will be treated as a kwargs call if key != nullptr or num_args==0, otherwise it is positional.
--- 
---@param sym number @(SymbolHandle) symbol handle
---@param num_args number @(mx_uint) numbe of input arguments.
---@param keys number @(const char * *) the key of keyword args (optional)
---@param arg_ind_ptr number @(const mx_uint *) the head pointer of the rows in CSR
---@param arg_shape_data number @(const mx_uint *) the content of the CSR
---@param in_shape_size number @(mx_uint *) sizeof the returning array of in_shapes
---@param in_shape_ndim number @(const mx_uint * *) returning array of shape dimensions of eachs input shape.
---@param in_shape_data number @(const mx_uint * * *) returning array of pointers to head of the input shape.
---@param out_shape_size number @(mx_uint *) sizeof the returning array of out_shapes
---@param out_shape_ndim number @(const mx_uint * *) returning array of shape dimensions of eachs input shape.
---@param out_shape_data number @(const mx_uint * * *) returning array of pointers to head of the input shape.
---@param aux_shape_size number @(mx_uint *) sizeof the returning array of aux_shapes
---@param aux_shape_ndim number @(const mx_uint * *) returning array of shape dimensions of eachs auxiliary shape.
---@param aux_shape_data number @(const mx_uint * * *) returning array of pointers to head of the auxiliary shape.
---@param complete number @(int *) whether infer shape completes or more information is needed.
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXSymbolInferShapePartial(sym, num_args, keys, arg_ind_ptr, arg_shape_data, in_shape_size, in_shape_ndim, in_shape_data, out_shape_size, out_shape_ndim, out_shape_data, aux_shape_size, aux_shape_ndim, aux_shape_data, complete)
    return _CALL("MXSymbolInferShapePartial", sym, num_args, keys, arg_ind_ptr, arg_shape_data, in_shape_size, in_shape_ndim, in_shape_data, out_shape_size, out_shape_ndim, out_shape_data, aux_shape_size, aux_shape_ndim, aux_shape_data, complete)
end
_FUNCDEF("MXSymbolInferShapePartial", { "SymbolHandle", "mx_uint", "const char * *", "const mx_uint *", "const mx_uint *", "mx_uint *", "const mx_uint * *", "const mx_uint * * *", "mx_uint *", "const mx_uint * *", "const mx_uint * * *", "mx_uint *", "const mx_uint * *", "const mx_uint * * *", "int *" }, "int")

--

--- 
---@brief partially infer shape of unknown input shapes given the known one.
--- 
---  Return partially inferred results if not all shapes could be inferred.
---  The shapes are packed into a CSR matrix represented by arg_ind_ptr and arg_shape_data
---  The call will be treated as a kwargs call if key != nullptr or num_args==0, otherwise it is positional.
--- 
---@param sym number @(SymbolHandle) symbol handle
---@param num_args number @(mx_uint) numbe of input arguments.
---@param keys number @(const char * *) the key of keyword args (optional)
---@param arg_ind_ptr number @(const mx_uint *) the head pointer of the rows in CSR
---@param arg_shape_data number @(const int *) the content of the CSR
---@param in_shape_size number @(mx_uint *) sizeof the returning array of in_shapes
---@param in_shape_ndim number @(const int * *) returning array of shape dimensions of eachs input shape.
---@param in_shape_data number @(const int * * *) returning array of pointers to head of the input shape.
---@param out_shape_size number @(mx_uint *) sizeof the returning array of out_shapes
---@param out_shape_ndim number @(const int * *) returning array of shape dimensions of eachs input shape.
---@param out_shape_data number @(const int * * *) returning array of pointers to head of the input shape.
---@param aux_shape_size number @(mx_uint *) sizeof the returning array of aux_shapes
---@param aux_shape_ndim number @(const int * *) returning array of shape dimensions of eachs auxiliary shape.
---@param aux_shape_data number @(const int * * *) returning array of pointers to head of the auxiliary shape.
---@param complete number @(int *) whether infer shape completes or more information is needed.
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXSymbolInferShapePartialEx(sym, num_args, keys, arg_ind_ptr, arg_shape_data, in_shape_size, in_shape_ndim, in_shape_data, out_shape_size, out_shape_ndim, out_shape_data, aux_shape_size, aux_shape_ndim, aux_shape_data, complete)
    return _CALL("MXSymbolInferShapePartialEx", sym, num_args, keys, arg_ind_ptr, arg_shape_data, in_shape_size, in_shape_ndim, in_shape_data, out_shape_size, out_shape_ndim, out_shape_data, aux_shape_size, aux_shape_ndim, aux_shape_data, complete)
end
_FUNCDEF("MXSymbolInferShapePartialEx", { "SymbolHandle", "mx_uint", "const char * *", "const mx_uint *", "const int *", "mx_uint *", "const int * *", "const int * * *", "mx_uint *", "const int * *", "const int * * *", "mx_uint *", "const int * *", "const int * * *", "int *" }, "int")

--

--- 
---@brief infer type of unknown input types given the known one.
---  The types are packed into a CSR matrix represented by arg_ind_ptr and arg_type_data
---  The call will be treated as a kwargs call if key != nullptr or num_args==0, otherwise it is positional.
--- 
---@param sym number @(SymbolHandle) symbol handle
---@param num_args number @(mx_uint) numbe of input arguments.
---@param keys number @(const char * *) the key of keyword args (optional)
---@param arg_type_data number @(const int *) the content of the CSR
---@param in_type_size number @(mx_uint *) sizeof the returning array of in_types
---@param in_type_data number @(const int * *) returning array of pointers to head of the input type.
---@param out_type_size number @(mx_uint *) sizeof the returning array of out_types
---@param out_type_data number @(const int * *) returning array of pointers to head of the input type.
---@param aux_type_size number @(mx_uint *) sizeof the returning array of aux_types
---@param aux_type_data number @(const int * *) returning array of pointers to head of the auxiliary type.
---@param complete number @(int *) whether infer type completes or more information is needed.
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXSymbolInferType(sym, num_args, keys, arg_type_data, in_type_size, in_type_data, out_type_size, out_type_data, aux_type_size, aux_type_data, complete)
    return _CALL("MXSymbolInferType", sym, num_args, keys, arg_type_data, in_type_size, in_type_data, out_type_size, out_type_data, aux_type_size, aux_type_data, complete)
end
_FUNCDEF("MXSymbolInferType", { "SymbolHandle", "mx_uint", "const char * *", "const int *", "mx_uint *", "const int * *", "mx_uint *", "const int * *", "mx_uint *", "const int * *", "int *" }, "int")

--

--- 
---@brief partially infer type of unknown input types given the known one.
--- 
---  Return partially inferred results if not all types could be inferred.
---  The types are packed into a CSR matrix represented by arg_ind_ptr and arg_type_data
---  The call will be treated as a kwargs call if key != nullptr or num_args==0, otherwise it is positional.
--- 
---@param sym number @(SymbolHandle) symbol handle
---@param num_args number @(mx_uint) numbe of input arguments.
---@param keys number @(const char * *) the key of keyword args (optional)
---@param arg_type_data number @(const int *) the content of the CSR
---@param in_type_size number @(mx_uint *) sizeof the returning array of in_types
---@param in_type_data number @(const int * *) returning array of pointers to head of the input type.
---@param out_type_size number @(mx_uint *) sizeof the returning array of out_types
---@param out_type_data number @(const int * *) returning array of pointers to head of the input type.
---@param aux_type_size number @(mx_uint *) sizeof the returning array of aux_types
---@param aux_type_data number @(const int * *) returning array of pointers to head of the auxiliary type.
---@param complete number @(int *) whether infer type completes or more information is needed.
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXSymbolInferTypePartial(sym, num_args, keys, arg_type_data, in_type_size, in_type_data, out_type_size, out_type_data, aux_type_size, aux_type_data, complete)
    return _CALL("MXSymbolInferTypePartial", sym, num_args, keys, arg_type_data, in_type_size, in_type_data, out_type_size, out_type_data, aux_type_size, aux_type_data, complete)
end
_FUNCDEF("MXSymbolInferTypePartial", { "SymbolHandle", "mx_uint", "const char * *", "const int *", "mx_uint *", "const int * *", "mx_uint *", "const int * *", "mx_uint *", "const int * *", "int *" }, "int")

--

--- 
---@brief Convert a symbol into a quantized symbol where FP32 operators are replaced with INT8
---@param sym_handle number @(SymbolHandle) symbol to be converted
---@param ret_sym_handle number @(SymbolHandle *) quantized symbol result
---@param num_excluded_symbols number @(const mx_uint) number of layers excluded from being quantized in the input symbol
---@param excluded_symbols number @(const char * *) op names to be excluded from being quantized
---@param num_offline number @(const mx_uint) number of parameters that are quantized offline
---@param offline_params number @(const char * *) array of c strings representing the names of params quantized offline
---@param quantized_dtype string @(const char *) the quantized destination type for input data
---@param calib_quantize boolean @(const bool) **Deprecated**. quantize op will always be calibrated if could
--- 
function M.MXQuantizeSymbol(sym_handle, ret_sym_handle, num_excluded_symbols, excluded_symbols, num_offline, offline_params, quantized_dtype, calib_quantize)
    return _CALL("MXQuantizeSymbol", sym_handle, ret_sym_handle, num_excluded_symbols, excluded_symbols, num_offline, offline_params, quantized_dtype, calib_quantize)
end
_FUNCDEF("MXQuantizeSymbol", { "SymbolHandle", "SymbolHandle *", "const mx_uint", "const char * *", "const mx_uint", "const char * *", "const char *", "const bool" }, "int")

--

--- 
---@brief Set calibration table to node attributes in the sym
---@param sym_handle number @(SymbolHandle) symbol whose node attributes are to be set by calibration table
---@param num_layers number @(const mx_uint) number of layers in the calibration table
---@param layer_names number @(const char * *) names stored as keys in the calibration table
---@param low_quantiles number @(const float *) low quantiles of layers stored in the calibration table
---@param high_quantiles number @(const float *) high quantiles of layers stored in the calibration table
---@param ret_sym_handle number @(SymbolHandle *) returned symbol
--- 
function M.MXSetCalibTableToQuantizedSymbol(sym_handle, num_layers, layer_names, low_quantiles, high_quantiles, ret_sym_handle)
    return _CALL("MXSetCalibTableToQuantizedSymbol", sym_handle, num_layers, layer_names, low_quantiles, high_quantiles, ret_sym_handle)
end
_FUNCDEF("MXSetCalibTableToQuantizedSymbol", { "SymbolHandle", "const mx_uint", "const char * *", "const float *", "const float *", "SymbolHandle *" }, "int")

--

--- 
---@brief Run subgraph pass based on the backend provided
---@param sym_handle number @(SymbolHandle) symbol to be converted
---@param backend string @(const char *) backend names for subgraph pass
---@param ret_sym_handle number @(SymbolHandle *) returned symbol
--- 
function M.MXGenBackendSubgraph(sym_handle, backend, ret_sym_handle)
    return _CALL("MXGenBackendSubgraph", sym_handle, backend, ret_sym_handle)
end
_FUNCDEF("MXGenBackendSubgraph", { "SymbolHandle", "const char *", "SymbolHandle *" }, "int")

--

--- 
---@brief Generate atomic symbol (able to be composed) from a source symbol
---@param sym_handle number @(SymbolHandle) source symbol
---@param ret_sym_handle number @(SymbolHandle *) returned atomic symbol
--- 
function M.MXGenAtomicSymbolFromSymbol(sym_handle, ret_sym_handle)
    return _CALL("MXGenAtomicSymbolFromSymbol", sym_handle, ret_sym_handle)
end
_FUNCDEF("MXGenAtomicSymbolFromSymbol", { "SymbolHandle", "SymbolHandle *" }, "int")

--

--- 
---@brief Delete the executor
---@param handle number @(ExecutorHandle) the executor.
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXExecutorFree(handle)
    return _CALL("MXExecutorFree", handle)
end
_FUNCDEF("MXExecutorFree", { "ExecutorHandle" }, "int")

--

--- 
---@brief Print the content of execution plan, used for debug.
---@param handle number @(ExecutorHandle) the executor.
---@param out_str number @(const char * *) pointer to hold the output string of the printing.
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXExecutorPrint(handle, out_str)
    return _CALL("MXExecutorPrint", handle, out_str)
end
_FUNCDEF("MXExecutorPrint", { "ExecutorHandle", "const char * *" }, "int")

--

--- 
---@brief Executor forward method
--- 
---@param handle number @(ExecutorHandle) executor handle
---@param is_train number @(int) int value to indicate whether the forward pass is for evaluation
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXExecutorForward(handle, is_train)
    return _CALL("MXExecutorForward", handle, is_train)
end
_FUNCDEF("MXExecutorForward", { "ExecutorHandle", "int" }, "int")

--

--- 
---@brief Excecutor run backward
--- 
---@param handle number @(ExecutorHandle) execute handle
---@param len number @(mx_uint) lenth
---@param head_grads number @(NDArrayHandle *) NDArray handle for heads' gradient
--- 
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXExecutorBackward(handle, len, head_grads)
    return _CALL("MXExecutorBackward", handle, len, head_grads)
end
_FUNCDEF("MXExecutorBackward", { "ExecutorHandle", "mx_uint", "NDArrayHandle *" }, "int")

--

--- 
---@brief Excecutor run backward
--- 
---@param handle number @(ExecutorHandle) execute handle
---@param len number @(mx_uint) lenth
---@param head_grads number @(NDArrayHandle *) NDArray handle for heads' gradient
---@param is_train number @(int) int value to indicate whether the backward pass is for evaluation
--- 
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXExecutorBackwardEx(handle, len, head_grads, is_train)
    return _CALL("MXExecutorBackwardEx", handle, len, head_grads, is_train)
end
_FUNCDEF("MXExecutorBackwardEx", { "ExecutorHandle", "mx_uint", "NDArrayHandle *", "int" }, "int")

--

--- 
---@brief Get executor's head NDArray
--- 
---@param handle number @(ExecutorHandle) executor handle
---@param out_size number @(mx_uint *) output narray vector size
---@param out number @(NDArrayHandle * *) out put narray handles
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXExecutorOutputs(handle, out_size, out)
    return _CALL("MXExecutorOutputs", handle, out_size, out)
end
_FUNCDEF("MXExecutorOutputs", { "ExecutorHandle", "mx_uint *", "NDArrayHandle * *" }, "int")

--

--- 
---@brief Generate Executor from symbol
--- 
---@param symbol_handle number @(SymbolHandle) symbol handle
---@param dev_type number @(int) device type
---@param dev_id number @(int) device id
---@param len number @(mx_uint) length
---@param in_args number @(NDArrayHandle *) in args array
---@param arg_grad_store number @(NDArrayHandle *) arg grads handle array
---@param grad_req_type number @(mx_uint *) grad req array
---@param aux_states_len number @(mx_uint) length of auxiliary states
---@param aux_states number @(NDArrayHandle *) auxiliary states array
---@param out number @(ExecutorHandle *) output executor handle
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXExecutorBind(symbol_handle, dev_type, dev_id, len, in_args, arg_grad_store, grad_req_type, aux_states_len, aux_states, out)
    return _CALL("MXExecutorBind", symbol_handle, dev_type, dev_id, len, in_args, arg_grad_store, grad_req_type, aux_states_len, aux_states, out)
end
_FUNCDEF("MXExecutorBind", { "SymbolHandle", "int", "int", "mx_uint", "NDArrayHandle *", "NDArrayHandle *", "mx_uint *", "mx_uint", "NDArrayHandle *", "ExecutorHandle *" }, "int")

--

--- 
---@brief Generate Executor from symbol,
---  This is advanced function, allow specify group2ctx map.
---  The user can annotate "ctx_group" attribute to name each group.
--- 
---@param symbol_handle number @(SymbolHandle) symbol handle
---@param dev_type number @(int) device type of default context
---@param dev_id number @(int) device id of default context
---@param num_map_keys number @(mx_uint) size of group2ctx map
---@param map_keys number @(const char * *) keys of group2ctx map
---@param map_dev_types number @(const int *) device type of group2ctx map
---@param map_dev_ids number @(const int *) device id of group2ctx map
---@param len number @(mx_uint) length
---@param in_args number @(NDArrayHandle *) in args array
---@param arg_grad_store number @(NDArrayHandle *) arg grads handle array
---@param grad_req_type number @(mx_uint *) grad req array
---@param aux_states_len number @(mx_uint) length of auxiliary states
---@param aux_states number @(NDArrayHandle *) auxiliary states array
---@param out number @(ExecutorHandle *) output executor handle
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXExecutorBindX(symbol_handle, dev_type, dev_id, num_map_keys, map_keys, map_dev_types, map_dev_ids, len, in_args, arg_grad_store, grad_req_type, aux_states_len, aux_states, out)
    return _CALL("MXExecutorBindX", symbol_handle, dev_type, dev_id, num_map_keys, map_keys, map_dev_types, map_dev_ids, len, in_args, arg_grad_store, grad_req_type, aux_states_len, aux_states, out)
end
_FUNCDEF("MXExecutorBindX", { "SymbolHandle", "int", "int", "mx_uint", "const char * *", "const int *", "const int *", "mx_uint", "NDArrayHandle *", "NDArrayHandle *", "mx_uint *", "mx_uint", "NDArrayHandle *", "ExecutorHandle *" }, "int")

--

--- 
---@brief Generate Executor from symbol,
---  This is advanced function, allow specify group2ctx map.
---  The user can annotate "ctx_group" attribute to name each group.
--- 
---@param symbol_handle number @(SymbolHandle) symbol handle
---@param dev_type number @(int) device type of default context
---@param dev_id number @(int) device id of default context
---@param num_map_keys number @(mx_uint) size of group2ctx map
---@param map_keys number @(const char * *) keys of group2ctx map
---@param map_dev_types number @(const int *) device type of group2ctx map
---@param map_dev_ids number @(const int *) device id of group2ctx map
---@param len number @(mx_uint) length
---@param in_args number @(NDArrayHandle *) in args array
---@param arg_grad_store number @(NDArrayHandle *) arg grads handle array
---@param grad_req_type number @(mx_uint *) grad req array
---@param aux_states_len number @(mx_uint) length of auxiliary states
---@param aux_states number @(NDArrayHandle *) auxiliary states array
---@param shared_exec number @(ExecutorHandle) input executor handle for memory sharing
---@param out number @(ExecutorHandle *) output executor handle
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXExecutorBindEX(symbol_handle, dev_type, dev_id, num_map_keys, map_keys, map_dev_types, map_dev_ids, len, in_args, arg_grad_store, grad_req_type, aux_states_len, aux_states, shared_exec, out)
    return _CALL("MXExecutorBindEX", symbol_handle, dev_type, dev_id, num_map_keys, map_keys, map_dev_types, map_dev_ids, len, in_args, arg_grad_store, grad_req_type, aux_states_len, aux_states, shared_exec, out)
end
_FUNCDEF("MXExecutorBindEX", { "SymbolHandle", "int", "int", "mx_uint", "const char * *", "const int *", "const int *", "mx_uint", "NDArrayHandle *", "NDArrayHandle *", "mx_uint *", "mx_uint", "NDArrayHandle *", "ExecutorHandle", "ExecutorHandle *" }, "int")

--

---@brief DEPRECATED. Use MXExecutorSimpleBindEx instead.
--- 
function M.MXExecutorSimpleBind(symbol_handle, dev_type, dev_id, num_g2c_keys, g2c_keys, g2c_dev_types, g2c_dev_ids, provided_grad_req_list_len, provided_grad_req_names, provided_grad_req_types, num_provided_arg_shapes, provided_arg_shape_names, provided_arg_shape_data, provided_arg_shape_idx, num_provided_arg_dtypes, provided_arg_dtype_names, provided_arg_dtypes, num_provided_arg_stypes, provided_arg_stype_names, provided_arg_stypes, num_shared_arg_names, shared_arg_name_list, shared_buffer_len, shared_buffer_name_list, shared_buffer_handle_list, updated_shared_buffer_name_list, updated_shared_buffer_handle_list, num_in_args, in_args, arg_grads, num_aux_states, aux_states, shared_exec_handle, out)
    return _CALL("MXExecutorSimpleBind", symbol_handle, dev_type, dev_id, num_g2c_keys, g2c_keys, g2c_dev_types, g2c_dev_ids, provided_grad_req_list_len, provided_grad_req_names, provided_grad_req_types, num_provided_arg_shapes, provided_arg_shape_names, provided_arg_shape_data, provided_arg_shape_idx, num_provided_arg_dtypes, provided_arg_dtype_names, provided_arg_dtypes, num_provided_arg_stypes, provided_arg_stype_names, provided_arg_stypes, num_shared_arg_names, shared_arg_name_list, shared_buffer_len, shared_buffer_name_list, shared_buffer_handle_list, updated_shared_buffer_name_list, updated_shared_buffer_handle_list, num_in_args, in_args, arg_grads, num_aux_states, aux_states, shared_exec_handle, out)
end
_FUNCDEF("MXExecutorSimpleBind", { "SymbolHandle", "int", "int", "const mx_uint", "const char * *", "const int *", "const int *", "const mx_uint", "const char * *", "const char * *", "const mx_uint", "const char * *", "const mx_uint *", "const mx_uint *", "const mx_uint", "const char * *", "const int *", "const mx_uint", "const char * *", "const int *", "const mx_uint", "const char * *", "int *", "const char * *", "NDArrayHandle *", "const char * * *", "NDArrayHandle * *", "mx_uint *", "NDArrayHandle * *", "NDArrayHandle * *", "mx_uint *", "NDArrayHandle * *", "ExecutorHandle", "ExecutorHandle *" }, "int")

--

---@param symbol_handle number @(SymbolHandle)
---@param dev_type number @(int)
---@param dev_id number @(int)
---@param num_g2c_keys number @(const mx_uint)
---@param g2c_keys number @(const char * *)
---@param g2c_dev_types number @(const int *)
---@param g2c_dev_ids number @(const int *)
---@param provided_grad_req_list_len number @(const mx_uint)
---@param provided_grad_req_names number @(const char * *)
---@param provided_grad_req_types number @(const char * *)
---@param num_provided_arg_shapes number @(const mx_uint)
---@param provided_arg_shape_names number @(const char * *)
---@param provided_arg_shape_data number @(const int *)
---@param provided_arg_shape_idx number @(const mx_uint *)
---@param num_provided_arg_dtypes number @(const mx_uint)
---@param provided_arg_dtype_names number @(const char * *)
---@param provided_arg_dtypes number @(const int *)
---@param num_provided_arg_stypes number @(const mx_uint)
---@param provided_arg_stype_names number @(const char * *)
---@param provided_arg_stypes number @(const int *)
---@param num_shared_arg_names number @(const mx_uint)
---@param shared_arg_name_list number @(const char * *)
---@param shared_buffer_len number @(int *)
---@param shared_buffer_name_list number @(const char * *)
---@param shared_buffer_handle_list number @(NDArrayHandle *)
---@param updated_shared_buffer_name_list number @(const char * * *)
---@param updated_shared_buffer_handle_list number @(NDArrayHandle * *)
---@param num_in_args number @(mx_uint *)
---@param in_args number @(NDArrayHandle * *)
---@param arg_grads number @(NDArrayHandle * *)
---@param num_aux_states number @(mx_uint *)
---@param aux_states number @(NDArrayHandle * *)
---@param shared_exec_handle number @(ExecutorHandle)
---@param out number @(ExecutorHandle *)
---@return number @(int)
function M.MXExecutorSimpleBindEx(symbol_handle, dev_type, dev_id, num_g2c_keys, g2c_keys, g2c_dev_types, g2c_dev_ids, provided_grad_req_list_len, provided_grad_req_names, provided_grad_req_types, num_provided_arg_shapes, provided_arg_shape_names, provided_arg_shape_data, provided_arg_shape_idx, num_provided_arg_dtypes, provided_arg_dtype_names, provided_arg_dtypes, num_provided_arg_stypes, provided_arg_stype_names, provided_arg_stypes, num_shared_arg_names, shared_arg_name_list, shared_buffer_len, shared_buffer_name_list, shared_buffer_handle_list, updated_shared_buffer_name_list, updated_shared_buffer_handle_list, num_in_args, in_args, arg_grads, num_aux_states, aux_states, shared_exec_handle, out)
    return _CALL("MXExecutorSimpleBindEx", symbol_handle, dev_type, dev_id, num_g2c_keys, g2c_keys, g2c_dev_types, g2c_dev_ids, provided_grad_req_list_len, provided_grad_req_names, provided_grad_req_types, num_provided_arg_shapes, provided_arg_shape_names, provided_arg_shape_data, provided_arg_shape_idx, num_provided_arg_dtypes, provided_arg_dtype_names, provided_arg_dtypes, num_provided_arg_stypes, provided_arg_stype_names, provided_arg_stypes, num_shared_arg_names, shared_arg_name_list, shared_buffer_len, shared_buffer_name_list, shared_buffer_handle_list, updated_shared_buffer_name_list, updated_shared_buffer_handle_list, num_in_args, in_args, arg_grads, num_aux_states, aux_states, shared_exec_handle, out)
end
_FUNCDEF("MXExecutorSimpleBindEx", { "SymbolHandle", "int", "int", "const mx_uint", "const char * *", "const int *", "const int *", "const mx_uint", "const char * *", "const char * *", "const mx_uint", "const char * *", "const int *", "const mx_uint *", "const mx_uint", "const char * *", "const int *", "const mx_uint", "const char * *", "const int *", "const mx_uint", "const char * *", "int *", "const char * *", "NDArrayHandle *", "const char * * *", "NDArrayHandle * *", "mx_uint *", "NDArrayHandle * *", "NDArrayHandle * *", "mx_uint *", "NDArrayHandle * *", "ExecutorHandle", "ExecutorHandle *" }, "int")

--

--- 
---@brief DEPRECATED. Use MXExecutorReshapeEx instead.
--- Return a new executor with the same symbol and shared memory,
--- but different input/output shapes.
--- 
---@param partial_shaping number @(int) Whether to allow changing the shape of unspecified arguments.
---@param allow_up_sizing number @(int) Whether to allow allocating new ndarrays that's larger than the original.
---@param dev_type number @(int) device type of default context
---@param dev_id number @(int) device id of default context
---@param num_map_keys number @(mx_uint) size of group2ctx map
---@param map_keys number @(const char * *) keys of group2ctx map
---@param map_dev_types number @(const int *) device type of group2ctx map
---@param map_dev_ids number @(const int *) device id of group2ctx map
---@param num_in_args number @(mx_uint *) length of in_args
---@param in_args number @(NDArrayHandle * *) in args array
---@param arg_grads number @(NDArrayHandle * *) arg grads handle array
---@param num_aux_states number @(mx_uint *) length of auxiliary states
---@param aux_states number @(NDArrayHandle * *) auxiliary states array
---@param shared_exec number @(ExecutorHandle) input executor handle for memory sharing
---@param out number @(ExecutorHandle *) output executor handle
---@return number @(int) a new executor
--- 
function M.MXExecutorReshape(partial_shaping, allow_up_sizing, dev_type, dev_id, num_map_keys, map_keys, map_dev_types, map_dev_ids, num_provided_arg_shapes, provided_arg_shape_names, provided_arg_shape_data, provided_arg_shape_idx, num_in_args, in_args, arg_grads, num_aux_states, aux_states, shared_exec, out)
    return _CALL("MXExecutorReshape", partial_shaping, allow_up_sizing, dev_type, dev_id, num_map_keys, map_keys, map_dev_types, map_dev_ids, num_provided_arg_shapes, provided_arg_shape_names, provided_arg_shape_data, provided_arg_shape_idx, num_in_args, in_args, arg_grads, num_aux_states, aux_states, shared_exec, out)
end
_FUNCDEF("MXExecutorReshape", { "int", "int", "int", "int", "mx_uint", "const char * *", "const int *", "const int *", "const mx_uint", "const char * *", "const mx_uint *", "const mx_uint *", "mx_uint *", "NDArrayHandle * *", "NDArrayHandle * *", "mx_uint *", "NDArrayHandle * *", "ExecutorHandle", "ExecutorHandle *" }, "int")

--

--- 
---@brief Return a new executor with the same symbol and shared memory,
--- but different input/output shapes.
--- 
---@param partial_shaping number @(int) Whether to allow changing the shape of unspecified arguments.
---@param allow_up_sizing number @(int) Whether to allow allocating new ndarrays that's larger than the original.
---@param dev_type number @(int) device type of default context
---@param dev_id number @(int) device id of default context
---@param num_map_keys number @(mx_uint) size of group2ctx map
---@param map_keys number @(const char * *) keys of group2ctx map
---@param map_dev_types number @(const int *) device type of group2ctx map
---@param map_dev_ids number @(const int *) device id of group2ctx map
---@param num_in_args number @(mx_uint *) length of in_args
---@param in_args number @(NDArrayHandle * *) in args array
---@param arg_grads number @(NDArrayHandle * *) arg grads handle array
---@param num_aux_states number @(mx_uint *) length of auxiliary states
---@param aux_states number @(NDArrayHandle * *) auxiliary states array
---@param shared_exec number @(ExecutorHandle) input executor handle for memory sharing
---@param out number @(ExecutorHandle *) output executor handle
---@return number @(int) a new executor
--- 
function M.MXExecutorReshapeEx(partial_shaping, allow_up_sizing, dev_type, dev_id, num_map_keys, map_keys, map_dev_types, map_dev_ids, num_provided_arg_shapes, provided_arg_shape_names, provided_arg_shape_data, provided_arg_shape_idx, num_in_args, in_args, arg_grads, num_aux_states, aux_states, shared_exec, out)
    return _CALL("MXExecutorReshapeEx", partial_shaping, allow_up_sizing, dev_type, dev_id, num_map_keys, map_keys, map_dev_types, map_dev_ids, num_provided_arg_shapes, provided_arg_shape_names, provided_arg_shape_data, provided_arg_shape_idx, num_in_args, in_args, arg_grads, num_aux_states, aux_states, shared_exec, out)
end
_FUNCDEF("MXExecutorReshapeEx", { "int", "int", "int", "int", "mx_uint", "const char * *", "const int *", "const int *", "const mx_uint", "const char * *", "const int *", "const mx_uint *", "mx_uint *", "NDArrayHandle * *", "NDArrayHandle * *", "mx_uint *", "NDArrayHandle * *", "ExecutorHandle", "ExecutorHandle *" }, "int")

--

--- 
---@brief get optimized graph from graph executor
--- 
function M.MXExecutorGetOptimizedSymbol(handle, out)
    return _CALL("MXExecutorGetOptimizedSymbol", handle, out)
end
_FUNCDEF("MXExecutorGetOptimizedSymbol", { "ExecutorHandle", "SymbolHandle *" }, "int")

--

--- 
---@brief set a call back to notify the completion of operation
--- 
function M.MXExecutorSetMonitorCallback(handle, callback, callback_handle)
    return _CALL("MXExecutorSetMonitorCallback", handle, callback, callback_handle)
end
_FUNCDEF("MXExecutorSetMonitorCallback", { "ExecutorHandle", "ExecutorMonitorCallback", "void *" }, "int")

--

--- 
---@brief set a call back to notify the completion of operation
---@param monitor_all boolean @(bool) If true, monitor both input and output, otherwise monitor output only.
--- 
function M.MXExecutorSetMonitorCallbackEX(handle, callback, callback_handle, monitor_all)
    return _CALL("MXExecutorSetMonitorCallbackEX", handle, callback, callback_handle, monitor_all)
end
_FUNCDEF("MXExecutorSetMonitorCallbackEX", { "ExecutorHandle", "ExecutorMonitorCallback", "void *", "bool" }, "int")

--

--- 
---@brief List all the available iterator entries
---@param out_size number @(mx_uint *) the size of returned iterators
---@param out_array number @(DataIterCreator * *) the output iteratos entries
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXListDataIters(out_size, out_array)
    return _CALL("MXListDataIters", out_size, out_array)
end
_FUNCDEF("MXListDataIters", { "mx_uint *", "DataIterCreator * *" }, "int")

--

--- 
---@brief Init an iterator, init with parameters
--- the array size of passed in arguments
---@param handle number @(DataIterCreator) of the iterator creator
---@param num_param number @(mx_uint) number of parameter
---@param keys number @(const char * *) parameter keys
---@param vals number @(const char * *) parameter values
---@param out number @(DataIterHandle *) resulting iterator
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXDataIterCreateIter(handle, num_param, keys, vals, out)
    return _CALL("MXDataIterCreateIter", handle, num_param, keys, vals, out)
end
_FUNCDEF("MXDataIterCreateIter", { "DataIterCreator", "mx_uint", "const char * *", "const char * *", "DataIterHandle *" }, "int")

--

--- 
---@brief Get the detailed information about data iterator.
---@param creator number @(DataIterCreator) the DataIterCreator.
---@param name number @(const char * *) The returned name of the creator.
---@param description number @(const char * *) The returned description of the symbol.
---@param num_args number @(mx_uint *) Number of arguments.
---@param arg_names number @(const char * * *) Name of the arguments.
---@param arg_type_infos number @(const char * * *) Type informations about the arguments.
---@param arg_descriptions number @(const char * * *) Description information about the arguments.
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXDataIterGetIterInfo(creator, name, description, num_args, arg_names, arg_type_infos, arg_descriptions)
    return _CALL("MXDataIterGetIterInfo", creator, name, description, num_args, arg_names, arg_type_infos, arg_descriptions)
end
_FUNCDEF("MXDataIterGetIterInfo", { "DataIterCreator", "const char * *", "const char * *", "mx_uint *", "const char * * *", "const char * * *", "const char * * *" }, "int")

--

--- 
---@brief Free the handle to the IO module
---@param handle number @(DataIterHandle) the handle pointer to the data iterator
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXDataIterFree(handle)
    return _CALL("MXDataIterFree", handle)
end
_FUNCDEF("MXDataIterFree", { "DataIterHandle" }, "int")

--

--- 
---@brief Move iterator to next position
---@param handle number @(DataIterHandle) the handle to iterator
---@param out number @(int *) return value of next
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXDataIterNext(handle, out)
    return _CALL("MXDataIterNext", handle, out)
end
_FUNCDEF("MXDataIterNext", { "DataIterHandle", "int *" }, "int")

--

--- 
---@brief Call iterator.Reset
---@param handle number @(DataIterHandle) the handle to iterator
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXDataIterBeforeFirst(handle)
    return _CALL("MXDataIterBeforeFirst", handle)
end
_FUNCDEF("MXDataIterBeforeFirst", { "DataIterHandle" }, "int")

--

--- 
---@brief Get the handle to the NDArray of underlying data
---@param handle number @(DataIterHandle) the handle pointer to the data iterator
---@param out number @(NDArrayHandle *) handle to underlying data NDArray
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXDataIterGetData(handle, out)
    return _CALL("MXDataIterGetData", handle, out)
end
_FUNCDEF("MXDataIterGetData", { "DataIterHandle", "NDArrayHandle *" }, "int")

--

--- 
---@brief Get the image index by array.
---@param handle number @(DataIterHandle) the handle pointer to the data iterator
---@param out_index number @(uint64_t * *) output index of the array.
---@param out_size number @(uint64_t *) output size of the array.
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXDataIterGetIndex(handle, out_index, out_size)
    return _CALL("MXDataIterGetIndex", handle, out_index, out_size)
end
_FUNCDEF("MXDataIterGetIndex", { "DataIterHandle", "uint64_t * *", "uint64_t *" }, "int")

--

--- 
---@brief Get the padding number in current data batch
---@param handle number @(DataIterHandle) the handle pointer to the data iterator
---@param pad number @(int *) pad number ptr
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXDataIterGetPadNum(handle, pad)
    return _CALL("MXDataIterGetPadNum", handle, pad)
end
_FUNCDEF("MXDataIterGetPadNum", { "DataIterHandle", "int *" }, "int")

--

--- 
---@brief Get the handle to the NDArray of underlying label
---@param handle number @(DataIterHandle) the handle pointer to the data iterator
---@param out number @(NDArrayHandle *) the handle to underlying label NDArray
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXDataIterGetLabel(handle, out)
    return _CALL("MXDataIterGetLabel", handle, out)
end
_FUNCDEF("MXDataIterGetLabel", { "DataIterHandle", "NDArrayHandle *" }, "int")

--

--- 
---@brief Initialized ps-lite environment variables
---@param num_vars number @(mx_uint) number of variables to initialize
---@param keys number @(const char * *) environment keys
---@param vals number @(const char * *) environment values
--- 
function M.MXInitPSEnv(num_vars, keys, vals)
    return _CALL("MXInitPSEnv", num_vars, keys, vals)
end
_FUNCDEF("MXInitPSEnv", { "mx_uint", "const char * *", "const char * *" }, "int")

--

--- 
---@brief Create a kvstore
---@param type string @(const char *) the type of KVStore
---@param out number @(KVStoreHandle *) The output type of KVStore
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXKVStoreCreate(type, out)
    return _CALL("MXKVStoreCreate", type, out)
end
_FUNCDEF("MXKVStoreCreate", { "const char *", "KVStoreHandle *" }, "int")

--

--- 
---@brief Set parameters to use low-bit compressed gradients
---@param handle number @(KVStoreHandle) handle to the kvstore
---@param keys number @(const char * *) keys for compression parameters
---@param vals number @(const char * *) values for compression parameters
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXKVStoreSetGradientCompression(handle, num_params, keys, vals)
    return _CALL("MXKVStoreSetGradientCompression", handle, num_params, keys, vals)
end
_FUNCDEF("MXKVStoreSetGradientCompression", { "KVStoreHandle", "mx_uint", "const char * *", "const char * *" }, "int")

--

--- 
---@brief Delete a KVStore handle.
---@param handle number @(KVStoreHandle) handle to the kvstore
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXKVStoreFree(handle)
    return _CALL("MXKVStoreFree", handle)
end
_FUNCDEF("MXKVStoreFree", { "KVStoreHandle" }, "int")

--

--- 
---@brief Init a list of (key,value) pairs in kvstore
---@param handle number @(KVStoreHandle) handle to the kvstore
---@param num number @(mx_uint) the number of key-value pairs
---@param keys number @(const int *) the list of keys
---@param vals number @(NDArrayHandle *) the list of values
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXKVStoreInit(handle, num, keys, vals)
    return _CALL("MXKVStoreInit", handle, num, keys, vals)
end
_FUNCDEF("MXKVStoreInit", { "KVStoreHandle", "mx_uint", "const int *", "NDArrayHandle *" }, "int")

--

--- 
---@brief Init a list of (key,value) pairs in kvstore, where each key is a string
---@param handle number @(KVStoreHandle) handle to the kvstore
---@param num number @(mx_uint) the number of key-value pairs
---@param keys number @(const char * *) the list of keys
---@param vals number @(NDArrayHandle *) the list of values
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXKVStoreInitEx(handle, num, keys, vals)
    return _CALL("MXKVStoreInitEx", handle, num, keys, vals)
end
_FUNCDEF("MXKVStoreInitEx", { "KVStoreHandle", "mx_uint", "const char * *", "NDArrayHandle *" }, "int")

--

--- 
---@brief Push a list of (key,value) pairs to kvstore
---@param handle number @(KVStoreHandle) handle to the kvstore
---@param num number @(mx_uint) the number of key-value pairs
---@param keys number @(const int *) the list of keys
---@param vals number @(NDArrayHandle *) the list of values
---@param priority number @(int) the priority of the action
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXKVStorePush(handle, num, keys, vals, priority)
    return _CALL("MXKVStorePush", handle, num, keys, vals, priority)
end
_FUNCDEF("MXKVStorePush", { "KVStoreHandle", "mx_uint", "const int *", "NDArrayHandle *", "int" }, "int")

--

--- 
---@brief Push a list of (key,value) pairs to kvstore, where each key is a string
---@param handle number @(KVStoreHandle) handle to the kvstore
---@param num number @(mx_uint) the number of key-value pairs
---@param keys number @(const char * *) the list of keys
---@param vals number @(NDArrayHandle *) the list of values
---@param priority number @(int) the priority of the action
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXKVStorePushEx(handle, num, keys, vals, priority)
    return _CALL("MXKVStorePushEx", handle, num, keys, vals, priority)
end
_FUNCDEF("MXKVStorePushEx", { "KVStoreHandle", "mx_uint", "const char * *", "NDArrayHandle *", "int" }, "int")

--

--- 
---@brief pull a list of (key, value) pairs from the kvstore
---@param handle number @(KVStoreHandle) handle to the kvstore
---@param num number @(mx_uint) the number of key-value pairs
---@param keys number @(const int *) the list of keys
---@param vals number @(NDArrayHandle *) the list of values
---@param priority number @(int) the priority of the action
---@param ignore_sparse boolean @(bool) whether to ignore sparse arrays in the request
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXKVStorePullWithSparse(handle, num, keys, vals, priority, ignore_sparse)
    return _CALL("MXKVStorePullWithSparse", handle, num, keys, vals, priority, ignore_sparse)
end
_FUNCDEF("MXKVStorePullWithSparse", { "KVStoreHandle", "mx_uint", "const int *", "NDArrayHandle *", "int", "bool" }, "int")

--

--- 
---@brief pull a list of (key, value) pairs from the kvstore, where each key is a string
---@param handle number @(KVStoreHandle) handle to the kvstore
---@param num number @(mx_uint) the number of key-value pairs
---@param keys number @(const char * *) the list of keys
---@param vals number @(NDArrayHandle *) the list of values
---@param priority number @(int) the priority of the action
---@param ignore_sparse boolean @(bool) whether to ignore sparse arrays in the request
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXKVStorePullWithSparseEx(handle, num, keys, vals, priority, ignore_sparse)
    return _CALL("MXKVStorePullWithSparseEx", handle, num, keys, vals, priority, ignore_sparse)
end
_FUNCDEF("MXKVStorePullWithSparseEx", { "KVStoreHandle", "mx_uint", "const char * *", "NDArrayHandle *", "int", "bool" }, "int")

--

--- 
---@brief pull a list of (key, value) pairs from the kvstore
---@param handle number @(KVStoreHandle) handle to the kvstore
---@param num number @(mx_uint) the number of key-value pairs
---@param keys number @(const int *) the list of keys
---@param vals number @(NDArrayHandle *) the list of values
---@param priority number @(int) the priority of the action
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXKVStorePull(handle, num, keys, vals, priority)
    return _CALL("MXKVStorePull", handle, num, keys, vals, priority)
end
_FUNCDEF("MXKVStorePull", { "KVStoreHandle", "mx_uint", "const int *", "NDArrayHandle *", "int" }, "int")

--

--- 
---@brief pull a list of (key, value) pairs from the kvstore, where each key is a string
---@param handle number @(KVStoreHandle) handle to the kvstore
---@param num number @(mx_uint) the number of key-value pairs
---@param keys number @(const char * *) the list of keys
---@param vals number @(NDArrayHandle *) the list of values
---@param priority number @(int) the priority of the action
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXKVStorePullEx(handle, num, keys, vals, priority)
    return _CALL("MXKVStorePullEx", handle, num, keys, vals, priority)
end
_FUNCDEF("MXKVStorePullEx", { "KVStoreHandle", "mx_uint", "const char * *", "NDArrayHandle *", "int" }, "int")

--

--- 
---@brief pull a list of (key, value) pairs from the kvstore, where each key is an integer.
---        The NDArray pulled back will be in row_sparse storage with only the specified
---        row_ids present based row_ids (others rows are zeros).
---@param handle number @(KVStoreHandle) handle to the kvstore
---@param num number @(mx_uint) the number of key-value pairs
---@param keys number @(const int *) the list of keys
---@param vals number @(NDArrayHandle *) the list of values
---@param row_ids number @(const NDArrayHandle *) the list of row_id NDArrays
---@param priority number @(int) the priority of the action
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXKVStorePullRowSparse(handle, num, keys, vals, row_ids, priority)
    return _CALL("MXKVStorePullRowSparse", handle, num, keys, vals, row_ids, priority)
end
_FUNCDEF("MXKVStorePullRowSparse", { "KVStoreHandle", "mx_uint", "const int *", "NDArrayHandle *", "const NDArrayHandle *", "int" }, "int")

--

--- 
---@brief pull a list of (key, value) pairs from the kvstore, where each key is a string.
---        The NDArray pulled back will be in row_sparse storage with only the specified
---        row_ids present based row_ids (others rows are zeros).
---@param handle number @(KVStoreHandle) handle to the kvstore
---@param num number @(mx_uint) the number of key-value pairs
---@param keys number @(const char * *) the list of keys
---@param vals number @(NDArrayHandle *) the list of values
---@param row_ids number @(const NDArrayHandle *) the list of row_id NDArrays
---@param priority number @(int) the priority of the action
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXKVStorePullRowSparseEx(handle, num, keys, vals, row_ids, priority)
    return _CALL("MXKVStorePullRowSparseEx", handle, num, keys, vals, row_ids, priority)
end
_FUNCDEF("MXKVStorePullRowSparseEx", { "KVStoreHandle", "mx_uint", "const char * *", "NDArrayHandle *", "const NDArrayHandle *", "int" }, "int")

--

--- 
---@brief user-defined updater for the kvstore
--- It's this updater's responsibility to delete \a recv and \a local
--@param the key
--@param recv the pushed value on this key
--@param local the value stored on local on this key
--@param handle The additional handle to the updater
--- 
_TYPEDEF("MXKVStoreUpdater", "void*")

--

--- 
---@brief user-defined updater for the kvstore with string keys
--- It's this updater's responsibility to delete \a recv and \a local
--@param the key
--@param recv the pushed value on this key
--@param local the value stored on local on this key
--@param handle The additional handle to the updater
--- 
_TYPEDEF("MXKVStoreStrUpdater", "void*")

--

--- 
---@brief register a push updater
---@param handle number @(KVStoreHandle) handle to the KVStore
---@param updater number @(MXKVStoreUpdater) udpater function
---@param updater_handle number @(void *) The additional handle used to invoke the updater
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXKVStoreSetUpdater(handle, updater, updater_handle)
    return _CALL("MXKVStoreSetUpdater", handle, updater, updater_handle)
end
_FUNCDEF("MXKVStoreSetUpdater", { "KVStoreHandle", "MXKVStoreUpdater", "void *" }, "int")

--

--- 
---@brief register a push updater with int keys and one with string keys
---@param handle number @(KVStoreHandle) handle to the KVStore
---@param updater number @(MXKVStoreUpdater) updater function with int keys
---@param str_updater number @(MXKVStoreStrUpdater) updater function with string keys
---@param updater_handle number @(void *) The additional handle used to invoke the updater
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXKVStoreSetUpdaterEx(handle, updater, str_updater, updater_handle)
    return _CALL("MXKVStoreSetUpdaterEx", handle, updater, str_updater, updater_handle)
end
_FUNCDEF("MXKVStoreSetUpdaterEx", { "KVStoreHandle", "MXKVStoreUpdater", "MXKVStoreStrUpdater", "void *" }, "int")

--

--- 
---@brief get the type of the kvstore
---@param handle number @(KVStoreHandle) handle to the KVStore
---@param type number @(const char * *) a string type
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXKVStoreGetType(handle, type)
    return _CALL("MXKVStoreGetType", handle, type)
end
_FUNCDEF("MXKVStoreGetType", { "KVStoreHandle", "const char * *" }, "int")

--

--- 
---@brief return The rank of this node in its group, which is in [0, GroupSize).
--- 
---@param handle number @(KVStoreHandle) handle to the KVStore
---@param ret number @(int *) the node rank
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXKVStoreGetRank(handle, ret)
    return _CALL("MXKVStoreGetRank", handle, ret)
end
_FUNCDEF("MXKVStoreGetRank", { "KVStoreHandle", "int *" }, "int")

--

--- 
---@brief return The number of nodes in this group, which is
--- - number of workers if if `IsWorkerNode() == true`,
--- - number of servers if if `IsServerNode() == true`,
--- - 1 if `IsSchedulerNode() == true`,
---@param handle number @(KVStoreHandle) handle to the KVStore
---@param ret number @(int *) the group size
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXKVStoreGetGroupSize(handle, ret)
    return _CALL("MXKVStoreGetGroupSize", handle, ret)
end
_FUNCDEF("MXKVStoreGetGroupSize", { "KVStoreHandle", "int *" }, "int")

--

--- 
---@brief return whether or not this process is a worker node.
---@param ret number @(int *) 1 for yes, 0 for no
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXKVStoreIsWorkerNode(ret)
    return _CALL("MXKVStoreIsWorkerNode", ret)
end
_FUNCDEF("MXKVStoreIsWorkerNode", { "int *" }, "int")

--

--- 
---@brief return whether or not this process is a server node.
---@param ret number @(int *) 1 for yes, 0 for no
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXKVStoreIsServerNode(ret)
    return _CALL("MXKVStoreIsServerNode", ret)
end
_FUNCDEF("MXKVStoreIsServerNode", { "int *" }, "int")

--

--- 
---@brief return whether or not this process is a scheduler node.
---@param ret number @(int *) 1 for yes, 0 for no
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXKVStoreIsSchedulerNode(ret)
    return _CALL("MXKVStoreIsSchedulerNode", ret)
end
_FUNCDEF("MXKVStoreIsSchedulerNode", { "int *" }, "int")

--

--- 
---@brief global barrier among all worker machines
--- 
---@param handle number @(KVStoreHandle) handle to the KVStore
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXKVStoreBarrier(handle)
    return _CALL("MXKVStoreBarrier", handle)
end
_FUNCDEF("MXKVStoreBarrier", { "KVStoreHandle" }, "int")

--

--- 
---@brief whether to do barrier when finalize
--- 
---@param handle number @(KVStoreHandle) handle to the KVStore
---@param barrier_before_exit number @(const int) whether to do barrier when kvstore finalize
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXKVStoreSetBarrierBeforeExit(handle, barrier_before_exit)
    return _CALL("MXKVStoreSetBarrierBeforeExit", handle, barrier_before_exit)
end
_FUNCDEF("MXKVStoreSetBarrierBeforeExit", { "KVStoreHandle", "const int" }, "int")

--

--- 
---@brief the prototype of a server controller
--@param head the head of the command
--@param body the body of the command
--@param controller_handle helper handle for implementing controller
--- 
_TYPEDEF("MXKVStoreServerController", "void*")

--

--- 
---@brief Run as server (or scheduler)
---@param handle number @(KVStoreHandle) handle to the KVStore
---@param controller number @(MXKVStoreServerController) the user-defined server controller
---@param controller_handle number @(void *) helper handle for implementing controller
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXKVStoreRunServer(handle, controller, controller_handle)
    return _CALL("MXKVStoreRunServer", handle, controller, controller_handle)
end
_FUNCDEF("MXKVStoreRunServer", { "KVStoreHandle", "MXKVStoreServerController", "void *" }, "int")

--

--- 
---@brief Send a command to all server nodes
---@param handle number @(KVStoreHandle) handle to the KVStore
---@param cmd_id number @(int) the head of the command
---@param cmd_body string @(const char *) the body of the command
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXKVStoreSendCommmandToServers(handle, cmd_id, cmd_body)
    return _CALL("MXKVStoreSendCommmandToServers", handle, cmd_id, cmd_body)
end
_FUNCDEF("MXKVStoreSendCommmandToServers", { "KVStoreHandle", "int", "const char *" }, "int")

--

--- 
---@brief Get the number of ps dead node(s) specified by {node_id}
--- 
---@param handle number @(KVStoreHandle) handle to the KVStore
---@param node_id number @(const int) Can be a node group or a single node.
---                kScheduler = 1, kServerGroup = 2, kWorkerGroup = 4
---@param number number @(int *) Ouptut number of dead nodes
---@param timeout_sec number @(const int) A node fails to send heartbeart in {timeout_sec} seconds
---                    will be presumed as 'dead'
--- 
function M.MXKVStoreGetNumDeadNode(handle, node_id, number, timeout_sec)
    return _CALL("MXKVStoreGetNumDeadNode", handle, node_id, number, timeout_sec)
end
_FUNCDEF("MXKVStoreGetNumDeadNode", { "KVStoreHandle", "const int", "int *", "const int" }, "int")

--

--- 
---@brief Create a RecordIO writer object
---@param uri string @(const char *) path to file
---@param out number @(RecordIOHandle *) handle pointer to the created object
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXRecordIOWriterCreate(uri, out)
    return _CALL("MXRecordIOWriterCreate", uri, out)
end
_FUNCDEF("MXRecordIOWriterCreate", { "const char *", "RecordIOHandle *" }, "int")

--

--- 
---@brief Delete a RecordIO writer object
---@param handle number @(RecordIOHandle) handle to RecordIO object
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXRecordIOWriterFree(handle)
    return _CALL("MXRecordIOWriterFree", handle)
end
_FUNCDEF("MXRecordIOWriterFree", { "RecordIOHandle" }, "int")

--

--- 
---@brief Write a record to a RecordIO object
---@param handle number @(RecordIOHandle) handle to RecordIO object
---@param buf string @(const char *) buffer to write
---@param size number @(size_t) size of buffer
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXRecordIOWriterWriteRecord(handle, buf, size)
    return _CALL("MXRecordIOWriterWriteRecord", handle, buf, size)
end
_FUNCDEF("MXRecordIOWriterWriteRecord", { "RecordIOHandle", "const char *", "size_t" }, "int")

--

--- 
---@brief Get the current writer pointer position
---@param handle number @(RecordIOHandle) handle to RecordIO object
---@param pos number @(size_t *) handle to output position
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXRecordIOWriterTell(handle, pos)
    return _CALL("MXRecordIOWriterTell", handle, pos)
end
_FUNCDEF("MXRecordIOWriterTell", { "RecordIOHandle", "size_t *" }, "int")

--

--- 
---@brief Create a RecordIO reader object
---@param uri string @(const char *) path to file
---@param out number @(RecordIOHandle *) handle pointer to the created object
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXRecordIOReaderCreate(uri, out)
    return _CALL("MXRecordIOReaderCreate", uri, out)
end
_FUNCDEF("MXRecordIOReaderCreate", { "const char *", "RecordIOHandle *" }, "int")

--

--- 
---@brief Delete a RecordIO reader object
---@param handle number @(RecordIOHandle) handle to RecordIO object
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXRecordIOReaderFree(handle)
    return _CALL("MXRecordIOReaderFree", handle)
end
_FUNCDEF("MXRecordIOReaderFree", { "RecordIOHandle" }, "int")

--

--- 
---@brief Write a record to a RecordIO object
---@param handle number @(RecordIOHandle) handle to RecordIO object
---@param buf number @(char const * *) pointer to return buffer
---@param size number @(size_t *) point to size of buffer
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXRecordIOReaderReadRecord(handle, buf, size)
    return _CALL("MXRecordIOReaderReadRecord", handle, buf, size)
end
_FUNCDEF("MXRecordIOReaderReadRecord", { "RecordIOHandle", "char const * *", "size_t *" }, "int")

--

--- 
---@brief Set the current reader pointer position
---@param handle number @(RecordIOHandle) handle to RecordIO object
---@param pos number @(size_t) target position
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXRecordIOReaderSeek(handle, pos)
    return _CALL("MXRecordIOReaderSeek", handle, pos)
end
_FUNCDEF("MXRecordIOReaderSeek", { "RecordIOHandle", "size_t" }, "int")

--

--- 
---@brief Get the current writer pointer position
---@param handle number @(RecordIOHandle) handle to RecordIO object
---@param pos number @(size_t *) handle to output position
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.MXRecordIOReaderTell(handle, pos)
    return _CALL("MXRecordIOReaderTell", handle, pos)
end
_FUNCDEF("MXRecordIOReaderTell", { "RecordIOHandle", "size_t *" }, "int")

--

--- 
---@brief Create a MXRtc object
--- 
function M.MXRtcCreate(name, num_input, num_output, input_names, output_names, inputs, outputs, kernel, out)
    return _CALL("MXRtcCreate", name, num_input, num_output, input_names, output_names, inputs, outputs, kernel, out)
end
_FUNCDEF("MXRtcCreate", { "char *", "mx_uint", "mx_uint", "char * *", "char * *", "NDArrayHandle *", "NDArrayHandle *", "char *", "RtcHandle *" }, "int")

--

--- 
---@brief Run cuda kernel
--- 
function M.MXRtcPush(handle, num_input, num_output, inputs, outputs, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ)
    return _CALL("MXRtcPush", handle, num_input, num_output, inputs, outputs, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ)
end
_FUNCDEF("MXRtcPush", { "RtcHandle", "mx_uint", "mx_uint", "NDArrayHandle *", "NDArrayHandle *", "mx_uint", "mx_uint", "mx_uint", "mx_uint", "mx_uint", "mx_uint" }, "int")

--

--- 
---@brief Delete a MXRtc object
--- 
function M.MXRtcFree(handle)
    return _CALL("MXRtcFree", handle)
end
_FUNCDEF("MXRtcFree", { "RtcHandle" }, "int")

--

--- 
---@brief register custom operators from frontend.
---@param op_type string @(const char *) name of custom op
---@param creator number @(CustomOpPropCreator)
--- 
function M.MXCustomOpRegister(op_type, creator)
    return _CALL("MXCustomOpRegister", op_type, creator)
end
_FUNCDEF("MXCustomOpRegister", { "const char *", "CustomOpPropCreator" }, "int")

--

--- 
---@brief record custom function for backward later.
---@param num_inputs number @(int) number of input NDArrays.
---@param inputs number @(NDArrayHandle *) handle to input NDArrays.
---@param num_outputs number @(int) number of output NDArrays.
---@param outputs number @(NDArrayHandle *) handle to output NDArrays.
---@param callbacks number @(struct MXCallbackList *) callbacks for backward function.
--- 
function M.MXCustomFunctionRecord(num_inputs, inputs, num_outputs, outputs, callbacks)
    return _CALL("MXCustomFunctionRecord", num_inputs, inputs, num_outputs, outputs, callbacks)
end
_FUNCDEF("MXCustomFunctionRecord", { "int", "NDArrayHandle *", "int", "NDArrayHandle *", "struct MXCallbackList *" }, "int")

--

--- 
---@brief create cuda rtc module
---@param source string @(const char *) cuda source code
---@param num_options number @(int) number of compiler flags
---@param options number @(const char * *) compiler flags
---@param num_exports number @(int) number of exported function names
---@param exported function names
---@param out number @(CudaModuleHandle *) handle to created module
--- 
function M.MXRtcCudaModuleCreate(source, num_options, options, num_exports, exports, out)
    return _CALL("MXRtcCudaModuleCreate", source, num_options, options, num_exports, exports, out)
end
_FUNCDEF("MXRtcCudaModuleCreate", { "const char *", "int", "const char * *", "int", "const char * *", "CudaModuleHandle *" }, "int")

--

--- 
---@brief delete cuda rtc module
---@param handle number @(CudaModuleHandle) handle to cuda module
--- 
function M.MXRtcCudaModuleFree(handle)
    return _CALL("MXRtcCudaModuleFree", handle)
end
_FUNCDEF("MXRtcCudaModuleFree", { "CudaModuleHandle" }, "int")

--

--- 
---@brief get kernel from module
---@param handle number @(CudaModuleHandle) handle to cuda module
---@param name string @(const char *) name of kernel function
---@param num_args number @(int) number of arguments
---@param is_ndarray number @(int *) whether argument is ndarray
---@param is_const number @(int *) whether argument is constant
---@param arg_types number @(int *) data type of arguments
---@param out number @(CudaKernelHandle *) created kernel
--- 
function M.MXRtcCudaKernelCreate(handle, name, num_args, is_ndarray, is_const, arg_types, out)
    return _CALL("MXRtcCudaKernelCreate", handle, name, num_args, is_ndarray, is_const, arg_types, out)
end
_FUNCDEF("MXRtcCudaKernelCreate", { "CudaModuleHandle", "const char *", "int", "int *", "int *", "int *", "CudaKernelHandle *" }, "int")

--

--- 
---@brief delete kernel
---@param handle number @(CudaKernelHandle) handle to previously created kernel
--- 
function M.MXRtcCudaKernelFree(handle)
    return _CALL("MXRtcCudaKernelFree", handle)
end
_FUNCDEF("MXRtcCudaKernelFree", { "CudaKernelHandle" }, "int")

--

--- 
---@brief launch cuda kernel
---@param handle number @(CudaKernelHandle) handle to kernel
---@param dev_id number @(int) (GPU) device id
---@param args number @(void * *) pointer to arguments
---@param grid_dim_x number @(mx_uint) grid dimension x
---@param grid_dim_y number @(mx_uint) grid dimension y
---@param grid_dim_z number @(mx_uint) grid dimension z
---@param block_dim_x number @(mx_uint) block dimension x
---@param block_dim_y number @(mx_uint) block dimension y
---@param block_dim_z number @(mx_uint) block dimension z
---@param shared_mem number @(mx_uint) size of dynamically allocated shared memory
--- 
function M.MXRtcCudaKernelCall(handle, dev_id, args, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x, block_dim_y, block_dim_z, shared_mem)
    return _CALL("MXRtcCudaKernelCall", handle, dev_id, args, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x, block_dim_y, block_dim_z, shared_mem)
end
_FUNCDEF("MXRtcCudaKernelCall", { "CudaKernelHandle", "int", "void * *", "mx_uint", "mx_uint", "mx_uint", "mx_uint", "mx_uint", "mx_uint", "mx_uint" }, "int")

--

--- 
---@brief Get shared memory handle from NDArray
---@param handle number @(NDArrayHandle) NDArray handle.
---@param shared_pid number @(int *) output PID
---@param shared_id number @(int *) output shared memory id.
--- 
function M.MXNDArrayGetSharedMemHandle(handle, shared_pid, shared_id)
    return _CALL("MXNDArrayGetSharedMemHandle", handle, shared_pid, shared_id)
end
_FUNCDEF("MXNDArrayGetSharedMemHandle", { "NDArrayHandle", "int *", "int *" }, "int")

--

--- 
---@brief DEPRECATED. Use MXNDArrayCreateFromSharedMemEx instead.
--- Reconstruct NDArray from shared memory handle
---@param shared_pid number @(int) shared PID
---@param shared_id number @(int) shared memory id
---@param shape number @(const mx_uint *) pointer to NDArray dimensions
---@param ndim number @(mx_uint) number of NDArray dimensions
---@param dtype number @(int) data type of NDArray
---@param out number @(NDArrayHandle *) constructed NDArray
--- 
function M.MXNDArrayCreateFromSharedMem(shared_pid, shared_id, shape, ndim, dtype, out)
    return _CALL("MXNDArrayCreateFromSharedMem", shared_pid, shared_id, shape, ndim, dtype, out)
end
_FUNCDEF("MXNDArrayCreateFromSharedMem", { "int", "int", "const mx_uint *", "mx_uint", "int", "NDArrayHandle *" }, "int")

--

--- 
---@brief Release all unreferenced memory from the devices storage managers memory pool
---@param dev_type number @(int) device type, specify device we want to take
---@param dev_id number @(int) the device id of the specific device
--- 
function M.MXStorageEmptyCache(dev_type, dev_id)
    return _CALL("MXStorageEmptyCache", dev_type, dev_id)
end
_FUNCDEF("MXStorageEmptyCache", { "int", "int" }, "int")

--

--- 
---@brief Reconstruct NDArray from shared memory handle
---@param shared_pid number @(int) shared PID
---@param shared_id number @(int) shared memory id
---@param shape number @(const int *) pointer to NDArray dimensions
---@param ndim number @(int) number of NDArray dimensions
---@param dtype number @(int) data type of NDArray
---@param out number @(NDArrayHandle *) constructed NDArray
--- 
function M.MXNDArrayCreateFromSharedMemEx(shared_pid, shared_id, shape, ndim, dtype, out)
    return _CALL("MXNDArrayCreateFromSharedMemEx", shared_pid, shared_id, shape, ndim, dtype, out)
end
_FUNCDEF("MXNDArrayCreateFromSharedMemEx", { "int", "int", "const int *", "int", "int", "NDArrayHandle *" }, "int")

--

--- 
---@brief Push an asynchronous operation to the engine.
---@param async_func number @(EngineAsyncFunc) Execution function whici takes a parameter on_complete
---                   that must be called when the execution ompletes.
---@param func_param number @(void *) The parameter set on calling async_func, can be NULL.
---@param deleter number @(EngineFuncParamDeleter) The callback to free func_param, can be NULL.
---@param ctx_handle number @(ContextHandle) Execution context.
---@param const_vars_handle number @(EngineVarHandle) The variables that current operation will use
---                          but not mutate.
---@param num_const_vars number @(int) The number of const_vars_handle.
---@param mutable_vars_handle number @(EngineVarHandle) The variables that current operation will mutate.
---@param num_mutable_vars number @(int) The number of mutable_vars_handle.
---@param prop_handle number @(EngineFnPropertyHandle) Property of the function.
---@param priority number @(int) Priority of the action, as hint to the engine.
---@param opr_name string @(const char *) The operation name.
---@param wait boolean @(bool) Whether this is a WaitForVar operation.
--- 
function M.MXEnginePushAsync(async_func, func_param, deleter, ctx_handle, const_vars_handle, num_const_vars, mutable_vars_handle, num_mutable_vars, prop_handle, priority, opr_name, wait)
    return _CALL("MXEnginePushAsync", async_func, func_param, deleter, ctx_handle, const_vars_handle, num_const_vars, mutable_vars_handle, num_mutable_vars, prop_handle, priority, opr_name, wait)
end
_FUNCDEF("MXEnginePushAsync", { "EngineAsyncFunc", "void *", "EngineFuncParamDeleter", "ContextHandle", "EngineVarHandle", "int", "EngineVarHandle", "int", "EngineFnPropertyHandle", "int", "const char *", "bool" }, "int")

--

--- 
---@brief Push a synchronous operation to the engine.
---@param sync_func number @(EngineSyncFunc) Execution function that executes the operation.
---@param func_param number @(void *) The parameter set on calling sync_func, can be NULL.
---@param deleter number @(EngineFuncParamDeleter) The callback to free func_param, can be NULL.
---@param ctx_handle number @(ContextHandle) Execution context.
---@param const_vars_handle number @(EngineVarHandle) The variables that current operation will use
---                          but not mutate.
---@param num_const_vars number @(int) The number of const_vars_handle.
---@param mutable_vars_handle number @(EngineVarHandle) The variables that current operation will mutate.
---@param num_mutable_vars number @(int) The number of mutable_vars_handle.
---@param prop_handle number @(EngineFnPropertyHandle) Property of the function.
---@param priority number @(int) Priority of the action, as hint to the engine.
---@param opr_name string @(const char *) The operation name.
--- 
function M.MXEnginePushSync(sync_func, func_param, deleter, ctx_handle, const_vars_handle, num_const_vars, mutable_vars_handle, num_mutable_vars, prop_handle, priority, opr_name)
    return _CALL("MXEnginePushSync", sync_func, func_param, deleter, ctx_handle, const_vars_handle, num_const_vars, mutable_vars_handle, num_mutable_vars, prop_handle, priority, opr_name)
end
_FUNCDEF("MXEnginePushSync", { "EngineSyncFunc", "void *", "EngineFuncParamDeleter", "ContextHandle", "EngineVarHandle", "int", "EngineVarHandle", "int", "EngineFnPropertyHandle", "int", "const char *" }, "int")

--

--- 
---@brief Push an asynchronous operation to the engine.
---@param async_func number @(EngineAsyncFunc) Execution function whici takes a parameter on_complete
---                   that must be called when the execution ompletes.
---@param func_param number @(void *) The parameter set on calling async_func, can be NULL.
---@param deleter number @(EngineFuncParamDeleter) The callback to free func_param, can be NULL.
---@param ctx_handle number @(ContextHandle) Execution context.
---@param const_nds_handle number @(NDArrayHandle) The NDArrays that current operation will use
---                          but not mutate.
---@param num_const_nds number @(int) The number of const_nds_handle.
---@param mutable_nds_handle number @(NDArrayHandle) The NDArrays that current operation will mutate.
---@param num_mutable_nds number @(int) The number of mutable_nds_handle.
---@param prop_handle number @(EngineFnPropertyHandle) Property of the function.
---@param priority number @(int) Priority of the action, as hint to the engine.
---@param opr_name string @(const char *) The operation name.
---@param wait boolean @(bool) Whether this is a WaitForVar operation.
--- 
function M.MXEnginePushAsyncND(async_func, func_param, deleter, ctx_handle, const_nds_handle, num_const_nds, mutable_nds_handle, num_mutable_nds, prop_handle, priority, opr_name, wait)
    return _CALL("MXEnginePushAsyncND", async_func, func_param, deleter, ctx_handle, const_nds_handle, num_const_nds, mutable_nds_handle, num_mutable_nds, prop_handle, priority, opr_name, wait)
end
_FUNCDEF("MXEnginePushAsyncND", { "EngineAsyncFunc", "void *", "EngineFuncParamDeleter", "ContextHandle", "NDArrayHandle", "int", "NDArrayHandle", "int", "EngineFnPropertyHandle", "int", "const char *", "bool" }, "int")

--

--- 
---@brief Push a synchronous operation to the engine.
---@param sync_func number @(EngineSyncFunc) Execution function that executes the operation.
---@param func_param number @(void *) The parameter set on calling sync_func, can be NULL.
---@param deleter number @(EngineFuncParamDeleter) The callback to free func_param, can be NULL.
---@param ctx_handle number @(ContextHandle) Execution context.
---@param const_nds_handle number @(NDArrayHandle) The NDArrays that current operation will use
---                          but not mutate.
---@param num_const_nds number @(int) The number of const_nds_handle.
---@param mutable_nds_handle number @(NDArrayHandle) The NDArrays that current operation will mutate.
---@param num_mutable_nds number @(int) The number of mutable_nds_handle.
---@param prop_handle number @(EngineFnPropertyHandle) Property of the function.
---@param priority number @(int) Priority of the action, as hint to the engine.
---@param opr_name string @(const char *) The operation name.
--- 
function M.MXEnginePushSyncND(sync_func, func_param, deleter, ctx_handle, const_nds_handle, num_const_nds, mutable_nds_handle, num_mutable_nds, prop_handle, priority, opr_name)
    return _CALL("MXEnginePushSyncND", sync_func, func_param, deleter, ctx_handle, const_nds_handle, num_const_nds, mutable_nds_handle, num_mutable_nds, prop_handle, priority, opr_name)
end
_FUNCDEF("MXEnginePushSyncND", { "EngineSyncFunc", "void *", "EngineFuncParamDeleter", "ContextHandle", "NDArrayHandle", "int", "NDArrayHandle", "int", "EngineFnPropertyHandle", "int", "const char *" }, "int")

--

-- header/c_api_nnvm.h

--

---@brief manually define unsigned int
_TYPEDEF("nn_uint", "unsigned int")

--

---@brief handle to a function that takes param and creates symbol
_TYPEDEF("OpHandle", "void *")

--

---@brief handle to a symbol that can be bind as operator
_TYPEDEF("SymbolHandle", "void *")

--

---@brief handle to Graph
_TYPEDEF("GraphHandle", "void *")

--

--- 
---@brief Set the last error message needed by C API
---@param msg string @(const char *) The error message to set.
--- 
function M.NNAPISetLastError(msg)
    return _CALL("NNAPISetLastError", msg)
end
_FUNCDEF("NNAPISetLastError", { "const char *" }, "void")

--

--- 
---@brief return str message of the last error
---  all function in this file will return 0 when success
---  and -1 when an error occured,
---  NNGetLastError can be called to retrieve the error
--- 
---  this function is threadsafe and can be called by different thread
---  \return error info
--- 
function M.NNGetLastError()
    return _CALL("NNGetLastError")
end
_FUNCDEF("NNGetLastError", {  }, "const char *")

--

--- 
---@brief list all the available operator names, include entries.
---@param out_size number @(nn_uint *) the size of returned array
---@param out_array number @(const char * * *) the output operator name array.
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.NNListAllOpNames(out_size, out_array)
    return _CALL("NNListAllOpNames", out_size, out_array)
end
_FUNCDEF("NNListAllOpNames", { "nn_uint *", "const char * * *" }, "int")

--

--- 
---@brief Get operator handle given name.
---@param op_name string @(const char *) The name of the operator.
---@param op_out number @(OpHandle *) The returnning op handle.
--- 
function M.NNGetOpHandle(op_name, op_out)
    return _CALL("NNGetOpHandle", op_name, op_out)
end
_FUNCDEF("NNGetOpHandle", { "const char *", "OpHandle *" }, "int")

--

--- 
---@brief list all the available operators.
---  This won't include the alias, use ListAllNames
---  instead to get all alias names.
--- 
---@param out_size number @(nn_uint *) the size of returned array
---@param out_array number @(OpHandle * *) the output AtomicSymbolCreator array
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.NNListUniqueOps(out_size, out_array)
    return _CALL("NNListUniqueOps", out_size, out_array)
end
_FUNCDEF("NNListUniqueOps", { "nn_uint *", "OpHandle * *" }, "int")

--

--- 
---@brief Get the detailed information about atomic symbol.
---@param op number @(OpHandle) The operator handle.
---@param real_name number @(const char * *) The returned name of the creator.
---   This name is not the alias name of the atomic symbol.
---@param description number @(const char * *) The returned description of the symbol.
---@param num_doc_args number @(nn_uint *) Number of arguments that contain documents.
---@param arg_names number @(const char * * *) Name of the arguments of doc args
---@param arg_type_infos number @(const char * * *) Type informations about the arguments.
---@param arg_descriptions number @(const char * * *) Description information about the arguments.
---@param return_type number @(const char * *) Return type of the function, if any.
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.NNGetOpInfo(op, real_name, description, num_doc_args, arg_names, arg_type_infos, arg_descriptions, return_type)
    return _CALL("NNGetOpInfo", op, real_name, description, num_doc_args, arg_names, arg_type_infos, arg_descriptions, return_type)
end
_FUNCDEF("NNGetOpInfo", { "OpHandle", "const char * *", "const char * *", "nn_uint *", "const char * * *", "const char * * *", "const char * * *", "const char * *" }, "int")

--

--- 
---@brief Create an AtomicSymbol functor.
---@param op number @(OpHandle) The operator handle
---@param num_param number @(nn_uint) the number of parameters
---@param keys number @(const char * *) the keys to the params
---@param vals number @(const char * *) the vals of the params
---@param out number @(SymbolHandle *) pointer to the created symbol handle
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.NNSymbolCreateAtomicSymbol(op, num_param, keys, vals, out)
    return _CALL("NNSymbolCreateAtomicSymbol", op, num_param, keys, vals, out)
end
_FUNCDEF("NNSymbolCreateAtomicSymbol", { "OpHandle", "nn_uint", "const char * *", "const char * *", "SymbolHandle *" }, "int")

--

--- 
---@brief Create a Variable Symbol.
---@param name string @(const char *) name of the variable
---@param out number @(SymbolHandle *) pointer to the created symbol handle
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.NNSymbolCreateVariable(name, out)
    return _CALL("NNSymbolCreateVariable", name, out)
end
_FUNCDEF("NNSymbolCreateVariable", { "const char *", "SymbolHandle *" }, "int")

--

--- 
---@brief Create a Symbol by grouping list of symbols together
---@param num_symbols number @(nn_uint) number of symbols to be grouped
---@param symbols number @(SymbolHandle *) array of symbol handles
---@param out number @(SymbolHandle *) pointer to the created symbol handle
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.NNSymbolCreateGroup(num_symbols, symbols, out)
    return _CALL("NNSymbolCreateGroup", num_symbols, symbols, out)
end
_FUNCDEF("NNSymbolCreateGroup", { "nn_uint", "SymbolHandle *", "SymbolHandle *" }, "int")

--

--- 
---@brief Add src_dep to the handle as control dep.
---@param handle number @(SymbolHandle) The symbol to add dependency edges on.
---@param src_dep number @(SymbolHandle) the source handles.
--- 
function M.NNAddControlDeps(handle, src_dep)
    return _CALL("NNAddControlDeps", handle, src_dep)
end
_FUNCDEF("NNAddControlDeps", { "SymbolHandle", "SymbolHandle" }, "int")

--

--- 
---@brief Free the symbol handle.
---@param symbol number @(SymbolHandle) the symbol
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.NNSymbolFree(symbol)
    return _CALL("NNSymbolFree", symbol)
end
_FUNCDEF("NNSymbolFree", { "SymbolHandle" }, "int")

--

--- 
---@brief Copy the symbol to another handle
---@param symbol number @(SymbolHandle) the source symbol
---@param out number @(SymbolHandle *) used to hold the result of copy
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.NNSymbolCopy(symbol, out)
    return _CALL("NNSymbolCopy", symbol, out)
end
_FUNCDEF("NNSymbolCopy", { "SymbolHandle", "SymbolHandle *" }, "int")

--

--- 
---@brief Print the content of symbol, used for debug.
---@param symbol number @(SymbolHandle) the symbol
---@param out_str number @(const char * *) pointer to hold the output string of the printing.
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.NNSymbolPrint(symbol, out_str)
    return _CALL("NNSymbolPrint", symbol, out_str)
end
_FUNCDEF("NNSymbolPrint", { "SymbolHandle", "const char * *" }, "int")

--

--- 
---@brief Get string attribute from symbol
---@param symbol number @(SymbolHandle) the source symbol
---@param key string @(const char *) The key of the symbol.
---@param out number @(const char * *) The result attribute, can be NULL if the attribute do not exist.
---@param success number @(int *) Whether the result is contained in out.
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.NNSymbolGetAttr(symbol, key, out, success)
    return _CALL("NNSymbolGetAttr", symbol, key, out, success)
end
_FUNCDEF("NNSymbolGetAttr", { "SymbolHandle", "const char *", "const char * *", "int *" }, "int")

--

--- 
---@brief Set string attribute from symbol.
---  NOTE: Setting attribute to a symbol can affect the semantics(mutable/immutable) of symbolic graph.
--- 
---  Safe recommendaton: use  immutable graph
---  - Only allow set attributes during creation of new symbol as optional parameter
--- 
---  Mutable graph (be careful about the semantics):
---  - Allow set attr at any point.
---  - Mutating an attribute of some common node of two graphs can cause confusion from user.
--- 
---@param symbol number @(SymbolHandle) the source symbol
---@param num_param number @(nn_uint) Number of parameters to set.
---@param keys number @(const char * *) The keys of the attribute
---@param values number @(const char * *) The value to be set
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.NNSymbolSetAttrs(symbol, num_param, keys, values)
    return _CALL("NNSymbolSetAttrs", symbol, num_param, keys, values)
end
_FUNCDEF("NNSymbolSetAttrs", { "SymbolHandle", "nn_uint", "const char * *", "const char * *" }, "int")

--

--- 
---@brief Get all attributes from symbol, including all descendents.
---@param symbol number @(SymbolHandle) the source symbol
---@param recursive_option number @(int) 0 for recursive, 1 for shallow.
---@param out_size number @(nn_uint *) The number of output attributes
---@param out number @(const char * * *) 2*out_size strings representing key value pairs.
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.NNSymbolListAttrs(symbol, recursive_option, out_size, out)
    return _CALL("NNSymbolListAttrs", symbol, recursive_option, out_size, out)
end
_FUNCDEF("NNSymbolListAttrs", { "SymbolHandle", "int", "nn_uint *", "const char * * *" }, "int")

--

--- 
---@brief List inputs variables in the symbol.
---@param symbol number @(SymbolHandle) the symbol
---@param option number @(int) The option to list the inputs
---   option=0 means list all arguments.
---   option=1 means list arguments that are readed only by the graph.
---   option=2 means list arguments that are mutated by the graph.
---@param out_size number @(nn_uint *) output size
---@param out_sym_array number @(SymbolHandle * *) the output array.
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.NNSymbolListInputVariables(symbol, option, out_size, out_sym_array)
    return _CALL("NNSymbolListInputVariables", symbol, option, out_size, out_sym_array)
end
_FUNCDEF("NNSymbolListInputVariables", { "SymbolHandle", "int", "nn_uint *", "SymbolHandle * *" }, "int")

--

--- 
---@brief List input names in the symbol.
---@param symbol number @(SymbolHandle) the symbol
---@param option number @(int) The option to list the inputs
---   option=0 means list all arguments.
---   option=1 means list arguments that are readed only by the graph.
---   option=2 means list arguments that are mutated by the graph.
---@param out_size number @(nn_uint *) output size
---@param out_str_array number @(const char * * *) pointer to hold the output string array
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.NNSymbolListInputNames(symbol, option, out_size, out_str_array)
    return _CALL("NNSymbolListInputNames", symbol, option, out_size, out_str_array)
end
_FUNCDEF("NNSymbolListInputNames", { "SymbolHandle", "int", "nn_uint *", "const char * * *" }, "int")

--

--- 
---@brief List returns names in the symbol.
---@param symbol number @(SymbolHandle) the symbol
---@param out_size number @(nn_uint *) output size
---@param out_str_array number @(const char * * *) pointer to hold the output string array
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.NNSymbolListOutputNames(symbol, out_size, out_str_array)
    return _CALL("NNSymbolListOutputNames", symbol, out_size, out_str_array)
end
_FUNCDEF("NNSymbolListOutputNames", { "SymbolHandle", "nn_uint *", "const char * * *" }, "int")

--

--- 
---@brief Supply number of outputs of the symbol.
---@param symbol number @(SymbolHandle) the symbol
---@param output_count number @(nn_uint *) number of outputs
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.NNSymbolGetNumOutputs(symbol, output_count)
    return _CALL("NNSymbolGetNumOutputs", symbol, output_count)
end
_FUNCDEF("NNSymbolGetNumOutputs", { "SymbolHandle", "nn_uint *" }, "int")

--

--- 
---@brief Get a symbol that contains all the internals.
---@param symbol number @(SymbolHandle) The symbol
---@param out number @(SymbolHandle *) The output symbol whose outputs are all the internals.
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.NNSymbolGetInternals(symbol, out)
    return _CALL("NNSymbolGetInternals", symbol, out)
end
_FUNCDEF("NNSymbolGetInternals", { "SymbolHandle", "SymbolHandle *" }, "int")

--

--- 
---@brief Get a symbol that contains only direct children.
---@param symbol number @(SymbolHandle) The symbol
---@param out number @(SymbolHandle *) The output symbol whose outputs are the direct children.
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.NNSymbolGetChildren(symbol, out)
    return _CALL("NNSymbolGetChildren", symbol, out)
end
_FUNCDEF("NNSymbolGetChildren", { "SymbolHandle", "SymbolHandle *" }, "int")

--

--- 
---@brief Get index-th outputs of the symbol.
---@param symbol number @(SymbolHandle) The symbol
---@param index number @(nn_uint) the Index of the output.
---@param out number @(SymbolHandle *) The output symbol whose outputs are the index-th symbol.
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.NNSymbolGetOutput(symbol, index, out)
    return _CALL("NNSymbolGetOutput", symbol, index, out)
end
_FUNCDEF("NNSymbolGetOutput", { "SymbolHandle", "nn_uint", "SymbolHandle *" }, "int")

--

--- 
---@brief Compose the symbol on other symbols.
--- 
---  This function will change the sym hanlde.
---  To achieve function apply behavior, copy the symbol first
---  before apply.
--- 
---@param sym number @(SymbolHandle) the symbol to apply
---@param name string @(const char *) the name of symbol
---@param num_args number @(nn_uint) number of arguments
---@param keys number @(const char * *) the key of keyword args (optional)
---@param args number @(SymbolHandle *) arguments to sym
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.NNSymbolCompose(sym, name, num_args, keys, args)
    return _CALL("NNSymbolCompose", sym, name, num_args, keys, args)
end
_FUNCDEF("NNSymbolCompose", { "SymbolHandle", "const char *", "nn_uint", "const char * *", "SymbolHandle *" }, "int")

--

--- 
---@brief create a graph handle from symbol
---@param symbol number @(SymbolHandle) The symbol representing the graph.
---@param graph number @(GraphHandle *) The graph handle created.
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.NNGraphCreate(symbol, graph)
    return _CALL("NNGraphCreate", symbol, graph)
end
_FUNCDEF("NNGraphCreate", { "SymbolHandle", "GraphHandle *" }, "int")

--

--- 
---@brief free the graph handle
---@param handle number @(GraphHandle) The handle to be freed.
--- 
function M.NNGraphFree(handle)
    return _CALL("NNGraphFree", handle)
end
_FUNCDEF("NNGraphFree", { "GraphHandle" }, "int")

--

--- 
---@brief Get a new symbol from the graph.
---@param graph number @(GraphHandle) The graph handle.
---@param symbol number @(SymbolHandle *) The corresponding symbol
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.NNGraphGetSymbol(graph, symbol)
    return _CALL("NNGraphGetSymbol", graph, symbol)
end
_FUNCDEF("NNGraphGetSymbol", { "GraphHandle", "SymbolHandle *" }, "int")

--

--- 
---@brief Get Set a attribute in json format.
--- This feature allows pass graph attributes back and forth in reasonable speed.
--- 
---@param handle number @(GraphHandle) The graph handle.
---@param key string @(const char *) The key to the attribute.
---@param json_value string @(const char *) The value need to be in format [type_name, value],
---  Where type_name is a registered type string in C++ side via DMLC_JSON_ENABLE_ANY.
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.NNGraphSetJSONAttr(handle, key, json_value)
    return _CALL("NNGraphSetJSONAttr", handle, key, json_value)
end
_FUNCDEF("NNGraphSetJSONAttr", { "GraphHandle", "const char *", "const char *" }, "int")

--

--- 
---@brief Get a serialized attrirbute from graph.
--- This feature allows pass graph attributes back and forth in reasonable speed.
--- 
---@param handle number @(GraphHandle) The graph handle.
---@param key string @(const char *) The key to the attribute.
---@param json_out number @(const char * *) The result attribute, can be NULL if the attribute do not exist.
---  The json_out is an array of [type_name, value].
---  Where the type_name is a registered type string in C++ side via DMLC_JSON_ENABLE_ANY.
---@param success number @(int *) Whether the result is contained in out.
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.NNGraphGetJSONAttr(handle, key, json_out, success)
    return _CALL("NNGraphGetJSONAttr", handle, key, json_out, success)
end
_FUNCDEF("NNGraphGetJSONAttr", { "GraphHandle", "const char *", "const char * *", "int *" }, "int")

--

--- 
---@brief Set a attribute whose type is std::vector<NodeEntry> in c++
--- This feature allows pass List of symbolic variables for gradient request.
--- 
---@note This is beta feature only used for test purpos
--- 
---@param handle number @(GraphHandle) The graph handle.
---@param key string @(const char *) The key to the attribute.
---@param list number @(SymbolHandle) The symbol whose outputs represents the list of NodeEntry to be passed.
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.NNGraphSetNodeEntryListAttr_(handle, key, list)
    return _CALL("NNGraphSetNodeEntryListAttr_", handle, key, list)
end
_FUNCDEF("NNGraphSetNodeEntryListAttr_", { "GraphHandle", "const char *", "SymbolHandle" }, "int")

--

--- 
---@brief Apply passes on the src graph.
---@param src number @(GraphHandle) The source graph handle.
---@param num_pass number @(nn_uint) The number of pass to be applied.
---@param pass_names number @(const char * *) The names of the pass.
---@param dst number @(GraphHandle *) The result graph.
---@return number @(int) 0 when success, -1 when failure happens
--- 
function M.NNGraphApplyPasses(src, num_pass, pass_names, dst)
    return _CALL("NNGraphApplyPasses", src, num_pass, pass_names, dst)
end
_FUNCDEF("NNGraphApplyPasses", { "GraphHandle", "nn_uint", "const char * *", "GraphHandle *" }, "int")

--

-- header/c_api_test.h

--

--- 
---@brief This API partitions a graph only by the operator names
--- provided by users. This will attach a DefaultSubgraphProperty
--- to the input graph for partitioning. This function should be
--- used only for the testing purpose.
--- 
function M.MXBuildSubgraphByOpNames(sym_handle, prop_name, num_ops, op_names, ret_sym_handle)
    return _CALL("MXBuildSubgraphByOpNames", sym_handle, prop_name, num_ops, op_names, ret_sym_handle)
end
_FUNCDEF("MXBuildSubgraphByOpNames", { "SymbolHandle", "const char *", "const mx_uint", "const char * *", "SymbolHandle *" }, "int")

--

--- 
---@brief Given a subgraph property name, use the provided op names
--- as the op_names attribute for that subgraph property, instead of
--- the predefined one. This is only for the purpose of testing.
--- 
function M.MXSetSubgraphPropertyOpNames(prop_name, num_ops, op_names)
    return _CALL("MXSetSubgraphPropertyOpNames", prop_name, num_ops, op_names)
end
_FUNCDEF("MXSetSubgraphPropertyOpNames", { "const char *", "const mx_uint", "const char * *" }, "int")

--

--- 
---@brief Given a subgraph property name, delete the op name set
--- in the SubgraphPropertyOpNameSet.
--- 
function M.MXRemoveSubgraphPropertyOpNames(prop_name)
    return _CALL("MXRemoveSubgraphPropertyOpNames", prop_name)
end
_FUNCDEF("MXRemoveSubgraphPropertyOpNames", { "const char *" }, "int")

--

-- header/c_predict_api.h

--

---@brief manually define unsigned int
--_TYPEDEF("mx_uint", "unsigned int")

--

---@brief manually define float
--_TYPEDEF("mx_float", "float")

--

---@brief handle to Predictor
_TYPEDEF("PredictorHandle", "void *")

--

---@brief handle to NDArray list
_TYPEDEF("NDListHandle", "void *")

--

--- 
---@brief create a predictor
---@param symbol_json_str string @(const char *) The JSON string of the symbol.
---@param param_bytes number @(const void *) The in-memory raw bytes of parameter ndarray file.
---@param param_size number @(int) The size of parameter ndarray file.
---@param dev_type number @(int) The device type, 1: cpu, 2:gpu
---@param dev_id number @(int) The device id of the predictor.
---@param num_input_nodes number @(mx_uint) Number of input nodes to the net,
---    For feedforward net, this is 1.
---@param input_keys number @(const char * *) The name of input argument.
---    For feedforward net, this is {"data"}
---@param input_shape_indptr number @(const mx_uint *) Index pointer of shapes of each input node.
---    The length of this array = num_input_nodes + 1.
---    For feedforward net that takes 4 dimensional input, this is {0, 4}.
---@param input_shape_data number @(const mx_uint *) A flattened data of shapes of each input node.
---    For feedforward net that takes 4 dimensional input, this is the shape data.
---@param out number @(PredictorHandle *) The created predictor handle.
---@return number @(int) 0 when success, -1 when failure.
--- 
function M.MXPredCreate(symbol_json_str, param_bytes, param_size, dev_type, dev_id, num_input_nodes, input_keys, input_shape_indptr, input_shape_data, out)
    return _CALL("MXPredCreate", symbol_json_str, param_bytes, param_size, dev_type, dev_id, num_input_nodes, input_keys, input_shape_indptr, input_shape_data, out)
end
_FUNCDEF("MXPredCreate", { "const char *", "const void *", "int", "int", "int", "mx_uint", "const char * *", "const mx_uint *", "const mx_uint *", "PredictorHandle *" }, "int")

--

--- 
---@brief create a predictor wich customized outputs
---@param symbol_json_str string @(const char *) The JSON string of the symbol.
---@param param_bytes number @(const void *) The in-memory raw bytes of parameter ndarray file.
---@param param_size number @(int) The size of parameter ndarray file.
---@param dev_type number @(int) The device type, 1: cpu, 2:gpu
---@param dev_id number @(int) The device id of the predictor.
---@param num_input_nodes number @(mx_uint) Number of input nodes to the net,
---    For feedforward net, this is 1.
---@param input_keys number @(const char * *) The name of input argument.
---    For feedforward net, this is {"data"}
---@param input_shape_indptr number @(const mx_uint *) Index pointer of shapes of each input node.
---    The length of this array = num_input_nodes + 1.
---    For feedforward net that takes 4 dimensional input, this is {0, 4}.
---@param input_shape_data number @(const mx_uint *) A flattened data of shapes of each input node.
---    For feedforward net that takes 4 dimensional input, this is the shape data.
---@param num_output_nodes number @(mx_uint) Number of output nodes to the net,
---@param output_keys number @(const char * *) The name of output argument.
---    For example {"global_pool"}
---@param out number @(PredictorHandle *) The created predictor handle.
---@return number @(int) 0 when success, -1 when failure.
--- 
function M.MXPredCreatePartialOut(symbol_json_str, param_bytes, param_size, dev_type, dev_id, num_input_nodes, input_keys, input_shape_indptr, input_shape_data, num_output_nodes, output_keys, out)
    return _CALL("MXPredCreatePartialOut", symbol_json_str, param_bytes, param_size, dev_type, dev_id, num_input_nodes, input_keys, input_shape_indptr, input_shape_data, num_output_nodes, output_keys, out)
end
_FUNCDEF("MXPredCreatePartialOut", { "const char *", "const void *", "int", "int", "int", "mx_uint", "const char * *", "const mx_uint *", "const mx_uint *", "mx_uint", "const char * *", "PredictorHandle *" }, "int")

--

--- 
---@brief create predictors for multiple threads. One predictor for a thread.
---@param symbol_json_str string @(const char *) The JSON string of the symbol.
---@param param_bytes number @(const void *) The in-memory raw bytes of parameter ndarray file.
---@param param_size number @(int) The size of parameter ndarray file.
---@param dev_type number @(int) The device type, 1: cpu, 2:gpu
---@param dev_id number @(int) The device id of the predictor.
---@param num_input_nodes number @(mx_uint) Number of input nodes to the net,
---    For feedforward net, this is 1.
---@param input_keys number @(const char * *) The name of input argument.
---    For feedforward net, this is {"data"}
---@param input_shape_indptr number @(const mx_uint *) Index pointer of shapes of each input node.
---    The length of this array = num_input_nodes + 1.
---    For feedforward net that takes 4 dimensional input, this is {0, 4}.
---@param input_shape_data number @(const mx_uint *) A flattened data of shapes of each input node.
---    For feedforward net that takes 4 dimensional input, this is the shape data.
---@param num_threads number @(int) The number of threads that we'll run the predictors.
---@param out number @(PredictorHandle *) An array of created predictor handles. The array has to be large
---   enough to keep `num_threads` predictors.
---@return number @(int) 0 when success, -1 when failure.
--- 
function M.MXPredCreateMultiThread(symbol_json_str, param_bytes, param_size, dev_type, dev_id, num_input_nodes, input_keys, input_shape_indptr, input_shape_data, num_threads, out)
    return _CALL("MXPredCreateMultiThread", symbol_json_str, param_bytes, param_size, dev_type, dev_id, num_input_nodes, input_keys, input_shape_indptr, input_shape_data, num_threads, out)
end
_FUNCDEF("MXPredCreateMultiThread", { "const char *", "const void *", "int", "int", "int", "mx_uint", "const char * *", "const mx_uint *", "const mx_uint *", "int", "PredictorHandle *" }, "int")

--

--- 
---@brief Change the input shape of an existing predictor.
---@param num_input_nodes number @(mx_uint) Number of input nodes to the net,
---    For feedforward net, this is 1.
---@param input_keys number @(const char * *) The name of input argument.
---    For feedforward net, this is {"data"}
---@param input_shape_indptr number @(const mx_uint *) Index pointer of shapes of each input node.
---    The length of this array = num_input_nodes + 1.
---    For feedforward net that takes 4 dimensional input, this is {0, 4}.
---@param input_shape_data number @(const mx_uint *) A flattened data of shapes of each input node.
---    For feedforward net that takes 4 dimensional input, this is the shape data.
---@param handle number @(PredictorHandle) The original predictor handle.
---@param out number @(PredictorHandle *) The reshaped predictor handle.
---@return number @(int) 0 when success, -1 when failure.
--- 
function M.MXPredReshape(num_input_nodes, input_keys, input_shape_indptr, input_shape_data, handle, out)
    return _CALL("MXPredReshape", num_input_nodes, input_keys, input_shape_indptr, input_shape_data, handle, out)
end
_FUNCDEF("MXPredReshape", { "mx_uint", "const char * *", "const mx_uint *", "const mx_uint *", "PredictorHandle", "PredictorHandle *" }, "int")

--

--- 
---@brief Get the shape of output node.
---  The returned shape_data and shape_ndim is only valid before next call to MXPred function.
---@param handle number @(PredictorHandle) The handle of the predictor.
---@param index number @(mx_uint) The index of output node, set to 0 if there is only one output.
---@param shape_data number @(mx_uint * *) Used to hold pointer to the shape data
---@param shape_ndim number @(mx_uint *) Used to hold shape dimension.
---@return number @(int) 0 when success, -1 when failure.
--- 
function M.MXPredGetOutputShape(handle, index, shape_data, shape_ndim)
    return _CALL("MXPredGetOutputShape", handle, index, shape_data, shape_ndim)
end
_FUNCDEF("MXPredGetOutputShape", { "PredictorHandle", "mx_uint", "mx_uint * *", "mx_uint *" }, "int")

--

--- 
---@brief Set the input data of predictor.
---@param handle number @(PredictorHandle) The predictor handle.
---@param key string @(const char *) The name of input node to set.
---     For feedforward net, this is "data".
---@param data number @(const mx_float *) The pointer to the data to be set, with the shape specified in MXPredCreate.
---@param size number @(mx_uint) The size of data array, used for safety check.
---@return number @(int) 0 when success, -1 when failure.
--- 
function M.MXPredSetInput(handle, key, data, size)
    return _CALL("MXPredSetInput", handle, key, data, size)
end
_FUNCDEF("MXPredSetInput", { "PredictorHandle", "const char *", "const mx_float *", "mx_uint" }, "int")

--

--- 
---@brief Run a forward pass to get the output.
---@param handle number @(PredictorHandle) The handle of the predictor.
---@return number @(int) 0 when success, -1 when failure.
---
function M.MXPredForward(handle)
    return _CALL("MXPredForward", handle)
end
_FUNCDEF("MXPredForward", { "PredictorHandle" }, "int")

--

---
---@brief Run a interactive forward pass to get the output.
---  This is helpful for displaying progress of prediction which can be slow.
---  User must call PartialForward from step=0, keep increasing it until step_left=0.
---@code
--- int step_left = 1;
--- for (int step = 0; step_left != 0; ++step) {
---    MXPredPartialForward(handle, step, &step_left);
---    printf("Current progress [%d/%d]\n", step, step + step_left + 1);
--- }
---@endcode
---@param handle number @(PredictorHandle) The handle of the predictor.
---@param step number @(int) The current step to run forward on.
---@param step_left number @(int *) The number of steps left
---@return number @(int) 0 when success, -1 when failure.
---
function M.MXPredPartialForward(handle, step, step_left)
    return _CALL("MXPredPartialForward", handle, step, step_left)
end
_FUNCDEF("MXPredPartialForward", { "PredictorHandle", "int", "int *" }, "int")

--

---
---@brief Get the output value of prediction.
---@param handle number @(PredictorHandle) The handle of the predictor.
---@param index number @(mx_uint) The index of output node, set to 0 if there is only one output.
---@param data number @(mx_float *) User allocated data to hold the output.
---@param size number @(mx_uint) The size of data array, used for safe checking.
---@return number @(int) 0 when success, -1 when failure.
---
function M.MXPredGetOutput(handle, index, data, size)
    return _CALL("MXPredGetOutput", handle, index, data, size)
end
_FUNCDEF("MXPredGetOutput", { "PredictorHandle", "mx_uint", "mx_float *", "mx_uint" }, "int")

--

---
---@brief Free a predictor handle.
---@param handle number @(PredictorHandle) The handle of the predictor.
---@return number @(int) 0 when success, -1 when failure.
---
function M.MXPredFree(handle)
    return _CALL("MXPredFree", handle)
end
_FUNCDEF("MXPredFree", { "PredictorHandle" }, "int")

--

---
---@brief Create a NDArray List by loading from ndarray file.
---     This can be used to load mean image file.
---@param nd_file_bytes string @(const char *) The byte contents of nd file to be loaded.
---@param nd_file_size number @(int) The size of the nd file to be loaded.
---@param out number @(NDListHandle *) The out put NDListHandle
---@param out_length number @(mx_uint *) Length of the list.
---@return number @(int) 0 when success, -1 when failure.
---
function M.MXNDListCreate(nd_file_bytes, nd_file_size, out, out_length)
    return _CALL("MXNDListCreate", nd_file_bytes, nd_file_size, out, out_length)
end
_FUNCDEF("MXNDListCreate", { "const char *", "int", "NDListHandle *", "mx_uint *" }, "int")

--

---
---@brief Get an element from list
---@param handle number @(NDListHandle) The handle to the NDArray
---@param index number @(mx_uint) The index in the list
---@param out_key number @(const char * *) The output key of the item
---@param out_data number @(const mx_float * *) The data region of the item
---@param out_shape number @(const mx_uint * *) The shape of the item.
---@param out_ndim number @(mx_uint *) The number of dimension in the shape.
---@return number @(int) 0 when success, -1 when failure.
---
function M.MXNDListGet(handle, index, out_key, out_data, out_shape, out_ndim)
    return _CALL("MXNDListGet", handle, index, out_key, out_data, out_shape, out_ndim)
end
_FUNCDEF("MXNDListGet", { "NDListHandle", "mx_uint", "const char * *", "const mx_float * *", "const mx_uint * *", "mx_uint *" }, "int")

--

---
---@brief Free a MXAPINDList
---@param handle number @(NDListHandle) The handle of the MXAPINDList.
---@return number @(int) 0 when success, -1 when failure.
---
function M.MXNDListFree(handle)
    return _CALL("MXNDListFree", handle)
end
_FUNCDEF("MXNDListFree", { "NDListHandle" }, "int")

--

return M
