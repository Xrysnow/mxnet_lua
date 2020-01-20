# mxnet_lua

This project is Lua language binding for [MXNet](https://github.com/apache/incubator-mxnet/) (unofficial).

It's still under construction, see `mx/__init__.lua` for progress. `NDArray` and `Symbol` is available.

## Requirements

- Luajit 2.0.5+
- MXNet 1.5.0 binaries (can get from a python distribution)

## Installation

- make sure path of this project is in the search path of Lua
- make sure `libmxnet.dll`/`libmxnet.so` can be loaded by Lua host

## Example

```lua
> mx = require('mxnet')
> a = mx.nd.arange(6):reshape({2,3})
> print(a)
[[0, 1, 2],
 [3, 4, 5]]
<NDArray 2x3 @cpu(0)>
> print((1+a.T^2/2):as_in_context(mx.gpu()))
[[1, 5.5],
 [1.5, 9],
 [3, 13.5]]
<NDArray 3x2 @gpu(0)>
> c = a:exp():floor():ascdata() -- convert to cdata
> t = {} for i = 0, 5 do t[#t+1] = c[i] end
> print(table.concat(t, ' '))
1 2 7 20 54 148
```
