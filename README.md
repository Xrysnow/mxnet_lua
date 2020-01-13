# mxnet_lua

This project is Lua language binding for [MXNet](https://github.com/apache/incubator-mxnet/).

It's still under construction. See `mx/__init__.lua` for progress.

## Requirements

- Luajit 2.0.5+
- MXNet 1.5.0 binaries (can get from a python distribution)

## Installation

- make sure path of this project is in the search path of Lua
- Windows: put `libmxnet.dll` and other binaries into `bin` folder
- Linux: make sure `libmxnet.so` is in search path of system

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
