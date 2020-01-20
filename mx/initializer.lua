--
local M = {}

local logging = require('mx_py.logging')
local json = require('mx_util.json')
local ndarray = require('mx.ndarray.__init__')
local NDArray, load = ndarray.NDArray, ndarray.load
local random = require('mx.random')
local registry = require('mx.registry')

---@class mx.initializer.InitDesc:string
local InitDesc = class('mx.initializer.InitDesc')
M.InitDesc = InitDesc

function InitDesc:ctor(name, attrs, global_init)
    attrs, global_init = default(attrs, {}, global_init, None)
    self.attrs = attrs
    self.global_init = global_init
    self._name = tostring(name)
end

function InitDesc:__tostring()
    return self._name
end

function InitDesc:__index(k)
    local v = rawget(self, k)
    if v ~= nil then
        return v
    end
    local tk = type(k)
    if tk == 'string' then
        if self._name[k] then
            return self._name[k](self._name)
        elseif not isnone(InitDesc[k]) then
            return InitDesc[k]
        end
    end
    return nil
end

---@class mx.initializer.Initializer
local Initializer = class('mx.initializer.Initializer')
M.Initializer = Initializer

function Initializer:ctor(kwargs)
    self._kwargs = kwargs or {}
    self._verbose = false
    self._print_func = None
end

function Initializer:set_verbosity(verbose, print_func)
    verbose, print_func = default(verbose, false, print_func, None)
    self._verbose = verbose
    if isnone(print_func) then
        local function asum_stat(x)
            return tostring((ndarray.op.norm(x) / math.sqrt(x.size)):asscalar())
        end
        print_func = asum_stat
    end
    self._print_func = print_func
    return self
end

function Initializer:_verbose_print(desc, init, arr)
    if self._verbose and self._print_func then
        logging.info('Initialized %s as %s: %s', desc, init, self._print_func(arr))
    end
end

function Initializer:dumps()
    return json.encode({ getclassname(self):lower(), self._kwargs })
end

function Initializer:__call(desc, arr)
    if not isinstance(desc, InitDesc) then
        self:_legacy_init(desc, arr)
        return
    end

    if isnone(desc.global_init) then
        desc.global_init = self
    end
    local init = desc.attrs.__init__
    if init then
        -- when calling Variable initializer
        M.create(init):_init_weight(desc, arr)
        self:_verbose_print(desc, init, arr)
    else
        -- register nnvm::FSetInputVariableAttrs in the backend for new patterns
        -- don't add new cases here.
        if desc:ends_with('weight') then
            self:_init_weight(desc, arr)
            self:_verbose_print(desc, 'weight', arr)
        elseif desc:ends_with('bias') then
            self:_init_bias(desc, arr)
            self:_verbose_print(desc, 'bias', arr)
        elseif desc:ends_with('gamma') then
            self:_init_gamma(desc, arr)
            self:_verbose_print(desc, 'gamma', arr)
        elseif desc:ends_with('beta') then
            self:_init_beta(desc, arr)
            self:_verbose_print(desc, 'beta', arr)
        elseif desc:ends_with('min') then
            self:_init_zero(desc, arr)
            self:_verbose_print(desc, 'min', arr)
        elseif desc:ends_with('max') then
            self:_init_one(desc, arr)
            self:_verbose_print(desc, 'max', arr)
        elseif desc:ends_with('weight_quantize') then
            self:_init_quantized_weight(desc, arr)
            self:_verbose_print(desc, 'weight_quantize', arr)
        elseif desc:ends_with('bias_quantize') then
            self:_init_quantized_bias(desc, arr)
            self:_verbose_print(desc, 'bias_quantize', arr)
        else
            self:_init_default(desc, arr)
        end
    end
end

function Initializer:_legacy_init(name, arr)
    logging.warning('Calling initializer with init(str, NDArray) has been deprecated.' ..
                            ' Please use init(mx.init.InitDesc(...), NDArray) instead.')
    if not isinstance(name, '') then
        raise('TypeError', 'name must be string')
    end
    if not isinstance(arr, NDArray) then
        raise('TypeError', 'arr must be NDArray')
    end
    if name:starts_with('upsampling') then
        self:_init_bilinear(name, arr)
    elseif name:starts_with('stn_loc') and name:ends_with('weight') then
        self:_init_zero(name, arr)
    elseif name:starts_with('stn_loc') and name:ends_with('bias') then
        self:_init_loc_bias(name, arr)
    elseif name:ends_with('bias') then
        self:_init_bias(name, arr)
    elseif name:ends_with('gamma') then
        self:_init_gamma(name, arr)
    elseif name:ends_with('beta') then
        self:_init_beta(name, arr)
    elseif name:ends_with('weight') then
        self:_init_weight(name, arr)
    elseif name:ends_with("moving_mean") then
        self:_init_zero(name, arr)
    elseif name:ends_with("moving_var") then
        self:_init_one(name, arr)
    elseif name:ends_with("moving_inv_var") then
        self:_init_zero(name, arr)
    elseif name:ends_with("moving_avg") then
        self:_init_zero(name, arr)
    elseif name:ends_with('min') then
        self:_init_zero(name, arr)
    elseif name:ends_with('max') then
        self:_init_one(name, arr)
    else
        self:_init_default(name, arr)
    end
end

function Initializer:_init_bilinear(_, arr)
    local shape = arr.shape
    local prod = table.prod(shape)
    local weight = ndarray.zeros({ prod }, nil, 'float32')
    local f = math.ceil(shape[4] / 2)
    local c = (2 * f - 1 - f % 2) / (2. * f)
    for i = 0, prod - 1 do
        local x = i % shape[4]
        local y = math.floor(i / shape[4]) % shape[3]
        weight[i - 1] = (1 - math.abs(x / f - c)) * (1 - math.abs(y / f - c))
    end
    arr[{}] = weight:reshape(shape)
end

function Initializer:_init_loc_bias(_, arr)
    local shape = arr.shape
    assert(shape[1] == 6)
    arr[{}] = ndarray.array({ 1.0, 0, 0, 0, 1.0, 0 })
end

function Initializer:_init_zero(_, arr)
    arr[{}] = 0
end

function Initializer:_init_one(_, arr)
    arr[{}] = 1
end

function Initializer:_init_bias(_, arr)
    arr[{}] = 0
end

function Initializer:_init_quantized_bias(_, arr)
    arr[{}] = 0
end

function Initializer:_init_gamma(_, arr)
    arr[{}] = 1
end

function Initializer:_init_beta(_, arr)
    arr[{}] = 0
end

function Initializer:_init_weight(name, arr)
    raise('NotImplementedError', 'Must override it')
end

function Initializer:_init_quantized_weight(_, arr)
    arr[{}] = random.randint(-127, 127, nil, 'int8')
end

function Initializer:_init_default(name, _)
    raise('ValueError', ('Unknown initialization pattern for %s. '):format(name) ..
            'Default initialization is now limited to ' ..
            '"weight", "bias", "gamma" (1.0), and "beta" (0.0).' ..
            'Please use mx.sym.Variable(init=mx.init.*) to set initialization pattern')
end

local _register = registry.get_register_func(Initializer, 'initializer')
M._register = _register
local alias = registry.get_alias_func(Initializer, 'initializer')
M.alias = alias
local create = registry.get_create_func(Initializer, 'initializer')
M.create = create

function M.register(klass)
    return _register(klass)
end
local register = M.register

---@class mx.initializer.Load
local Load = class('mx.initializer.Load')
M.Load = Load

function Load:ctor(param, default_init, verbose)
    verbose = default(verbose, false)
    if isinstance(param, '') then
        param = load(param)
    end
    assert(type(param) == 'table')
    self.param = {}
    for name, arr in pairs(param) do
        if name:starts_with('arg:') or name:starts_with('aux:') then
            self.param[name:sub(5, -1)] = arr
        else
            self.param[name] = arr
        end
    end
    self.default_init = default_init
    self.verbose = verbose
end

function Load:__call(name, arr)
    if self.param[name] then
        assert(table.equal(arr.shape, self.param[name].shape),
               ('Parameter %s cannot be initialized from loading. '):format(name) ..
                       'Shape mismatch.')
        arr[{}] = self.param[name]
        if self.verbose then
            logging.info('Initialized %s by loading', name)
        end
    else
        assert(not isnone(self.default_init),
               ('Cannot Initialize %s. Not found in loaded param '):format(name) ..
                       'and no default Initializer is provided.')
        self.default_init(name, arr)
        if self.verbose then
            logging.info('Initialized %s by default', name)
        end
    end
end

---@class mx.initializer.Mix
local Mix = class('mx.initializer.Mix')
M.Mix = Mix

function Mix:ctor(patterns, initializers)
    assert(len(patterns) == len(initializers))
    self.map = table.zip(patterns, initializers)
end

function Mix:__call(name, arr)
    for _, v in ipairs(self.map) do
        if name:match(v[1]) then
            v[2](name, arr)
            return
        end
    end
    raise('ValueError', ('Parameter name %s did not match any pattern. Consider'):format(name) ..
            'add a ".*" pattern at the and with default Initializer.')
end

---@class mx.initializer.Zero:mx.initializer.Initializer
local Zero = class('mx.initializer.Zero', Initializer)
M.Zero = Zero
register(Zero)
alias('zeros')(Zero)

function Zero:ctor()
    Initializer.ctor(self)
end

function Zero:_init_weight(_, arr)
    arr[{}] = 0
end

---@class mx.initializer.One:mx.initializer.Initializer
local One = class('mx.initializer.One', Initializer)
M.One = One
register(One)
alias('ones')(One)

function One:ctor()
    Initializer.ctor(self)
end

function One:_init_weight(_, arr)
    arr[{}] = 1
end

---@class mx.initializer.Constant:mx.initializer.Initializer
local Constant = class('mx.initializer.Constant', Initializer)
M.Constant = Constant
register(Constant)

function Constant:ctor(value)
    Initializer.ctor(self, { value = value })
    self.value = value
end

function Constant:_init_weight(_, arr)
    arr[{}] = self.value
end

---@class mx.initializer.Uniform:mx.initializer.Initializer
local Uniform = class('mx.initializer.Uniform', Initializer)
M.Uniform = Uniform
register(Uniform)

function Uniform:ctor(scale)
    scale = default(scale, 0.07)
    Initializer.ctor(self, { scale = scale })
    self.scale = scale
end

function Uniform:_init_weight(_, arr)
    random.uniform(-self.scale, self.scale, nil, nil, nil, arr)
end

---@class mx.initializer.Normal:mx.initializer.Initializer
local Normal = class('mx.initializer.Normal', Initializer)
M.Normal = Normal
register(Normal)

function Normal:ctor(sigma)
    sigma = default(sigma, 0.01)
    Initializer.ctor(self, { sigma = sigma })
    self.sigma = sigma
end

function Normal:_init_weight(_, arr)
    random.normal(0, self.sigma, nil, nil, nil, arr)
end

---@class mx.initializer.Orthogonal:mx.initializer.Initializer
local Orthogonal = class('mx.initializer.Orthogonal', Initializer)
M.Orthogonal = Orthogonal
register(Orthogonal)

function Orthogonal:ctor(scale, rand_type)
    scale, rand_type = default(scale, 1.414, rand_type, 'uniform')
    Initializer.ctor(self, { scale = scale, rand_type = rand_type })
    self.scale = scale
    self.rand_type = rand_type
end

function Orthogonal:_init_weight(_, arr)
    --TODO: need mxnet 1.6.0
    raise('NotImplementedError')
end

---@class mx.initializer.Xavier:mx.initializer.Initializer
local Xavier = class('mx.initializer.Xavier', Initializer)
M.Xavier = Xavier
register(Xavier)

function Xavier:ctor(rnd_type, factor_type, magnitude)
    rnd_type, factor_type, magnitude = default(
            rnd_type, 'uniform', factor_type, 'avg', magnitude, 3)
    Initializer.ctor(self, { rnd_type  = rnd_type, factor_type = factor_type,
                             magnitude = magnitude })
    self.rnd_type = rnd_type
    self.factor_type = factor_type
    self.magnitude = magnitude
end

function Xavier:_init_weight(name, arr)
    local shape = arr.shape
    local hw_scale = 1
    if len(shape) < 2 then
        raise('ValueError', ('Xavier initializer cannot be applied to vector %s.'):format(name) ..
                ' It requires at least 2D.')
    end
    if len(shape) > 2 then
        hw_scale = table.prod(table.slice(shape, 3))
    end
    local fan_in, fan_out = shape[1] * hw_scale, shape[0] * hw_scale
    local factor = 1
    if self.factor_type == 'avg' then
        factor = (fan_in + fan_out) / 2
    elseif self.factor_type == 'in' then
        factor = fan_in
    elseif self.factor_type == 'out' then
        factor = fan_out
    else
        raise('ValueError', 'Incorrect factor type')
    end
    local scale = math.sqrt(self.magnitude / factor)
    if self.rnd_type == 'uniform' then
        random.uniform(-scale, scale, nil, nil, nil, arr)
    elseif self.rnd_type == 'gaussian' then
        random.normal(0, scale, nil, nil, nil, arr)
    else
        raise('ValueError', 'Unknown random type: ' .. self.rnd_type)
    end
end

---@class mx.initializer.MSRAPrelu:mx.initializer.Xavier
local MSRAPrelu = class('mx.initializer.MSRAPrelu', Xavier)
M.MSRAPrelu = MSRAPrelu
register(MSRAPrelu)

function MSRAPrelu:ctor(factor_type, slope)
    factor_type, slope = default(factor_type, 'avg', slope, 0.25)
    local magnitude = 2 / (1 + slope ^ 2)
    Xavier.ctor(self, 'gaussian', factor_type, magnitude)
    self._kwargs = { factor_type = factor_type, slope = slope }
end

---@class mx.initializer.Bilinear:mx.initializer.Initializer
local Bilinear = class('mx.initializer.Bilinear', Initializer)
M.Bilinear = Bilinear
register(Bilinear)

function Bilinear:ctor()
    Initializer.ctor(self)
end

function Bilinear:_init_weight(name, arr)
    local shape = arr.shape
    local prod = table.prod(shape)
    local weight = ndarray.zeros({ prod }, nil, 'float32')
    local f = math.ceil(shape[4] / 2.)
    local c = (2 * f - 1 - f % 2) / (2. * f)
    for i = 0, prod - 1 do
        local x = i % shape[4]
        local y = math.floor(i / shape[4]) % shape[3]
        weight[i - 1] = (1 - math.abs(x / f - c)) * (1 - math.abs(y / f - c))
    end
    arr[{}] = weight:reshape(shape)
end

---@class mx.initializer.LSTMBias:mx.initializer.Initializer
local LSTMBias = class('mx.initializer.LSTMBias', Initializer)
M.LSTMBias = LSTMBias
register(LSTMBias)

function LSTMBias:ctor(forget_bias)
    forget_bias = default(forget_bias, 1.0)
    Initializer.ctor(self, { forget_bias = forget_bias })
    self.forget_bias = forget_bias
end

function LSTMBias:_init_weight(name, arr)
    arr[{}] = 0
    -- in the case of LSTMCell the forget gate is the second
    -- gate of the 4 LSTM gates, we modify the according values.
    local num_hidden = math.floor(arr.shape[1] / 4)
    arr[{ num_hidden, 2 * num_hidden }] = self.forget_bias
end

---@class mx.initializer.FusedRNN:mx.initializer.Initializer
local FusedRNN = class('mx.initializer.FusedRNN', Initializer)
M.FusedRNN = FusedRNN
register(FusedRNN)

function FusedRNN:ctor(init, num_hidden, num_layers, mode, bidirectional, forget_bias)
    bidirectional, forget_bias = default(bidirectional, false, forget_bias, 1.0)
    if isinstance(init, '') then
        local j = json.decode(init)
        local klass, kwargs = j[1], j[2]
        init = registry._REGISTRY[klass:lower()](kwargs)
    end
    Initializer.ctor(self, { init          = isnone(init) and None or init:dumps(),
                             num_hidden    = num_hidden, num_layers = num_layers, mode = mode,
                             bidirectional = bidirectional, forget_bias = forget_bias })
    self._init = init
    self._num_hidden = num_hidden
    self._num_layers = num_layers
    self._mode = mode
    self._bidirectional = bidirectional
    self._forget_bias = forget_bias
end

function FusedRNN:_init_weight(desc, arr)
    local rnn_cell = require('mx.rnn').rnn_cell
    local cell = rnn_cell.FusedRNNCell(self._num_hidden, self._num_layers,
                                       self._mode, self._bidirectional,
                                       nil, nil,
                                       self._forget_bias, '')
    local args = cell:unpack_weights({ parameters = arr })
    for name, a in pairs(args) do
        local arg_desc = InitDesc(name, nil, desc.global_init)
        -- for lstm bias, we use a custom initializer
        -- which adds a bias to the forget gate
        if self._mode == 'lstg' and name:ends_with('_f_bias') then
            a[{}] = self._forget_bias
        elseif isnone(self._init) then
            desc.global_init(arg_desc, a)
        else
            self._init(arg_desc, a)
        end
    end

    arr[{}] = cell:pack_weights(args)['parameters']
end

return M
