--
local logging = require('python.logging')
local cos, pi = math.cos, math.pi

--
local M = {}

---@class mx.lr_scheduler.LRScheduler
--- Base class of a learning rate scheduler.
---
--- A scheduler returns a new learning rate based on the number of updates that have
--- been performed.
local LRScheduler = class('mx.lr_scheduler.LRScheduler')
M.LRScheduler = LRScheduler

local _warmup_mode = { linear = true, constant = true }

function LRScheduler:ctor(base_lr, warmup_steps, warmup_begin_lr, warmup_mode)
    warmup_mode = warmup_mode or 'linear'
    self.base_lr = base_lr or 0.01
    self.warmup_steps = warmup_steps or 0
    self.warmup_final_lr = self.base_lr
    self.warmup_begin_lr = warmup_begin_lr or 0
    if self.warmup_begin_lr > self.warmup_final_lr then
        error("Base lr has to be higher than warmup_begin_lr")
    end
    if self.warmup_steps < 0 then
        error("Warmup steps has to be positive or 0")
    end
    if not _warmup_mode[warmup_mode] then
        error("Supports only linear and constant modes of warmup")
    end
    self.warmup_mode = warmup_mode
end

function LRScheduler:get_warmup_lr(num_update)
    assert(num_update < self.warmup_steps)
    if self.warmup_mode == 'linear' then
        local increase = (self.warmup_final_lr - self.warmup_begin_lr)
                * (num_update) / (self.warmup_steps)
        return self.warmup_begin_lr + increase
    elseif self.warmup_mode == 'constant' then
        return self.warmup_begin_lr
    else
        error(('Invalid warmup mode %s'):format(self.warmup_mode))
    end
end

function LRScheduler:__call(num_update)
    error('must override this')
end

---@class mx.lr_scheduler.FactorScheduler:mx.lr_scheduler.LRScheduler
local FactorScheduler = class('mx.lr_scheduler.FactorScheduler', LRScheduler)
M.FactorScheduler = FactorScheduler

function FactorScheduler:ctor(step, factor, stop_factor_lr, base_lr,
                              warmup_steps, warmup_begin_lr, warmup_mode)
    factor = factor or 1
    stop_factor_lr = stop_factor_lr or 1e-8
    base_lr = base_lr or 0.01
    warmup_steps = warmup_steps or 0
    warmup_begin_lr = warmup_begin_lr or 0
    warmup_mode = warmup_mode or 'linear'
    LRScheduler.ctor(self, base_lr, warmup_steps, warmup_begin_lr, warmup_mode)
    if step < 1 then
        error('Schedule step must be greater or equal than 1 round')
    end
    if factor > 1.0 then
        error('Factor must be no more than 1 to make lr reduce')
    end
    self.step = step
    self.factor = factor
    self.stop_factor_lr = stop_factor_lr
    self.count = 0
end

function FactorScheduler:__call(num_update)
    if num_update < self.warmup_steps then
        return self:get_warmup_lr(num_update)
    end
    -- NOTE: use while rather than if  (for continuing training via load_epoch)
    while num_update > self.count + self.step do
        self.count = self.count + self.step
        self.base_lr = self.base_lr * self.factor
        if self.base_lr < self.stop_factor_lr then
            self.base_lr = self.stop_factor_lr
            logging.info("Update[%d]: now learning rate arrived at %0.5e, will not change in the future",
                         num_update, self.base_lr)
        else
            logging.info("Update[%d]: Change learning rate to %0.5e",
                         num_update, self.base_lr)
        end
    end
    return self.base_lr
end

---@class mx.lr_scheduler.MultiFactorScheduler:mx.lr_scheduler.LRScheduler
local MultiFactorScheduler = class('mx.lr_scheduler.MultiFactorScheduler', LRScheduler)
M.MultiFactorScheduler = MultiFactorScheduler

function MultiFactorScheduler:ctor(step, factor, base_lr,
                                   warmup_steps, warmup_begin_lr, warmup_mode)
    factor = factor or 1
    base_lr = base_lr or 0.01
    warmup_steps = warmup_steps or 0
    warmup_begin_lr = warmup_begin_lr or 0
    warmup_mode = warmup_mode or 'linear'
    LRScheduler.ctor(self, base_lr, warmup_steps, warmup_begin_lr, warmup_mode)
    assert(#step >= 1)
    for i, _step in ipairs(step) do
        if i ~= 1 and step[i] <= step[i - 1] then
            error('Schedule step must be an increasing integer list')
        end
        if _step < 1 then
            error('Schedule step must be greater or equal than 1 round')
        end
    end
    if factor > 1.0 then
        error('Factor must be no more than 1 to make lr reduce')
    end
    self.step = step
    self.cur_step_ind = 1
    self.factor = factor
    self.count = 0
end

function MultiFactorScheduler:__call(num_update)
    if num_update < self.warmup_steps then
        return self:get_warmup_lr(num_update)
    end
    while self.cur_step_ind <= #self.step do
        if num_update > self.step[self.cur_step_ind] then
            self.count = self.step[self.cur_step_ind]
            self.cur_step_ind = self.cur_step_ind + 1
            self.base_lr = self.base_lr * self.factor
            logging.info("Update[%d]: Change learning rate to %0.5e",
                         num_update, self.base_lr)
        else
            return self.base_lr
        end
    end
    return self.base_lr
end

---@class mx.lr_scheduler.PolyScheduler:mx.lr_scheduler.LRScheduler
local PolyScheduler = class('mx.lr_scheduler.PolyScheduler', LRScheduler)
M.PolyScheduler = PolyScheduler

function PolyScheduler:ctor(max_update, base_lr, pwr, final_lr,
                            warmup_steps, warmup_begin_lr, warmup_mode)
    base_lr = base_lr or 0.01
    pwr = pwr or 2
    final_lr = final_lr or 0
    warmup_steps = warmup_steps or 0
    warmup_begin_lr = warmup_begin_lr or 0
    warmup_mode = warmup_mode or 'linear'
    LRScheduler.ctor(self, base_lr, warmup_steps, warmup_begin_lr, warmup_mode)
    if max_update < 1 then
        error('maximum number of updates must be strictly positive')
    end
    self.power = pwr
    self.base_lr_orig = self.base_lr
    self.max_update = max_update
    self.final_lr = final_lr
    self.max_steps = self.max_update - self.warmup_steps
end

function PolyScheduler:__call(num_update)
    if num_update < self.warmup_steps then
        return self:get_warmup_lr(num_update)
    end
    if num_update <= self.max_update then
        self.base_lr = self.final_lr + (self.base_lr_orig - self.final_lr) *
                math.pow(1 - (num_update - self.warmup_steps) / (self.max_steps), self.power)
    end
    return self.base_lr
end

---@class mx.lr_scheduler.CosineScheduler:mx.lr_scheduler.LRScheduler
local CosineScheduler = class('mx.lr_scheduler.CosineScheduler', LRScheduler)
M.CosineScheduler = CosineScheduler

function CosineScheduler:ctor(max_update, base_lr, final_lr,
                              warmup_steps, warmup_begin_lr, warmup_mode)
    base_lr = base_lr or 0.01
    final_lr = final_lr or 0
    warmup_steps = warmup_steps or 0
    warmup_begin_lr = warmup_begin_lr or 0
    warmup_mode = warmup_mode or 'linear'
    LRScheduler.ctor(self, base_lr, warmup_steps, warmup_begin_lr, warmup_mode)
    if max_update < 1 then
        error('maximum number of updates must be strictly positive')
    end
    self.base_lr_orig = base_lr
    self.max_update = max_update
    self.final_lr = final_lr
    self.max_steps = self.max_update - self.warmup_steps
end

function CosineScheduler:__call(num_update)
    if num_update < self.warmup_steps then
        return self:get_warmup_lr(num_update)
    end
    if num_update <= self.max_update then
        self.base_lr = self.final_lr + (self.base_lr_orig - self.final_lr) *
                (1 + cos(pi * (num_update - self.warmup_steps) / self.max_steps)) / 2
    end
    return self.base_lr
end

return M
