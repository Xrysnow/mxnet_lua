--
local logging = require('python.logging')

local M = {}

---@class mx.misc.LearningRateScheduler
local LearningRateScheduler = class('LearningRateScheduler')
M.LearningRateScheduler = LearningRateScheduler

function LearningRateScheduler:ctor()
    self.base_lr = 0.01
end

function LearningRateScheduler:__call(iteration)
    error('must override this')
end

---@class mx.misc.FactorScheduler:mx.misc.LearningRateScheduler
local FactorScheduler = class('FactorScheduler', LearningRateScheduler)
M.FactorScheduler = FactorScheduler

function FactorScheduler:ctor(step, factor)
    factor = factor or 0.1
    self.super.ctor(self)
    if step < 1 then
        error('Schedule step must be greater or equal than 1 round')
    end
    if factor >= 1.0 then
        error('Factor must be less than 1 to make lr reduce')
    end
    self.step = step
    self.factor = factor
    self.old_lr = self.base_lr
    self.init = false
end

function FactorScheduler:__call(iteration)
    if not self.init then
        self.init = true
        self.old_lr = self.base_lr
    end
    local lr = self.base_lr * math.pow(self.factor, math.floor(iteration / self.step))
    if lr ~= self.old_lr then
        self.old_lr = lr
        logging.info("At Iteration [%d]: Swith to new learning rate %.5f",
                     iteration, lr)
    end
    return lr
end

return M
