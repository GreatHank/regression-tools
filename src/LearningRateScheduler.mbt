///|
trait LearningRateScheduler {
  get_lr(Self, Int) -> Double // 根据迭代次数返回学习率
  step(Self) -> Unit // 更新内部状态
  name_to_string(Self) -> String
}

//指数衰减

///|
struct ExponentialLRScheduler {
  initial_lr : Double
  decay_rate : Double
  mut current_step : Int
}

///|
pub fn ExponentialLRScheduler::new(
  initial_lr : Double,
  decay_rate~ : Double = 0.95,
) -> ExponentialLRScheduler {
  { initial_lr, decay_rate, current_step: 0 }
}

///|
impl LearningRateScheduler for ExponentialLRScheduler with get_lr(self, step) {
  self.initial_lr * @math.pow(self.decay_rate, step.to_double())
}

///|
impl LearningRateScheduler for ExponentialLRScheduler with step(self) {
  self.current_step = self.current_step + 1
}

///|
impl LearningRateScheduler for ExponentialLRScheduler with name_to_string(_) {
  "ExponentialLRScheduler"
}

//e指数衰减

///|
struct NaturalExponentialLRScheduler {
  initial_lr : Double
  decay_rate : Double
  mut current_step : Int
}

///|
pub fn NaturalExponentialLRScheduler::new(
  initial_lr : Double,
  decay_rate~ : Double = 0.1,
) -> NaturalExponentialLRScheduler {
  { initial_lr, decay_rate, current_step: 0 }
}

///|
impl LearningRateScheduler for NaturalExponentialLRScheduler with get_lr(
  self,
  step,
) {
  self.initial_lr * @math.exp(-self.decay_rate * step.to_double())
}

///|
impl LearningRateScheduler for NaturalExponentialLRScheduler with step(self) {
  self.current_step = self.current_step + 1
}

///|
impl LearningRateScheduler for NaturalExponentialLRScheduler with name_to_string(
  _,
) {
  "NaturalExponentialLRScheduler"
}

//阶梯衰减

///|
struct StepLRScheduler {
  initial_lr : Double
  step_size : Int
  gamma : Double
  mut current_step : Int
}

///|
pub fn StepLRScheduler::new(
  initial_lr : Double,
  step_size : Int,
  gamma~ : Double = 0.1,
) -> StepLRScheduler {
  { initial_lr, step_size, gamma, current_step: 0 }
}

///|
impl LearningRateScheduler for StepLRScheduler with get_lr(self, step) {
  let decay_times = step / self.step_size
  self.initial_lr * @math.pow(self.gamma, decay_times.to_double())
}

///|
impl LearningRateScheduler for StepLRScheduler with step(self) {
  self.current_step = self.current_step + 1
}

///|
impl LearningRateScheduler for StepLRScheduler with name_to_string(_) {
  "StepLRScheduler"
}

//余弦退火

///|
struct CosineAnnealingLRScheduler {
  initial_lr : Double
  t_max : Int
  min_lr : Double
  mut current_step : Int
}

///|
pub fn CosineAnnealingLRScheduler::new(
  initial_lr : Double,
  t_max : Int,
  min_lr~ : Double = 0.0,
) -> CosineAnnealingLRScheduler {
  { initial_lr, t_max, min_lr, current_step: 0 }
}

///|
impl LearningRateScheduler for CosineAnnealingLRScheduler with get_lr(
  self,
  step,
) {
  let progress = (step % self.t_max).to_double() / self.t_max.to_double()
  let cosine_factor = (1.0 + @math.cos(@math.PI * progress)) / 2.0
  self.min_lr + (self.initial_lr - self.min_lr) * cosine_factor
}

///|
impl LearningRateScheduler for CosineAnnealingLRScheduler with step(self) {
  self.current_step = self.current_step + 1
}

///|
impl LearningRateScheduler for CosineAnnealingLRScheduler with name_to_string(_) {
  "CosineAnnealingLRScheduler"
}

//固定学习率

///|
struct ConstantLRScheduler {
  learning_rate : Double
  mut current_step : Int
}

///|
pub fn ConstantLRScheduler::new(learning_rate : Double) -> ConstantLRScheduler {
  { learning_rate, current_step: 0 }
}

///|
impl LearningRateScheduler for ConstantLRScheduler with get_lr(self, _) {
  self.learning_rate // 始终返回固定的学习率
}

///|
impl LearningRateScheduler for ConstantLRScheduler with step(self) {
  self.current_step = self.current_step + 1
}

///|
impl LearningRateScheduler for ConstantLRScheduler with name_to_string(_) {
  "ConstantLRScheduler"
}

//线性衰减

///|
struct LinearLRScheduler {
  initial_lr : Double
  final_lr : Double
  total_steps : Int
  mut current_step : Int
}

///|
pub fn LinearLRScheduler::new(
  initial_lr : Double,
  final_lr~ : Double = 0.0,
  total_steps : Int,
) -> LinearLRScheduler {
  { initial_lr, final_lr, total_steps, current_step: 0 }
}

///|
impl LearningRateScheduler for LinearLRScheduler with get_lr(self, step) {
  if step >= self.total_steps {
    self.final_lr
  } else {
    let progress = step.to_double() / self.total_steps.to_double()
    self.initial_lr + (self.final_lr - self.initial_lr) * progress
  }
}

///|
impl LearningRateScheduler for LinearLRScheduler with step(self) {
  self.current_step = self.current_step + 1
}

///|
impl LearningRateScheduler for LinearLRScheduler with name_to_string(_) {
  "LinearLRScheduler"
}

//逆向时间衰减

///|
struct InverseTimeDecayLRScheduler {
  initial_lr : Double
  decay_steps : Int
  decay_rate : Double
  staircase : Bool
  mut current_step : Int
}

///|
pub fn InverseTimeDecayLRScheduler::new(
  initial_lr : Double,
  decay_steps~ : Int = 100,
  decay_rate~ : Double = 0.5,
  staircase~ : Bool = false,
) -> InverseTimeDecayLRScheduler {
  { initial_lr, decay_steps, decay_rate, staircase, current_step: 0 }
}

///|
impl LearningRateScheduler for InverseTimeDecayLRScheduler with get_lr(
  self,
  step,
) {
  let time_factor = if self.staircase {
    (step / self.decay_steps).to_double()
  } else {
    step.to_double() / self.decay_steps.to_double()
  }
  self.initial_lr / (1.0 + self.decay_rate * time_factor)
}

///|
impl LearningRateScheduler for InverseTimeDecayLRScheduler with step(self) {
  self.current_step = self.current_step + 1
}

///|
impl LearningRateScheduler for InverseTimeDecayLRScheduler with name_to_string(
  _,
) {
  "InverseTimeDecayLRScheduler"
}

//多项式衰减

///|
struct PolynomialDecayLRScheduler {
  initial_lr : Double
  final_lr : Double
  decay_steps : Int
  power : Double
  cycle : Bool
  mut current_step : Int
}

///|
pub fn PolynomialDecayLRScheduler::new(
  initial_lr : Double,
  decay_steps : Int,
  final_lr~ : Double = 0.0001,
  power~ : Double = 2.0,
  cycle~ : Bool = false,
) -> PolynomialDecayLRScheduler {
  { initial_lr, final_lr, decay_steps, power, cycle, current_step: 0 }
}

///|
impl LearningRateScheduler for PolynomialDecayLRScheduler with get_lr(
  self,
  step,
) {
  let effective_step = if self.cycle {
    // 循环模式：重复衰减过程
    step % self.decay_steps
  } else {
    // 非循环模式：达到decay_steps后保持final_lr
    step.min(self.decay_steps)
  }
  if effective_step >= self.decay_steps {
    self.final_lr
  } else {
    // 多项式衰减公式：lr = (initial_lr - final_lr) * (1 - step/decay_steps)^power + final_lr
    let progress = effective_step.to_double() / self.decay_steps.to_double()
    let decay_factor = @math.pow(1.0 - progress, self.power)
    (self.initial_lr - self.final_lr) * decay_factor + self.final_lr
  }
}

///|
impl LearningRateScheduler for PolynomialDecayLRScheduler with step(self) {
  self.current_step = self.current_step + 1
}

///|
impl LearningRateScheduler for PolynomialDecayLRScheduler with name_to_string(_) {
  "PolynomialDecayLRScheduler"
}
