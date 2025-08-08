pub struct LogisticRegression[Scheduler]{
  mut coef_ : @lmut.Matrix[Double]
  mut intercept_ : Double
  scheduler : Scheduler
  max_iter : Int
  loss_history : Array[Double]
}

pub fn LogisticRegression::new[Scheduler : LearningRateScheduler](
  scheduler : Scheduler,
  max_iter~ : Int = 1000
) -> LogisticRegression[Scheduler]{
  return {
    coef_: @lmut.Matrix::from_2d_array([[]]),
    intercept_: 0.0,
    scheduler,
    max_iter,
    loss_history : []
  }
}

fn sigmoid(z : Double) -> Double {
  return 1.0 / (1.0 + @math.exp(-z))
}

pub fn mean(mtx : @lmut.Matrix[Double]) -> Double{
  let mut sum = 0.0
  for i in 0..<@lmut.Matrix::row(mtx){
    for j in 0..<@lmut.Matrix::col(mtx){
      sum = sum + mtx[i][j]
    }
  }
  return sum / (@lmut.Matrix::row(mtx).to_double() * @lmut.Matrix::col(mtx).to_double())
}

fn cost_function(
  y_true : @lmut.Matrix[Double], 
  y_pred : @lmut.Matrix[Double]
) -> Double {

  let cost = @lmut.Matrix::make(@lmut.Matrix::row(y_true), 1, (i, j) => y_true[i][0] * @math.log10(y_pred[i][0] + 1.0e-15) + (1 - y_true[i][0]) * @math.log10(1 - y_pred[i][0] + 1.0e-15))
  let avg = mean(cost)
  return -avg
}

fn grad_of_coef(
  x : @lmut.Matrix[Double],
  y_true : @lmut.Matrix[Double], 
  y_pred : @lmut.Matrix[Double]
) -> @lmut.Matrix[Double] {

  let rows = @lmut.Matrix::row(x).to_double()
  let x_T = @lmut.Matrix::transpose(x)
  let mul = x_T * (y_pred - y_true)
  let grad = @lmut.Matrix::scale(mul, 1.0/rows)
  return grad
}

fn grad_of_intercept(
  y_true : @lmut.Matrix[Double], 
  y_pred : @lmut.Matrix[Double]
) -> Double {
  return mean(y_pred - y_true)
}

pub fn LogisticRegression::predict_proba[Scheduler : LearningRateScheduler](
  self : LogisticRegression[Scheduler],
  x : @lmut.Matrix[Double]
) -> @lmut.Matrix[Double] {
  let z = @lmut.Matrix::add_constant(x * self.coef_, self.intercept_)
  return @lmut.Matrix::make(@lmut.Matrix::row(z), 1, (i, j) => sigmoid(z[i][j]))
}

pub fn standardization_vertical(x : @lmut.Matrix[Double]) -> @lmut.Matrix[Double] {
  let rows = @lmut.Matrix::row(x)
  let cols = @lmut.Matrix::col(x)
  let ans = x
  for j in 0..<cols{
    let mut avg = 0.0
    for i in 0..<rows{
      avg = avg + ans[i][j]
    }
    avg = avg / rows.to_double()
    let mut std = 0.0
    for i in 0..<rows{
      ans[i][j] = ans[i][j] - avg
      std = std + ans[i][j] * ans[i][j]
    }
    std = (std/rows.to_double()).pow(0.5)
    for i in 0..<rows{
      ans[i][j] = ans[i][j] / std
    }
  }
  return ans
}

pub fn LogisticRegression::fit[Scheduler : LearningRateScheduler](
  self : LogisticRegression[Scheduler],
  x : @lmut.Matrix[Double],
  y : @lmut.Matrix[Double],
) -> Unit {
  self.coef_ = @lmut.Matrix::new(@lmut.Matrix::col(x), 1, 0.0)
  for iter in 0..<self.max_iter {
    let current_lr = self.scheduler.get_lr(iter)
    // get y_pred and calculate cost
    let y_pred = self.predict_proba(x)
    let current_cost = cost_function(y, y_pred)
    self.loss_history.push(current_cost)

    // update coef and intercept
    let coef_grad = grad_of_coef(x, y, y_pred)
    self.coef_ = self.coef_ - @lmut.Matrix::scale(coef_grad, current_lr)
    self.intercept_ = self.intercept_ - current_lr * grad_of_intercept(y, y_pred)

    self.scheduler.step()

    // check
    if iter > 0 && (current_cost - self.loss_history[iter-1]).abs() <= 1.0e-6 {
      break
    }
  }
}

impl [Scheduler : LearningRateScheduler] Show for LogisticRegression[Scheduler] with to_string(self) {
  let res = StringBuilder::new()
  res.write_string("  LogisticRegression[\{self.scheduler.name_to_string()}]:\n")
  res.write_string("  Max Iterations: \{self.max_iter}\n")
  res.write_string("  Iterations:\{self.loss_history.length()}\n")
  if @lmut.Matrix::row(self.coef_) > 0 {
    res.write_string("  Coefficients: ")
    for i in 0..<@lmut.Matrix::row(self.coef_) {
      res.write_string("\{self.coef_[i][0]} ")
    }
    res.write_string("\n  Intercept: \{self.intercept_}")
  }
  return res.to_string()
}

impl [Scheduler : LearningRateScheduler] Show for LogisticRegression[Scheduler] with output(self, logger) {
  logger.write_string(self.to_string())
}


///|
test "LogisticRegression::fit" {
  // Test basic logistic regression with linearly separable data
  let scheduler = ConstantLRScheduler::new(0.1)
  let model = LogisticRegression::new(scheduler, max_iter=100)

  // Simple linearly separable data
  let x = @lmut.Matrix::from_2d_array([
    [1.0, 2.0],
    [2.0, 3.0],
    [3.0, 4.0],
    [-1.0, -2.0],
    [-2.0, -3.0],
    [-3.0, -4.0],
  ])
  let y = @lmut.Matrix::from_2d_array([[1.0], [1.0], [1.0], [0.0], [0.0], [0.0]])
  model.fit(x, y)

  // Check that coefficients are initialized and updated
  inspect(@lmut.Matrix::row(model.coef_), content="2")
  inspect(@lmut.Matrix::col(model.coef_), content="1")

  // Check that loss history is populated
  inspect(model.loss_history.length() > 0, content="true")

  inspect(model, content=(
    #|  LogisticRegression[ConstantLRScheduler]:
    #|  Max Iterations: 100
    #|  Iterations:100
    #|  Coefficients: 0.7427675147267799 1.2394112747625419 
    #|  Intercept: -2.3113383313770152e-17
  ))

  // Check that predictions are reasonable (should be close to 0 or 1)
  let predictions = model.predict_proba(x)
  let first_pred = predictions[0][0]
  let last_pred = predictions[5][0]
  inspect(first_pred > 0.5, content="true") // Should predict class 1
  inspect(last_pred < 0.5, content="true") // Should predict class 0
}
///|
test "LogisticRegression::fit/convergence_with_different_schedulers" {
  // Test convergence with exponential decay scheduler
  let exp_scheduler = ExponentialLRScheduler::new(0.1, decay_rate=0.9)
  let exp_model = LogisticRegression::new(exp_scheduler, max_iter=50)
  let x = @lmut.Matrix::from_2d_array([
    [1.0, 2.0],
    [2.0, 3.0],
    [3.0, 4.0],
    [-1.0, -2.0],
    [-2.0, -3.0],
    [-3.0, -4.0],
  ])
  let y = @lmut.Matrix::from_2d_array([[1.0], [1.0], [1.0], [0.0], [0.0], [0.0]])
  exp_model.fit(x, y)

  // Check that coefficients were updated from initial zeros
  let coef_sum = exp_model.coef_[0][0] + exp_model.coef_[1][0]
  inspect(coef_sum != 0.0, content="true")

  // Test with step scheduler
  let step_scheduler = StepLRScheduler::new(0.05, 10, gamma=0.5)
  let step_model = LogisticRegression::new(step_scheduler, max_iter=30)
  step_model.fit(x, y)

  // Check that loss history is populated
  inspect(step_model.loss_history.length() > 0, content="true")
  inspect(step_model.loss_history.length() <= 30, content="true")

  inspect(exp_model, content=(
    #|  LogisticRegression[ExponentialLRScheduler]:
    #|  Max Iterations: 50
    #|  Iterations:50
    #|  Coefficients: 0.368216139672241 0.5758466959546671 
    #|  Intercept: 1.752918514997948e-18
  ))
  inspect(step_model, content=(
    #|  LogisticRegression[StepLRScheduler]:
    #|  Max Iterations: 30
    #|  Iterations:30
    #|  Coefficients: 0.3445252520698235 0.5378845737888959 
    #|  Intercept: -3.7151994443836625e-18
  ))
}

///|
test "LogisticRegression::fit/perfect_separation" {
  // Test with perfectly linearly separable data
  let scheduler = ConstantLRScheduler::new(1.0)
  let model = LogisticRegression::new(scheduler, max_iter=200)

  // Perfectly separable data points
  let x = @lmut.Matrix::from_2d_array([
    [10.0, 10.0],
    [9.0, 9.0],
    [8.0, 8.0],
    [-10.0, -10.0],
    [-9.0, -9.0],
    [-8.0, -8.0],
  ])
  let y = @lmut.Matrix::from_2d_array([[1.0], [1.0], [1.0], [0.0], [0.0], [0.0]])
  model.fit(x, y)

  // Check that predictions are very confident (close to 0 or 1)
  let predictions = model.predict_proba(x)

  // First three should be close to 1
  inspect(predictions[0][0] > 0.9, content="true")
  inspect(predictions[1][0] > 0.9, content="true")
  inspect(predictions[2][0] > 0.9, content="true")

  // Last three should be close to 0
  inspect(predictions[3][0] < 0.1, content="true")
  inspect(predictions[4][0] < 0.1, content="true")
  inspect(predictions[5][0] < 0.1, content="true")

  inspect(model, content=(
    #|  LogisticRegression[ConstantLRScheduler]:
    #|  Max Iterations: 200
    #|  Iterations:3
    #|  Coefficients: 4.5 4.5 
    #|  Intercept: -1.7936167365604073e-32
  ))
}

///|
test "LogisticRegression::fit/early_convergence" {
  // Test early convergence with data that should converge quickly
  let scheduler = ConstantLRScheduler::new(0.01)
  let model = LogisticRegression::new(scheduler, max_iter=1000)

  // Simple binary classification with clear separation
  let x = @lmut.Matrix::from_2d_array([
    [5.0],
    [4.0],
    [3.0],
    [-3.0],
    [-4.0],
    [-5.0],
  ])
  let y = @lmut.Matrix::from_2d_array([[1.0], [1.0], [1.0], [0.0], [0.0], [0.0]])
  model.fit(x, y)

  // Should converge before max_iter due to early stopping condition
  inspect(model.loss_history.length() < 1000, content="true")
  inspect(model.loss_history.length() > 1, content="true")

  // Check that coefficients are reasonable (should be positive for this data)
  inspect(model.coef_[0][0] > 0.0, content="true")

  // Verify that scheduler's step was called
  inspect(model.scheduler.current_step > 0, content="true")

  inspect(model, content=(
    #|  LogisticRegression[ConstantLRScheduler]:
    #|  Max Iterations: 1000
    #|  Coefficients: 0.03916720065708247 
    #|  Intercept: 0
    #|
  ))
}

test {
  let x = @lmut.Matrix::from_2d_array([
    [1.0],
    [2.0],
    [3.0],
    [4.0],
    [5.0],
    [6.0]]
  )
  let y = @lmut.Matrix::from_2d_array([
    [1.0],
    [1],
    [1],
    [0],
    [0],
    [0]]
  )
  let exp = ExponentialLRScheduler::new(0.1)
  let exp_mod = LogisticRegression::new(exp, max_iter=1000)
  exp_mod.fit(x, y)
  inspect(exp_mod, content=(
    #|  LogisticRegression[ExponentialLRScheduler]:
    #|  Max Iterations: 1000
    #|  Iterations:139
    #|  Coefficients: -0.2699847165195138 
    #|  Intercept: 0.2760366281580831
  ))

  let ntr_exp = NaturalExponentialLRScheduler::new(0.1)
  let ntr_exp_mod = LogisticRegression::new(ntr_exp, max_iter=1000)
  ntr_exp_mod.fit(x, y)
  inspect(ntr_exp_mod, content=(
    #|  LogisticRegression[NaturalExponentialLRScheduler]:
    #|  Max Iterations: 1000
    #|  Iterations:73
    #|  Coefficients: -0.22951960861549717 
    #|  Intercept: 0.12770564568074996
  ))

  let stp = StepLRScheduler::new(0.1, 100, gamma = 0.1)
  let stp_mod = LogisticRegression::new(stp, max_iter=1000)
  stp_mod.fit(x, y)
  inspect(stp_mod, content=(
    #|  LogisticRegression[StepLRScheduler]:
    #|  Max Iterations: 1000
    #|  Iterations:302
    #|  Coefficients: -0.5653231180870748 
    #|  Intercept: 1.4477910564965517
  ))

  let cos = CosineAnnealingLRScheduler::new(0.1, 50)
  let cos_mod = LogisticRegression::new(cos, max_iter=1000)
  cos_mod.fit(x, y)
  inspect(cos_mod, content=(
    #|  LogisticRegression[CosineAnnealingLRScheduler]:
    #|  Max Iterations: 1000
    #|  Iterations:101
    #|  Coefficients: -0.38500871095199085 
    #|  Intercept: 0.7382048203161405
  ))

  let fix = ConstantLRScheduler::new(0.1)
  let fix_mod = LogisticRegression::new(fix, max_iter=1000)
  fix_mod.fit(x, y)
  inspect(fix_mod, content=(
    #|  LogisticRegression[ConstantLRScheduler]:
    #|  Max Iterations: 1000
    #|  Iterations:1000
    #|  Coefficients: -1.7036105175157013 
    #|  Intercept: 5.690294412813251
  ))

  let lnr = LinearLRScheduler::new(0.1, 1000)
  let lnr_mod = LogisticRegression::new(lnr, max_iter=1000)
  lnr_mod.fit(x, y)
  inspect(lnr_mod, content=(
    #|  LogisticRegression[LinearLRScheduler]:
    #|  Max Iterations: 1000
    #|  Iterations:991
    #|  Coefficients: -1.24720388463649 
    #|  Intercept: 4.0221957607190575
  ))

  let inv_time = InverseTimeDecayLRScheduler::new(0.1)
  let inv_mod = LogisticRegression::new(inv_time, max_iter=1000)
  inv_mod.fit(x, y)
  inspect(inv_mod, content=(
    #|  LogisticRegression[InverseTimeDecayLRScheduler]:
    #|  Max Iterations: 1000
    #|  Iterations:1000
    #|  Coefficients: -1.0593210254098036 
    #|  Intercept: 3.3251163304552795
  ))

  let plnm = PolynomialDecayLRScheduler::new(0.1, 1000)
  let plnm_mod = LogisticRegression::new(plnm, max_iter=1000)
  plnm_mod.fit(x, y)
  inspect(plnm_mod, content=(
    #|  LogisticRegression[PolynomialDecayLRScheduler]:
    #|  Max Iterations: 1000
    #|  Iterations:929
    #|  Coefficients: -1.022202178339939 
    #|  Intercept: 3.1865004616250583
  ))
}