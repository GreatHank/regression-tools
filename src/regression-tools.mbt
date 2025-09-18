///|
pub struct LinearRegressionResult {
  coef : @lmut.Matrix[Double]
  intercept : Double
  fitted : @lmut.Matrix[Double]
  residuals : @lmut.Matrix[Double]
  r_squared : Double
  stderr : @lmut.Matrix[Double]
  intercept_stderr : Double
}

///|
struct LinearRegression {
  fit_intercept : Bool
  mut coef_ : @lmut.Matrix[Double]
  mut intercept_ : Double
}

///|
test "mean" {
  let x = @lmut.Matrix::from_2d_array([
    [1.0, 1.0],
    [2.0, 3.0],
    [3.0, 4.0],
    [2.0, 3.0],
  ])
  inspect(mean_vertical(x), content="|2, 2.75|")
}

///|
pub fn LinearRegression::new(fit_intercept~ : Bool = true) -> LinearRegression {
  return {
    fit_intercept,
    coef_: @lmut.Matrix::from_2d_array([[]]),
    intercept_: 0.0,
  }
}

///|
fn clone_matrix(matrix : @lmut.Matrix[Double]) -> @lmut.Matrix[Double] {
  let rows = @lmut.Matrix::row(matrix)
  let cols = @lmut.Matrix::col(matrix)
  @lmut.Matrix::make(rows, cols, fn(i, j) {
    matrix[i][j]
  })
}

///|
fn build_design_matrix(
  x : @lmut.Matrix[Double],
  fit_intercept : Bool,
) -> @lmut.Matrix[Double] {
  if !fit_intercept {
    return x
  }
  let ones = @lmut.Matrix::new(@lmut.Matrix::row(x), 1, 1.0)
  @lmut.Matrix::horizontal_combine(ones, x)
}

///|
struct LinearRegressionSolve {
  coef : @lmut.Matrix[Double]
  intercept : Double
  fitted : @lmut.Matrix[Double]
  residuals : @lmut.Matrix[Double]
  xtx_inv : @lmut.Matrix[Double]
  rss : Double
  parameter_count : Int
}

///|
fn solve_linear_regression(
  x : @lmut.Matrix[Double],
  y : @lmut.Matrix[Double],
  fit_intercept : Bool,
) -> Option[LinearRegressionSolve] {
  let design = build_design_matrix(x, fit_intercept)
  let design_t = @lmut.Matrix::transpose(design)
  let xtx = design_t * design
  match @lmut.Matrix::inverse(xtx) {
    None => None
    Some(xtx_inv) => {
      let theta = xtx_inv * design_t * y
      let predictions = design * theta
      let residuals = y - predictions
      let parameter_count = @lmut.Matrix::row(theta)
      let slope_count = if fit_intercept {
        parameter_count - 1
      } else {
        parameter_count
      }
      let slopes = @lmut.Matrix::make(slope_count, 1, fn(i, _) {
        let idx = if fit_intercept { i + 1 } else { i }
        theta[idx][0]
      })
      let intercept = if fit_intercept { theta[0][0] } else { 0.0 }
      let rss = residual_sum_of_square(y, predictions)
      Some({
        coef: slopes,
        intercept,
        fitted: predictions,
        residuals,
        xtx_inv,
        rss,
        parameter_count,
      })
    }
  }
}

///|
fn compute_standard_errors(
  solve : LinearRegressionSolve,
  fit_intercept : Bool,
  sample_count : Int,
) -> (Double, @lmut.Matrix[Double]) {
  let dof = sample_count - solve.parameter_count
  if dof <= 0 {
    let zero = @lmut.Matrix::new(@lmut.Matrix::row(solve.coef), 1, 0.0)
    return (0.0, zero)
  }
  let variance = solve.rss / dof.to_double()
  let slopes = @lmut.Matrix::row(solve.coef)
  let stderr = @lmut.Matrix::make(slopes, 1, fn(i, _) {
    let diag_index = if fit_intercept { i + 1 } else { i }
    let value = variance * solve.xtx_inv[diag_index][diag_index]
    if value <= 0.0 { 0.0 } else { value.pow(0.5) }
  })
  let intercept_std = if fit_intercept {
    let value = variance * solve.xtx_inv[0][0]
    if value <= 0.0 { 0.0 } else { value.pow(0.5) }
  } else {
    0.0
  }
  (intercept_std, stderr)
}

///|
fn build_result(
  solve : LinearRegressionSolve,
  x : @lmut.Matrix[Double],
  y : @lmut.Matrix[Double],
  fit_intercept : Bool,
) -> LinearRegressionResult {
  let sample_count = @lmut.Matrix::row(x)
  let (intercept_stderr, stderr) =
    compute_standard_errors(solve, fit_intercept, sample_count)
  let r_squared = r2_score(x, y, solve.coef, solve.intercept)
  return {
    coef: clone_matrix(solve.coef),
    intercept: solve.intercept,
    fitted: clone_matrix(solve.fitted),
    residuals: clone_matrix(solve.residuals),
    r_squared,
    stderr,
    intercept_stderr,
  }
}

///|
pub fn LinearRegression::fit(
  self : LinearRegression,
  x : @lmut.Matrix[Double],
  y : @lmut.Matrix[Double],
) -> LinearRegression {
  match solve_linear_regression(x, y, self.fit_intercept) {
    None => abort("LinearRegression::fit: design matrix is singular")
    Some(solution) => {
      self.coef_ = clone_matrix(solution.coef)
      self.intercept_ = solution.intercept
      return self
    }
  }
}

///|
pub fn LinearRegression::predict(
  self : LinearRegression,
  x : @lmut.Matrix[Double],
) -> @lmut.Matrix[Double] {
  @lmut.Matrix::add_constant(x * self.coef_, self.intercept_)
}

///|
pub fn LinearRegression::score(
  self : LinearRegression,
  x : @lmut.Matrix[Double],
  y : @lmut.Matrix[Double],
) -> Double {
  r2_score(x, y, self.coef_, self.intercept_)
}

///|
impl Show for LinearRegression with output(self, logger) {
  logger.write_string("Linear regression:\n")
  logger.write_string("y = \{self.intercept_}")
  let rows = self.coef_.row()
  for i in 0..<rows {
    logger.write_string(" + \{self.coef_[i][0]} * x _ \{i + 1}")
  }
  logger.write_string(" + ϵ")
}

///|
pub fn linear_regression(
  x : @lmut.Matrix[Double],
  y : @lmut.Matrix[Double],
  fit_intercept~ : Bool = true,
) -> LinearRegressionResult {
  match solve_linear_regression(x, y, fit_intercept) {
    None => abort("linear_regression: design matrix is singular")
    Some(solution) => build_result(solution, x, y, fit_intercept)
  }
}

///|
test "LinearRegression::fit_estimates_coefficients" {
  let model = LinearRegression::new()
  let x = @lmut.Matrix::from_2d_array([
    [1.0, 1.0],
    [2.0, 3.0],
    [3.0, 4.0],
    [2.0, 3.0],
  ])
  let y = @lmut.Matrix::from_2d_array([[6.0], [14.0], [19.0], [15.0]])
  let ans = model.fit(x, y)
  inspect(
    ans.coef_,
    content=(
      #||0.500000000000103|
      #||3.999999999999954|
    ),
  )
  inspect(ans.intercept_, content="1.499999999999968")
  inspect(
    ans,
    content=(
      #|Linear regression:
      #|y = 1.499999999999968 + 0.500000000000103 * x _ 1 + 3.999999999999954 * x _ 2 + ϵ
    ),
  )
}

///|
test "LinearRegression::predict_and_score" {
  let model = LinearRegression::new()
  let x = @lmut.Matrix::from_2d_array([
    [1.0],
    [2.0],
    [3.0],
    [4.0],
  ])
  let y = @lmut.Matrix::from_2d_array([[3.0], [5.0], [7.0], [9.0]])
  let ans = model.fit(x, y)
  let predictions = ans.predict(x)
  inspect(
    predictions,
    content=(
      #||2.999999999999992|
      #||4.999999999999993|
      #||6.999999999999994|
      #||8.999999999999995|

    ),
  )
  inspect(ans.score(x, y), content="1")
}

///|
test "linear_regression_function_interface" {
  let x = @lmut.Matrix::from_2d_array([
    [1.0, 2.0],
    [2.0, 1.0],
    [3.0, 0.0],
    [4.0, 1.0],
  ])
  let y = @lmut.Matrix::from_2d_array([[5.0], [6.0], [7.0], [9.0]])
  let result = linear_regression(x, y)
  inspect(@lmut.Matrix::row(result.coef), content="2")
  inspect(result.r_squared > 0.0, content="true")
  inspect(result.stderr[0][0] >= 0.0, content="true")
}
