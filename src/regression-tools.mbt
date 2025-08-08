///|
struct LinearRegression {
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
pub fn LinearRegression::new() -> LinearRegression {
  return { coef_: @lmut.Matrix::from_2d_array([[]]), intercept_: 0.0 }
}

///|
impl Show for LinearRegression with output(self, logger) {
  logger.write_string("Linear regression:\n")
  let rows = self.coef_.row()
  logger.write_string("y = \{self.intercept_}")
  for i in 1..<rows {
    logger.write_string(" + \{self.coef_[i][0]} * x _ \{i}")
  }
  logger.write_string(" + ϵ")
}

///|
pub fn fit(
  self : LinearRegression,
  x : @lmut.Matrix[Double],
  y : @lmut.Matrix[Double],
) -> Unit {
  let one = @lmut.Matrix::new(@lmut.Matrix::row(x), 1, 1.0)
  let x_combine = @lmut.Matrix::horizontal_combine(one, x)
  let x_trans = @lmut.Matrix::transpose(x_combine)
  let xtx = x_trans * x_combine
  match @lmut.Matrix::inverse(xtx) {
    Some(xtx_inv) => {
      let coeff = xtx_inv * x_trans * y
      self.coef_ = coeff
      self.intercept_ = coeff[0][0]
    }
    None => abort("Unable to perform Linear Regression method.")
  }
  return
}

///|
test "LinearRegression::fit" {
  let lr = LinearRegression::new()
  // Simple linear regression: y = 2x1 + 3x2 + 1
  let x = @lmut.Matrix::from_2d_array([
    [1.0, 1.0],
    [2.0, 3.0],
    [3.0, 4.0],
    [2.0, 3.0],
  ])
  let y = @lmut.Matrix::from_2d_array([[6.0], [14.0], [19.0], [15.0]])
  lr.fit(x, y)
  inspect(
    lr.coef_,
    content=(
      #||1.499999999999968|
      #||0.500000000000103|
      #||3.999999999999954|
    ),
  )
  inspect(lr.intercept_, content="1.499999999999968")
  inspect(
    lr,
    content=(
      #|Linear regression:
      #|y = 1.499999999999968 + 0.500000000000103 * x _ 1 + 3.999999999999954 * x _ 2 + ϵ
    ),
  )
}

///|
test "LinearRegression:" {
  let lr = LinearRegression::new()
  // Simple linear regression: y = 2x1 + 3x2 + 1
  let x = @lmut.Matrix::from_2d_array([[1.0], [1.0], [1.0]])
  let y = @lmut.Matrix::from_2d_array([[6.0], [14.0], [19.0]])
  lr.fit(x, y)
  inspect(lr.coef_, content="||")
  inspect(lr.intercept_, content="0")
  inspect(
    lr,
    content=(
      #|Linear regression:
      #|y = 0 + ϵ
    ),
  )
}

///|
test "LinearRegression::fit_simple" {
  let lr = LinearRegression::new()
  // 更简单的测试数据：y = 2x + 1
  let x = @lmut.Matrix::from_2d_array([
    [1.0, 1.0],
    [2.0, 1.0],
    [1.0, 2.0],
    [2.0, 2.0],
  ])
  let y = @lmut.Matrix::from_2d_array([[5.0], [8.0], [9.0], [11.0]])
  lr.fit(x, y)
  let residual = get_residuals(
    y,
    lr_get_predict_data(x, lr.coef_, lr.intercept_),
  )
  inspect(
    lr.coef_,
    content=(
      #||-0.7500000000000142|
      #||2.500000000000003|
      #||3.5000000000000036|
    ),
  )
  inspect(lr.intercept_, content="-0.7500000000000142")
  inspect(
    lr,
    content=(
      #|Linear regression:
      #|y = -0.7500000000000142 + 2.500000000000003 * x _ 1 + 3.5000000000000036 * x _ 2 + ϵ
    ),
  )
  inspect(
    residual,
    content=(
      #||4.000000000000025|
      #||7.750000000000039|
      #||5.500000000000022|
      #||8.250000000000036|
    ),
  )
}
