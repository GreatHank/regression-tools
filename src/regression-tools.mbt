///|
struct LinearRegression {
  mut coef_ : @lmut.Matrix[Double]
  mut intercept_ : Double
}

///|
pub fn LinearRegression::new() -> LinearRegression {
  return { coef_: @lmut.Matrix::from_2d_array([[]]), intercept_: 0.0 }
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
      let rows = @lmut.Matrix::row(coeff)
      self.coef_ = @lmut.Matrix::make(rows - 1, 1, fn(i, j) { coeff[i + 1][j] })
      self.intercept_ = coeff[0][0]
    }
    None => abort("Matrix is not invertible")
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
  inspect(lr.coef_, content=(
    #||0.500000000000103|
    #||3.999999999999954|
  ))
  inspect(lr.intercept_, content="1.499999999999968")
}

test "LinearRegression::fit_simple" {
  let lr = LinearRegression::new()
  // 更简单的测试数据：y = 2x + 1
  let x = @lmut.Matrix::from_2d_array([
    [1.0, 1.0],
    [2.0, 1.0],
    [1.0, 2.0],
    [2.0, 2.0],
  ])
  let y = @lmut.Matrix::from_2d_array([[6.0], [8.0], [9.0], [11.0]])
  lr.fit(x, y)
  
  inspect(lr.coef_, content=(
    #||2.0000000000000027|
    #||3.0000000000000027|
  ))
  inspect(lr.intercept_, content="0.9999999999999893")
}
