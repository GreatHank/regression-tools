///|
pub fn mean_squared_error(
  real_data : @lmut.Matrix[Double],
  get_data : @lmut.Matrix[Double],
) -> Double {
  let n = @lmut.Matrix::row(real_data).to_double()
  let sum = residual_sum_of_square(real_data, get_data)
  sum / n
}

///|
pub fn residual_sum_of_square(
  real_data : @lmut.Matrix[Double],
  get_data : @lmut.Matrix[Double],
) -> Double {
  let residuals = real_data - get_data
  let pow_res = @lmut.Matrix::map(residuals, fn(x) { x * x })
  let sum = sum_mtr(pow_res)
  sum
}

///|
pub fn m_mean_squared_error(
  real_data : @lmut.Matrix[Double],
  get_data : @lmut.Matrix[Double],
) -> Double {
  let number = mean_squared_error(real_data, get_data)
  number.pow(0.5)
}

///|
pub fn mean_vertical(matrix : @lmut.Matrix[Double]) -> @lmut.Matrix[Double] {
  let row = @lmut.Matrix::row(matrix)
  let col = @lmut.Matrix::col(matrix)
  @lmut.Matrix::make(1, col, fn(_, j) {
    let mut sum = 0.0
    for k in 0..<row {
      sum = sum + matrix[k][j]
    }
    sum / row.to_double()
  })
}

///|
pub fn get_residuals(
  real_data : @lmut.Matrix[Double],
  get_data : @lmut.Matrix[Double],
) -> @lmut.Matrix[Double] {
  real_data - get_data
}

///|
pub fn mean_absolute_error(
  real_data : @lmut.Matrix[Double],
  get_data : @lmut.Matrix[Double],
) -> Double {
  let residuals = real_data - get_data
  let n = @lmut.Matrix::row(residuals).to_double()
  let abs_res = @lmut.Matrix::map(residuals, fn(x) { x.abs() })
  let sum = sum_mtr(abs_res)
  sum / n
}

///|
pub fn sum_mtr(data : @lmut.Matrix[Double]) -> Double {
  let mut sum = 0.0
  for i in data {
    sum = sum + i
  }
  sum
}

///|
pub fn total_sum_of_square(data : @lmut.Matrix[Double]) -> Double {
  let row = @lmut.Matrix::row(data).to_double()
  let sum = sum_mtr(data)
  let average = sum / row
  let mtr = @lmut.Matrix::map(data, fn(x) { (x - average).pow(2) })
  let total = sum_mtr(mtr)
  total / row
}

///|
pub fn r2_score(
  x : @lmut.Matrix[Double],
  y : @lmut.Matrix[Double],
  coef : @lmut.Matrix[Double],
  intercept : Double,
) -> Double {
  let mse = mean_squared_error(y, lr_get_predict_data(x, coef, intercept))
  let avr = total_sum_of_square(y)
  1 - mse / avr
}

///|
pub fn lr_get_predict_data(
  x : @lmut.Matrix[Double],
  coef : @lmut.Matrix[Double],
  intercept : Double,
) -> @lmut.Matrix[Double] {
  @lmut.Matrix::add_constant(x * coef, intercept)
}

///|
pub fn stantard_error(
  x : @lmut.Matrix[Double],
  y : @lmut.Matrix[Double],
  coef : @lmut.Matrix[Double],
  intercept : Double,
) -> @lmut.Matrix[Double] {
  let rows = @lmut.Matrix::row(x)
  let cols = @lmut.Matrix::col(x)
  let dow = rows - cols - 1
  if dow <= 0 {
    return @lmut.Matrix::new(@lmut.Matrix::row(coef), 1, 0.0)
  }
  let predictions = lr_get_predict_data(x, coef, intercept)
  let variance = residual_sum_of_square(y, predictions) / dow.to_double()
  let ones = @lmut.Matrix::new(rows, 1, 1.0)
  let design = @lmut.Matrix::horizontal_combine(ones, x)
  let xtx = @lmut.Matrix::transpose(design) * design
  match @lmut.Matrix::inverse(xtx) {
    None => abort("Unable to perform Linear Regression method.")
    Some(inv) => @lmut.Matrix::make(@lmut.Matrix::row(coef), 1, fn(i, _) {
      let diag_index = i + 1
      let value = variance * inv[diag_index][diag_index]
      if value <= 0.0 { 0.0 } else { value.pow(0.5) }
    })
  }
}

///|
pub fn t_statistic(
  x : @lmut.Matrix[Double],
  y : @lmut.Matrix[Double],
  coef : @lmut.Matrix[Double],
  intercept : Double,
) -> @lmut.Matrix[Double] {
  let se = stantard_error(x, y, coef, intercept)
  @lmut.Matrix::make(@lmut.Matrix::row(coef), 1, fn(i, _) {
    coef[i][0] / se[i][0]
  })
}

test "metrics_basic_mse_mae_rmse_rss_sum_tss" {
  let y = @lmut.Matrix::from_2d_array([[3.0],[5],[7]])
  let yhat = @lmut.Matrix::from_2d_array([[2.0],[6],[8]])

  assert_eq(residual_sum_of_square(y, yhat), 3.0)
  assert_eq(mean_squared_error(y, yhat), 1.0)
  assert_eq(m_mean_squared_error(y, yhat), 1.0)
  assert_eq(mean_absolute_error(y, yhat), 1.0)

  assert_eq(sum_mtr(y), 15.0)

  assert_eq(total_sum_of_square(y), 8.0 / 3.0)
}


test "mean_vertical_and_residuals" {
  let m = @lmut.Matrix::from_2d_array([[1.0,2],[3,4],[5,6]])
  let mv = mean_vertical(m)
  assert_eq(@lmut.Matrix::row(mv), 1)
  assert_eq(@lmut.Matrix::col(mv), 2)
  assert_eq(mv[0][0], 3.0)
  assert_eq(mv[0][1], 4.0)

  let y = @lmut.Matrix::from_2d_array([[3.0],[5],[7]])
  let yhat = @lmut.Matrix::from_2d_array([[2.0],[6],[8]])
  let r = get_residuals(y, yhat)
  assert_eq(r[0][0], 1.0)
  assert_eq(r[1][0], -1.0)
  assert_eq(r[2][0], -1.0)
}


test "lr_get_predict_data_and_r2" {
  // Simple 1D regression: y = 2*x + 1
  let x = @lmut.Matrix::make(3, 1, fn(i, _) { (i + 1).to_double() })
  let coef = @lmut.Matrix::from_2d_array([[2.0]])
  let intercept = 1.0

  let y_true = @lmut.Matrix::make(3, 1, fn(i, _) { 2.0 * (i + 1).to_double() + 1.0 })
  let y_pred = lr_get_predict_data(x, coef, intercept)

  assert_eq(y_pred[0][0], 3.0)
  assert_eq(y_pred[1][0], 5.0)
  assert_eq(y_pred[2][0], 7.0)

  assert_eq(r2_score(x, y_true, coef, intercept), 1.0)
}
