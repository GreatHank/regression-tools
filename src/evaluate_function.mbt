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
  let coeff = @lmut.Matrix::make(@lmut.Matrix::row(coef) - 1, 1, fn(i, _) {
    coef[i + 1][0]
  })
  @lmut.Matrix::add_constant(x * coeff, intercept)
}

///|
pub fn stantard_error(
  x : @lmut.Matrix[Double],
  y : @lmut.Matrix[Double],
  coef : @lmut.Matrix[Double],
  intercept : Double,
) -> @lmut.Matrix[Double] {
  let col = @lmut.Matrix::col(x)
  let dow = @lmut.Matrix::row(x) - col - 1
  let res = residual_sum_of_square(y, lr_get_predict_data(x, coef, intercept)) /
    dow.to_double()
  let x_trans = x.transpose()
  let xtx = x_trans * x
  match @lmut.Matrix::inverse(xtx) {
    None => abort("Unable to perform Linear Regression method.")
    Some(x) =>
      @lmut.Matrix::make(@lmut.Matrix::row(coef), 1, fn(i, _) {
        (res * x[i][i]).pow(0.5)
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
