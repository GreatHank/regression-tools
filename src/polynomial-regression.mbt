pub struct PolynomialRegression{
  mut coef_ : @lmut.Matrix[Double]
  mut intercept_ : Double
  mut degree : Int
}

pub fn PolynomialRegression::new() -> PolynomialRegression{
  return {
    coef_: @lmut.Matrix::from_2d_array([[]]),
    intercept_: 0.0,
    degree : 0
  }
}

fn fast_power(base: Double, exp: Int) -> Double {
  if exp == 0 {
    return 1.0
  }
  else if exp < 0 {
    return 1.0 / fast_power(base, -exp)
  }
  else {
    let mut result = 1.0
    let mut base_power = base
    let mut exponent = exp
    
    while exponent > 0 {
      if exponent % 2 == 1 {
        result = result * base_power
      }
      base_power = base_power * base_power
      exponent = exponent / 2
    }
    return result
  }
}

pub fn PolynomialRegression::fit(
    self : PolynomialRegression,
    x : @lmut.Matrix[Double],
    y : @lmut.Matrix[Double],
    degree : Int
) -> Unit {
  let x_pow = @lmut.Matrix::make(@lmut.Matrix::row(x), degree, (i, j) => fast_power(x[i][0], j + 1))
  let one = @lmut.Matrix::new(@lmut.Matrix::row(x), 1, 1.0)
  let x_combine = @lmut.Matrix::horizontal_combine(one, x_pow)
  let x_trans = @lmut.Matrix::transpose(x_combine)
  let xtx = x_trans * x_combine
  match @lmut.Matrix::inverse(xtx) {
    Some(xtx_inv) => {
      let coeff = xtx_inv * x_trans * y
      self.coef_ = coeff
      self.intercept_ = self.coef_[0][0]
      self.degree = degree
    }
    None => abort("Unable to perform polynomial regression")
  }
  return
}

impl Show for PolynomialRegression with output(self, logger) {
  logger.write_string("PolynomialRegression: degree = \{self.degree} \n")
  let rows = self.coef_.row()
  logger.write_string("y = \{self.intercept_}")
  for i in 1..<rows{
    logger.write_string(" + \{self.coef_[i][0]} * x^\{i}")
  }
  logger.write_string(" + ϵ")
}

test {
    let ans = PolynomialRegression::new()
    let x = @lmut.Matrix::from_2d_array([[1.0],[2.0],[3.0]])
    let y = @lmut.Matrix::from_2d_array([[4.0],[9.0],[16.0]])
    ans.fit(x, y, 2)
    inspect(ans, content = (
      #|PolynomialRegression: degree = 2 
      #|y = 1.0000000000001101 + 1.9999999999998437 * x^1 + 1.00000000000004 * x^2 + ϵ
    ))
}

test "PolynomialRegression::fit" {
  // Test degree 1 (linear): y = 2x + 1
  let pr1 = PolynomialRegression::new()
  let x1 = @lmut.Matrix::from_2d_array([[1.0], [2.0], [3.0]])
  let y1 = @lmut.Matrix::from_2d_array([[3.0], [5.0], [7.0]])
  pr1.fit(x1, y1, 1)
  inspect(pr1, content=(
    #|PolynomialRegression: degree = 1 
    #|y = 1 + 2.0000000000000018 * x^1 + ϵ

  ))

  // Test degree 2 (quadratic): y = x^2 + 1
  let pr2 = PolynomialRegression::new()
  let x2 = @lmut.Matrix::from_2d_array([[1.0], [2.0], [3.0], [4.0]])
  let y2 = @lmut.Matrix::from_2d_array([[2.0], [5.0], [10.0], [17.0]])
  pr2.fit(x2, y2, 2)
  inspect(pr2, content=(
    #|PolynomialRegression: degree = 2 
    #|y = 0.999999999999952 + -1.1013412404281553e-13 * x^1 + 1.0000000000000107 * x^2 + ϵ

  ))
}

test "PolynomialRegression::fit/degree_zero" {
  // Test degree 0 (constant): y = 5
  let pr = PolynomialRegression::new()
  let x = @lmut.Matrix::from_2d_array([[1.0], [2.0], [3.0]])
  let y = @lmut.Matrix::from_2d_array([[5.0], [5.0], [5.0]])
  pr.fit(x, y, 0)
  inspect(pr, content=(
    #|PolynomialRegression: degree = 0 
    #|y = 5 + ϵ

  ))
}

test "panic PolynomialRegression::fit/singular_matrix" {
  // Test with identical x values that should create a singular matrix
  let pr = PolynomialRegression::new()
  let x = @lmut.Matrix::from_2d_array([[1.0], [1.0], [1.0]])
  let y = @lmut.Matrix::from_2d_array([[2.0], [3.0], [4.0]])
  pr.fit(x, y, 2)
  inspect(pr, content="")
}