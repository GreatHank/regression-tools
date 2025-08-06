struct LinearRegression[T] {
  mut coef_ : @lmut.Matrix[T]
  mut intercept_ : T
}

pub fn new[T: @lg.Zero] () -> LinearRegression[T] {
  return {
    coef_: @lmut.Matrix::from_2d_array([[]]),
    intercept_: T::zero()
  }
}

pub fn[T: @lg.Semiring + @lmut.Tolerance + @lg.Num + Eq + Div + Sub + Compare + @lg.Inverse] fit(self: LinearRegression[T], x: @lmut.Matrix[T], y: @lmut.Matrix[T]) -> Unit {
  let one = @lmut.Matrix::new(@lmut.Matrix::row(x), 1, T::one())
  let x_combine = @lmut.Matrix::horizontal_combine(one, x)
  let x_trans = @lmut.Matrix::transpose(x_combine)
  let xtx = x_trans * x_combine
  match @lmut.Matrix::inverse(xtx) {
    Some(xtx_inv) => {
      let coeff = xtx_inv * x_trans * y
      let rows = @lmut.Matrix::row(coeff)
      self.coef_ = @lmut.Matrix::make(rows-1, 1,fn(i,j) {
        coeff[i+1][j]
      })
      self.intercept_ = T::zero()
    }
    None => {
      abort("Matrix is not invertible")
    }
  }
  return
}

test "LinearRegression::fit" {
  let lr = new()
  // Simple linear regression: y = 2x + 1
  let x = @lmut.Matrix::from_2d_array([[1.0], [2.0], [3.0]])
  let y = @lmut.Matrix::from_2d_array([[3.0] , [5.0], [7.0]])
  lr.fit(x, y)
  inspect(lr.coef_, content=(
    "|2.0000000000000018|"
  ))
  inspect(lr.intercept_, content="0")
}