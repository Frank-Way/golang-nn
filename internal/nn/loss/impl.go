package loss

import (
	"nn/pkg/mmath/matrix"
)

// NewMSELoss create new mean-squared loss module
func NewMSELoss() ILoss {
	logger.Debug("create new MSE loss")
	return &Loss{
		name: "MSE loss",
		output: func(t, y *matrix.Matrix) (float64, error) {
			// 1 / (2 * N) * sum[(y - t) ** 2]
			delta, err := y.Sub(t)
			if err != nil {
				return 0, err
			}
			return delta.Sqr().Sum() / (2 * float64(delta.Rows())), nil
		},
		gradient: func(t, y *matrix.Matrix) (*matrix.Matrix, error) {
			// (y - t) / N
			delta, err := y.Sub(t)
			if err != nil {
				return nil, err
			}

			return delta.DivNum(float64(delta.Rows())), nil
		},
	}
}
