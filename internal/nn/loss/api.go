package loss

import "nn/pkg/mmath/matrix"

type ILoss interface {
	Forward(t *matrix.Matrix, y *matrix.Matrix) (float64, error)
	Backward() (*matrix.Matrix, error)
}
