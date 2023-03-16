package operation

import "nn/pkg/mmath/matrix"

type IOperation interface {
	Forward(x *matrix.Matrix) (*matrix.Matrix, error)
	Backward(dy *matrix.Matrix) (*matrix.Matrix, error)
}
