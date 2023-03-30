package layer

import (
	"nn/internal/nn"
	"nn/internal/nn/operation"
	"nn/pkg/mmath/matrix"
)

type ILayer interface {
	nn.IModule
	Forward(x *matrix.Matrix) (*matrix.Matrix, error)
	Backward(dy *matrix.Matrix) (*matrix.Matrix, error)
	ApplyOptim(optimizer operation.Optimizer) error
}
