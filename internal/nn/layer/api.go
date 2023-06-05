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
	Output() *matrix.Matrix
	InputsCount() int
	Size() int
}

var layers = map[nn.Kind]struct{}{
	DenseLayer: {}, DenseDropLayer: {},
}

func IsLayer(kind nn.Kind) bool {
	_, ok := layers[kind]
	return ok
}
