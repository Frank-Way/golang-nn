package net

import (
	"nn/internal/nn"
	"nn/internal/nn/operation"
	"nn/pkg/mmath/matrix"
)

type INetwork interface {
	nn.IModule
	Forward(x *matrix.Matrix) (*matrix.Matrix, error)
	Loss(t *matrix.Matrix) (float64, error)
	Backward() (*matrix.Matrix, error)
	ApplyOptim(optimizer operation.Optimizer) error
}

var networks = map[nn.Kind]struct{}{
	FFNetwork: {},
}

func IsNetwork(kind nn.Kind) bool {
	_, ok := networks[kind]
	return ok
}
