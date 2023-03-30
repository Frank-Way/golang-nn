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
