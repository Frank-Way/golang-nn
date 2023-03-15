package operation

import (
	"math"
	"nn/pkg/mmath/matrix"
)

func NewLinearActivation() *Operation {
	return &Operation{
		name:     "linear activation",
		output:   func(x *matrix.Matrix) (*matrix.Matrix, error) { return x.Copy(), nil },
		gradient: func(dy *matrix.Matrix) (*matrix.Matrix, error) { return dy.Copy(), nil },
	}
}

func NewSigmoidActivation() *Operation {
	return &Operation{
		name: "sigmoid activation",
		output: func(x *matrix.Matrix) (*matrix.Matrix, error) {
			return x.ApplyFunc(func(value float64) float64 {
				return 1 / (1 + math.Exp(-value))
			}), nil
		},
		gradient: func(dy *matrix.Matrix) (*matrix.Matrix, error) {
			return dy.ApplyFunc(func(value float64) float64 {
				return value * (1 - value)
			}), nil
		},
	}
}

func NewTanhActivation() *Operation {
	return &Operation{
		name: "tanh activation",
		output: func(x *matrix.Matrix) (*matrix.Matrix, error) {
			return x.Tanh(), nil
		},
		gradient: func(dy *matrix.Matrix) (*matrix.Matrix, error) {
			return dy.ApplyFunc(func(value float64) float64 {
				return value * (1 - value)
			}), nil
		},
	}
}
