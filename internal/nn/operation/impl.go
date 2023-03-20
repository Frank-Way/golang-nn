package operation

import (
	"math"
	"nn/pkg/mmath/matrix"
)

// NewLinearActivation return operation:
//     y = f(x) = x;
//     dx = f(dy) = dy.
func NewLinearActivation() *Operation {
	logger.Debug("create new linear activation")
	return &Operation{
		name:     "linear activation",
		output:   func(x *matrix.Matrix) (*matrix.Matrix, error) { return x.Copy(), nil },
		gradient: func(dy *matrix.Matrix) (*matrix.Matrix, error) { return dy.Copy(), nil },
	}
}

// NewSigmoidActivation return operation:
//     y = f(x) = 1 / (1 + exp(-x));
//     dx = f(dy) = dy * (1 - dy).
func NewSigmoidActivation() *Operation {
	logger.Debug("create new sigmoid activation")
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

// NewTanhActivation return operation:
//     y = f(x) = tanh(x);
//     dx = f(dy) = dy * (1 - dy).
func NewTanhActivation() *Operation {
	logger.Debug("create new tanh activation")
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
