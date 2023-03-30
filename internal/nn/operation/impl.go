package operation

import (
	"math"
	"nn/internal/nn"
	"nn/pkg/mmath/matrix"
)

const (
	LinearActivation  nn.Kind = "linear activation"
	SigmoidActivation nn.Kind = "sigmoid activation"
	TanhActivation    nn.Kind = "tanh activation"
)

// NewLinearActivation return operation:
//     y = f(x) = x;
//     dx = f(dy) = dy.
func NewLinearActivation() IOperation {
	logger.Debug("create new linear activation")
	return &Operation{
		kind:       LinearActivation,
		activation: true,
		output:     func(x *matrix.Matrix) (*matrix.Matrix, error) { return x.Copy(), nil },
		gradient:   func(dy *matrix.Matrix) (*matrix.Matrix, error) { return dy.Copy(), nil },
	}
}

// NewSigmoidActivation return operation:
//     y = f(x) = 1 / (1 + exp(-x));
//     dx = f(dy) = dy * (1 - dy).
func NewSigmoidActivation() IOperation {
	logger.Debug("create new sigmoid activation")
	return &Operation{
		kind:       SigmoidActivation,
		activation: true,
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
func NewTanhActivation() IOperation {
	logger.Debug("create new tanh activation")
	return &Operation{
		kind:       TanhActivation,
		activation: true,
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
