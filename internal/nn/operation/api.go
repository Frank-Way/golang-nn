// Package operation provides functionality of IOperation and its implementations: Operation, ParamOperation,
// ConstOperation. Each operation is available by constructors (example: NewWeightOperation).
package operation

import (
	"nn/internal/nn"
	"nn/pkg/mmath/matrix"
)

// IOperation represents operation block of neural network
type IOperation interface {
	nn.IModule
	// Forward makes forward propagation step, consuming input and producing output
	Forward(x *matrix.Matrix) (*matrix.Matrix, error)

	// Backward makes backward propagation step, consuming output gradient and producing input gradient
	Backward(dy *matrix.Matrix) (*matrix.Matrix, error)

	Output() *matrix.Matrix
	IsActivation() bool
}
