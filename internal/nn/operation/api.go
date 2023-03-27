// Package operation provides functionality of IOperation and its implementations: Operation, ParamOperation,
// ConstOperation. Each operation is available by constructors (example: NewWeightOperation).
package operation

import "nn/pkg/mmath/matrix"

// IOperation represents operation block of neural network
type IOperation interface {
	// Forward makes forward propagation step, consuming input and producing output
	Forward(x *matrix.Matrix) (*matrix.Matrix, error)

	// Backward makes backward propagation step, consuming output gradient and producing input gradient
	Backward(dy *matrix.Matrix) (*matrix.Matrix, error)

	// Copy create deep-copy of IOperation
	Copy() IOperation

	// Equal return true if this and operation are deep-equal
	Equal(operation IOperation) bool

	// EqualApprox same as Equal, but it compares floats using some epsilon
	EqualApprox(operation IOperation) bool

	String() string
	PrettyString() string
	ShortString() string
}
