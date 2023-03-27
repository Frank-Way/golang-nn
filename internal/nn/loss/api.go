// Package loss provides functionality for ILoss and its implementations: Loss. Concrete losses available by constructors
// (for example, NewMSELoss)
package loss

import "nn/pkg/mmath/matrix"

// ILoss represents loss operation
type ILoss interface {
	// Forward calculates loss for given targets and outputs
	Forward(t *matrix.Matrix, y *matrix.Matrix) (float64, error)

	// Backward calculates input gradient for targets and outputs given during previous Forward() call
	Backward() (*matrix.Matrix, error)

	// Copy create deep-copy of ILoss
	Copy() ILoss

	// Equal return true if this and loss are deep-equal
	Equal(loss ILoss) bool

	// EqualApprox same as Equal, but it compares floats using some epsilon
	EqualApprox(loss ILoss) bool

	String() string
	PrettyString() string
	ShortString() string
}
