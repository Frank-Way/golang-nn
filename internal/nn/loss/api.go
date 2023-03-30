// Package loss provides functionality for ILoss and its implementations: Loss. Concrete losses available by constructors
// (for example, NewMSELoss)
package loss

import (
	"nn/internal/nn"
	"nn/pkg/mmath/matrix"
)

// ILoss represents loss operation
type ILoss interface {
	nn.IModule
	// Forward calculates loss for given targets and outputs
	Forward(t *matrix.Matrix, y *matrix.Matrix) (float64, error)

	// Backward calculates input gradient for targets and outputs given during previous Forward() call
	Backward() (*matrix.Matrix, error)

	Output() float64
}
