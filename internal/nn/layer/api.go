package layer

import (
	"nn/internal/nn/operation"
	"nn/pkg/mmath/matrix"
)

type ILayer interface {
	Forward(x *matrix.Matrix) (*matrix.Matrix, error)
	Backward(dy *matrix.Matrix) (*matrix.Matrix, error)
	ApplyOptim(optimizer operation.Optimizer) error

	// Copy create deep-copy of ILayer
	Copy() ILayer

	// Equal return true if this and layer are deep-equal
	Equal(layer ILayer) bool

	// EqualApprox same as Equal, but it compares floats using some epsilon
	EqualApprox(layer ILayer) bool

	String() string
	PrettyString() string
	ShortString() string

	Is(kind Kind) bool
}

type Kind string
