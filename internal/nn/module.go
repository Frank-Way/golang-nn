package nn

type IModule interface {
	// Copy create deep-copy of IModule
	Copy() IModule

	// Equal return true if this and argument are deep-equal
	Equal(module IModule) bool

	// EqualApprox same as Equal, but it compares floats using some epsilon
	EqualApprox(module IModule) bool

	String() string
	PrettyString() string
	ShortString() string

	Is(kind Kind) bool
	Kind() Kind
}

type Kind string
