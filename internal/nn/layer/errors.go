package layer

import "errors"

var (
	ErrNil     = errors.New("call nil layer")
	ErrCreate  = errors.New("can not create layer")
	ErrFabric  = errors.New("can not create layer using factory")
	ErrBuilder = errors.New("can not create layer using builder")
	ErrExec    = errors.New("can not execute step in layer")
)
