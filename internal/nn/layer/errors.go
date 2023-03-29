package layer

import "errors"

var (
	ErrNil    = errors.New("call nil layer")
	ErrCreate = errors.New("can not create layer")
	ErrExec   = errors.New("can not execute step in layer")
)
