package loss

import "errors"

var (
	ErrNil     = errors.New("call nil loss")
	ErrCreate  = errors.New("can not create loss")
	ErrFabric  = errors.New("can not create loss using factory")
	ErrBuilder = errors.New("can not create loss using builder")
	ErrExec    = errors.New("can not execute loss")
)
