package loss

import "errors"

var (
	ErrNil    = errors.New("call nil loss")
	ErrCreate = errors.New("can not create loss")
	ErrExec   = errors.New("can not execute loss")
)
