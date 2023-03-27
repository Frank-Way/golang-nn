package operation

import "errors"

var (
	ErrNil    = errors.New("call nil operation")
	ErrCreate = errors.New("can not create operation")
	ErrExec   = errors.New("can not execute operation")
)
