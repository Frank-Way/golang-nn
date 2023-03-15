package operation

import "errors"

var (
	ErrCreate = errors.New("can not create operation")
	ErrExec   = errors.New("can not execute operation")
)
