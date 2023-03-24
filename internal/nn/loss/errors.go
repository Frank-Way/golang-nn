package loss

import "errors"

var (
	ErrCreate = errors.New("can not create loss")
	ErrExec   = errors.New("can not execute loss")
)
