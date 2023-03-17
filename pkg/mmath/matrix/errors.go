package matrix

import "errors"

var (
	ErrCreate   = errors.New("can not create matrix")
	ErrNotFound = errors.New("can not find value in matrix")
	ErrExec     = errors.New("can not perform operation to matrix")
	ErrNil      = errors.New("calling nil matrix")
)
