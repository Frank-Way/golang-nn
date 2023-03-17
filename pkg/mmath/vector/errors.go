package vector

import "errors"

var (
	ErrCreate   = errors.New("can not create vector")
	ErrNotFound = errors.New("can not find value in vector")
	ErrExec     = errors.New("can not perform operation to vector")
	ErrNil      = errors.New("calling nil vector")
)
