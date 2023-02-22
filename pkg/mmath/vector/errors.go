package vector

import "errors"

var (
	ErrCreate        = errors.New("can not create vector")
	ErrNotFound      = errors.New("can not find value in vector")
	ErrOperationExec = errors.New("can not perform operation to vector")
)
