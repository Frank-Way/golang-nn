package operation

import "errors"

var (
	ErrNil     = errors.New("call nil operation")
	ErrCreate  = errors.New("can not create operation")
	ErrExec    = errors.New("can not execute operation")
	ErrFabric  = errors.New("can not create operation using fabric")
	ErrBuilder = errors.New("can not create operation using builder")
)
