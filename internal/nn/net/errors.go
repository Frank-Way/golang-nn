package net

import "errors"

var (
	ErrNil     = errors.New("calling nil network")
	ErrCreate  = errors.New("can not create network")
	ErrFabric  = errors.New("can not create network using factory")
	ErrBuilder = errors.New("can not create network using builder")
	ErrExec    = errors.New("can not perform step")
)
