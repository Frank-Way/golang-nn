package net

import "errors"

var (
	ErrNil    = errors.New("calling nil network")
	ErrCreate = errors.New("can not create network")
	ErrExec   = errors.New("can not perform step")
)
