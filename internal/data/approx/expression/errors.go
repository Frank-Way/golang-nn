package expression

import "github.com/pkg/errors"

var (
	ErrParse = errors.New("can not parse expression")
	ErrExec  = errors.New("can not execute expression")
)
