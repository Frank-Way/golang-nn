package train

import "errors"

var (
	ErrParameters = errors.New("error checking parameters")
	ErrPreTrain   = errors.New("error pre train checking")
	ErrExec       = errors.New("can not execute train")
)
