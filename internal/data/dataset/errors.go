package dataset

import "errors"

var (
	ErrCreate = errors.New("can not create data")
	ErrSplit  = errors.New("can not split data")
)
