package wraperr

import (
	"errors"
	"fmt"
)

type WrapErr struct {
	wrap error
	err  error
}

func NewWrapErr(wrap error, err error) *WrapErr {
	return &WrapErr{wrap: wrap, err: err}
}

func (e *WrapErr) Error() string {
	return fmt.Sprintf("%s: %s", e.wrap.Error(), e.err.Error())
}

func (e *WrapErr) Is(err error) bool {
	if errors.Is(e.wrap, err) {
		return true
	}

	return errors.Is(e.err, err)
}
