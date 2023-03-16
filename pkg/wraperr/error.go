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
	if err != nil && errors.Is(err, wrap) { // this check prevents multi-wrapping error with same wrap
		return &WrapErr{
			wrap: err,
			err:  errors.Unwrap(err),
		}
	}
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

func (e *WrapErr) Unwrap() error {
	return e.err
}

func WrapError(wrap error, err *error) {
	if *err != nil {
		*err = NewWrapErr(wrap, *err)
	}
}
