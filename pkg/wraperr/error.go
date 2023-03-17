// Package wraperr provides functionality for wrapping errors with cause and wrap in WrapErr struct
package wraperr

import (
	"errors"
	"fmt"
)

// WrapErr holds wrap and cause error. It implements error API.
type WrapErr struct {
	wrap error
	err  error
}

// NewWrapErr creates wrap on err. If err is WrapErr with given wrap, then NewWrapErr unwraps given err
// and wraps its cause in given wrap.
func NewWrapErr(wrap error, err error) *WrapErr {
	if err != nil && errors.Is(err, wrap) { // this check prevents multi-wrapping error with same wrap
		return &WrapErr{
			wrap: err,
			err:  errors.Unwrap(err),
		}
	}
	return &WrapErr{wrap: wrap, err: err}
}

// Error returns string representation of wrapped error in format `wrap: cause`
func (e *WrapErr) Error() string {
	return fmt.Sprintf("%s: %s", e.wrap.Error(), e.err.Error())
}

// Is detects if wrap or cause is target err
func (e *WrapErr) Is(err error) bool {
	if errors.Is(e.wrap, err) {
		return true
	}

	return errors.Is(e.err, err)
}

// Unwrap returns cause wrapped in WrapErr
func (e *WrapErr) Unwrap() error {
	return e.err
}

// WrapError wraps error if error pointed by err is not nil.
//
// NOTE: It modifies error that err pointing to!
func WrapError(wrap error, err *error) {
	if err != nil && *err != nil {
		*err = NewWrapErr(wrap, *err)
	}
}
