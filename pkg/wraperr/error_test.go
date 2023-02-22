package wraperr

import (
	"errors"
	"github.com/stretchr/testify/require"
	"testing"
)

func TestWrapErr(t *testing.T) {
	wrap := errors.New("wrap error")
	err := errors.New("base error")
	wrapped := NewWrapErr(wrap, err)

	unwrapped := errors.Unwrap(wrapped)
	require.True(t, errors.Is(unwrapped, err))

	doubleWrapped := NewWrapErr(wrap, wrapped)
	unwrappedDoubleWrapped := errors.Unwrap(doubleWrapped)
	require.True(t, errors.Is(unwrappedDoubleWrapped, err))

	doubleUnwrappedDoubleWrapped := errors.Unwrap(unwrappedDoubleWrapped)
	require.Nil(t, doubleUnwrappedDoubleWrapped)
}
