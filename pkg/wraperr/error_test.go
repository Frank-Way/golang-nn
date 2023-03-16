package wraperr

import (
	"errors"
	"github.com/stretchr/testify/require"
	"testing"
	"time"
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

func TestWrapError(t *testing.T) {
	wrap := errors.New("wrap")
	err := func() (err error) {
		defer WrapError(wrap, &err)
		return errors.New("cause")
	}()

	require.Error(t, err)
	require.ErrorIs(t, err, wrap)
}

func Benchmark_Wrapping(b *testing.B) {
	wrap := errors.New("wrap")
	f1 := func() error {
		err := func() error {
			time.Sleep(time.Duration(100) * time.Millisecond)
			return errors.New("cause")
		}()

		return NewWrapErr(wrap, err)
	}
	f2 := func() (err error) {
		defer WrapError(wrap, &err)
		time.Sleep(time.Duration(100) * time.Millisecond)
		return errors.New("cause")
	}
	b.Run("regular wrapping", func(b *testing.B) {
		f1()
	})
	b.Run("deferred wrapping", func(b *testing.B) {
		f2()
	})
}
