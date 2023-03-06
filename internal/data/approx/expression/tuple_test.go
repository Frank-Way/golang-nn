package expression

//
//import (
//	"github.com/stretchr/testify/require"
//	"testing"
//)
//
//func TestNewTuple(t *testing.T) {
//	tests := []struct {
//		name string
//		in   string
//		err  bool
//	}{
//		{name: "valid symbol x0", in: "x0"},
//		{name: "valid symbol x99", in: "x99"},
//		{name: "invalid symbol xx", in: "xx", err: true},
//		{name: "invalid symbol 0x", in: "0x", err: true},
//		{name: "invalid symbol x 0", in: "x 0", err: true},
//		{name: "invalid symbol x_0", in: "x_0", err: true},
//		{name: "invalid symbol x0.1", in: "x0.1", err: true},
//		{name: "invalid symbol x-1", in: "x-1", err: true},
//	}
//
//	for i := range tests {
//		t.Run(tests[i].name, func(t *testing.T) {
//			_, err := newSymbol(tests[i].in)
//			if tests[i].err {
//				require.Error(t, err)
//			} else {
//				require.NoError(t, err)
//			}
//		})
//	}
//}
//
//func TestSymbol_exec(t *testing.T) {
//	tests := []struct {
//		name     string
//		in       string
//		args     []float64
//		expected float64
//		err      bool
//	}{
//		{name: "symbol x0, args [2]", in: "x0", args: []float64{2}, expected: 2},
//		{name: "symbol x0, args [2 3]", in: "x0", args: []float64{2, 3}, expected: 2},
//		{name: "symbol x0, no args", in: "x0", err: true},
//		{name: "symbol x0, empty args", in: "x0", args: []float64{}, err: true},
//		{name: "symbol x5, args [2 3]", in: "x5", args: []float64{2, 3}, err: true},
//	}
//
//	for i := range tests {
//		t.Run(tests[i].name, func(t *testing.T) {
//			s, err := newSymbol(tests[i].in)
//			require.NoError(t, err)
//			actual, err := s.exec(tests[i].args)
//			if tests[i].err {
//				require.Error(t, err)
//			} else {
//				require.NoError(t, err)
//				require.Equal(t, tests[i].expected, actual)
//			}
//		})
//	}
//}
