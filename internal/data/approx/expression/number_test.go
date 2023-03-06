package expression

import (
	"github.com/stretchr/testify/require"
	"testing"
)

func TestNewNumber(t *testing.T) {
	tests := []struct {
		name string
		in   string
		err  bool
	}{
		{name: "valid number 1", in: "1"},
		{name: "valid number 1.5", in: "1.5"},
		{name: "valid number -1.5", in: "-1.5"},
		{name: "invalid number a", in: "a", err: true},
		{name: "invalid number -a", in: "-a", err: true},
		{name: "invalid number 1a", in: "1a", err: true},
		{name: "invalid number 1.a", in: "1.a", err: true},
		{name: "invalid number 1.1a", in: "1.1a", err: true},
		{name: "invalid number 1.1-", in: "1.1-", err: true},
		{name: "invalid number 1.-1", in: "1.-1", err: true},
		{name: "invalid number 1. 1", in: "1. 1", err: true},
		{name: "invalid number 1..1", in: "1..1", err: true},
		{name: "invalid number .11", in: ".11", err: true},
	}

	for i := range tests {
		t.Run(tests[i].name, func(t *testing.T) {
			_, err := newNumber(tests[i].in)
			if tests[i].err {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
			}
		})
	}
}

func TestNumber_exec(t *testing.T) {
	tests := []struct {
		name     string
		in       string
		args     []float64
		expected float64
		err      bool
	}{
		{name: "number 1, no args", in: "1", expected: 1},
		{name: "number 1, empty args", args: []float64{}, in: "1", expected: 1},
		{name: "number 1, args [2]", args: []float64{2}, in: "1", expected: 1},
		{name: "number 1, args [2 3]", args: []float64{2, 3}, in: "1", expected: 1},
	}

	for i := range tests {
		t.Run(tests[i].name, func(t *testing.T) {
			n, err := newNumber(tests[i].in)
			require.NoError(t, err)
			actual, err := n.exec(tests[i].args)
			if tests[i].err {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
				require.Equal(t, tests[i].expected, actual)
			}
		})
	}
}
