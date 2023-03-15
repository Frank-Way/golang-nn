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

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := newNumber(test.in)
			if test.err {
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

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			n, err := newNumber(test.in)
			require.NoError(t, err)
			actual, err := n.exec(test.args)
			if test.err {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
				require.Equal(t, test.expected, actual)
			}
		})
	}
}
