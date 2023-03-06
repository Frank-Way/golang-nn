package expression

import (
	"github.com/stretchr/testify/require"
	"math"
	"testing"
)

func TestOperation_exec(t *testing.T) {
	tests := []struct {
		name     string
		token    string
		args     []float64
		expected float64
		err1     bool
		err2     bool
	}{
		{name: "sin 2", token: "sin", args: []float64{2}, expected: math.Sin(2)},
		{name: "Sin 2", token: "Sin", args: []float64{2}, err1: true},
		{name: "sin 2 2", token: "sin", args: []float64{2, 2}, err2: true},
		{name: "+ 2 1", token: "+", args: []float64{2, 1}, expected: 2 + 1},
		{name: "++ 2 1", token: "++", args: []float64{2, 1}, err1: true},
		{name: "+ 2 1 3", token: "+", args: []float64{2, 1, 3}, err2: true},
		{name: "sum", token: "sum", args: []float64{}, expected: 0},
		{name: "sum 1", token: "sum", args: []float64{1}, expected: 1},
		{name: "sum 1 2 3 4 5", token: "sum", args: []float64{1, 2, 3, 4, 5}, expected: 1 + 2 + 3 + 4 + 5},
	}

	for i := range tests {
		t.Run(tests[i].name, func(t *testing.T) {
			operation, err := getOperation(tests[i].token)
			if tests[i].err1 {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
				exec, err := operation.exec(tests[i].args)
				if tests[i].err2 {
					require.Error(t, err)
				} else {
					require.NoError(t, err)
					require.Equal(t, tests[i].expected, exec)
				}
			}
		})
	}
}

func TestOperation_checkInputsCount(t *testing.T) {
	tests := []struct {
		name         string
		token        string
		valueToCheck int
		expected     bool
	}{
		{name: "sin 0", token: "sin", valueToCheck: 0, expected: false},
		{name: "sin 1", token: "sin", valueToCheck: 1, expected: true},
		{name: "sin 2", token: "sin", valueToCheck: 2, expected: false},
		{name: "+ 0", token: "+", valueToCheck: 0, expected: false},
		{name: "+ 1", token: "+", valueToCheck: 1, expected: false},
		{name: "+ 2", token: "+", valueToCheck: 2, expected: true},
		{name: "+ 3", token: "+", valueToCheck: 3, expected: false},
		{name: "sum 0", token: "sum", valueToCheck: 0, expected: true},
		{name: "sum 10", token: "sum", valueToCheck: 10, expected: true},
		{name: "max 0", token: "max", valueToCheck: 0, expected: false},
		{name: "max 1", token: "max", valueToCheck: 1, expected: true},
		{name: "max 10", token: "max", valueToCheck: 10, expected: true},
	}

	for i := range tests {
		t.Run(tests[i].name, func(t *testing.T) {
			operation, err := getOperation(tests[i].token)
			require.NoError(t, err)
			valid := operation.checkInputsCount(tests[i].valueToCheck)
			require.Equal(t, tests[i].expected, valid)
		})
	}
}
