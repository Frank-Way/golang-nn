package datagen

import (
	"github.com/stretchr/testify/require"
	"nn/internal/testutils"
	"testing"
)

type InputRangeParameters struct {
	*InputRange
}

func newInputRange(t *testing.T, parameters InputRangeParameters) *InputRange {
	if parameters.InputRange == nil {
		parameters.InputRange = &InputRange{
			Left:  0,
			Right: 1,
			Count: 11,
		}
	}
	inputRange, err := NewInputRange(parameters.Left, parameters.Right, parameters.Count)
	require.NoError(t, err)

	return inputRange
}

func TestNewInputRange(t *testing.T) {
	tests := []struct {
		testutils.Base
		left  float64
		right float64
		count int
	}{
		{Base: testutils.Base{Name: "0 to 1, 5 values"}, left: 0, right: 1, count: 5},
		{Base: testutils.Base{Name: "0 to 1, 50000 values"}, left: 0, right: 1, count: 50000},
		{Base: testutils.Base{Name: "2 to 1, 5 values", Err: ErrCreate}, left: 2, right: 1, count: 5},
		{Base: testutils.Base{Name: "0 to 1, 1 values", Err: ErrCreate}, left: 1, right: 1, count: 1},
	}

	for _, test := range tests {
		t.Run(test.Name, func(t *testing.T) {
			_, err := NewInputRange(test.left, test.right, test.count)
			if test.Err == nil {
				require.NoError(t, err)
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, test.Err)
			}
		})
	}
}

func TestInputRange_Strings(t *testing.T) {
	inputRange, err := NewInputRange(0.5, 2, 3)
	require.NoError(t, err)

	t.Log("ShortString\n" + inputRange.ShortString())
	t.Log("String\n" + inputRange.String())
	t.Log("PrettyString\n" + inputRange.PrettyString())
}
