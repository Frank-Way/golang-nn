package fabrics

import (
	"github.com/stretchr/testify/require"
	"nn/internal/data/approx/datagen"
	"testing"
)

type InputRangeParameters struct {
	*datagen.InputRange
}

func NewInputRange(t *testing.T, parameters InputRangeParameters) *datagen.InputRange {
	if parameters.InputRange == nil {
		parameters.InputRange = &datagen.InputRange{
			Left:  0,
			Right: 1,
			Count: 11,
		}
	}
	inputRange, err := datagen.NewInputRange(parameters.Left, parameters.Right, parameters.Count)
	require.NoError(t, err)

	return inputRange
}
