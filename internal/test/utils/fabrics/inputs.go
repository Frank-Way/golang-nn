package fabrics

import (
	"github.com/stretchr/testify/require"
	"nn/internal/data/approx/datagen"
	"testing"
)

type InputRangeParameters struct {
	*datagen.InputRange
}

func NewInputRange(t *testing.T, parameters *InputRangeParameters) *datagen.InputRange {
	var inputRange *datagen.InputRange
	var err error
	if parameters.Count == 0 {
		inputRange, err = datagen.NewInputRange(0, 1, 11)
	} else {
		inputRange, err = datagen.NewInputRange(parameters.Left, parameters.Right, parameters.Count)
	}
	require.NoError(t, err)

	return inputRange
}
