package datagentestutils

import (
	"github.com/stretchr/testify/require"
	"nn/internal/data/approx/datagen"
	"nn/internal/data/dataset"
	"testing"
)

type ParametersParameters struct {
	Expression string
	Ranges     []*datagen.InputRange
	Split      *dataset.DataSplitParameters
}

func NewParameters(t *testing.T, parameters ParametersParameters) *datagen.Parameters {
	if parameters.Expression == "" {
		parameters.Expression = "(sin x0)"
		parameters.Ranges = []*datagen.InputRange{NewInputRange(t, InputRangeParameters{})}
		parameters.Split = dataset.DefaultDataSplitParameters
	}
	params, err := datagen.NewParameters(parameters.Expression, parameters.Ranges, parameters.Split)
	require.NoError(t, err)

	return params
}

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
