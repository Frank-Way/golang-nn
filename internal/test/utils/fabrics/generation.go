package fabrics

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
	params, err := datagen.NewParameters(parameters.Expression, parameters.Ranges, parameters.Split)
	require.NoError(t, err)

	return params
}
