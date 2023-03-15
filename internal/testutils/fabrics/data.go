package fabrics

import (
	"github.com/stretchr/testify/require"
	"nn/internal/data/dataset"
	"testing"
)

type DataParameters struct {
	X MatrixParameters
	Y MatrixParameters
}

func NewData(t *testing.T, parameters DataParameters) *dataset.Data {
	x := NewMatrix(t, parameters.X)
	y := NewMatrix(t, parameters.Y)
	data, err := dataset.NewData(x, y)
	require.NoError(t, err)

	return data
}
