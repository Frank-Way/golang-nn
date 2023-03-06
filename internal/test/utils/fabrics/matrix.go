package fabrics

import (
	"github.com/stretchr/testify/require"
	"nn/internal/test/utils"
	"nn/pkg/mmath/matrix"
	"testing"
)

type MatrixParameters struct {
	Rows   int
	Cols   int
	Values []float64
}

func NewMatrix(t *testing.T, parameters MatrixParameters) *matrix.Matrix {
	if parameters.Values == nil {
		parameters.Values = utils.RandomArray(parameters.Rows * parameters.Cols)
	}
	mat, err := matrix.NewMatrixRawFlat(parameters.Rows, parameters.Cols, parameters.Values)
	require.NoError(t, err)

	return mat
}
