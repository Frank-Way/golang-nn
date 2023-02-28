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

func NewMatrix(t *testing.T, params MatrixParameters) *matrix.Matrix {
	if params.Values == nil {
		params.Values = utils.RandomArray(params.Rows * params.Cols)
	}
	mat, err := matrix.NewMatrixRawFlat(params.Rows, params.Cols, params.Values)
	require.NoError(t, err)

	return mat
}
