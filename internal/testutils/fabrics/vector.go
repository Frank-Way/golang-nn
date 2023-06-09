// Package fabrics provides fabrics for creating test structs from other packages
package fabrics

import (
	"github.com/stretchr/testify/require"
	"nn/internal/testutils"
	"nn/pkg/mmath/vector"
	"testing"
)

type VectorParameters struct {
	Values []float64
	Size   int
}

func NewVector(t *testing.T, parameters VectorParameters) *vector.Vector {
	var vec *vector.Vector
	var err error
	if parameters.Values == nil {
		vec, err = vector.NewVector(testutils.RandomArray(parameters.Size))
	} else {
		vec, err = vector.NewVector(parameters.Values)
	}
	require.NoError(t, err)

	return vec
}
