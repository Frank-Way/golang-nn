package fabrics

import (
	"github.com/stretchr/testify/require"
	"nn/internal/nn/layer"
	"testing"
)

type LayerParameters struct {
	DenseLayerParameters
	DenseDropLayerParameters
}

type DenseLayerParameters struct {
	Weight MatrixParameters
	Bias   VectorParameters
	Activation
	ActivationParameters
}

type DenseDropLayerParameters struct {
	DropoutParameters
}

func NewDenseLayer(t *testing.T, parameters LayerParameters) layer.ILayer {
	w := NewMatrix(t, parameters.Weight)
	b := NewVector(t, parameters.Bias)
	a := NewActivation(t, parameters.Activation, parameters.ActivationParameters)

	l, err := layer.NewDenseLayer(w, b, a)
	require.NoError(t, err)

	return l
}

func NewDenseDropLayer(t *testing.T, parameters LayerParameters) layer.ILayer {
	w := NewMatrix(t, parameters.Weight)
	b := NewVector(t, parameters.Bias)
	a := NewActivation(t, parameters.Activation, parameters.ActivationParameters)

	l, err := layer.NewDenseDropLayer(w, b, a, parameters.DropoutParameters.Percent)
	require.NoError(t, err)

	return l
}
