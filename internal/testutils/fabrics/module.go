package fabrics

import (
	"github.com/stretchr/testify/require"
	"nn/internal/nn"
	"nn/internal/nn/layer"
	"nn/internal/nn/loss"
	"nn/internal/nn/net"
	"nn/internal/nn/operation"
	"testing"
)

func NewOperation(t *testing.T, kind nn.Kind, args ...interface{}) operation.IOperation {
	o, err := operation.Create(kind, args...)
	require.NoError(t, err)
	return o
}

func NewLoss(t *testing.T, kind nn.Kind, args ...interface{}) loss.ILoss {
	l, err := loss.Create(kind, args...)
	require.NoError(t, err)
	return l
}

func NewLayer(t *testing.T, kind nn.Kind, args ...interface{}) layer.ILayer {
	l, err := layer.Create(kind, args...)
	require.NoError(t, err)
	return l
}

func NewNetwork(t *testing.T, kind nn.Kind, args ...interface{}) net.INetwork {
	n, err := net.Create(kind, args...)
	require.NoError(t, err)
	return n
}
