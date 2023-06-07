package layertestutils

import (
	"github.com/stretchr/testify/require"
	"nn/internal/nn"
	"nn/internal/nn/layer"
	"testing"
)

func NewLayer(t *testing.T, kind nn.Kind, args ...interface{}) layer.ILayer {
	l, err := layer.Create(kind, args...)
	require.NoError(t, err)
	return l
}
