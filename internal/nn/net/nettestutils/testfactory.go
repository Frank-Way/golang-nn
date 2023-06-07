package nettestutils

import (
	"github.com/stretchr/testify/require"
	"nn/internal/nn"
	"nn/internal/nn/net"
	"testing"
)

func NewNetwork(t *testing.T, kind nn.Kind, args ...interface{}) net.INetwork {
	n, err := net.Create(kind, args...)
	require.NoError(t, err)
	return n
}
