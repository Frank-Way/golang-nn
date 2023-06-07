package net

import (
	"github.com/stretchr/testify/require"
	"nn/internal/nn"
	"testing"
)

func newNetwork(t *testing.T, kind nn.Kind, args ...interface{}) INetwork {
	n, err := Create(kind, args...)
	require.NoError(t, err)
	return n
}
