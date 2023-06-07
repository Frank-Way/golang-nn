package losstestutils

import (
	"github.com/stretchr/testify/require"
	"nn/internal/nn"
	"nn/internal/nn/loss"
	"testing"
)

func NewLoss(t *testing.T, kind nn.Kind, args ...interface{}) loss.ILoss {
	l, err := loss.Create(kind, args...)
	require.NoError(t, err)
	return l
}
