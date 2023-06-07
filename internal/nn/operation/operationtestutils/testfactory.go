package operationtestutils

import (
	"github.com/stretchr/testify/require"
	"nn/internal/nn"
	"nn/internal/nn/operation"
	"testing"
)

func NewOperation(t *testing.T, kind nn.Kind, args ...interface{}) operation.IOperation {
	o, err := operation.Create(kind, args...)
	require.NoError(t, err)
	return o
}
