package fabrics

import (
	"github.com/stretchr/testify/require"
	"nn/internal/data/approx/expression"
	"testing"
)

type ExpressionParameters struct {
	Expression string
}

func NewExpression(t *testing.T, parameters ExpressionParameters) *expression.Expression {
	expr, err := expression.NewExpression(parameters.Expression)
	require.NoError(t, err)

	return expr
}
