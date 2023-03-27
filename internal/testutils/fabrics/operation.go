package fabrics

import (
	"github.com/stretchr/testify/require"
	"nn/internal/nn/operation"
	"nn/pkg/percent"
	"testing"
)

type ActivationParameters struct {
	SigmoidParamParameters VectorParameters
}

type Activation uint8

const (
	LinearAct Activation = iota
	SigmoidAct
	TanhAct
	SigmoidParamAct
)

func NewActivation(t *testing.T, act Activation, parameters ActivationParameters) operation.IOperation {
	switch act {
	case LinearAct:
		return operation.NewLinearActivation()
	case SigmoidAct:
		return operation.NewSigmoidActivation()
	case TanhAct:
		return operation.NewTanhActivation()
	case SigmoidParamAct:
		params := NewVector(t, parameters.SigmoidParamParameters)
		act, err := operation.NewSigmoidParam(params)
		require.NoError(t, err)
		return act
	}
	t.Errorf("unknown act: %d", act)
	return nil
}

type WeightParameters struct {
	MatrixParameters
}

func NewWeight(t *testing.T, parameters WeightParameters) operation.IOperation {
	weight := NewMatrix(t, parameters.MatrixParameters)

	res, err := operation.NewWeightOperation(weight)
	require.NoError(t, err)

	return res
}

type BiasParameters struct {
	VectorParameters
}

func NewBias(t *testing.T, parameters BiasParameters) operation.IOperation {
	bias := NewVector(t, parameters.VectorParameters)

	res, err := operation.NewBiasOperation(bias)
	require.NoError(t, err)

	return res
}

type DropoutParameters struct {
	percent.Percent
}

func NewDropout(t *testing.T, parameters DropoutParameters) operation.IOperation {
	res, err := operation.NewDropout(parameters.Percent)
	require.NoError(t, err)

	return res
}
