package test

import (
	"github.com/stretchr/testify/require"
	"nn/internal/nn"
	"nn/internal/nn/layer"
	"nn/internal/nn/loss"
	"nn/internal/nn/net"
	"nn/internal/nn/operation"
	"nn/internal/testutils"
	"nn/internal/testutils/fabrics"
	"nn/pkg/percent"
	"testing"
)

func TestNewBuilder(t *testing.T) {
	kinds := []nn.Kind{
		operation.LinearActivation,
		operation.SigmoidActivation,
		operation.TanhActivation,
		operation.SigmoidParamActivation,
		operation.Dropout,
		operation.WeightMultiply,
		operation.BiasAdd,
		"unknown kind",
		loss.MSELoss,
		layer.DenseLayer,
		layer.DenseDropLayer,
		net.FFNetwork,
	}
	pivot := 7
	for i, kind := range kinds {
		_, err := operation.NewBuilder(kind)
		if i < pivot {
			require.NoError(t, err)
		} else {
			require.Error(t, err)
			require.ErrorIs(t, err, operation.ErrBuilder)
		}
	}
}

func TestBuilder_Build(t *testing.T) {
	newBuilder := func(kind nn.Kind) *operation.Builder {
		res, err := operation.NewBuilder(kind)
		require.NoError(t, err)
		return res
	}
	factory := func(kind nn.Kind, args ...interface{}) operation.IOperation {
		res, err := operation.Create(kind, args...)
		require.NoError(t, err)
		return res
	}
	testcases := []struct {
		testutils.Base
		builder  *operation.Builder
		expected operation.IOperation
	}{
		{
			Base:     testutils.Base{Name: "build linear activation"},
			builder:  newBuilder(operation.LinearActivation),
			expected: factory(operation.LinearActivation),
		},
		{
			Base:     testutils.Base{Name: "build tanh activation"},
			builder:  newBuilder(operation.TanhActivation),
			expected: factory(operation.TanhActivation),
		},
		{
			Base:     testutils.Base{Name: "build sigmoid activation"},
			builder:  newBuilder(operation.SigmoidActivation),
			expected: factory(operation.SigmoidActivation),
		},
		{
			Base: testutils.Base{Name: "build sigmoid param activation, coeffs"},
			builder: newBuilder(operation.SigmoidParamActivation).
				SigmoidCoeffs(fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{1, 2, 3, 4}})),
			expected: factory(operation.SigmoidParamActivation,
				fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{1, 2, 3, 4}})),
		},
		{
			Base: testutils.Base{Name: "build sigmoid param activation, default range"},
			builder: newBuilder(operation.SigmoidParamActivation).
				NeuronsCount(4),
			expected: factory(operation.SigmoidParamActivation,
				fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{1, 2, 3, 4}})),
		},
		{
			Base: testutils.Base{Name: "build sigmoid param activation, range"},
			builder: newBuilder(operation.SigmoidParamActivation).
				NeuronsCount(4).
				SigmoidCoeffsRange(&operation.SigmoidCoeffsRange{Left: 5, Right: 8}),
			expected: factory(operation.SigmoidParamActivation,
				fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{5, 6, 7, 8}})),
		},
		{
			Base: testutils.Base{Name: "build sigmoid param activation, not enough arguments", Err: operation.ErrBuilder},
			builder: newBuilder(operation.SigmoidParamActivation).
				SigmoidCoeffsRange(&operation.SigmoidCoeffsRange{Left: 5, Right: 8}),
		},
		{
			Base:     testutils.Base{Name: "build dropout, 50%"},
			builder:  newBuilder(operation.Dropout).KeepProbability(percent.Percent50),
			expected: factory(operation.Dropout, percent.Percent50),
		},
		{
			Base:     testutils.Base{Name: "build dropout, default value"},
			builder:  newBuilder(operation.Dropout),
			expected: factory(operation.Dropout, percent.Percent100),
		},
		{
			Base: testutils.Base{Name: "build bias, biases"},
			builder: newBuilder(operation.BiasAdd).
				Bias(fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{1, 2, 3}})),
			expected: factory(operation.BiasAdd, fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{1, 2, 3}})),
		},
		{
			Base:    testutils.Base{Name: "build bias, random default"},
			builder: newBuilder(operation.BiasAdd).NeuronsCount(3),
		},
		{
			Base:    testutils.Base{Name: "build bias, random specified default"},
			builder: newBuilder(operation.BiasAdd).NeuronsCount(3).ParamInitType(operation.DefaultInit),
		},
		{
			Base:    testutils.Base{Name: "build bias, random glorot"},
			builder: newBuilder(operation.BiasAdd).NeuronsCount(3).InputsCount(2).ParamInitType(operation.GlorotInit),
		},
		{
			Base:    testutils.Base{Name: "build bias, random glorot no inputs count", Err: operation.ErrBuilder},
			builder: newBuilder(operation.BiasAdd).NeuronsCount(3).ParamInitType(operation.GlorotInit),
		},
		{
			Base:    testutils.Base{Name: "build bias, random no inputs count", Err: operation.ErrBuilder},
			builder: newBuilder(operation.BiasAdd),
		},
		{
			Base: testutils.Base{Name: "build weight, weights"},
			builder: newBuilder(operation.WeightMultiply).
				Weight(fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}})),
			expected: factory(operation.WeightMultiply,
				fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}})),
		},
		{
			Base:    testutils.Base{Name: "build weight, random default"},
			builder: newBuilder(operation.WeightMultiply).InputsCount(2).NeuronsCount(3),
		},
		{
			Base: testutils.Base{Name: "build weight, random specified default"},
			builder: newBuilder(operation.WeightMultiply).InputsCount(2).
				NeuronsCount(3).ParamInitType(operation.DefaultInit),
		},
		{
			Base: testutils.Base{Name: "build weight, random glorot"},
			builder: newBuilder(operation.WeightMultiply).InputsCount(2).
				NeuronsCount(3).ParamInitType(operation.GlorotInit),
		},
		{
			Base:    testutils.Base{Name: "build weight, random no inputs count", Err: operation.ErrBuilder},
			builder: newBuilder(operation.WeightMultiply).NeuronsCount(3),
		},
		{
			Base:    testutils.Base{Name: "build weight, random no neurons count", Err: operation.ErrBuilder},
			builder: newBuilder(operation.WeightMultiply).InputsCount(3),
		},
		{
			Base:    testutils.Base{Name: "build weight, random no inputs&neurons count", Err: operation.ErrBuilder},
			builder: newBuilder(operation.WeightMultiply),
		},
		{
			Base: testutils.Base{Name: "build activation with extra parameters"},
			builder: newBuilder(operation.TanhActivation).
				InputsCount(45).
				KeepProbability(percent.Percent50).
				ParamInitType(operation.GlorotInit),
			expected: factory(operation.TanhActivation),
		},
	}

	for _, tc := range testcases {
		t.Run(tc.Name, func(t *testing.T) {
			op, err := tc.builder.Build()
			if tc.Err == nil {
				require.NoError(t, err)
				if tc.expected != nil {
					require.True(t, op.EqualApprox(tc.expected))
				} else {
					t.Log("\n" + op.PrettyString())
				}
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, tc.Err)
			}
		})
	}
}
