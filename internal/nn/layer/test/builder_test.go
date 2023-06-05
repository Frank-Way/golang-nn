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
		layer.DenseLayer,
		layer.DenseDropLayer,
		"unknown kind",
		operation.LinearActivation,
		operation.SigmoidActivation,
		operation.TanhActivation,
		operation.SigmoidParamActivation,
		operation.Dropout,
		operation.WeightMultiply,
		operation.BiasAdd,
		loss.MSELoss,
		net.FFNetwork,
	}
	pivot := 2
	for i, kind := range kinds {
		_, err := layer.NewBuilder(kind)
		if i < pivot {
			require.NoError(t, err)
		} else {
			require.Error(t, err)
			require.ErrorIs(t, err, layer.ErrBuilder)
		}
	}
}

func TestBuilder_Build(t *testing.T) {
	newBuilder := func(kind nn.Kind, activationKind nn.Kind) *layer.Builder {
		res, err := layer.NewBuilder(kind)
		require.NoError(t, err)
		return res.ActivationKind(activationKind)
	}
	//factory := func(kind nn.Kind, args ...interface{}) layer.ILayer {
	//	res, err := layer.Create(kind, args...)
	//	require.NoError(t, err)
	//	return res
	//}
	//fabrics.NewOperation := func(kind nn.Kind, args ...interface{}) operation.IOperation {
	//	res, err := operation.Create(kind, args...)
	//	require.NoError(t, err)
	//	return res
	//}
	testcases := []struct {
		testutils.Base
		builder   *layer.Builder
		expected  layer.ILayer
		skipCheck bool
	}{
		{
			Base: testutils.Base{Name: "dense, all params"},
			builder: newBuilder(layer.DenseLayer, operation.SigmoidParamActivation).
				Weight(fabrics.NewOperation(t, operation.WeightMultiply, fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}}))).
				Bias(fabrics.NewOperation(t, operation.BiasAdd, fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{4, 5, 6}}))).
				Activation(fabrics.NewOperation(t, operation.SigmoidParamActivation, fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{1, 2, 3}}))),
			expected: fabrics.NewLayer(t, layer.DenseLayer,
				fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}}),
				fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{4, 5, 6}}),
				fabrics.NewOperation(t, operation.SigmoidParamActivation, fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{1, 2, 3}})),
			),
		},
		{
			Base: testutils.Base{Name: "dense, miss activation", Err: layer.ErrBuilder},
			builder: newBuilder(layer.DenseLayer, operation.SigmoidParamActivation).
				Weight(fabrics.NewOperation(t, operation.WeightMultiply, fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}}))).
				Bias(fabrics.NewOperation(t, operation.BiasAdd, fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{4, 5, 6}}))),
		},
		{
			Base: testutils.Base{Name: "dense, miss bias", Err: layer.ErrBuilder},
			builder: newBuilder(layer.DenseLayer, operation.SigmoidParamActivation).
				Weight(fabrics.NewOperation(t, operation.WeightMultiply, fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}}))).
				Activation(fabrics.NewOperation(t, operation.SigmoidParamActivation, fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{1, 2, 3}}))),
		},
		{
			Base: testutils.Base{Name: "dense, miss weight", Err: layer.ErrBuilder},
			builder: newBuilder(layer.DenseLayer, operation.SigmoidParamActivation).
				Bias(fabrics.NewOperation(t, operation.BiasAdd, fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{4, 5, 6}}))).
				Activation(fabrics.NewOperation(t, operation.SigmoidParamActivation, fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{1, 2, 3}}))),
		},
		{
			Base: testutils.Base{Name: "nil builder", Err: layer.ErrBuilder},
		},
		{
			Base: testutils.Base{Name: "dense, build weight"},
			builder: newBuilder(layer.DenseLayer, operation.SigmoidParamActivation).
				InputsCount(2).NeuronsCount(3).ParamInitType(operation.GlorotInit).
				Bias(fabrics.NewOperation(t, operation.BiasAdd, fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{4, 5, 6}}))).
				Activation(fabrics.NewOperation(t, operation.SigmoidParamActivation, fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{1, 2, 3}}))),
			skipCheck: true,
		},
		{
			Base: testutils.Base{Name: "dense, build bias"},
			builder: newBuilder(layer.DenseLayer, operation.SigmoidParamActivation).
				Weight(fabrics.NewOperation(t, operation.WeightMultiply, fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}}))).
				NeuronsCount(3).
				Activation(fabrics.NewOperation(t, operation.SigmoidParamActivation, fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{1, 2, 3}}))),
			skipCheck: true,
		},
		{
			Base: testutils.Base{Name: "dense, build activation"},
			builder: newBuilder(layer.DenseLayer, operation.SigmoidParamActivation).
				Weight(fabrics.NewOperation(t, operation.WeightMultiply, fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}}))).
				Bias(fabrics.NewOperation(t, operation.BiasAdd, fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{4, 5, 6}}))).
				NeuronsCount(3).SigmoidCoeffsRange(&operation.SigmoidCoeffsRange{Left: 1, Right: 2}),
			expected: fabrics.NewLayer(t, layer.DenseLayer,
				fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}}),
				fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{4, 5, 6}}),
				fabrics.NewOperation(t, operation.SigmoidParamActivation, fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{1, 1.5, 2}})),
			),
		},
		{
			Base: testutils.Base{Name: "dense, build another activation"},
			builder: newBuilder(layer.DenseLayer, operation.SigmoidParamActivation).
				Weight(fabrics.NewOperation(t, operation.WeightMultiply, fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}}))).
				Bias(fabrics.NewOperation(t, operation.BiasAdd, fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{4, 5, 6}}))).
				SigmoidCoeffs(fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{4, 5, 6}})),
			expected: fabrics.NewLayer(t, layer.DenseLayer,
				fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}}),
				fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{4, 5, 6}}),
				fabrics.NewOperation(t, operation.SigmoidParamActivation, fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{4, 5, 6}})),
			),
		},
		{
			Base: testutils.Base{Name: "dense, wrong weight size", Err: layer.ErrBuilder},
			builder: newBuilder(layer.DenseLayer, operation.SigmoidParamActivation).
				Weight(fabrics.NewOperation(t, operation.WeightMultiply, fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 2, Values: []float64{1, 2, 3, 4}}))).
				Bias(fabrics.NewOperation(t, operation.BiasAdd, fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{4, 5, 6}}))).
				Activation(fabrics.NewOperation(t, operation.SigmoidParamActivation, fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{1, 2, 3}}))),
		},
		{
			Base: testutils.Base{Name: "dense, wrong bias size", Err: layer.ErrBuilder},
			builder: newBuilder(layer.DenseLayer, operation.SigmoidParamActivation).
				Weight(fabrics.NewOperation(t, operation.WeightMultiply, fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}}))).
				Bias(fabrics.NewOperation(t, operation.BiasAdd, fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{4, 5}}))).
				Activation(fabrics.NewOperation(t, operation.SigmoidParamActivation, fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{1, 2, 3}}))),
		},
		{
			Base: testutils.Base{Name: "dense, wrong activation size", Err: layer.ErrBuilder},
			builder: newBuilder(layer.DenseLayer, operation.SigmoidParamActivation).
				Weight(fabrics.NewOperation(t, operation.WeightMultiply, fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}}))).
				Bias(fabrics.NewOperation(t, operation.BiasAdd, fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{4, 5, 6}}))).
				Activation(fabrics.NewOperation(t, operation.SigmoidParamActivation, fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{1, 2}}))),
		},
		{
			Base: testutils.Base{Name: "dense drop, all params"},
			builder: newBuilder(layer.DenseDropLayer, operation.SigmoidParamActivation).
				Weight(fabrics.NewOperation(t, operation.WeightMultiply, fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}}))).
				Bias(fabrics.NewOperation(t, operation.BiasAdd, fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{4, 5, 6}}))).
				Activation(fabrics.NewOperation(t, operation.SigmoidParamActivation, fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{1, 2, 3}}))).
				Dropout(fabrics.NewOperation(t, operation.Dropout, percent.Percent50)),
			expected: fabrics.NewLayer(t, layer.DenseDropLayer,
				fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}}),
				fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{4, 5, 6}}),
				fabrics.NewOperation(t, operation.SigmoidParamActivation, fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{1, 2, 3}})),
				percent.Percent50,
			),
		},
		{
			Base: testutils.Base{Name: "dense drop, build default dropout"},
			builder: newBuilder(layer.DenseDropLayer, operation.SigmoidParamActivation).
				Weight(fabrics.NewOperation(t, operation.WeightMultiply, fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}}))).
				Bias(fabrics.NewOperation(t, operation.BiasAdd, fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{4, 5, 6}}))).
				Activation(fabrics.NewOperation(t, operation.SigmoidParamActivation, fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{1, 2, 3}}))),
			expected: fabrics.NewLayer(t, layer.DenseDropLayer,
				fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}}),
				fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{4, 5, 6}}),
				fabrics.NewOperation(t, operation.SigmoidParamActivation, fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{1, 2, 3}})),
				percent.Percent100,
			),
		},
		{
			Base: testutils.Base{Name: "dense drop, build dropout"},
			builder: newBuilder(layer.DenseDropLayer, operation.SigmoidParamActivation).
				Weight(fabrics.NewOperation(t, operation.WeightMultiply, fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}}))).
				Bias(fabrics.NewOperation(t, operation.BiasAdd, fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{4, 5, 6}}))).
				Activation(fabrics.NewOperation(t, operation.SigmoidParamActivation, fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{1, 2, 3}}))).
				KeepProbability(percent.Percent50),
			expected: fabrics.NewLayer(t, layer.DenseDropLayer,
				fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}}),
				fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{4, 5, 6}}),
				fabrics.NewOperation(t, operation.SigmoidParamActivation, fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{1, 2, 3}})),
				percent.Percent50,
			),
		},
		{
			Base: testutils.Base{Name: "dense drop, fail on build base", Err: layer.ErrBuilder},
			builder: newBuilder(layer.DenseDropLayer, operation.SigmoidParamActivation).
				Weight(fabrics.NewOperation(t, operation.WeightMultiply, fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 2, Values: []float64{1, 2, 3, 4}}))).
				Bias(fabrics.NewOperation(t, operation.BiasAdd, fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{4, 5, 6}}))).
				Activation(fabrics.NewOperation(t, operation.SigmoidParamActivation, fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{1, 2, 3}}))),
		},
	}

	for _, tc := range testcases {
		t.Run(tc.Name, func(t *testing.T) {
			l, err := tc.builder.Build()
			if tc.Err == nil {
				require.NoError(t, err)
				if !tc.skipCheck {
					require.True(t, l.EqualApprox(tc.expected))
				}
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, tc.Err)
			}
		})
	}
}
