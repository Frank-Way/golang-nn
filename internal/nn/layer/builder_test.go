package layer

import (
	"github.com/stretchr/testify/require"
	"nn/internal/nn"
	"nn/internal/nn/loss"
	"nn/internal/nn/operation"
	"nn/internal/nn/operation/operationtestutils"
	"nn/internal/testutils"
	"nn/internal/testutils/testfactories"
	"nn/pkg/percent"
	"testing"
)

func TestNewBuilder(t *testing.T) {
	kinds := []nn.Kind{
		DenseLayer,
		DenseDropLayer,
		"unknown kind",
		operation.LinearActivation,
		operation.SigmoidActivation,
		operation.TanhActivation,
		operation.SigmoidParamActivation,
		operation.Dropout,
		operation.WeightMultiply,
		operation.BiasAdd,
		loss.MSELoss,
	}
	pivot := 2
	for i, kind := range kinds {
		_, err := NewBuilder(kind)
		if i < pivot {
			require.NoError(t, err)
		} else {
			require.Error(t, err)
			require.ErrorIs(t, err, ErrBuilder)
		}
	}
}

func TestBuilder_Build(t *testing.T) {
	newBuilder := func(kind nn.Kind, activationKind nn.Kind) *Builder {
		res, err := NewBuilder(kind)
		require.NoError(t, err)
		return res.ActivationKind(activationKind)
	}
	//factory := func(kind nn.Kind, args ...interface{}) layer.ILayer {
	//	res, err := layer.Create(kind, args...)
	//	require.NoError(t, err)
	//	return res
	//}
	//testfactories.NewOperation := func(kind nn.Kind, args ...interface{}) operation.IOperation {
	//	res, err := operation.Create(kind, args...)
	//	require.NoError(t, err)
	//	return res
	//}
	testcases := []struct {
		testutils.Base
		builder   *Builder
		expected  ILayer
		skipCheck bool
	}{
		{
			Base: testutils.Base{Name: "dense, all params"},
			builder: newBuilder(DenseLayer, operation.SigmoidParamActivation).
				Weight(operationtestutils.NewOperation(t, operation.WeightMultiply, testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}}))).
				Bias(operationtestutils.NewOperation(t, operation.BiasAdd, testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{4, 5, 6}}))).
				Activation(operationtestutils.NewOperation(t, operation.SigmoidParamActivation, testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{1, 2, 3}}))),
			expected: newLayer(t, DenseLayer,
				testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}}),
				testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{4, 5, 6}}),
				operationtestutils.NewOperation(t, operation.SigmoidParamActivation, testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{1, 2, 3}})),
			),
		},
		{
			Base: testutils.Base{Name: "dense, miss activation", Err: ErrBuilder},
			builder: newBuilder(DenseLayer, operation.SigmoidParamActivation).
				Weight(operationtestutils.NewOperation(t, operation.WeightMultiply, testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}}))).
				Bias(operationtestutils.NewOperation(t, operation.BiasAdd, testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{4, 5, 6}}))),
		},
		{
			Base: testutils.Base{Name: "dense, miss bias", Err: ErrBuilder},
			builder: newBuilder(DenseLayer, operation.SigmoidParamActivation).
				Weight(operationtestutils.NewOperation(t, operation.WeightMultiply, testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}}))).
				Activation(operationtestutils.NewOperation(t, operation.SigmoidParamActivation, testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{1, 2, 3}}))),
		},
		{
			Base: testutils.Base{Name: "dense, miss weight", Err: ErrBuilder},
			builder: newBuilder(DenseLayer, operation.SigmoidParamActivation).
				Bias(operationtestutils.NewOperation(t, operation.BiasAdd, testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{4, 5, 6}}))).
				Activation(operationtestutils.NewOperation(t, operation.SigmoidParamActivation, testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{1, 2, 3}}))),
		},
		{
			Base: testutils.Base{Name: "nil builder", Err: ErrBuilder},
		},
		{
			Base: testutils.Base{Name: "dense, build weight"},
			builder: newBuilder(DenseLayer, operation.SigmoidParamActivation).
				InputsCount(2).NeuronsCount(3).ParamInitType(operation.GlorotInit).
				Bias(operationtestutils.NewOperation(t, operation.BiasAdd, testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{4, 5, 6}}))).
				Activation(operationtestutils.NewOperation(t, operation.SigmoidParamActivation, testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{1, 2, 3}}))),
			skipCheck: true,
		},
		{
			Base: testutils.Base{Name: "dense, build bias"},
			builder: newBuilder(DenseLayer, operation.SigmoidParamActivation).
				Weight(operationtestutils.NewOperation(t, operation.WeightMultiply, testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}}))).
				NeuronsCount(3).
				Activation(operationtestutils.NewOperation(t, operation.SigmoidParamActivation, testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{1, 2, 3}}))),
			skipCheck: true,
		},
		{
			Base: testutils.Base{Name: "dense, build activation"},
			builder: newBuilder(DenseLayer, operation.SigmoidParamActivation).
				Weight(operationtestutils.NewOperation(t, operation.WeightMultiply, testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}}))).
				Bias(operationtestutils.NewOperation(t, operation.BiasAdd, testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{4, 5, 6}}))).
				NeuronsCount(3).SigmoidCoeffsRange(&operation.SigmoidCoeffsRange{Left: 1, Right: 2}),
			expected: newLayer(t, DenseLayer,
				testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}}),
				testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{4, 5, 6}}),
				operationtestutils.NewOperation(t, operation.SigmoidParamActivation, testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{1, 1.5, 2}})),
			),
		},
		{
			Base: testutils.Base{Name: "dense, build another activation"},
			builder: newBuilder(DenseLayer, operation.SigmoidParamActivation).
				Weight(operationtestutils.NewOperation(t, operation.WeightMultiply, testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}}))).
				Bias(operationtestutils.NewOperation(t, operation.BiasAdd, testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{4, 5, 6}}))).
				SigmoidCoeffs(testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{4, 5, 6}})),
			expected: newLayer(t, DenseLayer,
				testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}}),
				testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{4, 5, 6}}),
				operationtestutils.NewOperation(t, operation.SigmoidParamActivation, testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{4, 5, 6}})),
			),
		},
		{
			Base: testutils.Base{Name: "dense, wrong weight size", Err: ErrBuilder},
			builder: newBuilder(DenseLayer, operation.SigmoidParamActivation).
				Weight(operationtestutils.NewOperation(t, operation.WeightMultiply, testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 2, Values: []float64{1, 2, 3, 4}}))).
				Bias(operationtestutils.NewOperation(t, operation.BiasAdd, testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{4, 5, 6}}))).
				Activation(operationtestutils.NewOperation(t, operation.SigmoidParamActivation, testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{1, 2, 3}}))),
		},
		{
			Base: testutils.Base{Name: "dense, wrong bias size", Err: ErrBuilder},
			builder: newBuilder(DenseLayer, operation.SigmoidParamActivation).
				Weight(operationtestutils.NewOperation(t, operation.WeightMultiply, testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}}))).
				Bias(operationtestutils.NewOperation(t, operation.BiasAdd, testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{4, 5}}))).
				Activation(operationtestutils.NewOperation(t, operation.SigmoidParamActivation, testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{1, 2, 3}}))),
		},
		{
			Base: testutils.Base{Name: "dense, wrong activation size", Err: ErrBuilder},
			builder: newBuilder(DenseLayer, operation.SigmoidParamActivation).
				Weight(operationtestutils.NewOperation(t, operation.WeightMultiply, testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}}))).
				Bias(operationtestutils.NewOperation(t, operation.BiasAdd, testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{4, 5, 6}}))).
				Activation(operationtestutils.NewOperation(t, operation.SigmoidParamActivation, testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{1, 2}}))),
		},
		{
			Base: testutils.Base{Name: "dense drop, all params"},
			builder: newBuilder(DenseDropLayer, operation.SigmoidParamActivation).
				Weight(operationtestutils.NewOperation(t, operation.WeightMultiply, testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}}))).
				Bias(operationtestutils.NewOperation(t, operation.BiasAdd, testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{4, 5, 6}}))).
				Activation(operationtestutils.NewOperation(t, operation.SigmoidParamActivation, testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{1, 2, 3}}))).
				Dropout(operationtestutils.NewOperation(t, operation.Dropout, percent.Percent50)),
			expected: newLayer(t, DenseDropLayer,
				testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}}),
				testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{4, 5, 6}}),
				operationtestutils.NewOperation(t, operation.SigmoidParamActivation, testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{1, 2, 3}})),
				percent.Percent50,
			),
		},
		{
			Base: testutils.Base{Name: "dense drop, build default dropout"},
			builder: newBuilder(DenseDropLayer, operation.SigmoidParamActivation).
				Weight(operationtestutils.NewOperation(t, operation.WeightMultiply, testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}}))).
				Bias(operationtestutils.NewOperation(t, operation.BiasAdd, testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{4, 5, 6}}))).
				Activation(operationtestutils.NewOperation(t, operation.SigmoidParamActivation, testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{1, 2, 3}}))),
			expected: newLayer(t, DenseDropLayer,
				testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}}),
				testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{4, 5, 6}}),
				operationtestutils.NewOperation(t, operation.SigmoidParamActivation, testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{1, 2, 3}})),
				percent.Percent100,
			),
		},
		{
			Base: testutils.Base{Name: "dense drop, build dropout"},
			builder: newBuilder(DenseDropLayer, operation.SigmoidParamActivation).
				Weight(operationtestutils.NewOperation(t, operation.WeightMultiply, testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}}))).
				Bias(operationtestutils.NewOperation(t, operation.BiasAdd, testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{4, 5, 6}}))).
				Activation(operationtestutils.NewOperation(t, operation.SigmoidParamActivation, testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{1, 2, 3}}))).
				KeepProbability(percent.Percent50),
			expected: newLayer(t, DenseDropLayer,
				testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}}),
				testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{4, 5, 6}}),
				operationtestutils.NewOperation(t, operation.SigmoidParamActivation, testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{1, 2, 3}})),
				percent.Percent50,
			),
		},
		{
			Base: testutils.Base{Name: "dense drop, fail on build base", Err: ErrBuilder},
			builder: newBuilder(DenseDropLayer, operation.SigmoidParamActivation).
				Weight(operationtestutils.NewOperation(t, operation.WeightMultiply, testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 2, Values: []float64{1, 2, 3, 4}}))).
				Bias(operationtestutils.NewOperation(t, operation.BiasAdd, testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{4, 5, 6}}))).
				Activation(operationtestutils.NewOperation(t, operation.SigmoidParamActivation, testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{1, 2, 3}}))),
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
