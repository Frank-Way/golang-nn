package operation

import (
	"github.com/stretchr/testify/require"
	"nn/internal/nn"
	"nn/internal/testutils"
	"nn/internal/testutils/testfactories"
	"nn/pkg/percent"
	"testing"
)

func TestNewBuilder(t *testing.T) {
	kinds := []nn.Kind{
		LinearActivation,
		SigmoidActivation,
		TanhActivation,
		SigmoidParamActivation,
		Dropout,
		WeightMultiply,
		BiasAdd,
		"unknown kind",
	}
	pivot := 7
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
	newBuilder := func(kind nn.Kind) *Builder {
		res, err := NewBuilder(kind)
		require.NoError(t, err)
		return res
	}
	factory := func(kind nn.Kind, args ...interface{}) IOperation {
		res, err := Create(kind, args...)
		require.NoError(t, err)
		return res
	}
	testcases := []struct {
		testutils.Base
		builder  *Builder
		expected IOperation
	}{
		{
			Base:     testutils.Base{Name: "build linear activation"},
			builder:  newBuilder(LinearActivation),
			expected: factory(LinearActivation),
		},
		{
			Base:     testutils.Base{Name: "build tanh activation"},
			builder:  newBuilder(TanhActivation),
			expected: factory(TanhActivation),
		},
		{
			Base:     testutils.Base{Name: "build sigmoid activation"},
			builder:  newBuilder(SigmoidActivation),
			expected: factory(SigmoidActivation),
		},
		{
			Base: testutils.Base{Name: "build sigmoid param activation, coeffs"},
			builder: newBuilder(SigmoidParamActivation).
				SigmoidCoeffs(testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{1, 2, 3, 4}})),
			expected: factory(SigmoidParamActivation,
				testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{1, 2, 3, 4}})),
		},
		{
			Base: testutils.Base{Name: "build sigmoid param activation, default range"},
			builder: newBuilder(SigmoidParamActivation).
				NeuronsCount(4),
			expected: factory(SigmoidParamActivation,
				testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{1, 2, 3, 4}})),
		},
		{
			Base: testutils.Base{Name: "build sigmoid param activation, range"},
			builder: newBuilder(SigmoidParamActivation).
				NeuronsCount(4).
				SigmoidCoeffsRange(&SigmoidCoeffsRange{Left: 5, Right: 8}),
			expected: factory(SigmoidParamActivation,
				testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{5, 6, 7, 8}})),
		},
		{
			Base: testutils.Base{Name: "build sigmoid param activation, not enough arguments", Err: ErrBuilder},
			builder: newBuilder(SigmoidParamActivation).
				SigmoidCoeffsRange(&SigmoidCoeffsRange{Left: 5, Right: 8}),
		},
		{
			Base:     testutils.Base{Name: "build dropout, 50%"},
			builder:  newBuilder(Dropout).KeepProbability(percent.Percent50),
			expected: factory(Dropout, percent.Percent50),
		},
		{
			Base:     testutils.Base{Name: "build dropout, default value"},
			builder:  newBuilder(Dropout),
			expected: factory(Dropout, percent.Percent100),
		},
		{
			Base: testutils.Base{Name: "build bias, biases"},
			builder: newBuilder(BiasAdd).
				Bias(testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{1, 2, 3}})),
			expected: factory(BiasAdd, testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{1, 2, 3}})),
		},
		{
			Base:    testutils.Base{Name: "build bias, random default"},
			builder: newBuilder(BiasAdd).NeuronsCount(3),
		},
		{
			Base:    testutils.Base{Name: "build bias, random specified default"},
			builder: newBuilder(BiasAdd).NeuronsCount(3).ParamInitType(DefaultInit),
		},
		{
			Base:    testutils.Base{Name: "build bias, random glorot"},
			builder: newBuilder(BiasAdd).NeuronsCount(3).InputsCount(2).ParamInitType(GlorotInit),
		},
		{
			Base:    testutils.Base{Name: "build bias, random glorot no inputs count", Err: ErrBuilder},
			builder: newBuilder(BiasAdd).NeuronsCount(3).ParamInitType(GlorotInit),
		},
		{
			Base:    testutils.Base{Name: "build bias, random no inputs count", Err: ErrBuilder},
			builder: newBuilder(BiasAdd),
		},
		{
			Base: testutils.Base{Name: "build weight, weights"},
			builder: newBuilder(WeightMultiply).
				Weight(testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}})),
			expected: factory(WeightMultiply,
				testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}})),
		},
		{
			Base:    testutils.Base{Name: "build weight, random default"},
			builder: newBuilder(WeightMultiply).InputsCount(2).NeuronsCount(3),
		},
		{
			Base: testutils.Base{Name: "build weight, random specified default"},
			builder: newBuilder(WeightMultiply).InputsCount(2).
				NeuronsCount(3).ParamInitType(DefaultInit),
		},
		{
			Base: testutils.Base{Name: "build weight, random glorot"},
			builder: newBuilder(WeightMultiply).InputsCount(2).
				NeuronsCount(3).ParamInitType(GlorotInit),
		},
		{
			Base:    testutils.Base{Name: "build weight, random no inputs count", Err: ErrBuilder},
			builder: newBuilder(WeightMultiply).NeuronsCount(3),
		},
		{
			Base:    testutils.Base{Name: "build weight, random no neurons count", Err: ErrBuilder},
			builder: newBuilder(WeightMultiply).InputsCount(3),
		},
		{
			Base:    testutils.Base{Name: "build weight, random no inputs&neurons count", Err: ErrBuilder},
			builder: newBuilder(WeightMultiply),
		},
		{
			Base: testutils.Base{Name: "build activation with extra parameters"},
			builder: newBuilder(TanhActivation).
				InputsCount(45).
				KeepProbability(percent.Percent50).
				ParamInitType(GlorotInit),
			expected: factory(TanhActivation),
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
