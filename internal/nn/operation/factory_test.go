package operation

import (
	"github.com/stretchr/testify/require"
	"nn/internal/nn"
	"nn/internal/testutils"
	"nn/internal/testutils/testfactories"
	"nn/pkg/percent"
	"testing"
)

func TestCreate(t *testing.T) {
	type testcase struct {
		testutils.Base
		kind     nn.Kind
		args     []interface{}
		expected IOperation
	}
	testcases := make([]testcase, 0)
	testcases = append(testcases,
		testcase{
			Base:     testutils.Base{Name: "create linear activation"},
			kind:     LinearActivation,
			expected: NewLinearActivation(),
		},
		testcase{
			Base:     testutils.Base{Name: "create sigmoid activation"},
			kind:     SigmoidActivation,
			expected: NewSigmoidActivation(),
		},
		testcase{
			Base:     testutils.Base{Name: "create tanh activation"},
			kind:     TanhActivation,
			expected: NewTanhActivation(),
		},
	)
	o, err := NewSigmoidParam(testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{1, 2, 3, 4, 5}}))
	require.NoError(t, err)
	testcases = append(testcases,
		testcase{
			Base:     testutils.Base{Name: "create param sigmoid activation"},
			kind:     SigmoidParamActivation,
			args:     []interface{}{testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{1, 2, 3, 4, 5}})},
			expected: o,
		},
		testcase{
			Base: testutils.Base{Name: "create param sigmoid activation, no args", Err: ErrFabric},
			kind: SigmoidParamActivation,
		},
		testcase{
			Base: testutils.Base{Name: "create param sigmoid activation, empty args", Err: ErrFabric},
			kind: SigmoidParamActivation,
			args: []interface{}{},
		},
		testcase{
			Base: testutils.Base{Name: "create param sigmoid activation, wrong args", Err: ErrFabric},
			kind: SigmoidParamActivation,
			args: []interface{}{[]float64{1, 2, 3, 4, 5}},
		},
	)
	o, err = NewDropout(percent.Percent50)
	require.NoError(t, err)
	testcases = append(testcases,
		testcase{
			Base:     testutils.Base{Name: "create dropout"},
			kind:     Dropout,
			args:     []interface{}{percent.Percent50},
			expected: o,
		},
		testcase{
			Base: testutils.Base{Name: "create dropout, no args", Err: ErrFabric},
			kind: Dropout,
		},
		testcase{
			Base: testutils.Base{Name: "create dropout, empty args", Err: ErrFabric},
			kind: Dropout,
			args: []interface{}{},
		},
		testcase{
			Base: testutils.Base{Name: "create dropout, wrong args", Err: ErrFabric},
			kind: Dropout,
			args: []interface{}{50},
		},
	)
	o, err = NewWeightOperation(testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 2, Values: []float64{1, 2, 3, 4}}))
	require.NoError(t, err)
	testcases = append(testcases,
		testcase{
			Base:     testutils.Base{Name: "create weight"},
			kind:     WeightMultiply,
			args:     []interface{}{testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 2, Values: []float64{1, 2, 3, 4}})},
			expected: o,
		},
		testcase{
			Base: testutils.Base{Name: "create weight, no args", Err: ErrFabric},
			kind: WeightMultiply,
		},
		testcase{
			Base: testutils.Base{Name: "create weight, empty args", Err: ErrFabric},
			kind: WeightMultiply,
			args: []interface{}{},
		},
		testcase{
			Base: testutils.Base{Name: "create weight, wrong args", Err: ErrFabric},
			kind: WeightMultiply,
			args: []interface{}{50},
		},
	)
	o, err = NewBiasOperation(testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{1, 2, 3, 4}}))
	require.NoError(t, err)
	testcases = append(testcases,
		testcase{
			Base:     testutils.Base{Name: "create bias"},
			kind:     BiasAdd,
			args:     []interface{}{testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{1, 2, 3, 4}})},
			expected: o,
		},
		testcase{
			Base: testutils.Base{Name: "create bias, no args", Err: ErrFabric},
			kind: BiasAdd,
		},
		testcase{
			Base: testutils.Base{Name: "create bias, empty args", Err: ErrFabric},
			kind: BiasAdd,
			args: []interface{}{},
		},
		testcase{
			Base: testutils.Base{Name: "create bias, wrong args", Err: ErrFabric},
			kind: BiasAdd,
			args: []interface{}{50},
		},
		testcase{
			Base: testutils.Base{Name: "unknown operation", Err: ErrFabric},
			kind: "unknown operation",
		},
	)

	for _, tc := range testcases {
		t.Run(tc.Name, func(t *testing.T) {
			o, err := Create(tc.kind, tc.args...)
			if tc.Err == nil {
				require.NoError(t, err)
				require.True(t, o.EqualApprox(tc.expected))
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, tc.Err)
			}
		})
	}
}

func TestIs(t *testing.T) {
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

	testcases := []struct {
		testutils.Base
		op     IOperation
		hitIdx int
	}{
		{
			Base:   testutils.Base{Name: string(LinearActivation)},
			op:     newOperation(t, LinearActivation),
			hitIdx: 0,
		},
		{
			Base:   testutils.Base{Name: string(SigmoidActivation)},
			op:     newOperation(t, SigmoidActivation),
			hitIdx: 1,
		},
		{
			Base:   testutils.Base{Name: string(TanhActivation)},
			op:     newOperation(t, TanhActivation),
			hitIdx: 2,
		},
		{
			Base:   testutils.Base{Name: string(SigmoidParamActivation)},
			op:     newOperation(t, SigmoidParamActivation, testfactories.NewVector(t, testfactories.VectorParameters{Size: 5})),
			hitIdx: 3,
		},
		{
			Base:   testutils.Base{Name: string(Dropout)},
			op:     newOperation(t, Dropout, percent.Percent50),
			hitIdx: 4,
		},
		{
			Base:   testutils.Base{Name: string(WeightMultiply)},
			op:     newOperation(t, WeightMultiply, testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 2})),
			hitIdx: 5,
		},
		{
			Base:   testutils.Base{Name: string(BiasAdd)},
			op:     newOperation(t, BiasAdd, testfactories.NewVector(t, testfactories.VectorParameters{Size: 5})),
			hitIdx: 6,
		},
	}

	for _, tc := range testcases {
		t.Run(tc.Name, func(t *testing.T) {
			for idx, kind := range kinds {
				if tc.op.Is(kind) {
					require.Equal(t, tc.hitIdx, idx)
				} else {
					require.NotEqual(t, tc.hitIdx, idx)
				}
			}
		})
	}
}
