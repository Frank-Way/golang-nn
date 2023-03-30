package test

import (
	"github.com/stretchr/testify/require"
	"nn/internal/nn"
	"nn/internal/nn/operation"
	"nn/internal/testutils"
	"nn/internal/testutils/fabrics"
	"nn/pkg/percent"
	"testing"
)

func TestCreate(t *testing.T) {
	type testcase struct {
		testutils.Base
		kind     nn.Kind
		args     []interface{}
		expected operation.IOperation
	}
	testcases := make([]testcase, 0)
	testcases = append(testcases,
		testcase{
			Base:     testutils.Base{Name: "create linear activation"},
			kind:     operation.LinearActivation,
			expected: operation.NewLinearActivation(),
		},
		testcase{
			Base:     testutils.Base{Name: "create sigmoid activation"},
			kind:     operation.SigmoidActivation,
			expected: operation.NewSigmoidActivation(),
		},
		testcase{
			Base:     testutils.Base{Name: "create tanh activation"},
			kind:     operation.TanhActivation,
			expected: operation.NewTanhActivation(),
		},
	)
	o, err := operation.NewSigmoidParam(fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{1, 2, 3, 4, 5}}))
	require.NoError(t, err)
	testcases = append(testcases,
		testcase{
			Base:     testutils.Base{Name: "create param sigmoid activation"},
			kind:     operation.SigmoidParamActivation,
			args:     []interface{}{fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{1, 2, 3, 4, 5}})},
			expected: o,
		},
		testcase{
			Base: testutils.Base{Name: "create param sigmoid activation, no args", Err: operation.ErrCreate},
			kind: operation.SigmoidParamActivation,
		},
		testcase{
			Base: testutils.Base{Name: "create param sigmoid activation, empty args", Err: operation.ErrCreate},
			kind: operation.SigmoidParamActivation,
			args: []interface{}{},
		},
		testcase{
			Base: testutils.Base{Name: "create param sigmoid activation, wrong args", Err: operation.ErrCreate},
			kind: operation.SigmoidParamActivation,
			args: []interface{}{[]float64{1, 2, 3, 4, 5}},
		},
	)
	o, err = operation.NewDropout(percent.Percent50)
	require.NoError(t, err)
	testcases = append(testcases,
		testcase{
			Base:     testutils.Base{Name: "create dropout"},
			kind:     operation.Dropout,
			args:     []interface{}{percent.Percent50},
			expected: o,
		},
		testcase{
			Base: testutils.Base{Name: "create dropout, no args", Err: operation.ErrCreate},
			kind: operation.Dropout,
		},
		testcase{
			Base: testutils.Base{Name: "create dropout, empty args", Err: operation.ErrCreate},
			kind: operation.Dropout,
			args: []interface{}{},
		},
		testcase{
			Base: testutils.Base{Name: "create dropout, wrong args", Err: operation.ErrCreate},
			kind: operation.Dropout,
			args: []interface{}{50},
		},
	)
	o, err = operation.NewWeightOperation(fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 2, Values: []float64{1, 2, 3, 4}}))
	require.NoError(t, err)
	testcases = append(testcases,
		testcase{
			Base:     testutils.Base{Name: "create weight"},
			kind:     operation.WeightMultiply,
			args:     []interface{}{fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 2, Values: []float64{1, 2, 3, 4}})},
			expected: o,
		},
		testcase{
			Base: testutils.Base{Name: "create weight, no args", Err: operation.ErrCreate},
			kind: operation.WeightMultiply,
		},
		testcase{
			Base: testutils.Base{Name: "create weight, empty args", Err: operation.ErrCreate},
			kind: operation.WeightMultiply,
			args: []interface{}{},
		},
		testcase{
			Base: testutils.Base{Name: "create weight, wrong args", Err: operation.ErrCreate},
			kind: operation.WeightMultiply,
			args: []interface{}{50},
		},
	)
	o, err = operation.NewBiasOperation(fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{1, 2, 3, 4}}))
	require.NoError(t, err)
	testcases = append(testcases,
		testcase{
			Base:     testutils.Base{Name: "create bias"},
			kind:     operation.BiasAdd,
			args:     []interface{}{fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{1, 2, 3, 4}})},
			expected: o,
		},
		testcase{
			Base: testutils.Base{Name: "create bias, no args", Err: operation.ErrCreate},
			kind: operation.BiasAdd,
		},
		testcase{
			Base: testutils.Base{Name: "create bias, empty args", Err: operation.ErrCreate},
			kind: operation.BiasAdd,
			args: []interface{}{},
		},
		testcase{
			Base: testutils.Base{Name: "create bias, wrong args", Err: operation.ErrCreate},
			kind: operation.BiasAdd,
			args: []interface{}{50},
		},
		testcase{
			Base: testutils.Base{Name: "unknown operation", Err: operation.ErrCreate},
			kind: "unknown operation",
		},
	)

	for _, tc := range testcases {
		t.Run(tc.Name, func(t *testing.T) {
			o, err := operation.Create(tc.kind, tc.args...)
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
		operation.LinearActivation,
		operation.SigmoidActivation,
		operation.TanhActivation,
		operation.SigmoidParamActivation,
		operation.Dropout,
		operation.WeightMultiply,
		operation.BiasAdd,
		"unknown kind",
	}

	testcases := []struct {
		testutils.Base
		op     operation.IOperation
		hitIdx int
	}{
		{
			Base:   testutils.Base{Name: string(operation.LinearActivation)},
			op:     fabrics.NewOperation(t, operation.LinearActivation),
			hitIdx: 0,
		},
		{
			Base:   testutils.Base{Name: string(operation.SigmoidActivation)},
			op:     fabrics.NewOperation(t, operation.SigmoidActivation),
			hitIdx: 1,
		},
		{
			Base:   testutils.Base{Name: string(operation.TanhActivation)},
			op:     fabrics.NewOperation(t, operation.TanhActivation),
			hitIdx: 2,
		},
		{
			Base:   testutils.Base{Name: string(operation.SigmoidParamActivation)},
			op:     fabrics.NewOperation(t, operation.SigmoidParamActivation, fabrics.NewVector(t, fabrics.VectorParameters{Size: 5})),
			hitIdx: 3,
		},
		{
			Base:   testutils.Base{Name: string(operation.Dropout)},
			op:     fabrics.NewOperation(t, operation.Dropout, percent.Percent50),
			hitIdx: 4,
		},
		{
			Base:   testutils.Base{Name: string(operation.WeightMultiply)},
			op:     fabrics.NewOperation(t, operation.WeightMultiply, fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 2})),
			hitIdx: 5,
		},
		{
			Base:   testutils.Base{Name: string(operation.BiasAdd)},
			op:     fabrics.NewOperation(t, operation.BiasAdd, fabrics.NewVector(t, fabrics.VectorParameters{Size: 5})),
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
