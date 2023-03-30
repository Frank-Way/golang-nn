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
	testcases := []struct {
		testutils.Base
		kind     nn.Kind
		args     []interface{}
		expected operation.IOperation
	}{
		{Base: testutils.Base{Name: "create linear activation"},
			kind:     operation.LinearActivation,
			expected: fabrics.NewActivation(t, fabrics.LinearAct, fabrics.ActivationParameters{}),
		},
		{Base: testutils.Base{Name: "create sigmoid activation"},
			kind:     operation.SigmoidActivation,
			expected: fabrics.NewActivation(t, fabrics.SigmoidAct, fabrics.ActivationParameters{}),
		},
		{Base: testutils.Base{Name: "create tanh activation"},
			kind:     operation.TanhActivation,
			expected: fabrics.NewActivation(t, fabrics.TanhAct, fabrics.ActivationParameters{}),
		},
		{Base: testutils.Base{Name: "create param sigmoid activation"},
			kind: operation.SigmoidParamActivation,
			args: []interface{}{fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{1, 2, 3, 4, 5}})},
			expected: fabrics.NewActivation(t, fabrics.SigmoidParamAct, fabrics.ActivationParameters{
				SigmoidParamParameters: fabrics.VectorParameters{Values: []float64{1, 2, 3, 4, 5}},
			}),
		},
		{Base: testutils.Base{Name: "create param sigmoid activation, no args", Err: operation.ErrCreate},
			kind: operation.SigmoidParamActivation,
		},
		{Base: testutils.Base{Name: "create param sigmoid activation, empty args", Err: operation.ErrCreate},
			args: []interface{}{},
			kind: operation.SigmoidParamActivation,
		},
		{Base: testutils.Base{Name: "create param sigmoid activation, wrong args", Err: operation.ErrCreate},
			kind: operation.SigmoidParamActivation,
			args: []interface{}{[]float64{1, 2, 3, 4, 5}},
		},
		{Base: testutils.Base{Name: "create dropout"},
			kind:     operation.Dropout,
			args:     []interface{}{percent.Percent10},
			expected: fabrics.NewDropout(t, fabrics.DropoutParameters{Percent: percent.Percent10}),
		},
		{Base: testutils.Base{Name: "create dropout, no args", Err: operation.ErrCreate},
			kind: operation.Dropout,
		},
		{Base: testutils.Base{Name: "create dropout, empty args", Err: operation.ErrCreate},
			kind: operation.Dropout,
			args: []interface{}{},
		},
		{Base: testutils.Base{Name: "create dropout, wrong args", Err: operation.ErrCreate},
			kind: operation.Dropout,
			args: []interface{}{10},
		},
		{Base: testutils.Base{Name: "create weight"},
			kind: operation.WeightMultiply,
			args: []interface{}{fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 2, Values: []float64{1, 2, 3, 4}})},
			expected: fabrics.NewWeight(t, fabrics.WeightParameters{
				MatrixParameters: fabrics.MatrixParameters{Rows: 2, Cols: 2, Values: []float64{1, 2, 3, 4}},
			}),
		},
		{Base: testutils.Base{Name: "create weight, no args", Err: operation.ErrCreate},
			kind: operation.WeightMultiply,
		},
		{Base: testutils.Base{Name: "create weight, empty args", Err: operation.ErrCreate},
			kind: operation.WeightMultiply,
			args: []interface{}{},
		},
		{Base: testutils.Base{Name: "create weight, wrong args", Err: operation.ErrCreate},
			kind: operation.WeightMultiply,
			args: []interface{}{fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{1, 2, 3, 4}})},
		},
		{Base: testutils.Base{Name: "create bias"},
			kind: operation.BiasAdd,
			args: []interface{}{fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{1, 2, 3, 4, 5}})},
			expected: fabrics.NewBias(t, fabrics.BiasParameters{
				VectorParameters: fabrics.VectorParameters{Values: []float64{1, 2, 3, 4, 5}},
			}),
		},
		{Base: testutils.Base{Name: "create bias, no args", Err: operation.ErrCreate},
			kind: operation.BiasAdd,
		},
		{Base: testutils.Base{Name: "create bias, empty args", Err: operation.ErrCreate},
			kind: operation.BiasAdd,
			args: []interface{}{},
		},
		{Base: testutils.Base{Name: "create bias, wrong args", Err: operation.ErrCreate},
			kind: operation.BiasAdd,
			args: []interface{}{fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 2})},
		},
		{
			Base: testutils.Base{Name: "unknown operation", Err: operation.ErrCreate},
			kind: "abcde",
		},
	}

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
			op:     fabrics.NewActivation(t, fabrics.LinearAct, fabrics.ActivationParameters{}),
			hitIdx: 0,
		},
		{
			Base:   testutils.Base{Name: string(operation.SigmoidActivation)},
			op:     fabrics.NewActivation(t, fabrics.SigmoidAct, fabrics.ActivationParameters{}),
			hitIdx: 1,
		},
		{
			Base:   testutils.Base{Name: string(operation.TanhActivation)},
			op:     fabrics.NewActivation(t, fabrics.TanhAct, fabrics.ActivationParameters{}),
			hitIdx: 2,
		},
		{
			Base: testutils.Base{Name: string(operation.SigmoidParamActivation)},
			op: fabrics.NewActivation(t, fabrics.SigmoidParamAct, fabrics.ActivationParameters{
				SigmoidParamParameters: fabrics.VectorParameters{Size: 5},
			}),
			hitIdx: 3,
		},
		{
			Base:   testutils.Base{Name: string(operation.Dropout)},
			op:     fabrics.NewDropout(t, fabrics.DropoutParameters{Percent: percent.Percent10}),
			hitIdx: 4,
		},
		{
			Base: testutils.Base{Name: string(operation.WeightMultiply)},
			op: fabrics.NewWeight(t, fabrics.WeightParameters{
				MatrixParameters: fabrics.MatrixParameters{Rows: 2, Cols: 2},
			}),
			hitIdx: 5,
		},
		{
			Base: testutils.Base{Name: string(operation.BiasAdd)},
			op: fabrics.NewBias(t, fabrics.BiasParameters{
				VectorParameters: fabrics.VectorParameters{Size: 5},
			}),
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
