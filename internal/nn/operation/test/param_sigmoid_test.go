package test

import (
	"github.com/stretchr/testify/require"
	"nn/internal/nn/operation"
	"nn/internal/testutils"
	"nn/internal/testutils/fabrics"
	"nn/pkg/mmath/matrix"
	"testing"
)

func TestNewSigmoidParam(t *testing.T) {
	_, err := operation.NewSigmoidParam(nil)
	require.Error(t, err)
	require.ErrorIs(t, err, operation.ErrCreate)

	_, err = operation.NewSigmoidParam(fabrics.NewVector(t, fabrics.VectorParameters{Size: 5}))
	require.NoError(t, err)
}

func TestSigmoidParam_Forward(t *testing.T) {
	tests := []struct {
		testutils.Base
		act *operation.ConstOperation
		in  *matrix.Matrix
	}{
		{
			Base: testutils.Base{Name: "5x1 input, [2] params"},
			act:  fabrics.NewOperation(t, operation.SigmoidParamActivation, fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{2}})).(*operation.ConstOperation),
			in:   fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 5, Cols: 1, Values: []float64{1, 2, 3, 4, 5}}),
		},
		{
			Base: testutils.Base{Name: "5x1 input, [2,3] params, error", Err: operation.ErrExec},
			act:  fabrics.NewOperation(t, operation.SigmoidParamActivation, fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{2, 3}})).(*operation.ConstOperation),
			in:   fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 5, Cols: 1, Values: []float64{1, 2, 3, 4, 5}}),
		},
		{
			Base: testutils.Base{Name: "5x2 input, [2] params, error", Err: operation.ErrExec},
			act:  fabrics.NewOperation(t, operation.SigmoidParamActivation, fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{2}})).(*operation.ConstOperation),
			in:   fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 5, Cols: 2, Values: []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}}),
		},
	}

	for _, test := range tests {
		t.Run(test.Name, func(t *testing.T) {
			out, err := test.act.Forward(test.in)
			if test.Err == nil {
				require.NoError(t, err)
				params := test.act.Parameters()[0]
				multiplied, err := test.in.MulRowM(params)
				require.NoError(t, err)
				sigmoid := fabrics.NewOperation(t, operation.SigmoidActivation)
				expected, err := sigmoid.Forward(multiplied)
				require.NoError(t, err)
				require.True(t, out.EqualApprox(expected))
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, test.Err)
			}
		})
	}
}
