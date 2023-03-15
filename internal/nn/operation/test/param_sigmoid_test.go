package test

import (
	"github.com/stretchr/testify/require"
	"nn/internal/nn/operation"
	"nn/internal/testutils"
	"nn/internal/testutils/fabrics"
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
		params fabrics.ActivationParameters
		in     fabrics.MatrixParameters
	}{
		{
			Base:   testutils.Base{Name: "5x1 input, [2] params"},
			params: fabrics.ActivationParameters{SigmoidParamParameters: fabrics.VectorParameters{Values: []float64{2}}},
			in:     fabrics.MatrixParameters{Rows: 5, Cols: 1, Values: []float64{1, 2, 3, 4, 5}},
		},
		{
			Base:   testutils.Base{Name: "5x1 input, [2,3] params, error", Err: operation.ErrExec},
			params: fabrics.ActivationParameters{SigmoidParamParameters: fabrics.VectorParameters{Values: []float64{2, 3}}},
			in:     fabrics.MatrixParameters{Rows: 5, Cols: 1, Values: []float64{1, 2, 3, 4, 5}},
		},
		{
			Base:   testutils.Base{Name: "5x2 input, [2] params, error", Err: operation.ErrExec},
			params: fabrics.ActivationParameters{SigmoidParamParameters: fabrics.VectorParameters{Values: []float64{2}}},
			in:     fabrics.MatrixParameters{Rows: 5, Cols: 2, Values: []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}},
		},
	}

	for _, test := range tests {
		t.Run(test.Name, func(t *testing.T) {
			activation := fabrics.NewActivation(t, fabrics.SigmoidParamAct, test.params)
			in := fabrics.NewMatrix(t, test.in)
			out, err := activation.Forward(in)
			if test.Err == nil {
				require.NoError(t, err)
				params := fabrics.NewVector(t, test.params.SigmoidParamParameters)
				multiplied, err := in.MulRow(params)
				require.NoError(t, err)
				sigmoid := fabrics.NewActivation(t, fabrics.SigmoidAct, fabrics.ActivationParameters{})
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

//func TestSigmoidParam_Backward(t *testing.T) {
//	prob := percent.Percent30
//	dropout := fabrics.NewDropout(t, fabrics.DropoutParameters{Percent: prob})
//	in, err := matrix.NewMatrixOf(10, 10, 1)
//	require.NoError(t, err)
//	outGrad := in.Copy()
//	tries := 5
//	for try := 0; try < tries; try++ {
//		out, err := dropout.Forward(in)
//		require.NoError(t, err)
//		inGrad, err := dropout.Backward(outGrad)
//		require.NoError(t, err)
//		// ensure dropping out the same neurons (mask keeps the same until new Forward call)
//		require.True(t, out.Equal(inGrad))
//	}
//}
