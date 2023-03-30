package test

import (
	"github.com/stretchr/testify/require"
	"nn/internal/nn/operation"
	"nn/internal/testutils"
	"nn/internal/testutils/fabrics"
	"nn/pkg/mmath/matrix"
	"testing"
)

func TestNewWeightOperation(t *testing.T) {
	tests := []struct {
		testutils.Base
		weight *matrix.Matrix
	}{
		{
			Base:   testutils.Base{Name: "2x8 weight operation"},
			weight: fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 8}),
		},
		{
			Base: testutils.Base{Name: "nil weight operation", Err: operation.ErrCreate},
		},
	}

	for _, test := range tests {
		t.Run(test.Name, func(t *testing.T) {
			_, err := operation.NewWeightOperation(test.weight)
			if test.Err == nil {
				require.NoError(t, err)
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, test.Err)
			}
		})
	}
}

func TestWeight_Forward(t *testing.T) {
	tests := []struct {
		testutils.Base
		in       *matrix.Matrix
		weight   operation.IOperation
		expected *matrix.Matrix
	}{
		{
			Base:     testutils.Base{Name: "2x1 input, 1x4 weight"},
			in:       fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{5, 6}}),
			weight:   fabrics.NewOperation(t, operation.WeightMultiply, fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 1, Cols: 4, Values: []float64{1, 2, 3, 4}})),
			expected: fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 4, Values: []float64{5, 10, 15, 20, 6, 12, 18, 24}}),
		},
		{
			Base:   testutils.Base{Name: "2x2 input, 1x4 weight, error", Err: operation.ErrExec},
			in:     fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 2, Values: []float64{5, 6, 7, 8}}),
			weight: fabrics.NewOperation(t, operation.WeightMultiply, fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 1, Cols: 4, Values: []float64{1, 2, 3, 4}})),
		},
		{
			Base:   testutils.Base{Name: "2x1 input, 2x4 weight, error", Err: operation.ErrExec},
			in:     fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{5, 6}}),
			weight: fabrics.NewOperation(t, operation.WeightMultiply, fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 4, Values: []float64{1, 2, 3, 4, 5, 6, 7, 8}})),
		},
		{
			Base:   testutils.Base{Name: "nil input, 1x4 weight, error", Err: operation.ErrExec},
			weight: fabrics.NewOperation(t, operation.WeightMultiply, fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 1, Cols: 4, Values: []float64{1, 2, 3, 4}})),
		},
	}

	for _, test := range tests {
		t.Run(test.Name, func(t *testing.T) {
			out, err := test.weight.Forward(test.in)
			if test.Err == nil {
				require.NoError(t, err)
				require.True(t, out.Equal(test.expected))
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, test.Err)
			}
		})
	}
}

func TestWeight_Backward(t *testing.T) {
	tests := []struct {
		testutils.Base
		in       *matrix.Matrix
		weight   operation.IOperation
		outGrad  *matrix.Matrix
		expected *matrix.Matrix
	}{
		{
			Base:     testutils.Base{Name: "2x1 input, 1x4 weight, 2x4 out grad"},
			in:       fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{1, 2}}),
			weight:   fabrics.NewOperation(t, operation.WeightMultiply, fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 1, Cols: 4, Values: []float64{3, 4, 5, 6}})),
			outGrad:  fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 4, Values: []float64{7, 8, 9, 10, 11, 12, 13, 14}}),
			expected: fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{158, 230}}),
		},
		{
			Base:    testutils.Base{Name: "2x1 input, 1x4 weight, 2x4 out grad, no forward", Err: operation.ErrExec},
			weight:  fabrics.NewOperation(t, operation.WeightMultiply, fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 1, Cols: 4, Values: []float64{3, 4, 5, 6}})),
			outGrad: fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 4, Values: []float64{7, 8, 9, 10, 11, 12, 13, 14}}),
		},
		{
			Base:   testutils.Base{Name: "2x1 input, 1x4 weight, nil out grad", Err: operation.ErrExec},
			in:     fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{1, 2}}),
			weight: fabrics.NewOperation(t, operation.WeightMultiply, fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 1, Cols: 4, Values: []float64{3, 4, 5, 6}})),
		},
	}

	for _, test := range tests {
		t.Run(test.Name, func(t *testing.T) {
			if test.in != nil {
				_, err := test.weight.Forward(test.in)
				require.NoError(t, err)
			}
			inGrad, err := test.weight.Backward(test.outGrad)
			if test.Err == nil {
				require.NoError(t, err)
				require.True(t, inGrad.Equal(test.expected))
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, test.Err)
			}
		})
	}
}

func TestWeight_ApplyOptim(t *testing.T) {
	optimizer := func(param, grad *matrix.Matrix) (*matrix.Matrix, error) {
		return param.Sub(grad)
	}
	tests := []struct {
		testutils.Base
		in        *matrix.Matrix
		weight    *operation.ParamOperation
		outGrad   *matrix.Matrix
		expected  *matrix.Matrix
		optimizer operation.Optimizer
	}{
		{
			Base:      testutils.Base{Name: "2x1 input, 1x4 weight, 2x4 out grad"},
			in:        fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{1, 2}}),
			weight:    fabrics.NewOperation(t, operation.WeightMultiply, fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 1, Cols: 4, Values: []float64{3, 4, 5, 6}})).(*operation.ParamOperation),
			outGrad:   fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 4, Values: []float64{7, 8, 9, 10, 11, 12, 13, 14}}),
			expected:  fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 1, Cols: 4, Values: []float64{3 - 29, 4 - 32, 5 - 35, 6 - 38}}),
			optimizer: optimizer,
		},
		{
			Base:      testutils.Base{Name: "2x1 input, 1x4 weight, 2x4 out grad, no backward", Err: operation.ErrExec},
			in:        fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{1, 2}}),
			weight:    fabrics.NewOperation(t, operation.WeightMultiply, fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 1, Cols: 4, Values: []float64{3, 4, 5, 6}})).(*operation.ParamOperation),
			expected:  fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 1, Cols: 4, Values: []float64{3 - 29, 4 - 32, 5 - 35, 6 - 38}}),
			optimizer: optimizer,
		},
		{
			Base:     testutils.Base{Name: "2x1 input, 1x4 weight, 2x4 out grad, incorrect optimizer", Err: operation.ErrExec},
			in:       fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{1, 2}}),
			weight:   fabrics.NewOperation(t, operation.WeightMultiply, fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 1, Cols: 4, Values: []float64{3, 4, 5, 6}})).(*operation.ParamOperation),
			outGrad:  fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 4, Values: []float64{7, 8, 9, 10, 11, 12, 13, 14}}),
			expected: fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 1, Cols: 4, Values: []float64{3 - 29, 4 - 32, 5 - 35, 6 - 38}}),
			optimizer: func(param, grad *matrix.Matrix) (*matrix.Matrix, error) {
				return nil, nil
			},
		},
		{
			Base:     testutils.Base{Name: "2x1 input, 1x4 weight, 2x4 out grad", Err: operation.ErrExec},
			in:       fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{1, 2}}),
			weight:   fabrics.NewOperation(t, operation.WeightMultiply, fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 1, Cols: 4, Values: []float64{3, 4, 5, 6}})).(*operation.ParamOperation),
			outGrad:  fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 4, Values: []float64{7, 8, 9, 10, 11, 12, 13, 14}}),
			expected: fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 1, Cols: 4, Values: []float64{3 - 29, 4 - 32, 5 - 35, 6 - 38}}),
		},
	}

	for _, test := range tests {
		t.Run(test.Name, func(t *testing.T) {
			_, err := test.weight.Forward(test.in)
			require.NoError(t, err)
			if test.outGrad != nil {
				_, err = test.weight.Backward(test.outGrad)
				require.NoError(t, err)
			}
			err = test.weight.ApplyOptim(test.optimizer)
			if test.Err == nil {
				require.NoError(t, err)
				actual := test.weight.Parameter()
				require.True(t, actual.Equal(test.expected))
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, test.Err)
			}
		})
	}
}
