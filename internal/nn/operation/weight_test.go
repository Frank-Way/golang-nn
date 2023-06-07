package operation

import (
	"github.com/stretchr/testify/require"
	"nn/internal/testutils"
	"nn/internal/testutils/testfactories"
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
			weight: testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 8}),
		},
		{
			Base: testutils.Base{Name: "nil weight operation", Err: ErrCreate},
		},
	}

	for _, test := range tests {
		t.Run(test.Name, func(t *testing.T) {
			_, err := NewWeightOperation(test.weight)
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
		weight   IOperation
		expected *matrix.Matrix
	}{
		{
			Base:     testutils.Base{Name: "2x1 input, 1x4 weight"},
			in:       testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{5, 6}}),
			weight:   newOperation(t, WeightMultiply, testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 1, Cols: 4, Values: []float64{1, 2, 3, 4}})),
			expected: testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 4, Values: []float64{5, 10, 15, 20, 6, 12, 18, 24}}),
		},
		{
			Base:   testutils.Base{Name: "2x2 input, 1x4 weight, error", Err: ErrExec},
			in:     testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 2, Values: []float64{5, 6, 7, 8}}),
			weight: newOperation(t, WeightMultiply, testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 1, Cols: 4, Values: []float64{1, 2, 3, 4}})),
		},
		{
			Base:   testutils.Base{Name: "2x1 input, 2x4 weight, error", Err: ErrExec},
			in:     testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{5, 6}}),
			weight: newOperation(t, WeightMultiply, testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 4, Values: []float64{1, 2, 3, 4, 5, 6, 7, 8}})),
		},
		{
			Base:   testutils.Base{Name: "nil input, 1x4 weight, error", Err: ErrExec},
			weight: newOperation(t, WeightMultiply, testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 1, Cols: 4, Values: []float64{1, 2, 3, 4}})),
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
		weight   IOperation
		outGrad  *matrix.Matrix
		expected *matrix.Matrix
	}{
		{
			Base:     testutils.Base{Name: "2x1 input, 1x4 weight, 2x4 out grad"},
			in:       testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{1, 2}}),
			weight:   newOperation(t, WeightMultiply, testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 1, Cols: 4, Values: []float64{3, 4, 5, 6}})),
			outGrad:  testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 4, Values: []float64{7, 8, 9, 10, 11, 12, 13, 14}}),
			expected: testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{158, 230}}),
		},
		{
			Base:    testutils.Base{Name: "2x1 input, 1x4 weight, 2x4 out grad, no forward", Err: ErrExec},
			weight:  newOperation(t, WeightMultiply, testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 1, Cols: 4, Values: []float64{3, 4, 5, 6}})),
			outGrad: testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 4, Values: []float64{7, 8, 9, 10, 11, 12, 13, 14}}),
		},
		{
			Base:   testutils.Base{Name: "2x1 input, 1x4 weight, nil out grad", Err: ErrExec},
			in:     testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{1, 2}}),
			weight: newOperation(t, WeightMultiply, testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 1, Cols: 4, Values: []float64{3, 4, 5, 6}})),
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
		weight    *ParamOperation
		outGrad   *matrix.Matrix
		expected  *matrix.Matrix
		optimizer Optimizer
	}{
		{
			Base:      testutils.Base{Name: "2x1 input, 1x4 weight, 2x4 out grad"},
			in:        testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{1, 2}}),
			weight:    newOperation(t, WeightMultiply, testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 1, Cols: 4, Values: []float64{3, 4, 5, 6}})).(*ParamOperation),
			outGrad:   testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 4, Values: []float64{7, 8, 9, 10, 11, 12, 13, 14}}),
			expected:  testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 1, Cols: 4, Values: []float64{3 - 29, 4 - 32, 5 - 35, 6 - 38}}),
			optimizer: optimizer,
		},
		{
			Base:      testutils.Base{Name: "2x1 input, 1x4 weight, 2x4 out grad, no backward", Err: ErrExec},
			in:        testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{1, 2}}),
			weight:    newOperation(t, WeightMultiply, testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 1, Cols: 4, Values: []float64{3, 4, 5, 6}})).(*ParamOperation),
			expected:  testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 1, Cols: 4, Values: []float64{3 - 29, 4 - 32, 5 - 35, 6 - 38}}),
			optimizer: optimizer,
		},
		{
			Base:     testutils.Base{Name: "2x1 input, 1x4 weight, 2x4 out grad, incorrect optimizer", Err: ErrExec},
			in:       testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{1, 2}}),
			weight:   newOperation(t, WeightMultiply, testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 1, Cols: 4, Values: []float64{3, 4, 5, 6}})).(*ParamOperation),
			outGrad:  testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 4, Values: []float64{7, 8, 9, 10, 11, 12, 13, 14}}),
			expected: testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 1, Cols: 4, Values: []float64{3 - 29, 4 - 32, 5 - 35, 6 - 38}}),
			optimizer: func(param, grad *matrix.Matrix) (*matrix.Matrix, error) {
				return nil, nil
			},
		},
		{
			Base:     testutils.Base{Name: "2x1 input, 1x4 weight, 2x4 out grad", Err: ErrExec},
			in:       testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{1, 2}}),
			weight:   newOperation(t, WeightMultiply, testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 1, Cols: 4, Values: []float64{3, 4, 5, 6}})).(*ParamOperation),
			outGrad:  testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 4, Values: []float64{7, 8, 9, 10, 11, 12, 13, 14}}),
			expected: testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 1, Cols: 4, Values: []float64{3 - 29, 4 - 32, 5 - 35, 6 - 38}}),
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
