package test

import (
	"github.com/stretchr/testify/require"
	"nn/internal/nn/operation"
	"nn/internal/testutils"
	"nn/internal/testutils/fabrics"
	"nn/pkg/mmath/matrix"
	"nn/pkg/mmath/vector"
	"testing"
)

func TestNewBiasOperation(t *testing.T) {
	tests := []struct {
		testutils.Base
		bias *vector.Vector
	}{
		{
			Base: testutils.Base{Name: "1x8 bias operation"},
			bias: fabrics.NewVector(t, fabrics.VectorParameters{Size: 8}),
		},
		{
			Base: testutils.Base{Name: "nil bias operation", Err: operation.ErrCreate},
		},
	}

	for _, test := range tests {
		t.Run(test.Name, func(t *testing.T) {
			_, err := operation.NewBiasOperation(test.bias)
			if test.Err == nil {
				require.NoError(t, err)
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, test.Err)
			}
		})
	}
}

func TestBias_Forward(t *testing.T) {
	tests := []struct {
		testutils.Base
		in       *matrix.Matrix
		bias     operation.IOperation
		expected *matrix.Matrix
	}{
		{
			Base:     testutils.Base{Name: "2x1 input, 1x1 bias"},
			in:       fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{1, 2}}),
			bias:     fabrics.NewOperation(t, operation.BiasAdd, fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{1}})),
			expected: fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{2, 3}}),
		},
		{
			Base: testutils.Base{Name: "2x2 input, 1x1 bias, error", Err: operation.ErrExec},
			in:   fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 2, Values: []float64{5, 6, 7, 8}}),
			bias: fabrics.NewOperation(t, operation.BiasAdd, fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{1}})),
		},
		{
			Base: testutils.Base{Name: "2x1 input, 1x2 bias, error", Err: operation.ErrExec},
			in:   fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{5, 6}}),
			bias: fabrics.NewOperation(t, operation.BiasAdd, fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{1, 2}})),
		},
		{
			Base: testutils.Base{Name: "nil input, 1x1 bias, error", Err: operation.ErrExec},
			bias: fabrics.NewOperation(t, operation.BiasAdd, fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{1}})),
		},
	}

	for _, test := range tests {
		t.Run(test.Name, func(t *testing.T) {
			out, err := test.bias.Forward(test.in)
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

func TestBias_Backward(t *testing.T) {
	tests := []struct {
		testutils.Base
		in       *matrix.Matrix
		bias     operation.IOperation
		outGrad  *matrix.Matrix
		expected *matrix.Matrix
	}{
		{
			Base:     testutils.Base{Name: "2x1 input, 1x1 bias, 2x1 out grad"},
			in:       fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{1, 2}}),
			bias:     fabrics.NewOperation(t, operation.BiasAdd, fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{3}})),
			outGrad:  fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{6, 7}}),
			expected: fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{6, 7}}),
		},
		{
			Base:     testutils.Base{Name: "2x1 input, 1x1 bias, 2x1 out grad, no forward", Err: operation.ErrExec},
			bias:     fabrics.NewOperation(t, operation.BiasAdd, fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{3}})),
			outGrad:  fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{6, 7}}),
			expected: fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{6, 7}}),
		},
		{
			Base:     testutils.Base{Name: "2x1 input, 1x1 bias, nil out grad", Err: operation.ErrExec},
			in:       fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{1, 2}}),
			bias:     fabrics.NewOperation(t, operation.BiasAdd, fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{3}})),
			expected: fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{6, 7}}),
		},
	}

	for _, test := range tests {
		t.Run(test.Name, func(t *testing.T) {
			if test.in != nil {
				_, err := test.bias.Forward(test.in)
				require.NoError(t, err)
			}
			inGrad, err := test.bias.Backward(test.outGrad)
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

func TestBias_ApplyOptim(t *testing.T) {
	optimizer := func(param, grad *matrix.Matrix) (*matrix.Matrix, error) {
		return param.Sub(grad)
	}
	tests := []struct {
		testutils.Base
		in        *matrix.Matrix
		bias      *operation.ParamOperation
		outGrad   *matrix.Matrix
		expected  *matrix.Matrix
		optimizer operation.Optimizer
	}{
		{
			Base:      testutils.Base{Name: "2x1 input, 1x4 bias, 2x4 out grad"},
			in:        fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{1, 2}}),
			bias:      fabrics.NewOperation(t, operation.BiasAdd, fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{3}})).(*operation.ParamOperation),
			outGrad:   fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{6, 7}}),
			expected:  fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 1, Cols: 1, Values: []float64{3 - (6 + 7)}}),
			optimizer: optimizer,
		},
		{
			Base:      testutils.Base{Name: "2x1 input, 1x4 bias, 2x4 out grad, no backward", Err: operation.ErrExec},
			in:        fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{1, 2}}),
			bias:      fabrics.NewOperation(t, operation.BiasAdd, fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{3}})).(*operation.ParamOperation),
			expected:  fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 1, Cols: 1, Values: []float64{3 - (6 + 7)}}),
			optimizer: optimizer,
		},
		{
			Base:     testutils.Base{Name: "2x1 input, 1x4 bias, 2x4 out grad, incorrect optimizer", Err: operation.ErrExec},
			in:       fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{1, 2}}),
			bias:     fabrics.NewOperation(t, operation.BiasAdd, fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{3}})).(*operation.ParamOperation),
			outGrad:  fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{6, 7}}),
			expected: fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 1, Cols: 1, Values: []float64{3 - (6 + 7)}}),
			optimizer: func(param, grad *matrix.Matrix) (*matrix.Matrix, error) {
				return nil, nil
			},
		},
		{
			Base:     testutils.Base{Name: "2x1 input, 1x4 bias, 2x4 out grad", Err: operation.ErrExec},
			in:       fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{1, 2}}),
			bias:     fabrics.NewOperation(t, operation.BiasAdd, fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{3}})).(*operation.ParamOperation),
			outGrad:  fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{6, 7}}),
			expected: fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 1, Cols: 1, Values: []float64{3 - (6 + 7)}}),
		},
	}

	for _, test := range tests {
		t.Run(test.Name, func(t *testing.T) {
			_, err := test.bias.Forward(test.in)
			require.NoError(t, err)
			if test.outGrad != nil {
				_, err = test.bias.Backward(test.outGrad)
				require.NoError(t, err)
			}
			err = test.bias.ApplyOptim(test.optimizer)
			if test.Err == nil {
				require.NoError(t, err)
				actual := test.bias.Parameter()
				require.True(t, actual.Equal(test.expected))
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, test.Err)
			}
		})
	}
}
