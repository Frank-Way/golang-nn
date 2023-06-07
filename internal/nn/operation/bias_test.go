package operation

import (
	"github.com/stretchr/testify/require"
	"nn/internal/testutils"
	"nn/internal/testutils/testfactories"
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
			bias: testfactories.NewVector(t, testfactories.VectorParameters{Size: 8}),
		},
		{
			Base: testutils.Base{Name: "nil bias operation", Err: ErrCreate},
		},
	}

	for _, test := range tests {
		t.Run(test.Name, func(t *testing.T) {
			_, err := NewBiasOperation(test.bias)
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
		bias     IOperation
		expected *matrix.Matrix
	}{
		{
			Base:     testutils.Base{Name: "2x1 input, 1x1 bias"},
			in:       testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{1, 2}}),
			bias:     newOperation(t, BiasAdd, testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{1}})),
			expected: testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{2, 3}}),
		},
		{
			Base: testutils.Base{Name: "2x2 input, 1x1 bias, error", Err: ErrExec},
			in:   testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 2, Values: []float64{5, 6, 7, 8}}),
			bias: newOperation(t, BiasAdd, testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{1}})),
		},
		{
			Base: testutils.Base{Name: "2x1 input, 1x2 bias, error", Err: ErrExec},
			in:   testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{5, 6}}),
			bias: newOperation(t, BiasAdd, testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{1, 2}})),
		},
		{
			Base: testutils.Base{Name: "nil input, 1x1 bias, error", Err: ErrExec},
			bias: newOperation(t, BiasAdd, testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{1}})),
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
		bias     IOperation
		outGrad  *matrix.Matrix
		expected *matrix.Matrix
	}{
		{
			Base:     testutils.Base{Name: "2x1 input, 1x1 bias, 2x1 out grad"},
			in:       testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{1, 2}}),
			bias:     newOperation(t, BiasAdd, testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{3}})),
			outGrad:  testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{6, 7}}),
			expected: testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{6, 7}}),
		},
		{
			Base:     testutils.Base{Name: "2x1 input, 1x1 bias, 2x1 out grad, no forward", Err: ErrExec},
			bias:     newOperation(t, BiasAdd, testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{3}})),
			outGrad:  testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{6, 7}}),
			expected: testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{6, 7}}),
		},
		{
			Base:     testutils.Base{Name: "2x1 input, 1x1 bias, nil out grad", Err: ErrExec},
			in:       testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{1, 2}}),
			bias:     newOperation(t, BiasAdd, testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{3}})),
			expected: testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{6, 7}}),
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
		bias      *ParamOperation
		outGrad   *matrix.Matrix
		expected  *matrix.Matrix
		optimizer Optimizer
	}{
		{
			Base:      testutils.Base{Name: "2x1 input, 1x4 bias, 2x4 out grad"},
			in:        testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{1, 2}}),
			bias:      newOperation(t, BiasAdd, testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{3}})).(*ParamOperation),
			outGrad:   testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{6, 7}}),
			expected:  testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 1, Cols: 1, Values: []float64{3 - (6 + 7)}}),
			optimizer: optimizer,
		},
		{
			Base:      testutils.Base{Name: "2x1 input, 1x4 bias, 2x4 out grad, no backward", Err: ErrExec},
			in:        testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{1, 2}}),
			bias:      newOperation(t, BiasAdd, testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{3}})).(*ParamOperation),
			expected:  testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 1, Cols: 1, Values: []float64{3 - (6 + 7)}}),
			optimizer: optimizer,
		},
		{
			Base:     testutils.Base{Name: "2x1 input, 1x4 bias, 2x4 out grad, incorrect optimizer", Err: ErrExec},
			in:       testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{1, 2}}),
			bias:     newOperation(t, BiasAdd, testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{3}})).(*ParamOperation),
			outGrad:  testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{6, 7}}),
			expected: testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 1, Cols: 1, Values: []float64{3 - (6 + 7)}}),
			optimizer: func(param, grad *matrix.Matrix) (*matrix.Matrix, error) {
				return nil, nil
			},
		},
		{
			Base:     testutils.Base{Name: "2x1 input, 1x4 bias, 2x4 out grad", Err: ErrExec},
			in:       testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{1, 2}}),
			bias:     newOperation(t, BiasAdd, testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{3}})).(*ParamOperation),
			outGrad:  testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{6, 7}}),
			expected: testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 1, Cols: 1, Values: []float64{3 - (6 + 7)}}),
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
