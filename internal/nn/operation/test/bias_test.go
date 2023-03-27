package test

import (
	"github.com/stretchr/testify/require"
	"nn/internal/nn/operation"
	"nn/internal/testutils"
	"nn/internal/testutils/fabrics"
	"nn/pkg/mmath/matrix"
	"testing"
)

func TestNewBiasOperation(t *testing.T) {
	tests := []struct {
		testutils.Base
		bias     fabrics.VectorParameters
		nilCheck bool
	}{
		{
			Base: testutils.Base{Name: "1x8 bias operation"},
			bias: fabrics.VectorParameters{Size: 8},
		},
		{
			Base:     testutils.Base{Name: "nil bias operation", Err: operation.ErrCreate},
			bias:     fabrics.VectorParameters{Size: 8},
			nilCheck: true,
		},
	}

	for _, test := range tests {
		t.Run(test.Name, func(t *testing.T) {
			bias := fabrics.NewVector(t, test.bias)
			if test.nilCheck {
				bias = nil
			}
			_, err := operation.NewBiasOperation(bias)
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
		in       fabrics.MatrixParameters
		bias     fabrics.BiasParameters
		expected fabrics.MatrixParameters
		nilCheck bool
	}{
		{
			Base:     testutils.Base{Name: "2x1 input, 1x1 bias"},
			in:       fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{1, 2}},
			bias:     fabrics.BiasParameters{VectorParameters: fabrics.VectorParameters{Values: []float64{1}}},
			expected: fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{2, 3}},
		},
		{
			Base: testutils.Base{Name: "2x2 input, 1x1 bias, error", Err: operation.ErrExec},
			in:   fabrics.MatrixParameters{Rows: 2, Cols: 2, Values: []float64{5, 6, 7, 8}},
			bias: fabrics.BiasParameters{VectorParameters: fabrics.VectorParameters{Values: []float64{1}}},
		},
		{
			Base: testutils.Base{Name: "2x1 input, 1x2 bias, error", Err: operation.ErrExec},
			in:   fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{5, 6}},
			bias: fabrics.BiasParameters{VectorParameters: fabrics.VectorParameters{Values: []float64{1, 2}}},
		},
		{
			Base:     testutils.Base{Name: "nil input, 1x1 bias, error", Err: operation.ErrExec},
			in:       fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{5, 6}},
			bias:     fabrics.BiasParameters{VectorParameters: fabrics.VectorParameters{Values: []float64{1}}},
			nilCheck: true,
		},
	}

	for _, test := range tests {
		t.Run(test.Name, func(t *testing.T) {
			bias := fabrics.NewBias(t, test.bias)
			in := fabrics.NewMatrix(t, test.in)
			if test.nilCheck {
				in = nil
			}
			out, err := bias.Forward(in)
			if test.Err == nil {
				require.NoError(t, err)
				expected := fabrics.NewMatrix(t, test.expected)
				require.True(t, out.Equal(expected))
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
		in       fabrics.MatrixParameters
		bias     fabrics.BiasParameters
		out      fabrics.MatrixParameters
		outGrad  fabrics.MatrixParameters
		expected fabrics.MatrixParameters
		forward  bool
		nilCheck bool
	}{
		{
			Base:     testutils.Base{Name: "2x1 input, 1x1 bias, 2x1 out grad"},
			in:       fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{1, 2}},
			bias:     fabrics.BiasParameters{VectorParameters: fabrics.VectorParameters{Values: []float64{3}}},
			out:      fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{4, 5}},
			outGrad:  fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{6, 7}},
			expected: fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{6, 7}},
			forward:  true,
		},
		{
			Base:     testutils.Base{Name: "2x1 input, 1x1 bias, 2x1 out grad, no forward", Err: operation.ErrExec},
			in:       fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{1, 2}},
			bias:     fabrics.BiasParameters{VectorParameters: fabrics.VectorParameters{Values: []float64{3}}},
			out:      fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{4, 5}},
			outGrad:  fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{6, 7}},
			expected: fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{6, 7}},
		},
		{
			Base:     testutils.Base{Name: "2x1 input, 1x1 bias, nil out grad", Err: operation.ErrExec},
			in:       fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{1, 2}},
			bias:     fabrics.BiasParameters{VectorParameters: fabrics.VectorParameters{Values: []float64{3}}},
			out:      fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{4, 5}},
			outGrad:  fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{6, 7}},
			expected: fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{6, 7}},
			forward:  true,
			nilCheck: true,
		},
	}

	for _, test := range tests {
		t.Run(test.Name, func(t *testing.T) {
			bias := fabrics.NewBias(t, test.bias)
			in := fabrics.NewMatrix(t, test.in)
			if test.forward {
				out, err := bias.Forward(in)
				require.NoError(t, err)
				outExpected := fabrics.NewMatrix(t, test.out)
				require.True(t, out.Equal(outExpected))
			}
			outGrad := fabrics.NewMatrix(t, test.outGrad)
			if test.nilCheck {
				outGrad = nil
			}
			inGrad, err := bias.Backward(outGrad)
			if test.Err == nil {
				require.NoError(t, err)
				expected := fabrics.NewMatrix(t, test.expected)
				require.True(t, inGrad.Equal(expected))
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, test.Err)
			}
		})
	}
}

func TestBias_ApplyOptim(t *testing.T) {
	tests := []struct {
		testutils.Base
		in       fabrics.MatrixParameters
		bias     fabrics.BiasParameters
		out      fabrics.MatrixParameters
		outGrad  fabrics.MatrixParameters
		expected fabrics.MatrixParameters
		optim    operation.Optimizer
		backward bool
		nilCheck bool
	}{
		{
			Base:     testutils.Base{Name: "2x1 input, 1x4 bias, 2x4 out grad"},
			in:       fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{1, 2}},
			bias:     fabrics.BiasParameters{VectorParameters: fabrics.VectorParameters{Values: []float64{3}}},
			out:      fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{4, 5}},
			outGrad:  fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{6, 7}},
			expected: fabrics.MatrixParameters{Rows: 1, Cols: 1, Values: []float64{3 - (6 + 7)}},
			optim: func(param, grad *matrix.Matrix) (*matrix.Matrix, error) {
				return param.Sub(grad)
			},
			backward: true,
		},
		{
			Base:     testutils.Base{Name: "2x1 input, 1x4 bias, 2x4 out grad, no backward", Err: operation.ErrExec},
			in:       fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{1, 2}},
			bias:     fabrics.BiasParameters{VectorParameters: fabrics.VectorParameters{Values: []float64{3}}},
			out:      fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{4, 5}},
			outGrad:  fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{6, 7}},
			expected: fabrics.MatrixParameters{Rows: 1, Cols: 1, Values: []float64{3 - (6 + 7)}},
			optim: func(param, grad *matrix.Matrix) (*matrix.Matrix, error) {
				return param.Sub(grad)
			},
		},
		{
			Base:     testutils.Base{Name: "2x1 input, 1x4 bias, 2x4 out grad, incorrect optim", Err: operation.ErrExec},
			in:       fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{1, 2}},
			bias:     fabrics.BiasParameters{VectorParameters: fabrics.VectorParameters{Values: []float64{3}}},
			out:      fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{4, 5}},
			outGrad:  fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{6, 7}},
			expected: fabrics.MatrixParameters{Rows: 1, Cols: 1, Values: []float64{3 - (6 + 7)}},
			optim: func(param, grad *matrix.Matrix) (*matrix.Matrix, error) {
				return nil, nil
			},
			backward: true,
		},
		{
			Base:     testutils.Base{Name: "2x1 input, 1x4 bias, 2x4 out grad", Err: operation.ErrExec},
			in:       fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{1, 2}},
			bias:     fabrics.BiasParameters{VectorParameters: fabrics.VectorParameters{Values: []float64{3}}},
			out:      fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{4, 5}},
			outGrad:  fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{6, 7}},
			expected: fabrics.MatrixParameters{Rows: 1, Cols: 1, Values: []float64{3 - (6 + 7)}},
			optim: func(param, grad *matrix.Matrix) (*matrix.Matrix, error) {
				return param.Sub(grad)
			},
			backward: true,
			nilCheck: true,
		},
	}

	for _, test := range tests {
		t.Run(test.Name, func(t *testing.T) {
			bias := fabrics.NewBias(t, test.bias).(*operation.ParamOperation)
			in := fabrics.NewMatrix(t, test.in)
			out, err := bias.Forward(in)
			require.NoError(t, err)
			outExpected := fabrics.NewMatrix(t, test.out)
			require.True(t, out.Equal(outExpected))
			outGrad := fabrics.NewMatrix(t, test.outGrad)
			if test.backward {
				_, err = bias.Backward(outGrad)
				require.NoError(t, err)
			}
			if test.nilCheck {
				err = bias.ApplyOptim(nil)
			} else {
				err = bias.ApplyOptim(test.optim)
			}
			if test.Err == nil {
				require.NoError(t, err)
				actual := bias.Parameter()
				expected := fabrics.NewMatrix(t, test.expected)
				require.True(t, actual.Equal(expected))
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, test.Err)
			}
		})
	}
}
