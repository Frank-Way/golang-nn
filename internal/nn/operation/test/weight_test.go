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
		weight   fabrics.MatrixParameters
		nilCheck bool
	}{
		{
			Base:   testutils.Base{Name: "2x8 weight operation"},
			weight: fabrics.MatrixParameters{Rows: 2, Cols: 8},
		},
		{
			Base:     testutils.Base{Name: "nil weight operation", Err: operation.ErrCreate},
			weight:   fabrics.MatrixParameters{Rows: 2, Cols: 8},
			nilCheck: true,
		},
	}

	for _, test := range tests {
		t.Run(test.Name, func(t *testing.T) {
			weight := fabrics.NewMatrix(t, test.weight)
			if test.nilCheck {
				weight = nil
			}
			_, err := operation.NewWeightOperation(weight)
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
		in       fabrics.MatrixParameters
		weight   fabrics.WeightParameters
		expected fabrics.MatrixParameters
		nilCheck bool
	}{
		{
			Base:     testutils.Base{Name: "2x1 input, 1x4 weight"},
			in:       fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{5, 6}},
			weight:   fabrics.WeightParameters{MatrixParameters: fabrics.MatrixParameters{Rows: 1, Cols: 4, Values: []float64{1, 2, 3, 4}}},
			expected: fabrics.MatrixParameters{Rows: 2, Cols: 4, Values: []float64{5, 10, 15, 20, 6, 12, 18, 24}},
		},
		{
			Base:   testutils.Base{Name: "2x2 input, 1x4 weight, error", Err: operation.ErrExec},
			in:     fabrics.MatrixParameters{Rows: 2, Cols: 2, Values: []float64{5, 6, 7, 8}},
			weight: fabrics.WeightParameters{MatrixParameters: fabrics.MatrixParameters{Rows: 1, Cols: 4, Values: []float64{1, 2, 3, 4}}},
		},
		{
			Base:   testutils.Base{Name: "2x1 input, 2x4 weight, error", Err: operation.ErrExec},
			in:     fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{5, 6}},
			weight: fabrics.WeightParameters{MatrixParameters: fabrics.MatrixParameters{Rows: 2, Cols: 4, Values: []float64{1, 2, 3, 4, 5, 6, 7, 8}}},
		},
		{
			Base:     testutils.Base{Name: "nil input, 1x4 weight, error", Err: operation.ErrExec},
			in:       fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{5, 6}},
			weight:   fabrics.WeightParameters{MatrixParameters: fabrics.MatrixParameters{Rows: 1, Cols: 4, Values: []float64{1, 2, 3, 4}}},
			nilCheck: true,
		},
	}

	for _, test := range tests {
		t.Run(test.Name, func(t *testing.T) {
			weight := fabrics.NewWeight(t, test.weight)
			in := fabrics.NewMatrix(t, test.in)
			if test.nilCheck {
				in = nil
			}
			out, err := weight.Forward(in)
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

func TestWeight_Backward(t *testing.T) {
	tests := []struct {
		testutils.Base
		in       fabrics.MatrixParameters
		weight   fabrics.WeightParameters
		out      fabrics.MatrixParameters
		outGrad  fabrics.MatrixParameters
		expected fabrics.MatrixParameters
		forward  bool
		nilCheck bool
	}{
		{
			Base:     testutils.Base{Name: "2x1 input, 1x4 weight, 2x4 out grad"},
			in:       fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{1, 2}},
			weight:   fabrics.WeightParameters{MatrixParameters: fabrics.MatrixParameters{Rows: 1, Cols: 4, Values: []float64{3, 4, 5, 6}}},
			out:      fabrics.MatrixParameters{Rows: 2, Cols: 4, Values: []float64{3, 4, 5, 6, 6, 8, 10, 12}},
			outGrad:  fabrics.MatrixParameters{Rows: 2, Cols: 4, Values: []float64{7, 8, 9, 10, 11, 12, 13, 14}},
			expected: fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{158, 230}},
			forward:  true,
		},
		{
			Base:     testutils.Base{Name: "2x1 input, 1x4 weight, 2x4 out grad, no forward", Err: operation.ErrExec},
			in:       fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{1, 2}},
			weight:   fabrics.WeightParameters{MatrixParameters: fabrics.MatrixParameters{Rows: 1, Cols: 4, Values: []float64{3, 4, 5, 6}}},
			out:      fabrics.MatrixParameters{Rows: 2, Cols: 4, Values: []float64{3, 4, 5, 6, 6, 8, 10, 12}},
			outGrad:  fabrics.MatrixParameters{Rows: 2, Cols: 4, Values: []float64{7, 8, 9, 10, 11, 12, 13, 14}},
			expected: fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{158, 230}},
		},
		{
			Base:     testutils.Base{Name: "2x1 input, 1x4 weight, nil out grad", Err: operation.ErrExec},
			in:       fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{1, 2}},
			weight:   fabrics.WeightParameters{MatrixParameters: fabrics.MatrixParameters{Rows: 1, Cols: 4, Values: []float64{3, 4, 5, 6}}},
			out:      fabrics.MatrixParameters{Rows: 2, Cols: 4, Values: []float64{3, 4, 5, 6, 6, 8, 10, 12}},
			outGrad:  fabrics.MatrixParameters{Rows: 2, Cols: 4, Values: []float64{7, 8, 9, 10, 11, 12, 13, 14}},
			expected: fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{158, 230}},
			forward:  true,
			nilCheck: true,
		},
	}

	for _, test := range tests {
		t.Run(test.Name, func(t *testing.T) {
			weight := fabrics.NewWeight(t, test.weight)
			in := fabrics.NewMatrix(t, test.in)
			if test.forward {
				out, err := weight.Forward(in)
				require.NoError(t, err)
				outExpected := fabrics.NewMatrix(t, test.out)
				require.True(t, out.Equal(outExpected))
			}
			outGrad := fabrics.NewMatrix(t, test.outGrad)
			if test.nilCheck {
				outGrad = nil
			}
			inGrad, err := weight.Backward(outGrad)
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

func TestWeight_ApplyOptim(t *testing.T) {
	tests := []struct {
		testutils.Base
		in       fabrics.MatrixParameters
		weight   fabrics.WeightParameters
		out      fabrics.MatrixParameters
		outGrad  fabrics.MatrixParameters
		expected fabrics.MatrixParameters
		optim    operation.Optimizer
		backward bool
		nilCheck bool
	}{
		{
			Base:     testutils.Base{Name: "2x1 input, 1x4 weight, 2x4 out grad"},
			in:       fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{1, 2}},
			weight:   fabrics.WeightParameters{MatrixParameters: fabrics.MatrixParameters{Rows: 1, Cols: 4, Values: []float64{3, 4, 5, 6}}},
			out:      fabrics.MatrixParameters{Rows: 2, Cols: 4, Values: []float64{3, 4, 5, 6, 6, 8, 10, 12}},
			outGrad:  fabrics.MatrixParameters{Rows: 2, Cols: 4, Values: []float64{7, 8, 9, 10, 11, 12, 13, 14}},
			expected: fabrics.MatrixParameters{Rows: 1, Cols: 4, Values: []float64{3 - 29, 4 - 32, 5 - 35, 6 - 38}},
			optim: func(param, grad *matrix.Matrix) (*matrix.Matrix, error) {
				return param.Sub(grad)
			},
			backward: true,
		},
		{
			Base:     testutils.Base{Name: "2x1 input, 1x4 weight, 2x4 out grad, no backward", Err: operation.ErrExec},
			in:       fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{1, 2}},
			weight:   fabrics.WeightParameters{MatrixParameters: fabrics.MatrixParameters{Rows: 1, Cols: 4, Values: []float64{3, 4, 5, 6}}},
			out:      fabrics.MatrixParameters{Rows: 2, Cols: 4, Values: []float64{3, 4, 5, 6, 6, 8, 10, 12}},
			outGrad:  fabrics.MatrixParameters{Rows: 2, Cols: 4, Values: []float64{7, 8, 9, 10, 11, 12, 13, 14}},
			expected: fabrics.MatrixParameters{Rows: 1, Cols: 4, Values: []float64{3 - 29, 4 - 32, 5 - 35, 6 - 38}},
			optim: func(param, grad *matrix.Matrix) (*matrix.Matrix, error) {
				return param.Sub(grad)
			},
		},
		{
			Base:     testutils.Base{Name: "2x1 input, 1x4 weight, 2x4 out grad, incorrect optim", Err: operation.ErrExec},
			in:       fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{1, 2}},
			weight:   fabrics.WeightParameters{MatrixParameters: fabrics.MatrixParameters{Rows: 1, Cols: 4, Values: []float64{3, 4, 5, 6}}},
			out:      fabrics.MatrixParameters{Rows: 2, Cols: 4, Values: []float64{3, 4, 5, 6, 6, 8, 10, 12}},
			outGrad:  fabrics.MatrixParameters{Rows: 2, Cols: 4, Values: []float64{7, 8, 9, 10, 11, 12, 13, 14}},
			expected: fabrics.MatrixParameters{Rows: 1, Cols: 4, Values: []float64{3 - 29, 4 - 32, 5 - 35, 6 - 38}},
			optim: func(param, grad *matrix.Matrix) (*matrix.Matrix, error) {
				return nil, nil
			},
			backward: true,
		},
		{
			Base:     testutils.Base{Name: "2x1 input, 1x4 weight, 2x4 out grad", Err: operation.ErrExec},
			in:       fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{1, 2}},
			weight:   fabrics.WeightParameters{MatrixParameters: fabrics.MatrixParameters{Rows: 1, Cols: 4, Values: []float64{3, 4, 5, 6}}},
			out:      fabrics.MatrixParameters{Rows: 2, Cols: 4, Values: []float64{3, 4, 5, 6, 6, 8, 10, 12}},
			outGrad:  fabrics.MatrixParameters{Rows: 2, Cols: 4, Values: []float64{7, 8, 9, 10, 11, 12, 13, 14}},
			expected: fabrics.MatrixParameters{Rows: 1, Cols: 4, Values: []float64{3 - 29, 4 - 32, 5 - 35, 6 - 38}},
			optim: func(param, grad *matrix.Matrix) (*matrix.Matrix, error) {
				return param.Sub(grad)
			},
			backward: true,
			nilCheck: true,
		},
	}

	for _, test := range tests {
		t.Run(test.Name, func(t *testing.T) {
			weight := fabrics.NewWeight(t, test.weight).(*operation.ParamOperation)
			in := fabrics.NewMatrix(t, test.in)
			out, err := weight.Forward(in)
			require.NoError(t, err)
			outExpected := fabrics.NewMatrix(t, test.out)
			require.True(t, out.Equal(outExpected))
			outGrad := fabrics.NewMatrix(t, test.outGrad)
			if test.backward {
				_, err = weight.Backward(outGrad)
				require.NoError(t, err)
			}
			if test.nilCheck {
				err = weight.ApplyOptim(nil)
			} else {
				err = weight.ApplyOptim(test.optim)
			}
			if test.Err == nil {
				require.NoError(t, err)
				actual := weight.Parameter()
				expected := fabrics.NewMatrix(t, test.expected)
				require.True(t, actual.Equal(expected))
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, test.Err)
			}
		})
	}
}
