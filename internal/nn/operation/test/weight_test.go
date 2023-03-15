package test

import (
	"github.com/stretchr/testify/require"
	"nn/internal/nn/operation"
	"nn/internal/test/utils"
	"nn/internal/test/utils/fabrics"
	"nn/pkg/mmath/matrix"
	"testing"
)

func TestNewWeightOperation(t *testing.T) {
	tests := []struct {
		utils.Base
		weight   fabrics.MatrixParameters
		nilCheck bool
	}{
		{
			Base:   utils.Base{Name: "2x8 weight operation"},
			weight: fabrics.MatrixParameters{Rows: 2, Cols: 8},
		},
		{
			Base:     utils.Base{Name: "nil weight operation", Err: operation.ErrCreate},
			weight:   fabrics.MatrixParameters{Rows: 2, Cols: 8},
			nilCheck: true,
		},
	}

	for i := range tests {
		t.Run(tests[i].Name, func(t *testing.T) {
			weight := fabrics.NewMatrix(t, tests[i].weight)
			if tests[i].nilCheck {
				weight = nil
			}
			_, err := operation.NewWeightOperation(weight)
			if tests[i].Err == nil {
				require.NoError(t, err)
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, tests[i].Err)
			}
		})
	}
}

func TestWeight_Forward(t *testing.T) {
	tests := []struct {
		utils.Base
		in       fabrics.MatrixParameters
		weight   fabrics.WeightParameters
		expected fabrics.MatrixParameters
		nilCheck bool
	}{
		{
			Base:     utils.Base{Name: "2x1 input, 1x4 weight"},
			in:       fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{5, 6}},
			weight:   fabrics.WeightParameters{MatrixParameters: fabrics.MatrixParameters{Rows: 1, Cols: 4, Values: []float64{1, 2, 3, 4}}},
			expected: fabrics.MatrixParameters{Rows: 2, Cols: 4, Values: []float64{5, 10, 15, 20, 6, 12, 18, 24}},
		},
		{
			Base:   utils.Base{Name: "2x2 input, 1x4 weight, error", Err: operation.ErrExec},
			in:     fabrics.MatrixParameters{Rows: 2, Cols: 2, Values: []float64{5, 6, 7, 8}},
			weight: fabrics.WeightParameters{MatrixParameters: fabrics.MatrixParameters{Rows: 1, Cols: 4, Values: []float64{1, 2, 3, 4}}},
		},
		{
			Base:   utils.Base{Name: "2x1 input, 2x4 weight, error", Err: operation.ErrExec},
			in:     fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{5, 6}},
			weight: fabrics.WeightParameters{MatrixParameters: fabrics.MatrixParameters{Rows: 2, Cols: 4, Values: []float64{1, 2, 3, 4, 5, 6, 7, 8}}},
		},
		{
			Base:     utils.Base{Name: "nil input, 1x4 weight, error", Err: operation.ErrExec},
			in:       fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{5, 6}},
			weight:   fabrics.WeightParameters{MatrixParameters: fabrics.MatrixParameters{Rows: 1, Cols: 4, Values: []float64{1, 2, 3, 4}}},
			nilCheck: true,
		},
	}

	for i := range tests {
		t.Run(tests[i].Name, func(t *testing.T) {
			weight := fabrics.NewWeight(t, tests[i].weight)
			in := fabrics.NewMatrix(t, tests[i].in)
			if tests[i].nilCheck {
				in = nil
			}
			out, err := weight.Forward(in)
			if tests[i].Err == nil {
				require.NoError(t, err)
				expected := fabrics.NewMatrix(t, tests[i].expected)
				require.True(t, out.Equal(expected))
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, tests[i].Err)
			}
		})
	}
}

func TestWeight_Backward(t *testing.T) {
	tests := []struct {
		utils.Base
		in       fabrics.MatrixParameters
		weight   fabrics.WeightParameters
		out      fabrics.MatrixParameters
		outGrad  fabrics.MatrixParameters
		expected fabrics.MatrixParameters
		forward  bool
		nilCheck bool
	}{
		{
			Base:     utils.Base{Name: "2x1 input, 1x4 weight, 2x4 out grad"},
			in:       fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{1, 2}},
			weight:   fabrics.WeightParameters{MatrixParameters: fabrics.MatrixParameters{Rows: 1, Cols: 4, Values: []float64{3, 4, 5, 6}}},
			out:      fabrics.MatrixParameters{Rows: 2, Cols: 4, Values: []float64{3, 4, 5, 6, 6, 8, 10, 12}},
			outGrad:  fabrics.MatrixParameters{Rows: 2, Cols: 4, Values: []float64{7, 8, 9, 10, 11, 12, 13, 14}},
			expected: fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{158, 230}},
			forward:  true,
		},
		{
			Base:     utils.Base{Name: "2x1 input, 1x4 weight, 2x4 out grad, no forward", Err: operation.ErrExec},
			in:       fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{1, 2}},
			weight:   fabrics.WeightParameters{MatrixParameters: fabrics.MatrixParameters{Rows: 1, Cols: 4, Values: []float64{3, 4, 5, 6}}},
			out:      fabrics.MatrixParameters{Rows: 2, Cols: 4, Values: []float64{3, 4, 5, 6, 6, 8, 10, 12}},
			outGrad:  fabrics.MatrixParameters{Rows: 2, Cols: 4, Values: []float64{7, 8, 9, 10, 11, 12, 13, 14}},
			expected: fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{158, 230}},
		},
		{
			Base:     utils.Base{Name: "2x1 input, 1x4 weight, nil out grad", Err: operation.ErrExec},
			in:       fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{1, 2}},
			weight:   fabrics.WeightParameters{MatrixParameters: fabrics.MatrixParameters{Rows: 1, Cols: 4, Values: []float64{3, 4, 5, 6}}},
			out:      fabrics.MatrixParameters{Rows: 2, Cols: 4, Values: []float64{3, 4, 5, 6, 6, 8, 10, 12}},
			outGrad:  fabrics.MatrixParameters{Rows: 2, Cols: 4, Values: []float64{7, 8, 9, 10, 11, 12, 13, 14}},
			expected: fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{158, 230}},
			forward:  true,
			nilCheck: true,
		},
	}

	for i := range tests {
		t.Run(tests[i].Name, func(t *testing.T) {
			weight := fabrics.NewWeight(t, tests[i].weight)
			in := fabrics.NewMatrix(t, tests[i].in)
			if tests[i].forward {
				out, err := weight.Forward(in)
				require.NoError(t, err)
				outExpected := fabrics.NewMatrix(t, tests[i].out)
				require.True(t, out.Equal(outExpected))
			}
			outGrad := fabrics.NewMatrix(t, tests[i].outGrad)
			if tests[i].nilCheck {
				outGrad = nil
			}
			inGrad, err := weight.Backward(outGrad)
			if tests[i].Err == nil {
				require.NoError(t, err)
				expected := fabrics.NewMatrix(t, tests[i].expected)
				require.True(t, inGrad.Equal(expected))
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, tests[i].Err)
			}
		})
	}
}

func TestWeight_ApplyOptim(t *testing.T) {
	tests := []struct {
		utils.Base
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
			Base:     utils.Base{Name: "2x1 input, 1x4 weight, 2x4 out grad"},
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
			Base:     utils.Base{Name: "2x1 input, 1x4 weight, 2x4 out grad, no backward", Err: operation.ErrExec},
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
			Base:     utils.Base{Name: "2x1 input, 1x4 weight, 2x4 out grad, incorrect optim", Err: operation.ErrExec},
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
			Base:     utils.Base{Name: "2x1 input, 1x4 weight, 2x4 out grad", Err: operation.ErrExec},
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

	for i := range tests {
		t.Run(tests[i].Name, func(t *testing.T) {
			weight := fabrics.NewWeight(t, tests[i].weight)
			in := fabrics.NewMatrix(t, tests[i].in)
			out, err := weight.Forward(in)
			require.NoError(t, err)
			outExpected := fabrics.NewMatrix(t, tests[i].out)
			require.True(t, out.Equal(outExpected))
			outGrad := fabrics.NewMatrix(t, tests[i].outGrad)
			if tests[i].backward {
				_, err = weight.Backward(outGrad)
				require.NoError(t, err)
			}
			if tests[i].nilCheck {
				err = weight.ApplyOptim(nil)
			} else {
				err = weight.ApplyOptim(tests[i].optim)
			}
			if tests[i].Err == nil {
				require.NoError(t, err)
				actual := weight.Parameter()
				expected := fabrics.NewMatrix(t, tests[i].expected)
				require.True(t, actual.Equal(expected))
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, tests[i].Err)
			}
		})
	}
}
