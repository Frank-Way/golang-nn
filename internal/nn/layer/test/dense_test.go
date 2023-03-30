package test

import (
	"github.com/stretchr/testify/require"
	"math"
	"nn/internal/nn/layer"
	"nn/internal/nn/operation"
	"nn/internal/testutils"
	"nn/internal/testutils/fabrics"
	"nn/pkg/mmath/matrix"
	"nn/pkg/mmath/vector"
	"nn/pkg/percent"
	"testing"
)

func TestNewDenseLayer(t *testing.T) {
	testcases := []struct {
		testutils.Base
		weight     *matrix.Matrix
		bias       *vector.Vector
		activation operation.IOperation
	}{
		{
			Base:       testutils.Base{Name: "correct parameters"},
			weight:     fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 1, Cols: 2, Values: []float64{1, 2}}),
			bias:       fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{3, 4}}),
			activation: fabrics.NewOperation(t, operation.TanhActivation),
		},
		{
			Base:       testutils.Base{Name: "neurons count mismatch", Err: layer.ErrCreate},
			weight:     fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 1, Cols: 2, Values: []float64{1, 2}}),
			bias:       fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{3, 4, 5}}),
			activation: fabrics.NewOperation(t, operation.TanhActivation),
		},
		{
			Base:       testutils.Base{Name: "not activation", Err: layer.ErrCreate},
			weight:     fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 1, Cols: 2, Values: []float64{1, 2}}),
			bias:       fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{3, 4}}),
			activation: fabrics.NewOperation(t, operation.Dropout, percent.Percent80),
		},
		{
			Base:       testutils.Base{Name: "wrong sigmoid param coeffs count", Err: layer.ErrCreate},
			weight:     fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 1, Cols: 2, Values: []float64{1, 2}}),
			bias:       fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{3, 4}}),
			activation: fabrics.NewOperation(t, operation.SigmoidParamActivation, fabrics.NewVector(t, fabrics.VectorParameters{Size: 3})),
		},
	}

	for _, tc := range testcases {
		t.Run(tc.Name, func(t *testing.T) {
			_, err := layer.NewDenseLayer(tc.weight, tc.bias, tc.activation)
			if tc.Err == nil {
				require.NoError(t, err)
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, tc.Err)
			}
		})
	}
}

func TestDenseLayer_Forward(t *testing.T) {
	testcases := []struct {
		testutils.Base
		l        layer.ILayer
		in       *matrix.Matrix
		expected *matrix.Matrix
	}{
		{
			Base: testutils.Base{Name: "correct parameters"},
			l: fabrics.NewLayer(t, layer.DenseLayer,
				fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}}),
				fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{7, 8, 9}}),
				fabrics.NewOperation(t, operation.TanhActivation),
			),
			in: fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 1, Cols: 2, Values: []float64{10, 11}}),
			expected: fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 1, Cols: 3,
				Values: []float64{math.Tanh(10*1 + 11*4 + 7), math.Tanh(10*2 + 11*5 + 8), math.Tanh(10*3 + 11*6 + 9)},
			}),
		},
		{
			Base: testutils.Base{Name: "nil input", Err: layer.ErrExec},
			l: fabrics.NewLayer(t, layer.DenseLayer,
				fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}}),
				fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{7, 8, 9}}),
				fabrics.NewOperation(t, operation.TanhActivation),
			),
		},
		{
			Base: testutils.Base{Name: "incorrect input shape", Err: layer.ErrExec},
			l: fabrics.NewLayer(t, layer.DenseLayer,
				fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}}),
				fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{7, 8, 9}}),
				fabrics.NewOperation(t, operation.TanhActivation),
			),
			in: fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 1, Cols: 3}),
		},
	}

	for _, tc := range testcases {
		t.Run(tc.Name, func(t *testing.T) {
			y, err := tc.l.Forward(tc.in)
			if tc.Err == nil {
				require.NoError(t, err)
				require.True(t, y.EqualApprox(tc.expected))
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, tc.Err)
			}
		})
	}
}

func TestDenseLayer_Backward(t *testing.T) {
	testcases := []struct {
		testutils.Base
		l        layer.ILayer
		in       *matrix.Matrix
		outGrad  *matrix.Matrix
		expected *matrix.Matrix
	}{
		{
			Base: testutils.Base{Name: "correct parameters"},
			l: fabrics.NewLayer(t, layer.DenseLayer,
				fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}}),
				fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{7, 8, 9}}),
				fabrics.NewOperation(t, operation.TanhActivation),
			),
			in:      fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 1, Cols: 2, Values: []float64{10, 11}}),
			outGrad: fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 1, Cols: 3, Values: []float64{12, 13, 14}}),
			expected: fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 1, Cols: 2,
				Values: []float64{(12-12*12)*1 + (13-13*13)*2 + (14-14*14)*3, (12-12*12)*4 + (13-13*13)*5 + (14-14*14)*6},
			}),
		},
		{
			Base: testutils.Base{Name: "nil out grad", Err: layer.ErrExec},
			l: fabrics.NewLayer(t, layer.DenseLayer,
				fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}}),
				fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{7, 8, 9}}),
				fabrics.NewOperation(t, operation.TanhActivation),
			),
			in: fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 1, Cols: 2, Values: []float64{10, 11}}),
		},
		{
			Base: testutils.Base{Name: "no Forward() call", Err: layer.ErrExec},
			l: fabrics.NewLayer(t, layer.DenseLayer,
				fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}}),
				fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{7, 8, 9}}),
				fabrics.NewOperation(t, operation.TanhActivation),
			),
			outGrad: fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 1, Cols: 3, Values: []float64{12, 13, 14}}),
		},
		{
			Base: testutils.Base{Name: "wrong shape of out grad", Err: layer.ErrExec},
			l: fabrics.NewLayer(t, layer.DenseLayer,
				fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}}),
				fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{7, 8, 9}}),
				fabrics.NewOperation(t, operation.TanhActivation),
			),
			in:      fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 1, Cols: 2, Values: []float64{10, 11}}),
			outGrad: fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 1, Cols: 4}),
		},
	}

	for _, tc := range testcases {
		t.Run(tc.Name, func(t *testing.T) {
			x := tc.in
			if tc.in != nil {
				_, err := tc.l.Forward(x)
				require.NoError(t, err)
			}
			inGrad, err := tc.l.Backward(tc.outGrad)
			if tc.Err == nil {
				require.NoError(t, err)
				require.True(t, inGrad.EqualApprox(tc.expected))
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, tc.Err)
			}
		})
	}
}

func TestDenseLayer_ApplyOptim(t *testing.T) {
	optimizer := func(param, grad *matrix.Matrix) (*matrix.Matrix, error) {
		return param.Sub(grad)
	}
	testcases := []struct {
		testutils.Base
		l         layer.ILayer
		in        *matrix.Matrix
		outGrad   *matrix.Matrix
		optimizer operation.Optimizer
	}{
		{
			Base: testutils.Base{Name: "correct parameters"},
			l: fabrics.NewLayer(t, layer.DenseLayer,
				fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}}),
				fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{7, 8, 9}}),
				fabrics.NewOperation(t, operation.TanhActivation),
			),
			in:        fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 1, Cols: 2, Values: []float64{10, 11}}),
			outGrad:   fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 1, Cols: 3, Values: []float64{12, 13, 14}}),
			optimizer: optimizer,
		},
		{
			Base: testutils.Base{Name: "no optimizer", Err: layer.ErrExec},
			l: fabrics.NewLayer(t, layer.DenseLayer,
				fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}}),
				fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{7, 8, 9}}),
				fabrics.NewOperation(t, operation.TanhActivation),
			),
			in:      fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 1, Cols: 2, Values: []float64{10, 11}}),
			outGrad: fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 1, Cols: 3, Values: []float64{12, 13, 14}}),
		},
		{
			Base: testutils.Base{Name: "no Backward() call", Err: layer.ErrExec},
			l: fabrics.NewLayer(t, layer.DenseLayer,
				fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}}),
				fabrics.NewVector(t, fabrics.VectorParameters{Values: []float64{7, 8, 9}}),
				fabrics.NewOperation(t, operation.TanhActivation),
			),
			in:        fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 1, Cols: 2, Values: []float64{10, 11}}),
			optimizer: optimizer,
		},
	}

	for _, tc := range testcases {
		t.Run(tc.Name, func(t *testing.T) {
			_, err := tc.l.Forward(tc.in)
			require.NoError(t, err)

			if tc.outGrad != nil {
				_, err := tc.l.Backward(tc.outGrad)
				require.NoError(t, err)
			}

			err = tc.l.ApplyOptim(tc.optimizer)
			if tc.Err == nil {
				require.NoError(t, err)
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, tc.Err)
			}
		})
	}
}
