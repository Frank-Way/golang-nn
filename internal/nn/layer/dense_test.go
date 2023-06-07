package layer

import (
	"github.com/stretchr/testify/require"
	"math"
	"nn/internal/nn/operation"
	"nn/internal/nn/operation/operationtestutils"
	"nn/internal/testutils"
	"nn/internal/testutils/testfactories"
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
			weight:     testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 1, Cols: 2, Values: []float64{1, 2}}),
			bias:       testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{3, 4}}),
			activation: operationtestutils.NewOperation(t, operation.TanhActivation),
		},
		{
			Base:       testutils.Base{Name: "neurons count mismatch", Err: ErrCreate},
			weight:     testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 1, Cols: 2, Values: []float64{1, 2}}),
			bias:       testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{3, 4, 5}}),
			activation: operationtestutils.NewOperation(t, operation.TanhActivation),
		},
		{
			Base:       testutils.Base{Name: "not activation", Err: ErrCreate},
			weight:     testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 1, Cols: 2, Values: []float64{1, 2}}),
			bias:       testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{3, 4}}),
			activation: operationtestutils.NewOperation(t, operation.Dropout, percent.Percent80),
		},
		{
			Base:       testutils.Base{Name: "wrong sigmoid param coeffs count", Err: ErrCreate},
			weight:     testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 1, Cols: 2, Values: []float64{1, 2}}),
			bias:       testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{3, 4}}),
			activation: operationtestutils.NewOperation(t, operation.SigmoidParamActivation, testfactories.NewVector(t, testfactories.VectorParameters{Size: 3})),
		},
	}

	for _, tc := range testcases {
		t.Run(tc.Name, func(t *testing.T) {
			_, err := NewDenseLayer(tc.weight, tc.bias, tc.activation)
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
		l        ILayer
		in       *matrix.Matrix
		expected *matrix.Matrix
	}{
		{
			Base: testutils.Base{Name: "correct parameters"},
			l: newLayer(t, DenseLayer,
				testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}}),
				testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{7, 8, 9}}),
				operationtestutils.NewOperation(t, operation.TanhActivation),
			),
			in: testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 1, Cols: 2, Values: []float64{10, 11}}),
			expected: testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 1, Cols: 3,
				Values: []float64{math.Tanh(10*1 + 11*4 + 7), math.Tanh(10*2 + 11*5 + 8), math.Tanh(10*3 + 11*6 + 9)},
			}),
		},
		{
			Base: testutils.Base{Name: "nil input", Err: ErrExec},
			l: newLayer(t, DenseLayer,
				testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}}),
				testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{7, 8, 9}}),
				operationtestutils.NewOperation(t, operation.TanhActivation),
			),
		},
		{
			Base: testutils.Base{Name: "incorrect input shape", Err: ErrExec},
			l: newLayer(t, DenseLayer,
				testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}}),
				testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{7, 8, 9}}),
				operationtestutils.NewOperation(t, operation.TanhActivation),
			),
			in: testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 1, Cols: 3}),
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
		l        ILayer
		in       *matrix.Matrix
		outGrad  *matrix.Matrix
		expected *matrix.Matrix
	}{
		{
			Base: testutils.Base{Name: "correct parameters"},
			l: newLayer(t, DenseLayer,
				testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}}),
				testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{7, 8, 9}}),
				operationtestutils.NewOperation(t, operation.TanhActivation),
			),
			in:      testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 1, Cols: 2, Values: []float64{10, 11}}),
			outGrad: testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 1, Cols: 3, Values: []float64{12, 13, 14}}),
			expected: testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 1, Cols: 2,
				Values: []float64{(12-12*12)*1 + (13-13*13)*2 + (14-14*14)*3, (12-12*12)*4 + (13-13*13)*5 + (14-14*14)*6},
			}),
		},
		{
			Base: testutils.Base{Name: "nil out grad", Err: ErrExec},
			l: newLayer(t, DenseLayer,
				testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}}),
				testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{7, 8, 9}}),
				operationtestutils.NewOperation(t, operation.TanhActivation),
			),
			in: testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 1, Cols: 2, Values: []float64{10, 11}}),
		},
		{
			Base: testutils.Base{Name: "no Forward() call", Err: ErrExec},
			l: newLayer(t, DenseLayer,
				testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}}),
				testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{7, 8, 9}}),
				operationtestutils.NewOperation(t, operation.TanhActivation),
			),
			outGrad: testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 1, Cols: 3, Values: []float64{12, 13, 14}}),
		},
		{
			Base: testutils.Base{Name: "wrong shape of out grad", Err: ErrExec},
			l: newLayer(t, DenseLayer,
				testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}}),
				testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{7, 8, 9}}),
				operationtestutils.NewOperation(t, operation.TanhActivation),
			),
			in:      testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 1, Cols: 2, Values: []float64{10, 11}}),
			outGrad: testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 1, Cols: 4}),
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
		l         ILayer
		in        *matrix.Matrix
		outGrad   *matrix.Matrix
		optimizer operation.Optimizer
	}{
		{
			Base: testutils.Base{Name: "correct parameters"},
			l: newLayer(t, DenseLayer,
				testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}}),
				testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{7, 8, 9}}),
				operationtestutils.NewOperation(t, operation.TanhActivation),
			),
			in:        testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 1, Cols: 2, Values: []float64{10, 11}}),
			outGrad:   testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 1, Cols: 3, Values: []float64{12, 13, 14}}),
			optimizer: optimizer,
		},
		{
			Base: testutils.Base{Name: "no optimizer", Err: ErrExec},
			l: newLayer(t, DenseLayer,
				testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}}),
				testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{7, 8, 9}}),
				operationtestutils.NewOperation(t, operation.TanhActivation),
			),
			in:      testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 1, Cols: 2, Values: []float64{10, 11}}),
			outGrad: testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 1, Cols: 3, Values: []float64{12, 13, 14}}),
		},
		{
			Base: testutils.Base{Name: "no Backward() call", Err: ErrExec},
			l: newLayer(t, DenseLayer,
				testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}}),
				testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{7, 8, 9}}),
				operationtestutils.NewOperation(t, operation.TanhActivation),
			),
			in:        testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 1, Cols: 2, Values: []float64{10, 11}}),
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
