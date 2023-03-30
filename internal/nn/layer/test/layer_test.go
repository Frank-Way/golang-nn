package test

import (
	"fmt"
	"github.com/stretchr/testify/require"
	"nn/internal/nn/layer"
	"nn/internal/nn/operation"
	"nn/internal/testutils"
	"nn/internal/testutils/fabrics"
	"nn/pkg/mmath/matrix"
	"nn/pkg/percent"
	"testing"
)

func TestLayer_Strings(t *testing.T) {
	optimizer := func(param, grad *matrix.Matrix) (*matrix.Matrix, error) {
		return param.Sub(grad)
	}
	testcases := []struct {
		l         layer.ILayer
		in        *matrix.Matrix
		outGrad   *matrix.Matrix
		optimizer operation.Optimizer
	}{
		{
			l: fabrics.NewLayer(t, layer.DenseLayer,
				fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 10}),
				fabrics.NewVector(t, fabrics.VectorParameters{Size: 10}),
				fabrics.NewOperation(t, operation.SigmoidParamActivation, fabrics.NewVector(t, fabrics.VectorParameters{Size: 10})),
			),
			in:        fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 1, Cols: 2}),
			outGrad:   fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 1, Cols: 10}),
			optimizer: optimizer,
		},
		{
			l: fabrics.NewLayer(t, layer.DenseLayer,
				fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 10}),
				fabrics.NewVector(t, fabrics.VectorParameters{Size: 10}),
				fabrics.NewOperation(t, operation.SigmoidParamActivation, fabrics.NewVector(t, fabrics.VectorParameters{Size: 10})),
			),
			in:      fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 1, Cols: 2}),
			outGrad: fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 1, Cols: 10}),
		},
		{
			l: fabrics.NewLayer(t, layer.DenseLayer,
				fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 10}),
				fabrics.NewVector(t, fabrics.VectorParameters{Size: 10}),
				fabrics.NewOperation(t, operation.SigmoidParamActivation, fabrics.NewVector(t, fabrics.VectorParameters{Size: 10})),
			),
			in: fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 1, Cols: 2}),
		},
		{
			l: fabrics.NewLayer(t, layer.DenseLayer,
				fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 10}),
				fabrics.NewVector(t, fabrics.VectorParameters{Size: 10}),
				fabrics.NewOperation(t, operation.SigmoidParamActivation, fabrics.NewVector(t, fabrics.VectorParameters{Size: 10})),
			),
		},
		{
			l: fabrics.NewLayer(t, layer.DenseDropLayer,
				fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 10}),
				fabrics.NewVector(t, fabrics.VectorParameters{Size: 10}),
				fabrics.NewOperation(t, operation.SigmoidParamActivation, fabrics.NewVector(t, fabrics.VectorParameters{Size: 10})),
				percent.Percent50,
			),
			in:        fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 1, Cols: 2}),
			outGrad:   fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 1, Cols: 10}),
			optimizer: optimizer,
		},
		{
			l: fabrics.NewLayer(t, layer.DenseDropLayer,
				fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 10}),
				fabrics.NewVector(t, fabrics.VectorParameters{Size: 10}),
				fabrics.NewOperation(t, operation.SigmoidParamActivation, fabrics.NewVector(t, fabrics.VectorParameters{Size: 10})),
				percent.Percent50,
			),
			in:      fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 1, Cols: 2}),
			outGrad: fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 1, Cols: 10}),
		},
		{
			l: fabrics.NewLayer(t, layer.DenseDropLayer,
				fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 10}),
				fabrics.NewVector(t, fabrics.VectorParameters{Size: 10}),
				fabrics.NewOperation(t, operation.SigmoidParamActivation, fabrics.NewVector(t, fabrics.VectorParameters{Size: 10})),
				percent.Percent50,
			),
			in: fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 1, Cols: 2}),
		},
		{
			l: fabrics.NewLayer(t, layer.DenseDropLayer,
				fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 10}),
				fabrics.NewVector(t, fabrics.VectorParameters{Size: 10}),
				fabrics.NewOperation(t, operation.SigmoidParamActivation, fabrics.NewVector(t, fabrics.VectorParameters{Size: 10})),
				percent.Percent50,
			),
		},
	}

	testutils.SetupLogger()
	for i, tc := range testcases {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			if tc.in != nil {
				_, err := tc.l.Forward(tc.in)
				require.NoError(t, err)

				if tc.outGrad != nil {
					_, err = tc.l.Backward(tc.outGrad)
					require.NoError(t, err)

					if tc.optimizer != nil {
						err = tc.l.ApplyOptim(tc.optimizer)
						require.NoError(t, err)
					}
				}
			}

			t.Log("ShortString()\n" + tc.l.ShortString())
			t.Log("String()\n" + tc.l.String())
			t.Log("PrettyString()\n" + tc.l.PrettyString())
		})
	}
}
