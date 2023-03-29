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
	dlp := fabrics.DenseLayerParameters{
		Weight:     fabrics.MatrixParameters{Rows: 2, Cols: 10},
		Bias:       fabrics.VectorParameters{Size: 10},
		Activation: fabrics.SigmoidParamAct,
		ActivationParameters: fabrics.ActivationParameters{
			SigmoidParamParameters: fabrics.VectorParameters{Size: 10},
		},
	}
	ddlp := fabrics.DenseDropLayerParameters{DropoutParameters: fabrics.DropoutParameters{Percent: percent.Percent70}}
	testcases := []struct {
		l         layer.ILayer
		in        *matrix.Matrix
		outGrad   *matrix.Matrix
		optimizer operation.Optimizer
	}{
		{
			l: fabrics.NewDenseLayer(t, fabrics.LayerParameters{
				DenseLayerParameters:     dlp,
				DenseDropLayerParameters: fabrics.DenseDropLayerParameters{},
			}),
			in:        fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 1, Cols: 2}),
			outGrad:   fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 1, Cols: 10}),
			optimizer: optimizer,
		},
		{
			l: fabrics.NewDenseLayer(t, fabrics.LayerParameters{
				DenseLayerParameters:     dlp,
				DenseDropLayerParameters: fabrics.DenseDropLayerParameters{},
			}),
			in:      fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 1, Cols: 2}),
			outGrad: fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 1, Cols: 10}),
		},
		{
			l: fabrics.NewDenseLayer(t, fabrics.LayerParameters{
				DenseLayerParameters:     dlp,
				DenseDropLayerParameters: fabrics.DenseDropLayerParameters{},
			}),
			in: fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 1, Cols: 2}),
		},
		{
			l: fabrics.NewDenseLayer(t, fabrics.LayerParameters{
				DenseLayerParameters:     dlp,
				DenseDropLayerParameters: fabrics.DenseDropLayerParameters{},
			}),
		},
		{
			l: fabrics.NewDenseDropLayer(t, fabrics.LayerParameters{
				DenseLayerParameters:     dlp,
				DenseDropLayerParameters: ddlp,
			}),
			in:        fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 1, Cols: 2}),
			outGrad:   fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 1, Cols: 10}),
			optimizer: optimizer,
		},
		{
			l: fabrics.NewDenseDropLayer(t, fabrics.LayerParameters{
				DenseLayerParameters:     dlp,
				DenseDropLayerParameters: ddlp,
			}),
			in:      fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 1, Cols: 2}),
			outGrad: fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 1, Cols: 10}),
		},
		{
			l: fabrics.NewDenseDropLayer(t, fabrics.LayerParameters{
				DenseLayerParameters:     dlp,
				DenseDropLayerParameters: ddlp,
			}),
			in: fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 1, Cols: 2}),
		},
		{
			l: fabrics.NewDenseDropLayer(t, fabrics.LayerParameters{
				DenseLayerParameters:     dlp,
				DenseDropLayerParameters: ddlp,
			}),
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
