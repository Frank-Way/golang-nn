package test

import (
	"github.com/stretchr/testify/require"
	"nn/internal/nn/layer"
	"nn/internal/nn/loss"
	"nn/internal/nn/net"
	"nn/internal/nn/operation"
	"nn/internal/testutils"
	"nn/internal/testutils/fabrics"
	"nn/pkg/percent"
	"testing"
)

func TestNewFFNetwork(t *testing.T) {
	testcases := []struct {
		testutils.Base
		loss   loss.ILoss
		layers []layer.ILayer
	}{
		{
			Base: testutils.Base{Name: "correct parameters"},
			loss: fabrics.NewLoss(t, loss.MSELoss),
			layers: []layer.ILayer{
				fabrics.NewLayer(t, layer.DenseDropLayer,
					fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 3}),
					fabrics.NewVector(t, fabrics.VectorParameters{Size: 3}),
					fabrics.NewOperation(t, operation.SigmoidParamActivation,
						fabrics.NewVector(t, fabrics.VectorParameters{Size: 3}),
					),
					percent.Percent80,
				),
				fabrics.NewLayer(t, layer.DenseLayer,
					fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 3, Cols: 1}),
					fabrics.NewVector(t, fabrics.VectorParameters{Size: 1}),
					fabrics.NewOperation(t, operation.LinearActivation),
				),
			},
		},
		{
			Base: testutils.Base{Name: "nil loss", Err: net.ErrCreate},
			layers: []layer.ILayer{
				fabrics.NewLayer(t, layer.DenseDropLayer,
					fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 3}),
					fabrics.NewVector(t, fabrics.VectorParameters{Size: 3}),
					fabrics.NewOperation(t, operation.SigmoidParamActivation,
						fabrics.NewVector(t, fabrics.VectorParameters{Size: 3}),
					),
					percent.Percent80,
				),
				fabrics.NewLayer(t, layer.DenseLayer,
					fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 3, Cols: 1}),
					fabrics.NewVector(t, fabrics.VectorParameters{Size: 1}),
					fabrics.NewOperation(t, operation.LinearActivation),
				),
			},
		},
		{
			Base: testutils.Base{Name: "nil layers", Err: net.ErrCreate},
			loss: fabrics.NewLoss(t, loss.MSELoss),
		},
		{
			Base:   testutils.Base{Name: "empty layers", Err: net.ErrCreate},
			loss:   fabrics.NewLoss(t, loss.MSELoss),
			layers: []layer.ILayer{},
		},
		{
			Base: testutils.Base{Name: "layers sizes mismatch", Err: net.ErrCreate},
			loss: fabrics.NewLoss(t, loss.MSELoss),
			layers: []layer.ILayer{
				fabrics.NewLayer(t, layer.DenseDropLayer,
					fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 3}),
					fabrics.NewVector(t, fabrics.VectorParameters{Size: 3}),
					fabrics.NewOperation(t, operation.SigmoidParamActivation,
						fabrics.NewVector(t, fabrics.VectorParameters{Size: 3}),
					),
					percent.Percent80,
				),
				fabrics.NewLayer(t, layer.DenseLayer,
					fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 1}),
					fabrics.NewVector(t, fabrics.VectorParameters{Size: 1}),
					fabrics.NewOperation(t, operation.LinearActivation),
				),
			},
		},
	}

	for _, tc := range testcases {
		t.Run(tc.Name, func(t *testing.T) {
			_, err := net.NewFFNetwork(tc.loss, tc.layers...)
			if tc.Err == nil {
				require.NoError(t, err)
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, tc.Err)
			}
		})
	}
}
