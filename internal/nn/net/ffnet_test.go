package net

import (
	"github.com/stretchr/testify/require"
	"nn/internal/nn/layer"
	"nn/internal/nn/layer/layertestutils"
	"nn/internal/nn/loss"
	"nn/internal/nn/loss/losstestutils"
	"nn/internal/nn/operation"
	"nn/internal/nn/operation/operationtestutils"
	"nn/internal/testutils"
	"nn/internal/testutils/testfactories"
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
			loss: losstestutils.NewLoss(t, loss.MSELoss),
			layers: []layer.ILayer{
				layertestutils.NewLayer(t, layer.DenseDropLayer,
					testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 3}),
					testfactories.NewVector(t, testfactories.VectorParameters{Size: 3}),
					operationtestutils.NewOperation(t, operation.SigmoidParamActivation,
						testfactories.NewVector(t, testfactories.VectorParameters{Size: 3}),
					),
					percent.Percent80,
				),
				layertestutils.NewLayer(t, layer.DenseLayer,
					testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 3, Cols: 1}),
					testfactories.NewVector(t, testfactories.VectorParameters{Size: 1}),
					operationtestutils.NewOperation(t, operation.LinearActivation),
				),
			},
		},
		{
			Base: testutils.Base{Name: "nil loss", Err: ErrCreate},
			layers: []layer.ILayer{
				layertestutils.NewLayer(t, layer.DenseDropLayer,
					testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 3}),
					testfactories.NewVector(t, testfactories.VectorParameters{Size: 3}),
					operationtestutils.NewOperation(t, operation.SigmoidParamActivation,
						testfactories.NewVector(t, testfactories.VectorParameters{Size: 3}),
					),
					percent.Percent80,
				),
				layertestutils.NewLayer(t, layer.DenseLayer,
					testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 3, Cols: 1}),
					testfactories.NewVector(t, testfactories.VectorParameters{Size: 1}),
					operationtestutils.NewOperation(t, operation.LinearActivation),
				),
			},
		},
		{
			Base: testutils.Base{Name: "nil layers", Err: ErrCreate},
			loss: losstestutils.NewLoss(t, loss.MSELoss),
		},
		{
			Base:   testutils.Base{Name: "empty layers", Err: ErrCreate},
			loss:   losstestutils.NewLoss(t, loss.MSELoss),
			layers: []layer.ILayer{},
		},
		{
			Base: testutils.Base{Name: "layers sizes mismatch", Err: ErrCreate},
			loss: losstestutils.NewLoss(t, loss.MSELoss),
			layers: []layer.ILayer{
				layertestutils.NewLayer(t, layer.DenseDropLayer,
					testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 3}),
					testfactories.NewVector(t, testfactories.VectorParameters{Size: 3}),
					operationtestutils.NewOperation(t, operation.SigmoidParamActivation,
						testfactories.NewVector(t, testfactories.VectorParameters{Size: 3}),
					),
					percent.Percent80,
				),
				layertestutils.NewLayer(t, layer.DenseLayer,
					testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 1}),
					testfactories.NewVector(t, testfactories.VectorParameters{Size: 1}),
					operationtestutils.NewOperation(t, operation.LinearActivation),
				),
			},
		},
	}

	for _, tc := range testcases {
		t.Run(tc.Name, func(t *testing.T) {
			_, err := NewFFNetwork(tc.loss, tc.layers...)
			if tc.Err == nil {
				require.NoError(t, err)
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, tc.Err)
			}
		})
	}
}
