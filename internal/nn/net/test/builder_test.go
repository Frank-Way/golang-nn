package test

import (
	"github.com/stretchr/testify/require"
	"nn/internal/nn"
	"nn/internal/nn/layer"
	"nn/internal/nn/loss"
	"nn/internal/nn/net"
	"nn/internal/nn/operation"
	"nn/internal/testutils"
	"nn/internal/testutils/fabrics"
	"nn/internal/utils"
	"nn/pkg/percent"
	"testing"
)

func TestNewBuilder(t *testing.T) {
	kinds := []nn.Kind{
		net.FFNetwork,
		"unknown kind",
		operation.LinearActivation,
		operation.SigmoidActivation,
		operation.TanhActivation,
		operation.SigmoidParamActivation,
		operation.Dropout,
		operation.WeightMultiply,
		operation.BiasAdd,
		loss.MSELoss,
		layer.DenseLayer,
		layer.DenseDropLayer,
	}
	pivot := 1
	for i, kind := range kinds {
		_, err := net.NewBuilder(kind)
		if i < pivot {
			require.NoError(t, err)
		} else {
			require.Error(t, err)
			require.ErrorIs(t, err, net.ErrBuilder)
		}
	}
}

func TestBuilder_Build(t *testing.T) {
	newBuilder := func(kind nn.Kind) *net.Builder {
		res, err := net.NewBuilder(kind)
		require.NoError(t, err)
		return res
	}
	rawW_6 := utils.RandNormArray(6, 0, 1)
	rawW_3 := utils.RandNormArray(3, 0, 1)
	//rawW_1 := utils.RandNormArray(3, 0, 1)
	//rawB_6 := utils.RandNormArray(6, 0, 1)
	rawB_3 := utils.RandNormArray(3, 0, 1)
	rawB_1 := utils.RandNormArray(1, 0, 1)
	testcases := []struct {
		testutils.Base
		builder   *net.Builder
		expected  net.INetwork
		skipCheck bool
	}{
		{
			Base: testutils.Base{Name: "ffnet, basic build"},
			builder: newBuilder(net.FFNetwork).
				AddLayerKind(layer.DenseDropLayer).
				AddInputsCount(2).
				AddNeuronsCount(3).
				AddParamInitType(operation.GlorotInit).
				AddActivationKind(operation.TanhActivation).
				AddKeepProbability(percent.Percent80).
				AddLayerKind(layer.DenseLayer).
				AddInputsCount(3).
				AddNeuronsCount(1).
				AddActivationKind(operation.LinearActivation).
				LossKind(loss.MSELoss),
			skipCheck: true,
		},
		{
			Base: testutils.Base{Name: "ffnet, build exact"},
			builder: newBuilder(net.FFNetwork).
				AddLayerKind(layer.DenseDropLayer).
				AddWeight(fabrics.NewOperation(t, operation.WeightMultiply,
					fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 3, Values: rawW_6}))).
				AddBias(fabrics.NewOperation(t, operation.BiasAdd,
					fabrics.NewVector(t, fabrics.VectorParameters{Values: rawB_3}))).
				AddActivation(fabrics.NewOperation(t, operation.TanhActivation)).
				AddDropout(fabrics.NewOperation(t, operation.Dropout, percent.Percent80)).
				AddLayerKind(layer.DenseLayer).
				AddWeight(fabrics.NewOperation(t, operation.WeightMultiply,
					fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 3, Cols: 1, Values: rawW_3}))).
				AddBias(fabrics.NewOperation(t, operation.BiasAdd,
					fabrics.NewVector(t, fabrics.VectorParameters{Values: rawB_1}))).
				AddActivation(fabrics.NewOperation(t, operation.LinearActivation)).
				Loss(fabrics.NewLoss(t, loss.MSELoss)),
			expected: fabrics.NewNetwork(t, net.FFNetwork,
				fabrics.NewLoss(t, loss.MSELoss),
				fabrics.NewLayer(t, layer.DenseDropLayer,
					fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 2, Cols: 3, Values: rawW_6}),
					fabrics.NewVector(t, fabrics.VectorParameters{Values: rawB_3}),
					fabrics.NewOperation(t, operation.TanhActivation),
					percent.Percent80),
				fabrics.NewLayer(t, layer.DenseLayer,
					fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 3, Cols: 1, Values: rawW_3}),
					fabrics.NewVector(t, fabrics.VectorParameters{Values: rawB_1}),
					fabrics.NewOperation(t, operation.LinearActivation))),
		},
	}

	for _, tc := range testcases {
		t.Run(tc.Name, func(t *testing.T) {
			l, err := tc.builder.Build()
			if tc.Err == nil {
				require.NoError(t, err)
				if !tc.skipCheck {
					require.True(t, l.EqualApprox(tc.expected))
				}
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, tc.Err)
			}
		})
	}
}
