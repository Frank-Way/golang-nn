package net

import (
	"github.com/stretchr/testify/require"
	"nn/internal/nn"
	"nn/internal/nn/layer"
	"nn/internal/nn/layer/layertestutils"
	"nn/internal/nn/loss"
	"nn/internal/nn/loss/losstestutils"
	"nn/internal/nn/operation"
	"nn/internal/nn/operation/operationtestutils"
	"nn/internal/testutils"
	"nn/internal/testutils/testfactories"
	"nn/internal/utils"
	"nn/pkg/percent"
	"testing"
)

func TestNewBuilder(t *testing.T) {
	kinds := []nn.Kind{
		FFNetwork,
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
		_, err := NewBuilder(kind)
		if i < pivot {
			require.NoError(t, err)
		} else {
			require.Error(t, err)
			require.ErrorIs(t, err, ErrBuilder)
		}
	}
}

func TestBuilder_Build(t *testing.T) {
	newBuilder := func(kind nn.Kind) *Builder {
		res, err := NewBuilder(kind)
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
		builder   *Builder
		expected  INetwork
		skipCheck bool
	}{
		{
			Base: testutils.Base{Name: "ffnet, basic build"},
			builder: newBuilder(FFNetwork).
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
			builder: newBuilder(FFNetwork).
				AddLayerKind(layer.DenseDropLayer).
				AddWeight(operationtestutils.NewOperation(t, operation.WeightMultiply,
					testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 3, Values: rawW_6}))).
				AddBias(operationtestutils.NewOperation(t, operation.BiasAdd,
					testfactories.NewVector(t, testfactories.VectorParameters{Values: rawB_3}))).
				AddActivation(operationtestutils.NewOperation(t, operation.TanhActivation)).
				AddDropout(operationtestutils.NewOperation(t, operation.Dropout, percent.Percent80)).
				AddLayerKind(layer.DenseLayer).
				AddWeight(operationtestutils.NewOperation(t, operation.WeightMultiply,
					testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 3, Cols: 1, Values: rawW_3}))).
				AddBias(operationtestutils.NewOperation(t, operation.BiasAdd,
					testfactories.NewVector(t, testfactories.VectorParameters{Values: rawB_1}))).
				AddActivation(operationtestutils.NewOperation(t, operation.LinearActivation)).
				Loss(losstestutils.NewLoss(t, loss.MSELoss)),
			expected: newNetwork(t, FFNetwork,
				losstestutils.NewLoss(t, loss.MSELoss),
				layertestutils.NewLayer(t, layer.DenseDropLayer,
					testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 3, Values: rawW_6}),
					testfactories.NewVector(t, testfactories.VectorParameters{Values: rawB_3}),
					operationtestutils.NewOperation(t, operation.TanhActivation),
					percent.Percent80),
				layertestutils.NewLayer(t, layer.DenseLayer,
					testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 3, Cols: 1, Values: rawW_3}),
					testfactories.NewVector(t, testfactories.VectorParameters{Values: rawB_1}),
					operationtestutils.NewOperation(t, operation.LinearActivation))),
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
