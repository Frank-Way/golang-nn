package train

import (
	"github.com/stretchr/testify/require"
	"nn/internal/data/approx/datagen"
	"nn/internal/data/dataset"
	"nn/internal/nn/layer"
	"nn/internal/nn/loss"
	"nn/internal/nn/net"
	"nn/internal/nn/operation"
	"nn/internal/optim"
	"testing"
)

func TestTrain(t *testing.T) {
	nb, err := net.NewBuilder(net.FFNetwork)
	require.NoError(t, err)
	nb = nb.
		AddLayerKind(layer.DenseLayer).
		AddInputsCount(1).
		AddNeuronsCount(8).
		AddParamInitType(operation.GlorotInit).
		AddActivationKind(operation.TanhActivation).
		AddLayerKind(layer.DenseLayer).
		AddInputsCount(8).
		AddNeuronsCount(1).
		AddActivationKind(operation.LinearActivation).
		LossKind(loss.MSELoss)

	ir, err := datagen.NewInputRange(0, 1, 128)
	require.NoError(t, err)

	dp, err := datagen.NewParameters(
		"(sin (* 3.14 (/ x0 4)))",
		[]*datagen.InputRange{ir},
		nil,
	)
	require.NoError(t, err)

	ds, err := dp.Generate()
	require.NoError(t, err)
	ds = ds.Shuffle()

	ec := 200

	p := &Parameters{
		EpochsCount: ec,
		NetProvider: func() (net.INetwork, error) {
			return nb.Build()
		},
		DatasetProvider: func() (*dataset.Dataset, error) {
			return ds.Copy(), nil
		},
		OptimizerProvider: func() (operation.Optimizer, optim.PostOptimizeFunc, error) {
			sgd, f := optim.NewSGD(&optim.SGDParameters{
				LearnRate:     0.1,
				StopLearnRate: 0.0001,
				EpochsCount:   ec,
				DecrementType: optim.LinearDecrement,
			})
			return sgd, f, nil
		},
	}
	_, err = Train(p)
	require.NoError(t, err)
}
