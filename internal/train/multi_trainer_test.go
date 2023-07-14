package train

import (
	"github.com/stretchr/testify/require"
	"nn/internal/data/approx/datagen"
	"nn/internal/data/approx/estimate"
	"nn/internal/data/dataset"
	"nn/internal/nn/layer"
	"nn/internal/nn/loss"
	"nn/internal/nn/net"
	"nn/internal/nn/operation"
	"nn/internal/optim"
	"nn/pkg/percent"
	"nn/pkg/prettytable"
	"testing"
)

func TestTrain(t *testing.T) {
	//mylog.Setup(mylog.LeveledWriter{
	//	Level:  mylog.Debug,
	//	Writer: os.Stdout,
	//})
	nb, err := net.NewBuilder(net.FFNetwork)
	require.NoError(t, err)
	nb = nb.
		SetResetAfterBuild(true).
		AddLayerKind(layer.DenseLayer).
		AddInputsCount(1).
		AddNeuronsCount(16).
		AddParamInitType(operation.GlorotInit).
		AddActivationKind(operation.TanhActivation).
		AddLayerKind(layer.DenseLayer).
		AddInputsCount(16).
		AddNeuronsCount(1).
		AddActivationKind(operation.LinearActivation).
		LossKind(loss.MSELoss)

	ir1 := &datagen.InputRange{Left: 0, Right: 1,
		TrainParameters: &datagen.InputsGenerationParameters{Count: 100,
			Extension: &datagen.ExtendParameters{Left: percent.Percent10, Right: percent.Percent20},
		},
		TestsParameters: &datagen.InputsGenerationParameters{Count: 50},
		ValidParameters: &datagen.InputsGenerationParameters{Count: 25},
	}

	//ir2 := &datagen.InputRange{Left:  0, Right: 1,
	//	TrainParameters: &datagen.InputsGenerationParameters{Count: 16},
	//	TestsParameters: &datagen.InputsGenerationParameters{Count: 8},
	//	ValidParameters: &datagen.InputsGenerationParameters{Count: 6},
	//}

	dp, err := datagen.NewParameters(
		"(sin (* x0 (/ 3.14 4)))",
		ir1,
	)

	//dp, err := datagen.NewParameters(
	//	"(+ (sin (* x0 (/ 3.14 4))) x1)",
	//	ir1,
	//	ir2,
	//)
	require.NoError(t, err)

	ds, err := datagen.Generate(dp)
	require.NoError(t, err)

	ec := 500
	retries := 2

	p := &MultiParameters{
		SingleParameters: SingleParameters{EpochsCount: ec},
		RetriesCount:     retries,
		Parallel:         true,
		NetProvider: func() (net.INetwork, error) {
			return nb.Build()
		},
		DatasetProvider: func() (*dataset.Dataset, error) {
			return ds.Copy(), nil
		},
		OptimizerProvider: func() (operation.Optimizer, optim.PostOptimizeFunc, error) {
			sgd, f := optim.NewSGD(&optim.SGDParameters{
				LearnRate:     0.1,
				StopLearnRate: 0.00001,
				EpochsCount:   ec,
				DecrementType: optim.LinearDecrement,
			})
			return sgd, f, nil
		},
	}
	results, err := MultiTrain(p)
	require.NoError(t, err)

	require.Equal(t, retries, len(results.NetworkResults))

	outputs, err := results.BestNetwork.Forward(ds.Valid.X)
	require.NoError(t, err)
	estimation, err := estimate.Estimate(outputs, ds.Valid.Y)
	require.NoError(t, err)
	t.Logf("\nmae %f, mrep %f %%, aae %f",
		estimation.MaxAbsoluteError, estimation.MaxRelativeErrorPercents, estimation.AvgAbsoluteError)

	table, err := prettytable.Build([]*prettytable.Group{
		prettytable.MatrixToGroup("inputs", ds.Valid.X),
		prettytable.MatrixToGroup("targets", ds.Valid.Y),
		prettytable.MatrixToGroup("outputs", outputs),
		prettytable.MatrixToGroup("deltas", estimation.Deltas),
		prettytable.MatrixToGroup("abs deltas", estimation.AbsoluteDeltas),
		prettytable.MatrixToGroup("rel deltas", estimation.RelativeDeltas),
	}, false, 95)
	require.NoError(t, err)
	t.Logf("\n" + table)
}
