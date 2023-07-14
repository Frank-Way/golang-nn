package train

import (
	"fmt"
	"math"
	"nn/internal/data/dataset"
	"nn/internal/nn/net"
	"nn/internal/nn/operation"
	"nn/internal/optim"
	"nn/pkg/mylog"
	"nn/pkg/wraperr"
)

type SingleParameters struct {
	EpochsCount      int
	Network          net.INetwork
	Dataset          *dataset.Dataset
	Optimizer        operation.Optimizer
	PostOptimizeFunc optim.PostOptimizeFunc
}

func checkSingleParameters(p *SingleParameters) (err error) {
	defer wraperr.WrapError(ErrParameters, &err)

	if p == nil {
		return fmt.Errorf("no parameters provided")
	} else if p.Network == nil {
		return fmt.Errorf("no network provided")
	} else if p.Dataset == nil {
		return fmt.Errorf("no dataset provided")
	} else if p.Optimizer == nil {
		return fmt.Errorf("no optimizer provided")
	} else if p.PostOptimizeFunc == nil {
		return fmt.Errorf("no post optimize func provided")
	} else if p.EpochsCount < 1 {
		return fmt.Errorf("invalid epochs count provided: %d", p.EpochsCount)
	}
	return nil
}

type SingleResult struct {
	Network net.INetwork
	Loss    float64
}

func SingleTrain(parameters *SingleParameters) (r *SingleResult, err error) {
	defer logger.CatchErr(&err)
	defer wraperr.WrapError(ErrExec, &err)

	if err = checkSingleParameters(parameters); err != nil {
		return nil, fmt.Errorf("error checking parameters for single train run: %w", err)
	}

	logger.Infof("start single train run for parameters: epochs count [%d], network [%s], dataset [%s]",
		parameters.EpochsCount, parameters.Network.ShortString(), parameters.Dataset.ShortString())

	trainData := parameters.Dataset.Train.Copy()

	var loss float64
	bestLoss := math.MaxFloat64
	var bestNetwork net.INetwork
	var bestEpoch int

	for i := 0; i < parameters.EpochsCount; i++ {
		if i%(parameters.EpochsCount/10) == 0 {
			loss, err = calcAndPrintLoss(parameters.Network, parameters.Dataset.Tests, mylog.Debug,
				fmt.Sprintf("loss on tests data on [%d/%d] epoch", i, parameters.EpochsCount))
			if err != nil {
				return nil, fmt.Errorf("error calculating loss on epoch [%d]: %w", i, err)
			}
			if loss < bestLoss {
				bestEpoch = i
				bestLoss = loss
				bestNetwork = parameters.Network.Copy().(net.INetwork)
			} else {
				logger.Debugf("loss [%e] became worse for epoch [%d] comparing to [%e] at epoch [%d]",
					loss, i, bestLoss, bestEpoch)
			}
		}

		trainData, _ = trainData.Shuffle()
		if _, err = parameters.Network.Forward(trainData.X); err != nil {
			return nil, err
		}
		if _, err = parameters.Network.Loss(trainData.Y); err != nil {
			return nil, err
		}
		if _, err = parameters.Network.Backward(); err != nil {
			return nil, err
		}
		if err = parameters.Network.ApplyOptim(parameters.Optimizer); err != nil {
			return nil, err
		}
		parameters.PostOptimizeFunc()
	}
	if validLossEndTrained, err := calcAndPrintLoss(parameters.Network, parameters.Dataset.Valid, mylog.Info, "loss on valid data after train"); err != nil {
		return nil, err
	} else if loss != bestLoss {
		validLossSaved, err := calcAndPrintLoss(bestNetwork, parameters.Dataset.Valid, mylog.Trace, "")
		if err == nil && validLossEndTrained < validLossSaved {
			bestNetwork = parameters.Network
		} else {
			logger.Debugf("using network saved on [%d] epoch as it produces better loss on valid data: [%e] < [%e]",
				bestEpoch, validLossSaved, validLossEndTrained)
		}
	}

	loss, err = calcAndPrintLoss(bestNetwork, parameters.Dataset.Combine(), mylog.Info, "loss on all (combined) data after train")
	if err != nil {
		return nil, err
	}

	logger.Infof("done train network [%s], loss: %e", parameters.Network.ShortString(), loss)

	return &SingleResult{
		Network: parameters.Network,
		Loss:    loss,
	}, nil
}

func calcAndPrintLoss(network net.INetwork, data *dataset.Data, level mylog.Level, msg string) (l float64, err error) {
	if _, err = network.Forward(data.X); err != nil {
		return 0, err
	}
	loss, err := network.Loss(data.Y)
	if err != nil {
		return 0, err
	}
	logger.Logf(level, "%s: %e", msg, loss)
	return loss, nil
}
