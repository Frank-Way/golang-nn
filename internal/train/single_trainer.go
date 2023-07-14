package train

import (
	"fmt"
	"github.com/google/uuid"
	"math"
	"nn/internal/data/dataset"
	"nn/internal/nn/net"
	"nn/internal/nn/operation"
	"nn/internal/optim"
	"nn/pkg/mmath/matrix"
	"nn/pkg/mylog"
	"nn/pkg/wraperr"
)

type SingleParameters struct {
	TrainId

	EpochsCount      int
	Network          net.INetwork
	Dataset          *dataset.Dataset
	Optimizer        operation.Optimizer
	PostOptimizeFunc optim.PostOptimizeFunc

	TestEpochPicker func(epoch, epochs int) bool

	SaveBest  bool
	SaveStats bool
}

type TrainId struct {
	Id       uuid.UUID
	ParentId uuid.UUID
}

type SingleResult struct {
	TrainId
	*dataset.Dataset
	MainSingleResult
	*BestSingleResult
	*StatsSingleResult
}

type MainSingleResult struct {
	Network net.INetwork
	Forward *matrix.Matrix
	Loss    float64
}

type BestSingleResult struct {
	MainSingleResult
	Epoch int
}

type StatsSingleResult struct {
	ResultsPerEpoch map[int]*MainSingleResult
	Epochs          int
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
	} else if p.TrainId.Id.String() == "" {
		return fmt.Errorf("no single train uuid provided")
	} else if p.TestEpochPicker == nil {
		return fmt.Errorf("no test epoch picker provided")
	}

	return nil
}

func SingleTrain(parameters *SingleParameters) (r *SingleResult, err error) {
	defer logger.CatchErr(&err)
	defer wraperr.WrapError(ErrExec, &err)

	if err = checkSingleParameters(parameters); err != nil {
		return nil, fmt.Errorf("error checking parameters for single train run: %w", err)
	}

	id := parameters.Id.String()
	logger.Infof("start single train run for parameters: parent id [%s], single train id [%s], epochs count "+
		"[%d], network [%s], dataset [%s]",
		parameters.ParentId.String(), id, parameters.EpochsCount, parameters.Network.ShortString(), parameters.Dataset.ShortString())

	trainData := parameters.Dataset.Train.Copy()

	result := &SingleResult{
		TrainId: parameters.TrainId,
		Dataset: parameters.Dataset,
	}
	if parameters.SaveBest {
		result.BestSingleResult = &BestSingleResult{
			MainSingleResult: MainSingleResult{
				Loss: math.MaxFloat64,
			},
		}
	}
	if parameters.SaveStats {
		result.StatsSingleResult = &StatsSingleResult{
			ResultsPerEpoch: make(map[int]*MainSingleResult),
			Epochs:          parameters.EpochsCount,
		}
	}

	for i := 0; i < parameters.EpochsCount; i++ {
		if parameters.TestEpochPicker(i, parameters.EpochsCount) {
			logger.Tracef("evaluating current results on epoch: %d", i)
			loss, forward, err := calcAndPrintLoss(parameters.Network, parameters.Dataset.Tests, mylog.Debug,
				fmt.Sprintf("loss on tests data on [%d/%d] epoch", i, parameters.EpochsCount))
			if err != nil {
				return nil, fmt.Errorf("error calculating loss on epoch [%d]: %w", i, err)
			}

			if parameters.SaveStats {
				logger.Tracef("saving stats on epoch: %d", i)
				result.StatsSingleResult.ResultsPerEpoch[i] = &MainSingleResult{
					Network: parameters.Network.Copy().(net.INetwork),
					Forward: forward,
					Loss:    loss,
				}
			}

			if parameters.SaveBest {
				logger.Tracef("check best results on epoch: %d", i)
				if loss < result.BestSingleResult.Loss {
					if i > 0 {
						logger.Tracef("loss [%e] became better for epoch [%d] comparing to [%e] at epoch [%d]",
							loss, i, result.BestSingleResult.Loss, result.BestSingleResult.Epoch)
					}
					result.BestSingleResult.Epoch = i
					result.BestSingleResult.Network = parameters.Network.Copy().(net.INetwork)
					result.BestSingleResult.Forward = forward.Copy()
					result.BestSingleResult.Loss = loss
				} else {
					logger.Debugf("loss [%e] became worse for epoch [%d] comparing to [%e] at epoch [%d]",
						loss, i, result.BestSingleResult.Loss, result.BestSingleResult.Epoch)
				}
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

	loss, forward, err := calcAndPrintLoss(parameters.Network, parameters.Dataset.Valid, mylog.Info, "loss on valid data after train")
	if err != nil {
		return nil, err
	}

	logger.Infof("done train network [%s], loss: %e", parameters.Network.ShortString(), loss)

	result.Network = parameters.Network.Copy().(net.INetwork)
	result.Loss = loss
	result.Forward = forward

	return result, nil
}

func calcAndPrintLoss(network net.INetwork, data *dataset.Data, level mylog.Level, msg string) (l float64, m *matrix.Matrix, err error) {
	if m, err = network.Forward(data.X); err != nil {
		return 0, nil, err
	}
	l, err = network.Loss(data.Y)
	if err != nil {
		return 0, nil, err
	}
	logger.Logf(level, "%s: %e", msg, l)
	return l, m, nil
}
