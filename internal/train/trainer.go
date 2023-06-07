package train

import (
	"fmt"
	"nn/internal/data/dataset"
	"nn/internal/nn/net"
	"nn/internal/nn/operation"
	"nn/internal/optim"
	"nn/pkg/mylog"
	"nn/pkg/wraperr"
)

type Parameters struct {
	EpochsCount       int
	NetProvider       func() (net.INetwork, error)
	DatasetProvider   func() (*dataset.Dataset, error)
	OptimizerProvider func() (operation.Optimizer, optim.PostOptimizeFunc, error)
}

func checkParameters(p *Parameters) (err error) {
	defer wraperr.WrapError(ErrParameters, &err)

	if p == nil {
		return fmt.Errorf("no parameters provided")
	} else if p.NetProvider == nil {
		return fmt.Errorf("no network provider")
	} else if p.DatasetProvider == nil {
		return fmt.Errorf("no dataset provider")
	} else if p.OptimizerProvider == nil {
		return fmt.Errorf("no optimizer provider")
	} else if p.EpochsCount < 1 {
		return fmt.Errorf("invalid epochs count provided: %d", p.EpochsCount)
	}
	return nil
}

type Results struct {
	Net net.INetwork
}

func preTrain(parameters *Parameters) (n net.INetwork, ds *dataset.Dataset, o operation.Optimizer, f optim.PostOptimizeFunc, err error) {
	defer wraperr.WrapError(ErrPreTrain, &err)

	if err = checkParameters(parameters); err != nil {
		return nil, nil, nil, nil, err
	} else if n, err = parameters.NetProvider(); err != nil {
		return nil, nil, nil, nil, err
	} else if ds, err = parameters.DatasetProvider(); err != nil {
		return nil, nil, nil, nil, err
	} else if o, f, err = parameters.OptimizerProvider(); err != nil {
		return nil, nil, nil, nil, err
	} else {
		return n, ds, o, f, nil
	}
}

func Train(parameters *Parameters) (r *Results, err error) {
	defer logger.CatchErr(&err)
	defer wraperr.WrapError(ErrExec, &err)

	n, ds, o, f, err := preTrain(parameters)
	if err != nil {
		return nil, err
	}

	trainData := ds.Train
	testsData := ds.Tests
	validData := ds.Valid

	for i := 0; i < parameters.EpochsCount; i++ {
		if i%(parameters.EpochsCount/10) == 0 {
			err = calcAndPrintLoss(n, testsData, mylog.Debug,
				fmt.Sprintf("loss on tests data on [%d/%d] epoch", i, parameters.EpochsCount))
			if err != nil {
				return nil, err
			}
		}

		trainData, _ = trainData.Shuffle()
		if _, err = n.Forward(trainData.X); err != nil {
			return nil, err
		}
		if _, err = n.Loss(trainData.Y); err != nil {
			return nil, err
		}
		if _, err = n.Backward(); err != nil {
			return nil, err
		}
		if err = n.ApplyOptim(o); err != nil {
			return nil, err
		}
		f()
	}
	if err = calcAndPrintLoss(n, validData, mylog.Info, "loss on valid data after train"); err != nil {
		return nil, err
	}

	if err = calcAndPrintLoss(n, ds.Combine(), mylog.Info, "loss on all (combined) data after train"); err != nil {
		return nil, err
	}

	return &Results{
		Net: n,
	}, nil
}

func calcAndPrintLoss(network net.INetwork, data *dataset.Data, level mylog.Level, msg string) (err error) {
	_, err = network.Forward(data.X)
	if err != nil {
		return err
	}
	loss, err := network.Loss(data.Y)
	if err != nil {
		return err
	}
	logger.Logf(level, "%s: %e", msg, loss)
	return nil
}
