package train

import (
	"fmt"
	"math"
	"nn/internal/data/dataset"
	"nn/internal/nn/net"
	"nn/internal/nn/operation"
	"nn/internal/optim"
	"nn/pkg/wraperr"
	"sync"
	"time"
)

type MultiParameters struct {
	SingleParameters
	RetriesCount      int
	Parallel          bool
	NetProvider       func() (net.INetwork, error)
	DatasetProvider   func() (*dataset.Dataset, error)
	OptimizerProvider func() (operation.Optimizer, optim.PostOptimizeFunc, error)
}

func checkMultiParameters(p *MultiParameters) (err error) {
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
	} else if p.RetriesCount < 1 {
		return fmt.Errorf("invalid retries count provided: %d", p.RetriesCount)
	}
	return nil
}

func preMultiTrain(parameters *MultiParameters) (sp *SingleParameters, err error) {
	defer wraperr.WrapError(ErrPreTrain, &err)

	if n, err := parameters.NetProvider(); err != nil {
		return nil, err
	} else if ds, err := parameters.DatasetProvider(); err != nil {
		return nil, err
	} else if o, f, err := parameters.OptimizerProvider(); err != nil {
		return nil, err
	} else {
		return &SingleParameters{
			EpochsCount:      parameters.EpochsCount,
			Network:          n,
			Dataset:          ds,
			Optimizer:        o,
			PostOptimizeFunc: f,
		}, nil
	}
}

type MultiResults struct {
	BestNetwork    net.INetwork
	NetworkResults map[net.INetwork]float64
}

func MultiTrain(parameters *MultiParameters) (r *MultiResults, err error) {
	defer logger.CatchErr(&err)
	defer wraperr.WrapError(ErrExec, &err)

	if err = checkMultiParameters(parameters); err != nil {
		return nil, fmt.Errorf("error checking parameters for multi train run")
	}

	if parameters.Parallel {
		return multiTrainParallel(parameters)
	}
	return multiTrainNonParallel(parameters)
}

func multiTrainNonParallel(parameters *MultiParameters) (r *MultiResults, err error) {
	bestLoss := math.MaxFloat64
	var bestNetwork net.INetwork
	networkResults := make(map[net.INetwork]float64)

	for i := 0; i < parameters.RetriesCount; i++ {
		sp, err := preMultiTrain(parameters)
		if err != nil {
			return nil, fmt.Errorf("error preparing for [%d] train: %w", i, err)
		}

		trainResults, err := SingleTrain(sp)
		if err != nil {
			return nil, fmt.Errorf("error running [%d] train: %w", i, err)
		}

		if trainResults.Loss < bestLoss {
			bestLoss = trainResults.Loss
			bestNetwork = trainResults.Network
		}

		networkResults[trainResults.Network] = trainResults.Loss

	}

	return &MultiResults{
		BestNetwork:    bestNetwork,
		NetworkResults: networkResults,
	}, nil
}

func multiTrainParallel(parameters *MultiParameters) (r *MultiResults, err error) {
	bestLoss := math.MaxFloat64
	var bestNetwork net.INetwork
	networkResults := make(map[net.INetwork]float64)

	var wg sync.WaitGroup
	var mu sync.Mutex
	wg.Add(parameters.RetriesCount)
	errCh := make(chan error, parameters.RetriesCount)

	for i := 0; i < parameters.RetriesCount; i++ {
		//<- time.After(50 * time.Millisecond)
		go func(wait *sync.WaitGroup, mutex *sync.Mutex, errorChanel chan<- error, iter int) {
			defer wait.Done()
			//defer close(errorChanel)

			sp, err := preMultiTrain(parameters)
			if err != nil {
				errorChanel <- fmt.Errorf("error preparing for [%d] train: %w", iter, err)
				return
			}

			trainResults, err := SingleTrain(sp)
			if err != nil {
				errorChanel <- fmt.Errorf("error running [%d] train: %w", iter, err)
				return
			}

			mutex.Lock()
			defer mutex.Unlock()

			if trainResults.Loss < bestLoss {
				bestLoss = trainResults.Loss
				bestNetwork = trainResults.Network
			}

			networkResults[trainResults.Network] = trainResults.Loss
		}(&wg, &mu, errCh, i)
	}

	wg.Wait()
	stop := false
	for !stop {
		select {
		case err := <-errCh:
			close(errCh)
			return nil, err
		case <-time.After(10 * time.Millisecond):
			close(errCh)
			stop = true
		}
	}

	return &MultiResults{
		BestNetwork:    bestNetwork,
		NetworkResults: networkResults,
	}, nil
}
