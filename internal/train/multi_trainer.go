package train

import (
	"fmt"
	"github.com/google/uuid"
	"nn/internal/data/dataset"
	"nn/internal/nn/net"
	"nn/internal/nn/operation"
	"nn/internal/optim"
	"nn/internal/utils"
	"nn/pkg/wraperr"
	"sync"
	"time"
)

type MultiParameters struct {
	SingleParameters

	RetriesCount int
	Parallel     bool

	NetProvider       func() (net.INetwork, error)
	DatasetProvider   func() (*dataset.Dataset, error)
	OptimizerProvider func() (operation.Optimizer, optim.PostOptimizeFunc, error)
}

type MultiResults struct {
	TrainId

	BestResults *SingleResult
	AllResults  []*SingleResult
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
	} else if p.TrainId.Id.String() == "" {
		return fmt.Errorf("no multi train uuid provided")
	} else if p.TestEpochPicker == nil {
		return fmt.Errorf("no test epoch picker provided")
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
			TrainId: TrainId{
				Id:       uuid.New(),
				ParentId: parameters.Id,
			},
			EpochsCount:      parameters.EpochsCount,
			Network:          n,
			Dataset:          ds,
			Optimizer:        o,
			PostOptimizeFunc: f,
			TestEpochPicker:  parameters.TestEpochPicker,
			SaveBest:         parameters.SaveBest,
			SaveStats:        parameters.SaveStats,
		}, nil
	}
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

func getBestResultFinder() func(result interface{}) interface{} {
	var bestResult *SingleResult
	first := true
	return func(result interface{}) interface{} {
		if casted, ok := result.(*SingleResult); !ok {
			return result
		} else if first || bestResult.Loss < casted.Loss {
			bestResult = casted
		}
		return bestResult
	}

}

func getBestResult(results []*SingleResult) *SingleResult {
	resultsAsInterface := make([]interface{}, len(results))
	for i, result := range results {
		resultsAsInterface[i] = result
	}
	return utils.ApplySequentially(resultsAsInterface, getBestResultFinder()).(*SingleResult)
}

func multiTrainNonParallel(parameters *MultiParameters) (r *MultiResults, err error) {
	r = &MultiResults{
		TrainId:    parameters.TrainId,
		AllResults: make([]*SingleResult, parameters.RetriesCount),
	}

	for i := 0; i < parameters.RetriesCount; i++ {
		sp, err := preMultiTrain(parameters)
		if err != nil {
			return r, fmt.Errorf("error preparing for [%d] train: %w", i, err)
		}

		r.AllResults[i], err = SingleTrain(sp)
		if err != nil {
			return r, fmt.Errorf("error running [%d] train: %w", i, err)
		}
	}

	r.BestResults = getBestResult(r.AllResults)

	return r, nil
}

func multiTrainParallel(parameters *MultiParameters) (r *MultiResults, err error) {
	r = &MultiResults{
		TrainId:    parameters.TrainId,
		AllResults: make([]*SingleResult, 0),
	}

	var wg sync.WaitGroup
	var mu sync.Mutex
	wg.Add(parameters.RetriesCount)
	errCh := make(chan error, parameters.RetriesCount)

	for i := 0; i < parameters.RetriesCount; i++ {
		go func(wait *sync.WaitGroup, mutex *sync.Mutex, errorChanel chan<- error, iter int) {
			defer wait.Done()

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

			r.AllResults = append(r.AllResults, trainResults)
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

	r.BestResults = getBestResult(r.AllResults)

	return r, nil
}
