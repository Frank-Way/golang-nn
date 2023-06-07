package optim

import (
	"fmt"
	"math"
	"nn/internal/nn/operation"
	"nn/pkg/mmath/matrix"
	"nn/pkg/wraperr"
)

type PostOptimizeFunc func()

func NewSGD(parameters *SGDParameters) (operation.Optimizer, PostOptimizeFunc) {
	var learnRate float64
	var stopLearnRate float64
	var epochsCount int
	var decrement func(value *float64)
	paramStringer := func(lr float64, slr float64, ec int) string {
		return fmt.Sprintf("learn rate [%v], stop learn rate [%v], epochs count [%v]", lr, slr, ec)
	}

	if parameters == nil {
		parameters = &SGDParameters{}
	}

	if learnRate == parameters.LearnRate {
		learnRate = defaultLearnRate
		logger.Debugf("no learn rate provided, using default value: %v", defaultLearnRate)
	} else {
		learnRate = parameters.LearnRate
	}

	needsDecrease := parameters.StopLearnRate != stopLearnRate || parameters.EpochsCount != epochsCount || parameters.DecrementType == ExponentialDecrement
	if needsDecrease {
		if stopLearnRate == parameters.StopLearnRate {
			stopLearnRate = defaultStopLearnRate
			logger.Debugf("no stop learn rate provided, using default value: %v", defaultStopLearnRate)
		} else {
			stopLearnRate = parameters.StopLearnRate
		}
		if epochsCount == parameters.EpochsCount {
			epochsCount = defaultEpochsCount
			logger.Debugf("no epochs count provided, using default value: %v", defaultEpochsCount)
		} else {
			epochsCount = parameters.EpochsCount
		}
		if learnRate < stopLearnRate {
			logger.Warnf("learn rate [%v] less than stop learn rate [%v], using default values: %v and %v",
				learnRate, stopLearnRate, defaultLearnRate, defaultStopLearnRate)
			learnRate = defaultLearnRate
			stopLearnRate = defaultStopLearnRate
		}
		if parameters.DecrementType == ExponentialDecrement {
			factor := math.Pow(stopLearnRate/learnRate, 1.0/float64(epochsCount))
			logger.Tracef("set exponent learn rate decrement with per-epoch factor [%e] for parameters [%s]", factor,
				paramStringer(learnRate, stopLearnRate, epochsCount))
			decrement = func(value *float64) {
				if *value > stopLearnRate {
					*value = *value * factor
				}
			}
		} else {
			delta := (learnRate - stopLearnRate) / float64(epochsCount-1)
			logger.Tracef("set linear learn rate decrement with per-epoch delta [%e] for parameters [%s]", delta,
				paramStringer(learnRate, stopLearnRate, epochsCount))
			decrement = func(value *float64) {
				if *value > stopLearnRate {
					*value = *value - delta
				}
			}
		}
	} else {
		decrement = func(value *float64) {}
	}

	return func(param, grad *matrix.Matrix) (res *matrix.Matrix, err error) {
			defer logger.CatchErr(&err)
			defer wraperr.WrapError(ErrExec, &err)

			return param.Sub(grad.MulNum(learnRate))
		}, func() {
			decrement(&learnRate)
		}
}
