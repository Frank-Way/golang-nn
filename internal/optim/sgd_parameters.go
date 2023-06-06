package optim

type LearnRateDecrementType uint8

const (
	LinearDecrement LearnRateDecrementType = iota
	ExponentialDecrement
)

type SGDParameters struct {
	LearnRate     float64
	StopLearnRate float64
	EpochsCount   int
	DecrementType LearnRateDecrementType
}

const (
	defaultLearnRate     = 0.05
	defaultStopLearnRate = 0.001
	defaultEpochsCount   = 1000
)
