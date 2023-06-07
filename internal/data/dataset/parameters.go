package dataset

import (
	"nn/internal/utils"
	"nn/pkg/percent"
)

// DataSplitParameters represent distribution of data by train, tests and validation categories
type DataSplitParameters struct {
	TrainPercent percent.Percent
	TestsPercent percent.Percent
	ValidPercent percent.Percent
}

var DefaultDataSplitParameters = &DataSplitParameters{
	TrainPercent: percent.Percent60,
	TestsPercent: percent.Percent30,
	ValidPercent: percent.Percent10,
}

func parametersFromDataset(dataset *Dataset) *DataSplitParameters {
	trainCount := float64(dataset.Train.X.Rows())
	testsCount := float64(dataset.Tests.X.Rows())
	validCount := float64(dataset.Valid.X.Rows())
	totalCount := trainCount + testsCount + validCount
	trainPercent := percent.GetApproximate(trainCount / totalCount)
	testsPercent := percent.GetApproximate(testsCount / totalCount)
	validPercent := percent.GetApproximate(validCount / totalCount)
	return &DataSplitParameters{
		TrainPercent: trainPercent,
		TestsPercent: testsPercent,
		ValidPercent: validPercent,
	}
}

func (p *DataSplitParameters) Copy() *DataSplitParameters {
	return &DataSplitParameters{
		TrainPercent: p.TrainPercent,
		TestsPercent: p.TestsPercent,
		ValidPercent: p.ValidPercent,
	}
}

func (p *DataSplitParameters) toMap(stringer func(spStringer utils.SPStringer) string) map[string]string {
	return map[string]string{
		"TrainPercent": stringer(p.TrainPercent),
		"TestsPercent": stringer(p.TestsPercent),
		"ValidPercent": stringer(p.ValidPercent),
	}
}

func (p *DataSplitParameters) String() string {
	if p == nil {
		return "<nil>"
	}
	return utils.FormatObject(p.toMap(utils.String), utils.BaseFormat)
}

func (p *DataSplitParameters) PrettyString() string {
	if p == nil {
		return "<nil>"
	}
	return utils.FormatObject(p.toMap(utils.PrettyString), utils.PrettyFormat)
}

func (p *DataSplitParameters) ShortString() string {
	if p == nil {
		return "<nil>"
	}
	return utils.FormatObject(p.toMap(utils.ShortString), utils.ShortFormat)
}
