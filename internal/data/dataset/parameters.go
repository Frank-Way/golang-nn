package dataset

import (
	"nn/internal/utils"
)

type DataSplitParameters struct {
	TrainPercent Percent
	TestsPercent Percent
	ValidPercent Percent
}

var DefaultDataSplitParameters = &DataSplitParameters{
	TrainPercent: Percent60,
	TestsPercent: Percent30,
	ValidPercent: Percent10,
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
