package datagen

import (
	"fmt"
	"nn/internal/data/approx/expression"
	"nn/internal/data/dataset"
	"nn/internal/utils"
	"nn/pkg/mmath/matrix"
	"nn/pkg/mmath/vector"
	"nn/pkg/wraperr"
)

type Parameters struct {
	Expression string
	Ranges     []*InputRange
	*dataset.DataSplitParameters
}

func NewParameters(
	expression string,
	ranges []*InputRange,
	dataSplitParameters *dataset.DataSplitParameters,
) (params *Parameters, err error) {
	defer wraperr.WrapError(ErrCreate, &err)

	if expression == "" {
		return nil, fmt.Errorf("no expression provided in datagen parameters: %s", expression)
	} else if ranges == nil || len(ranges) < 1 {
		return nil, fmt.Errorf("no input ranges provided in datagen parameters: %v", ranges)
	} else if dataSplitParameters == nil {
		dataSplitParameters = dataset.DefaultDataSplitParameters
	}
	return &Parameters{Expression: expression, Ranges: ranges, DataSplitParameters: dataSplitParameters}, nil
}

func (p *Parameters) Generate() (*dataset.Dataset, error) {
	expr, err := expression.NewExpression(p.Expression)
	if err != nil {
		return nil, err
	}

	inCols := make([]*vector.Vector, len(p.Ranges))
	for i, inRange := range p.Ranges {
		if inCols[i], err = inRange.inputs(); err != nil {
			return nil, err
		}
	}

	inMat, err := matrix.CartesianProduct(inCols)
	if err != nil {
		return nil, err
	}
	inRaw := inMat.Raw()

	outRows := make([]*vector.Vector, inMat.Rows())
	for i := 0; i < inMat.Rows(); i++ {
		exec, err := expr.Exec(inRaw[i])
		if err != nil {
			return nil, err
		}
		if outRows[i], err = vector.NewVector([]float64{exec}); err != nil {
			return nil, err
		}
	}

	outMat, err := matrix.NewMatrix(outRows)
	if err != nil {
		return nil, err
	}

	data, err := dataset.NewData(inMat, outMat)
	if err != nil {
		return nil, err
	}

	return dataset.NewDatasetSplit(data, p.DataSplitParameters)
}

func (p *Parameters) rangesAsSPStringers() []utils.SPStringer {
	res := make([]utils.SPStringer, len(p.Ranges))
	for i, rng := range p.Ranges {
		res[i] = rng
	}
	return res
}

func (p *Parameters) toMap(
	stringer func(spStringer utils.SPStringer) string,
	stringers func(spStringers []utils.SPStringer) string,
) map[string]string {
	return map[string]string{
		"Expression":          p.Expression,
		"Ranges":              stringers(p.rangesAsSPStringers()),
		"DataSplitParameters": stringer(p.DataSplitParameters),
	}
}

func (p *Parameters) String() string {
	if p == nil {
		return "<nil>"
	}
	return utils.FormatObject(p.toMap(utils.String, utils.Strings), utils.BaseFormat)
}

func (p *Parameters) PrettyString() string {
	if p == nil {
		return "<nil>"
	}
	return utils.FormatObject(p.toMap(utils.PrettyString, utils.PrettyStrings), utils.PrettyFormat)
}

func (p *Parameters) ShortString() string {
	if p == nil {
		return "<nil>"
	}
	return utils.FormatObject(p.toMap(utils.ShortString, utils.ShortStrings), utils.ShortFormat)
}
