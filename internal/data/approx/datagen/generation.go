// Package datagen provides functionality for generation data using Parameters
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

// Parameters represent expression, input ranges and data split parameters. On generation:
//     * Expression will be parsed using expression.NewExpression();
//     * Ranges will be used to generate inputs for expression.Exec();
//     * DataSplitParameters tells which part of generated data will be stored for training, tests and validation.
type Parameters struct {
	*expression.Expression
	InputRanges []*InputRange
}

func NewParameters(
	rawExpression string,
	ranges ...*InputRange,
) (params *Parameters, err error) {
	defer logger.CatchErr(&err)
	defer wraperr.WrapError(ErrCreate, &err)

	if rawExpression == "" {
		return nil, fmt.Errorf("no expression provided in datagen parameters: %s", rawExpression)
	}

	expr, err := expression.NewExpression(rawExpression) // parse expression
	if err != nil {
		return nil, fmt.Errorf("error parsing expression %q: %w", rawExpression, err)
	}
	if ranges == nil {
		return nil, fmt.Errorf("no input ranges provided")
	}

	for i, inRange := range ranges {
		if err = checkInputRange(inRange); err != nil {
			return nil, fmt.Errorf("error checking input %d'th range: %w", i, err)
		}
	}

	return &Parameters{Expression: expr, InputRanges: ranges}, nil
}

// Generate provides Dataset from given Parameters.
//
// Throws ErrCreate error.
func Generate(p *Parameters) (ds *dataset.Dataset, err error) {
	defer logger.CatchErr(&err)
	defer wraperr.WrapError(ErrCreate, &err)

	if p == nil {
		return nil, fmt.Errorf("no generation parameters provided")
	}

	trainData, err := p.generateData(func(inRange *InputRange) (*vector.Vector, error) {
		return inRange.trainInputs()
	})
	if err != nil {
		return nil, fmt.Errorf("error generating train data: %w", err)
	}

	testsData, err := p.generateData(func(inRange *InputRange) (*vector.Vector, error) {
		return inRange.testsInputs()
	})
	if err != nil {
		return nil, fmt.Errorf("error generating tests data: %w", err)
	}

	validData, err := p.generateData(func(inRange *InputRange) (*vector.Vector, error) {
		return inRange.validInputs()
	})
	if err != nil {
		return nil, fmt.Errorf("error generating valid data: %w", err)
	}

	return dataset.NewDataset(trainData, testsData, validData)
}

func (p *Parameters) generateData(inGen func(inRange *InputRange) (*vector.Vector, error)) (d *dataset.Data, err error) {

	inCols := make([]*vector.Vector, len(p.InputRanges)) // generate inputs
	for i, inRange := range p.InputRanges {
		if inCols[i], err = inGen(inRange); err != nil {
			return nil, fmt.Errorf("error generating %d'th inputs: %w", i, err)
		}
	}

	logger.Debug("combine inputs")
	inMat, err := matrix.CartesianProduct(inCols) // combine inputs
	if err != nil {
		return nil, fmt.Errorf("error combining inputs: %w", err)
	}
	inRaw := inMat.Raw()

	logger.Debug("compute outputs of expression")
	outRows := make([]*vector.Vector, inMat.Rows()) // compute outputs using parsed expression
	for i := 0; i < inMat.Rows(); i++ {
		exec, err := p.Expression.Exec(inRaw[i])
		if err != nil {
			return nil, fmt.Errorf("error computing outputs: %w", err)
		}
		if outRows[i], err = vector.NewVector([]float64{exec}); err != nil {
			return nil, fmt.Errorf("error wrapping outputs to Vector: %w", err)
		}
	}

	outMat, err := matrix.NewMatrix(outRows)
	if err != nil {
		return nil, fmt.Errorf("error wrapping outputs to Matrix: %w", err)
	}

	d, err = dataset.NewData(inMat, outMat)
	if err != nil {
		return nil, fmt.Errorf("error wrapping inputs and outputs to Data: %w", err)
	}

	return d, nil
}

func (p *Parameters) rangesAsSPStringers() []utils.SPStringer {
	res := make([]utils.SPStringer, len(p.InputRanges))
	for i, rng := range p.InputRanges {
		res[i] = rng
	}
	return res
}

func (p *Parameters) toMap(
	stringer func(spStringer utils.SPStringer) string,
	stringers func(spStringers []utils.SPStringer) string,
) map[string]string {
	return map[string]string{
		"Expression":  stringer(p.Expression),
		"InputRanges": stringers(p.rangesAsSPStringers()),
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
