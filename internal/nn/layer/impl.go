package layer

import (
	"fmt"
	"nn/internal/nn"
	"nn/internal/nn/operation"
	"nn/pkg/mmath/matrix"
	"nn/pkg/mmath/vector"
	"nn/pkg/percent"
	"nn/pkg/wraperr"
)

const (
	DenseLayer     nn.Kind = "dense layer"
	DenseDropLayer nn.Kind = "densedrop layer"
)

func NewDenseLayer(weight *matrix.Matrix, bias *vector.Vector, activation operation.IOperation) (l ILayer, err error) {
	defer logger.CatchErr(&err)
	defer wraperr.WrapError(ErrCreate, &err)

	logger.Debug("create dense layer")
	w, err := operation.NewWeightOperation(weight)
	if err != nil {
		return nil, err
	}

	b, err := operation.NewBiasOperation(bias)
	if err != nil {
		return nil, err
	}

	if weight.Cols() != bias.Size() {
		return nil, fmt.Errorf("weight cols count must match bias size (as it is layer's size): %d != %d",
			weight.Cols(), bias.Size())
	}

	logger.Debug("check activation")
	if !activation.IsActivation() {
		return nil, fmt.Errorf("provided operation it is not an activation: %s", activation.ShortString())
	}

	if activation.Is(operation.SigmoidParamActivation) {
		casted, ok := activation.(*operation.ConstOperation)
		if !ok {
			return nil, fmt.Errorf("error casting activation")
		}
		p := casted.Parameters()
		if p[0].Cols() != weight.Cols() {
			return nil, fmt.Errorf("parametrized sigmoid parameters count does not match weight cols count (layer's size): %d != %d",
				p[0].Cols(), weight.Cols())
		}

	}

	return &Layer{
		kind:        DenseLayer,
		inputsCount: weight.Rows(),
		size:        weight.Cols(),
		operations:  []operation.IOperation{w, b, activation.Copy().(operation.IOperation)},
	}, nil
}

func NewDenseDropLayer(
	weight *matrix.Matrix,
	bias *vector.Vector,
	activation operation.IOperation,
	keepProb percent.Percent,
) (l ILayer, err error) {
	defer logger.CatchErr(&err)
	defer wraperr.WrapError(ErrCreate, &err)

	logger.Debug("create densedrop layer")

	l, err = NewDenseLayer(weight, bias, activation)
	if err != nil {
		return nil, err
	}

	drop, err := operation.NewDropout(keepProb)
	if err != nil {
		return nil, err
	}

	logger.Debug("combine dense layer with dropout operation")

	casted, ok := l.(*Layer)
	if !ok {
		panic("could not cast layer.ILayer to *layer.Layer")
	}

	casted.operations = append(casted.operations, drop)
	casted.kind = DenseDropLayer

	return casted, nil
}
