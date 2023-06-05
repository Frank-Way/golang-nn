package operation

import (
	"fmt"
	"nn/internal/nn"
	"nn/pkg/mmath/matrix"
	"nn/pkg/mmath/vector"
	"nn/pkg/percent"
	"nn/pkg/wraperr"
)

func Create(kind nn.Kind, args ...interface{}) (o IOperation, err error) {
	defer logger.CatchErr(&err)
	defer wraperr.WrapError(ErrFabric, &err)

	switch kind {
	case LinearActivation:
		return NewLinearActivation(), nil
	case TanhActivation:
		return NewTanhActivation(), nil
	case SigmoidActivation:
		return NewSigmoidActivation(), nil
	case SigmoidParamActivation:
		if len(args) < 1 {
			return nil, fmt.Errorf("no coefficients provided for %s", kind)
		} else if v, ok := args[0].(*vector.Vector); !ok {
			return nil, fmt.Errorf("first argument for %s is not a *vector.Vector: %T", kind, args[0])
		} else {
			return NewSigmoidParam(v)
		}
	case Dropout:
		if len(args) < 1 {
			return nil, fmt.Errorf("no keep probability provided for %s", kind)
		} else if p, ok := args[0].(percent.Percent); !ok {
			return nil, fmt.Errorf("first argument for %s is not a percent.Percent: %T", kind, args[0])
		} else {
			return NewDropout(p)
		}
	case WeightMultiply:
		if len(args) < 1 {
			return nil, fmt.Errorf("no weight provided for %s", kind)
		} else if w, ok := args[0].(*matrix.Matrix); !ok {
			return nil, fmt.Errorf("first argument for %s is not a *matrix.Matrix: %T", kind, args[0])
		} else {
			return NewWeightOperation(w)
		}
	case BiasAdd:
		if len(args) < 1 {
			return nil, fmt.Errorf("no bias provided for %s", kind)
		} else if b, ok := args[0].(*vector.Vector); !ok {
			return nil, fmt.Errorf("first argument for %s is not a *vector.Vector: %T", kind, args[0])
		} else {
			return NewBiasOperation(b)
		}
	}

	return nil, fmt.Errorf("unknown operation: %s", kind)
}
