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

func Create(kind nn.Kind, args ...interface{}) (layer ILayer, err error) {
	defer logger.CatchErr(&err)
	defer wraperr.WrapError(ErrFabric, &err)

	switch kind {
	case DenseLayer:
		if len(args) < 3 {
			return nil, fmt.Errorf("not enough arguments to create %q, required %d, provided %d", DenseLayer, 3, len(args))
		} else if w, ok := args[0].(*matrix.Matrix); !ok {
			return nil, fmt.Errorf("first argument is not *matrix.Matrix: %T", args[0])
		} else if b, ok := args[1].(*vector.Vector); !ok {
			return nil, fmt.Errorf("second argument is not *vector.Vector: %T", args[1])
		} else if a, ok := args[2].(operation.IOperation); !ok {
			return nil, fmt.Errorf("third argument is not operation.IOperation: %T", args[2])
		} else {
			return NewDenseLayer(w, b, a)
		}
	case DenseDropLayer:
		if len(args) < 4 {
			return nil, fmt.Errorf("not enough arguments to create %q, required %d, provided %d", DenseDropLayer, 4, len(args))
		} else if w, ok := args[0].(*matrix.Matrix); !ok {
			return nil, fmt.Errorf("first argument is not *matrix.Matrix: %T", args[0])
		} else if b, ok := args[1].(*vector.Vector); !ok {
			return nil, fmt.Errorf("second argument is not *vector.Vector: %T", args[1])
		} else if a, ok := args[2].(operation.IOperation); !ok {
			return nil, fmt.Errorf("third argument is not operation.IOperation: %T", args[2])
		} else if d, ok := args[3].(percent.Percent); !ok {
			return nil, fmt.Errorf("fourth argument is not percent.Percent: %T", args[3])
		} else {
			return NewDenseDropLayer(w, b, a, d)
		}
	}

	return nil, fmt.Errorf("unknown layer: %s", kind)
}
