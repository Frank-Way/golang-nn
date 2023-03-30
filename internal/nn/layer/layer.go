package layer

import (
	"fmt"
	"nn/internal/nn/operation"
	"nn/internal/utils"
	"nn/pkg/mmath/matrix"
	"nn/pkg/wraperr"
)

var _ ILayer = (*Layer)(nil)

type Layer struct {
	kind       Kind
	operations []operation.IOperation
}

func (l *Layer) Forward(x *matrix.Matrix) (y *matrix.Matrix, err error) {
	defer logger.CatchErr(&err)
	defer wraperr.WrapError(ErrExec, &err)
	defer wraperr.WrapError(fmt.Errorf("error during Forward propagation on %s", l.kind), &err)

	if l == nil {
		return nil, ErrNil
	}

	y = x.Copy()
	for i, op := range l.operations {
		y, err = op.Forward(y)
		if err != nil {
			return nil, fmt.Errorf("error computing %d'th operation: %w", i, err)
		}
	}

	return y, nil
}

func (l *Layer) Backward(dy *matrix.Matrix) (dx *matrix.Matrix, err error) {
	defer logger.CatchErr(&err)
	defer wraperr.WrapError(ErrExec, &err)
	defer wraperr.WrapError(fmt.Errorf("error during Backward propagation on %s", l.kind), &err)

	if l == nil {
		return nil, ErrNil
	}

	dx = dy.Copy()
	length := len(l.operations)
	for i := length - 1; i >= 0; i-- {
		dx, err = l.operations[i].Backward(dx)
		if err != nil {
			return nil, fmt.Errorf("error computing %d'th operation: %w", i, err)
		}
	}

	return dx, nil
}

func (l *Layer) ApplyOptim(optimizer operation.Optimizer) (err error) {
	defer logger.CatchErr(&err)
	defer wraperr.WrapError(ErrExec, &err)
	defer wraperr.WrapError(fmt.Errorf("error during apply Optimizer on %s", l.kind), &err)

	if l == nil {
		return ErrNil
	}

	for i, op := range l.operations {
		if paramOp, ok := op.(*operation.ParamOperation); ok {
			err = paramOp.ApplyOptim(optimizer)
			if err != nil {
				return fmt.Errorf("error optimizing %d'th operation's parameter: %w", i, err)
			}
		}
	}
	return nil
}

func (l *Layer) Is(kind Kind) bool {
	return l.kind == kind
}

func (l *Layer) Copy() ILayer {
	if l == nil {
		return nil
	}
	res := &Layer{
		kind: l.kind,
	}
	res.operations = make([]operation.IOperation, len(l.operations))
	for i, op := range l.operations {
		res.operations[i] = op.Copy()
	}
	return res
}

func (l *Layer) Equal(layer ILayer) bool {
	if l == nil || layer == nil {
		if (l != nil && layer == nil) || (l == nil && layer != nil) {
			return false // non-nil != nil and nil != non-nil
		} else {
			return true // nil == nil
		}
	}
	if la, ok := layer.(*Layer); !ok {
		return false
	} else if l.kind != la.kind {
		return false
	} else if l.operations != nil {
		if len(l.operations) != len(la.operations) {
			return false
		}

		for i, op := range l.operations {
			if !op.Equal(la.operations[i]) {
				return false
			}
		}
	}

	return true
}

func (l *Layer) EqualApprox(layer ILayer) bool {
	if l == nil || layer == nil {
		if (l != nil && layer == nil) || (l == nil && layer != nil) {
			return false // non-nil != nil and nil != non-nil
		} else {
			return true // nil == nil
		}
	}
	if la, ok := layer.(*Layer); !ok {
		return false
	} else if l.kind != la.kind {
		return false
	} else if l.operations != nil {
		if len(l.operations) != len(la.operations) {
			return false
		}

		for i, op := range l.operations {
			if !op.EqualApprox(la.operations[i]) {
				return false
			}
		}
	}

	return true
}

func (l *Layer) operationsAsSPStringers() []utils.SPStringer {
	res := make([]utils.SPStringer, len(l.operations))
	for i, op := range l.operations {
		res[i] = op
	}
	return res
}

func (l *Layer) toMap(stringers func(s []utils.SPStringer) string) map[string]string {
	return map[string]string{
		"kind":       string(l.kind),
		"operations": stringers(l.operationsAsSPStringers()),
	}
}

func (l *Layer) String() string {
	if l == nil {
		return "<nil>"
	}
	return utils.FormatObject(l.toMap(utils.Strings), utils.BaseFormat)
}

func (l *Layer) PrettyString() string {
	if l == nil {
		return "<nil>"
	}
	return utils.FormatObject(l.toMap(utils.PrettyStrings), utils.PrettyFormat)
}

func (l *Layer) ShortString() string {
	if l == nil {
		return "<nil>"
	}
	return utils.FormatObject(l.toMap(utils.ShortStrings), utils.ShortFormat)
}
