package operation

import (
	"fmt"
	"nn/internal/utils"
	"nn/pkg/mmath/matrix"
	"nn/pkg/wraperr"
)

var _ IOperation = (*Operation)(nil)

// Operation represents main part of all operations. It stores all given and computed data. It defines operation
// behavior.
type Operation struct {
	kind       Kind
	activation bool

	x *matrix.Matrix
	y *matrix.Matrix

	dx *matrix.Matrix
	dy *matrix.Matrix

	output   func(x *matrix.Matrix) (*matrix.Matrix, error)
	gradient func(dy *matrix.Matrix) (*matrix.Matrix, error)
}

func (o *Operation) Forward(x *matrix.Matrix) (y *matrix.Matrix, err error) {
	defer logger.CatchErr(&err)
	defer wraperr.WrapError(ErrExec, &err)

	if o == nil {
		return nil, ErrNil
	} else if x == nil {
		return nil, fmt.Errorf("no input provided: %v", x)
	}

	o.x = x.Copy()
	y, err = o.output(x)
	if err != nil {
		return nil, err
	}
	o.y = y.Copy()
	return y, nil
}

func (o *Operation) Backward(dy *matrix.Matrix) (dx *matrix.Matrix, err error) {
	defer logger.CatchErr(&err)
	defer wraperr.WrapError(ErrExec, &err)

	if o == nil {
		return nil, ErrNil
	} else if dy == nil {
		return nil, fmt.Errorf("no output gradient provided: %v", dy)
	} else if o.y == nil || o.x == nil {
		return nil, fmt.Errorf("call Backward() before Forward()")
	}

	o.dy = dy.Copy()
	if err := o.y.CheckEqualShape(dy); err != nil {
		return nil, err
	}
	dx, err = o.gradient(dy)
	if err != nil {
		return nil, err
	} else if err = o.x.CheckEqualShape(dx); err != nil {
		return nil, err
	}
	o.dx = dx.Copy()
	return dx, nil
}

func (o *Operation) Is(kind Kind) bool {
	return o.kind == kind
}

func (o *Operation) IsActivation() bool {
	return o.activation
}

func (o *Operation) Copy() IOperation {
	if o == nil {
		return nil
	}
	res := &Operation{
		kind:     o.kind,
		output:   o.output,
		gradient: o.gradient,
	}
	if o.x != nil {
		res.x = o.x.Copy()
	}
	if o.y != nil {
		res.y = o.y.Copy()
	}
	if o.dx != nil {
		res.dx = o.dx.Copy()
	}
	if o.dy != nil {
		res.dy = o.dy.Copy()
	}
	return res
}

func (o *Operation) Equal(operation IOperation) bool {
	if o == nil || operation == nil {
		if (o != nil && operation == nil) || (o == nil && operation != nil) {
			return false // non-nil != nil and nil != non-nil
		} else {
			return true // nil == nil
		}
	}
	if op, ok := operation.(*Operation); !ok {
		return false
	} else if o.kind != op.kind {
		return false
	} else if o.x != nil && !o.x.Equal(op.x) {
		return false
	} else if o.y != nil && !o.y.Equal(op.y) {
		return false
	} else if o.dx != nil && !o.dx.Equal(op.dx) {
		return false
	} else if o.dy != nil && !o.dy.Equal(op.dy) {
		return false
	}

	return true
}

func (o *Operation) EqualApprox(operation IOperation) bool {
	if o == nil || operation == nil {
		if (o != nil && operation == nil) || (o == nil && operation != nil) {
			return false // non-nil != nil and nil != non-nil
		} else {
			return true // nil == nil
		}
	}
	if op, ok := operation.(*Operation); !ok {
		return false
	} else if o.kind != op.kind {
		return false
	} else if o.x != nil && !o.x.EqualApprox(op.x) {
		return false
	} else if o.y != nil && !o.y.EqualApprox(op.y) {
		return false
	} else if o.dx != nil && !o.dx.EqualApprox(op.dx) {
		return false
	} else if o.dy != nil && !o.dy.EqualApprox(op.dy) {
		return false
	}

	return true
}

func (o *Operation) toMap(stringer func(spStringer utils.SPStringer) string) map[string]string {
	return map[string]string{
		"kind": string(o.kind),
		"x":    stringer(o.x),
		"y":    stringer(o.y),
		"dx":   stringer(o.dx),
		"dy":   stringer(o.dy),
	}
}

func (o *Operation) String() string {
	if o == nil {
		return "<nil>"
	}
	return utils.FormatObject(o.toMap(utils.String), utils.BaseFormat)
}

func (o *Operation) PrettyString() string {
	if o == nil {
		return "<nil>"
	}
	return utils.FormatObject(o.toMap(utils.PrettyString), utils.PrettyFormat)
}

func (o *Operation) ShortString() string {
	if o == nil {
		return "<nil>"
	}
	return utils.FormatObject(o.toMap(utils.ShortString), utils.ShortFormat)
}
