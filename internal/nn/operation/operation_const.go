package operation

import (
	"fmt"
	"nn/internal/nn"
	"nn/internal/utils"
	"nn/pkg/mmath/matrix"
	"nn/pkg/wraperr"
)

var _ IOperation = (*ConstOperation)(nil)

// ConstOperation represents Operation with additional constant (not modified during training) parameters
type ConstOperation struct {
	*Operation

	p []*matrix.Matrix

	output   func(x *matrix.Matrix, p []*matrix.Matrix) (*matrix.Matrix, error)
	gradient func(dy *matrix.Matrix, p []*matrix.Matrix, x *matrix.Matrix) (*matrix.Matrix, error)
}

func (o *ConstOperation) Forward(x *matrix.Matrix) (y *matrix.Matrix, err error) {
	defer logger.CatchErr(&err)
	defer wraperr.WrapError(ErrExec, &err)
	defer wraperr.WrapError(fmt.Errorf("error during Forward propagation on %s", o.kind), &err)

	if o == nil {
		return nil, ErrNil
	} else if x == nil {
		return nil, fmt.Errorf("no input provided: %v", x)
	}

	o.x = x.Copy()
	y, err = o.output(x, o.p)
	if err != nil {
		return nil, fmt.Errorf("error computing output: %w", err)
	}
	o.y = y.Copy()
	return y, nil
}

func (o *ConstOperation) Backward(dy *matrix.Matrix) (dx *matrix.Matrix, err error) {
	defer logger.CatchErr(&err)
	defer wraperr.WrapError(ErrExec, &err)
	defer wraperr.WrapError(fmt.Errorf("error during Backward propagation on %s", o.kind), &err)

	if o == nil {
		return nil, ErrNil
	} else if dy == nil {
		return nil, fmt.Errorf("no output gradient provided: %v", dy)
	} else if o.x == nil || o.y == nil {
		return nil, fmt.Errorf("call Backward() before Forward()")
	}

	o.dy = dy.Copy()
	if err := o.y.CheckEqualShape(dy); err != nil {
		return nil, err
	}
	dx, err = o.gradient(dy, o.p, o.x)
	if err != nil {
		return nil, fmt.Errorf("error computing input gradient: %w", err)
	} else if o.x != nil {
		if err = o.x.CheckEqualShape(dx); err != nil {
			return nil, err
		}
	}
	o.dx = dx.Copy()
	return dx, nil
}

func (o *ConstOperation) Parameters() []*matrix.Matrix {
	p := make([]*matrix.Matrix, len(o.p))
	for i, param := range o.p {
		if param != nil {
			p[i] = param.Copy()
		}
	}
	return p
}

func (o *ConstOperation) Copy() nn.IModule {
	if o == nil {
		return nil
	}
	res := &ConstOperation{
		Operation: o.Operation.Copy().(*Operation),
		output:    o.output,
		gradient:  o.gradient,
	}
	if o.p != nil {
		res.p = make([]*matrix.Matrix, len(o.p))
		for i, param := range o.p {
			if param != nil {
				res.p[i] = param.Copy()
			}
		}
	}
	return res
}

func (o *ConstOperation) Equal(operation nn.IModule) bool {
	if o == nil || operation == nil {
		if (o != nil && operation == nil) || (o == nil && operation != nil) {
			return false // non-nil != nil and nil != non-nil
		} else {
			return true // nil == nil
		}
	}
	if op, ok := operation.(*ConstOperation); !ok {
		return false
	} else if !o.Operation.Equal(op.Operation) {
		return false
	} else if o.p == nil || op.p == nil {
		if (o.p != nil && op.p == nil) || (o.p == nil && op.p != nil) {
			return false // non-nil != nil and nil != non-nil
		} else {
			if len(o.p) != len(op.p) {
				return false
			}
			for i := 0; i < len(o.p); i++ {
				if !o.p[i].Equal(op.p[i]) {
					return false
				}
			}
		}
	}

	return true
}

func (o *ConstOperation) EqualApprox(operation nn.IModule) bool {
	if o == nil || operation == nil {
		if (o != nil && operation == nil) || (o == nil && operation != nil) {
			return false // non-nil != nil and nil != non-nil
		} else {
			return true // nil == nil
		}
	}
	if op, ok := operation.(*ConstOperation); !ok {
		return false
	} else if !o.Operation.EqualApprox(op.Operation) {
		return false
	} else if o.p == nil || op.p == nil {
		if (o.p != nil && op.p == nil) || (o.p == nil && op.p != nil) {
			return false // non-nil != nil and nil != non-nil
		} else {
			if len(o.p) != len(op.p) {
				return false
			}
			for i := 0; i < len(o.p); i++ {
				if !o.p[i].EqualApprox(op.p[i]) {
					return false
				}
			}
		}
	}

	return true
}

func (o *ConstOperation) paramsAsSPStringers() []utils.SPStringer {
	res := make([]utils.SPStringer, len(o.p))
	for i, p := range o.p {
		res[i] = p
	}
	return res
}

func (o *ConstOperation) toMap(
	stringer func(spStringer utils.SPStringer) string,
	stringers func(spStringers []utils.SPStringer) string,
) map[string]string {
	return map[string]string{
		"operation": stringer(o.Operation),
		"p":         stringers(o.paramsAsSPStringers()),
	}
}

func (o *ConstOperation) String() string {
	if o == nil {
		return "<nil>"
	}
	return utils.FormatObject(o.toMap(utils.String, utils.Strings), utils.BaseFormat)
}

func (o *ConstOperation) PrettyString() string {
	if o == nil {
		return "<nil>"
	}
	return utils.FormatObject(o.toMap(utils.PrettyString, utils.PrettyStrings), utils.PrettyFormat)
}

func (o *ConstOperation) ShortString() string {
	if o == nil {
		return "<nil>"
	}
	return utils.FormatObject(o.toMap(utils.ShortString, utils.ShortStrings), utils.ShortFormat)
}
