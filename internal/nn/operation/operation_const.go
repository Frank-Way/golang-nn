package operation

import (
	"fmt"
	"nn/internal/utils"
	"nn/pkg/mmath/matrix"
	"nn/pkg/wraperr"
)

var _ IOperation = (*ConstOperation)(nil)

type ConstOperation struct {
	*Operation

	p []*matrix.Matrix

	output   func(x *matrix.Matrix, p []*matrix.Matrix) (*matrix.Matrix, error)
	gradient func(dy *matrix.Matrix, p []*matrix.Matrix, x *matrix.Matrix) (*matrix.Matrix, error)
}

func (o *ConstOperation) Forward(x *matrix.Matrix) (y *matrix.Matrix, err error) {
	defer wraperr.WrapError(ErrExec, &err)

	if x == nil {
		return nil, fmt.Errorf("no input provided: %v", x)
	}
	o.x = x.Copy()
	y, err = o.output(x, o.p)
	if err != nil {
		return nil, err
	}
	o.y = y.Copy()
	return y, nil
}

func (o *ConstOperation) Backward(dy *matrix.Matrix) (dx *matrix.Matrix, err error) {
	defer wraperr.WrapError(ErrExec, &err)

	if dy == nil {
		return nil, fmt.Errorf("no out gradient provided: %v", dy)
	} else if o.x == nil {
		return nil, fmt.Errorf("backward before forward: %v", o.x)
	}

	o.dy = dy.Copy()
	if err := o.y.CheckEqualShape(dy); err != nil {
		return nil, err
	}
	dx, err = o.gradient(dy, o.p, o.x)
	if err != nil {
		return nil, err
	} else if o.x != nil {
		if err = o.x.CheckEqualShape(dx); err != nil {
			return nil, err
		}
	}
	o.dx = dx.Copy()
	return dx, nil
}

func (o *ConstOperation) Copy() *ConstOperation {
	res := &ConstOperation{
		Operation: o.Operation.Copy(),
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

func (o *ConstOperation) Equal(operation *ConstOperation) bool {
	if o == nil || operation == nil {
		if (o != nil && operation == nil) || (o == nil && operation != nil) {
			return false // non-nil != nil and nil != non-nil
		} else {
			return true // nil == nil
		}
	} else if !o.Operation.Equal(operation.Operation) {
		return false
	} else if o.p == nil || operation.p == nil {
		if (o.p != nil && operation.p == nil) || (o.p == nil && operation.p != nil) {
			return false // non-nil != nil and nil != non-nil
		} else {
			if len(o.p) != len(operation.p) {
				return false
			}
			for i := 0; i < len(o.p); i++ {
				if !o.p[i].Equal(operation.p[i]) {
					return false
				}
			}
		}
	}

	return true
}

func (o *ConstOperation) EqualApprox(operation *ConstOperation) bool {
	if o == nil || operation == nil {
		if (o != nil && operation == nil) || (o == nil && operation != nil) {
			return false // non-nil != nil and nil != non-nil
		} else {
			return true // nil == nil
		}
	} else if !o.Operation.EqualApprox(operation.Operation) {
		return false
	} else if o.p == nil || operation.p == nil {
		if (o.p != nil && operation.p == nil) || (o.p == nil && operation.p != nil) {
			return false // non-nil != nil and nil != non-nil
		} else {
			if len(o.p) != len(operation.p) {
				return false
			}
			for i := 0; i < len(o.p); i++ {
				if !o.p[i].EqualApprox(operation.p[i]) {
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
