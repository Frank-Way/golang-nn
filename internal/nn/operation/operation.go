package operation

import (
	"nn/internal/utils"
	"nn/pkg/mmath/matrix"
	"nn/pkg/wraperr"
)

var _ IOperation = (*Operation)(nil)

// Operation represents main part of all operations. It stores all given and computed data. It defines operation
// behavior.
type Operation struct {
	name string

	x *matrix.Matrix
	y *matrix.Matrix

	dx *matrix.Matrix
	dy *matrix.Matrix

	output   func(x *matrix.Matrix) (*matrix.Matrix, error)
	gradient func(dy *matrix.Matrix) (*matrix.Matrix, error)
}

func (o *Operation) Forward(x *matrix.Matrix) (y *matrix.Matrix, err error) {
	defer wraperr.WrapError(ErrExec, &err)

	o.x = x.Copy()
	y, err = o.output(x)
	if err != nil {
		return nil, err
	}
	o.y = y.Copy()
	return y, nil
}

func (o *Operation) Backward(dy *matrix.Matrix) (dx *matrix.Matrix, err error) {
	defer wraperr.WrapError(ErrExec, &err)

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

// Copy returns deep copy of Operation
func (o *Operation) Copy() *Operation {
	res := &Operation{
		name:     o.name,
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

func (o *Operation) Equal(operation *Operation) bool {
	if o == nil || operation == nil {
		if (o != nil && operation == nil) || (o == nil && operation != nil) {
			return false // non-nil != nil and nil != non-nil
		} else {
			return true // nil == nil
		}
	} else if o.name != operation.name {
		return false
	} else if o.x != nil && !o.x.Equal(operation.x) {
		return false
	} else if o.y != nil && !o.y.Equal(operation.y) {
		return false
	} else if o.dx != nil && !o.dx.Equal(operation.dx) {
		return false
	} else if o.dy != nil && !o.dy.Equal(operation.dy) {
		return false
	}

	return true
}

func (o *Operation) EqualApprox(operation *Operation) bool {
	if o == nil || operation == nil {
		if (o != nil && operation == nil) || (o == nil && operation != nil) {
			return false // non-nil != nil and nil != non-nil
		} else {
			return true // nil == nil
		}
	} else if o.name != operation.name {
		return false
	} else if o.x != nil && !o.x.EqualApprox(operation.x) {
		return false
	} else if o.y != nil && !o.y.EqualApprox(operation.y) {
		return false
	} else if o.dx != nil && !o.dx.EqualApprox(operation.dx) {
		return false
	} else if o.dy != nil && !o.dy.EqualApprox(operation.dy) {
		return false
	}

	return true
}

func (o *Operation) toMap(stringer func(spStringer utils.SPStringer) string) map[string]string {
	return map[string]string{
		"name": o.name,
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
