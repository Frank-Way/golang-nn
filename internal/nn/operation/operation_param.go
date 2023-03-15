package operation

import (
	"fmt"
	"nn/internal/utils"
	"nn/pkg/mmath/matrix"
	"nn/pkg/wraperr"
)

var _ IOperation = (*ParamOperation)(nil)

type ParamOperation struct {
	*Operation

	p  *matrix.Matrix
	dp *matrix.Matrix

	output    func(x *matrix.Matrix, p *matrix.Matrix) (*matrix.Matrix, error)
	gradient  func(dy *matrix.Matrix, p *matrix.Matrix, x *matrix.Matrix) (*matrix.Matrix, error)
	gradParam func(dy *matrix.Matrix, p *matrix.Matrix, x *matrix.Matrix) (*matrix.Matrix, error)
}

func (o *ParamOperation) Forward(x *matrix.Matrix) (*matrix.Matrix, error) {
	res, err := func(x *matrix.Matrix) (*matrix.Matrix, error) {
		if x == nil {
			return nil, fmt.Errorf("no input provided: %v", x)
		}
		o.x = x.Copy()
		y, err := o.output(x, o.p)
		if err != nil {
			return nil, err
		}
		o.y = y.Copy()
		return y, nil
	}(x)

	if err != nil {
		return nil, wraperr.NewWrapErr(ErrExec, err)
	}

	return res, nil
}

func (o *ParamOperation) Backward(dy *matrix.Matrix) (*matrix.Matrix, error) {
	res, err := func(dy *matrix.Matrix) (*matrix.Matrix, error) {
		if dy == nil {
			return nil, fmt.Errorf("no out gradient provided: %v", dy)
		} else if o.x == nil {
			return nil, fmt.Errorf("backward before forward: %v", o.x)
		}
		dp, err := o.gradParam(dy, o.p, o.x)
		if err != nil {
			return nil, err
		} else if err = o.p.CheckEqualShape(dp); err != nil {
			return nil, err
		}
		o.dp = dp.Copy()

		o.dy = dy.Copy()
		if err := o.y.CheckEqualShape(dy); err != nil {
			return nil, err
		}
		dx, err := o.gradient(dy, o.p, o.x)
		if err != nil {
			return nil, err
		} else if o.x != nil {
			if err = o.x.CheckEqualShape(dx); err != nil {
				return nil, err
			}
		}
		o.dx = dx.Copy()
		return dx, nil
	}(dy)

	if err != nil {
		return nil, wraperr.NewWrapErr(ErrExec, err)
	}

	return res, nil
}

type Optimizer func(param, grad *matrix.Matrix) (*matrix.Matrix, error)

func (o *ParamOperation) ApplyOptim(optim Optimizer) error {
	err := func(optim func(param, grad *matrix.Matrix) (*matrix.Matrix, error)) error {
		if optim == nil {
			return fmt.Errorf("no optimizer provided")
		} else if o.dp == nil {
			return fmt.Errorf("can not apply optimizer before gradiend computation: %v", o.dp)
		}
		newP, err := optim(o.p, o.dp)
		if err != nil {
			return err
		} else if err := o.p.CheckEqualShape(newP); err != nil {
			return err
		} else if newP == nil {
			return fmt.Errorf("nil parameter after optimization: %v", newP)
		}

		o.p = newP
		return nil
	}(optim)

	if err != nil {
		return wraperr.NewWrapErr(ErrExec, err)
	}

	return nil
}

func (o *ParamOperation) Parameter() *matrix.Matrix {
	return o.p.Copy()
}

func (o *ParamOperation) Copy() *ParamOperation {
	res := &ParamOperation{
		Operation: o.Operation.Copy(),
		output:    o.output,
		gradient:  o.gradient,
	}
	if o.p != nil {
		res.p = o.p.Copy()
	}
	if o.dp != nil {
		res.dp = o.dp.Copy()
	}
	return res
}

func (o *ParamOperation) Equal(operation *ParamOperation) bool {
	if o == nil || operation == nil {
		if (o != nil && operation == nil) || (o == nil && operation != nil) {
			return false // non-nil != nil and nil != non-nil
		} else {
			return true // nil == nil
		}
	} else if !o.Operation.Equal(operation.Operation) {
		return false
	} else if o.p != nil && !o.p.Equal(operation.p) {
		return false
	} else if o.dp != nil && !o.dp.Equal(operation.dp) {
		return false
	}

	return true
}

func (o *ParamOperation) EqualApprox(operation *ParamOperation) bool {
	if operation == nil {
		return false
	} else if !o.Operation.EqualApprox(operation.Operation) {
		return false
	} else if o.p != nil && !o.p.EqualApprox(operation.p) {
		return false
	} else if o.dp != nil && !o.dp.EqualApprox(operation.dp) {
		return false
	}

	return true
}

func (o *ParamOperation) toMap(stringer func(spStringer utils.SPStringer) string) map[string]string {
	return map[string]string{
		"operation": stringer(o.Operation),
		"p":         stringer(o.p),
		"dp":        stringer(o.dp),
	}
}

func (o *ParamOperation) String() string {
	if o == nil {
		return "<nil>"
	}
	return utils.FormatObject(o.toMap(utils.String), utils.BaseFormat)
}

func (o *ParamOperation) PrettyString() string {
	if o == nil {
		return "<nil>"
	}
	return utils.FormatObject(o.toMap(utils.PrettyString), utils.PrettyFormat)
}

func (o *ParamOperation) ShortString() string {
	if o == nil {
		return "<nil>"
	}
	return utils.FormatObject(o.toMap(utils.ShortString), utils.ShortFormat)
}
