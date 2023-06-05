package loss

import (
	"fmt"
	"nn/internal/nn"
	"nn/internal/utils"
	"nn/pkg/mmath/matrix"
	"nn/pkg/wraperr"
)

var _ ILoss = (*Loss)(nil)

// Loss represent loss module that holds inputs and computed outputs
type Loss struct {
	kind nn.Kind

	t *matrix.Matrix
	y *matrix.Matrix

	l float64
	d *matrix.Matrix

	output   func(t, y *matrix.Matrix) (float64, error)
	gradient func(t, y *matrix.Matrix) (*matrix.Matrix, error)
}

func (l *Loss) Forward(t *matrix.Matrix, y *matrix.Matrix) (loss float64, err error) {
	defer logger.CatchErr(&err)
	defer wraperr.WrapError(ErrExec, &err)
	defer wraperr.WrapError(fmt.Errorf("error during Forward propagation on %s", l.kind), &err)

	if l == nil {
		return 0, ErrNil
	} else if t == nil {
		return 0, fmt.Errorf("no targets provided: %v", t)
	} else if y == nil {
		return 0, fmt.Errorf("no outputs provided: %v", y)
	} else if t.Rows() != y.Rows() || t.Cols() != y.Cols() {
		return 0, fmt.Errorf("targets and outputs sizes mismatch: %dx%d != %dx%d",
			t.Rows(), y.Rows(), t.Cols(), y.Cols())
	}

	l.t = t.Copy()
	l.y = y.Copy()

	loss, err = l.output(t, y)
	if err != nil {
		return 0, fmt.Errorf("error computing output: %w", err)
	}

	l.l = loss
	return loss, nil
}

func (l *Loss) Backward() (grad *matrix.Matrix, err error) {
	defer logger.CatchErr(&err)
	defer wraperr.WrapError(ErrExec, &err)
	defer wraperr.WrapError(fmt.Errorf("error during Backward propagation on %s", l.kind), &err)

	if l == nil {
		return nil, ErrNil
	} else if l.t == nil || l.y == nil {
		return nil, fmt.Errorf("calling backward before forward (missing targets or outputs): %v, %v", l.t, l.y)
	}

	grad, err = l.gradient(l.t, l.y)
	if err != nil {
		return nil, fmt.Errorf("error computing input gradient: %w", err)
	}

	l.d = grad.Copy()
	return grad, nil
}

func (l *Loss) Is(kind nn.Kind) bool {
	if l == nil {
		return false
	}
	return l.kind == kind
}

func (l *Loss) Kind() nn.Kind {
	return l.kind
}

func (l *Loss) Output() float64 {
	return l.l
}

func (l *Loss) Copy() nn.IModule {
	if l == nil {
		return nil
	}
	res := &Loss{
		kind:     l.kind,
		l:        l.l,
		output:   l.output,
		gradient: l.gradient,
	}
	if l.t != nil {
		res.t = l.t.Copy()
	}
	if l.y != nil {
		res.y = l.y.Copy()
	}
	if l.d != nil {
		res.d = l.d.Copy()
	}
	return res
}

func (l *Loss) Equal(loss nn.IModule) bool {
	if l == nil || loss == nil {
		if (l != nil && loss == nil) || (l == nil && loss != nil) {
			return false // non-nil != nil and nil != non-nil
		} else {
			return true // nil == nil
		}
	}
	if lo, ok := loss.(*Loss); !ok {
		return false
	} else if l.kind != lo.kind {
		return false
	} else if l.t != nil && !l.t.Equal(lo.t) {
		return false
	} else if l.y != nil && !l.y.Equal(lo.y) {
		return false
	} else if l.d != nil && !l.d.Equal(lo.d) {
		return false
	} else if l.l != lo.l {
		return false
	}

	return true
}

func (l *Loss) EqualApprox(loss nn.IModule) bool {
	if l == nil || loss == nil {
		if (l != nil && loss == nil) || (l == nil && loss != nil) {
			return false // non-nil != nil and nil != non-nil
		} else {
			return true // nil == nil
		}
	}
	if lo, ok := loss.(*Loss); !ok {
		return false
	} else if l.kind != lo.kind {
		return false
	} else if l.t != nil && !l.t.EqualApprox(lo.t) {
		return false
	} else if l.y != nil && !l.y.EqualApprox(lo.y) {
		return false
	} else if l.d != nil && !l.d.EqualApprox(lo.d) {
		return false
	} else if l.l != lo.l {
		return false
	}

	return true
}

func (l *Loss) toMap(stringer func(spStringer utils.SPStringer) string) map[string]string {
	return map[string]string{
		"kind": string(l.kind),
		"t":    stringer(l.t),
		"y":    stringer(l.y),
		"d":    stringer(l.d),
		"l":    fmt.Sprintf("%f", l.l),
	}
}

func (l *Loss) String() string {
	if l == nil {
		return "<nil>"
	}
	return utils.FormatObject(l.toMap(utils.String), utils.BaseFormat)
}

func (l *Loss) PrettyString() string {
	if l == nil {
		return "<nil>"
	}
	return utils.FormatObject(l.toMap(utils.PrettyString), utils.PrettyFormat)
}

func (l *Loss) ShortString() string {
	if l == nil {
		return "<nil>"
	}
	return utils.FormatObject(l.toMap(utils.ShortString), utils.ShortFormat)
}
