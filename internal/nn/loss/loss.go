package loss

import (
	"fmt"
	"nn/internal/utils"
	"nn/pkg/mmath/matrix"
	"nn/pkg/wraperr"
)

var _ ILoss = (*Loss)(nil)

type Loss struct {
	name string

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

	if l == nil {
		return 0, fmt.Errorf("calling nil")
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
		return 0, err
	}

	l.l = loss
	return loss, nil
}

func (l *Loss) Backward() (grad *matrix.Matrix, err error) {
	defer logger.CatchErr(&err)
	defer wraperr.WrapError(ErrExec, &err)

	if l == nil {
		return nil, fmt.Errorf("calling nil")
	} else if l.t == nil || l.y == nil {
		return nil, fmt.Errorf("calling backward before forward (missing targets or outputs): %v, %v", l.t, l.y)
	}

	grad, err = l.gradient(l.t, l.y)
	if err != nil {
		return nil, err
	}

	l.d = grad.Copy()
	return grad, nil
}

func (l *Loss) Copy() *Loss {
	res := &Loss{
		name:     l.name,
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

func (l *Loss) Equal(loss *Loss) bool {
	if l == nil || loss == nil {
		if (l != nil && loss == nil) || (l == nil && loss != nil) {
			return false // non-nil != nil and nil != non-nil
		} else {
			return true // nil == nil
		}
	} else if l.name != loss.name {
		return false
	} else if l.t != nil && !l.t.Equal(loss.t) {
		return false
	} else if l.y != nil && !l.y.Equal(loss.y) {
		return false
	} else if l.d != nil && !l.d.Equal(loss.d) {
		return false
	} else if l.l != loss.l {
		return false
	}

	return true
}

func (l *Loss) EqualApprox(loss *Loss) bool {
	if l == nil || loss == nil {
		if (l != nil && loss == nil) || (l == nil && loss != nil) {
			return false // non-nil != nil and nil != non-nil
		} else {
			return true // nil == nil
		}
	} else if l.name != loss.name {
		return false
	} else if l.t != nil && !l.t.EqualApprox(loss.t) {
		return false
	} else if l.y != nil && !l.y.EqualApprox(loss.y) {
		return false
	} else if l.d != nil && !l.d.EqualApprox(loss.d) {
		return false
	} else if l.l != loss.l {
		return false
	}

	return true
}

func (l *Loss) toMap(stringer func(spStringer utils.SPStringer) string) map[string]string {
	return map[string]string{
		"name": l.name,
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
