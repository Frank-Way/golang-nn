package loss

import (
	"fmt"
	"nn/internal/nn"
	"nn/pkg/wraperr"
)

type Builder struct {
	kind nn.Kind
}

func NewBuilder(kind nn.Kind) (b *Builder, err error) {
	defer logger.CatchErr(&err)
	defer wraperr.WrapError(ErrBuilder, &err)
	defer wraperr.WrapError(fmt.Errorf("error creating Builder"), &err)

	logger.Debugf("create new builder for %s", kind)

	if _, ok := losses[kind]; !ok {
		return nil, fmt.Errorf("not a loss: %s", kind)
	}

	return &Builder{
		kind: kind,
	}, nil
}

func (b *Builder) Build() (l ILoss, err error) {
	defer logger.CatchErr(&err)
	defer wraperr.WrapError(ErrBuilder, &err)
	defer wraperr.WrapError(fmt.Errorf("error building"), &err)

	if b == nil {
		return nil, ErrNil
	}

	logger.Debugf("build loss %s", b.kind)
	return NewMSELoss(), nil
}
