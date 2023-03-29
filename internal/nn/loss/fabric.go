package loss

import (
	"fmt"
	"nn/pkg/wraperr"
)

func Create(kind Kind, args ...interface{}) (l ILoss, err error) {
	defer logger.CatchErr(&err)
	defer wraperr.WrapError(ErrCreate, &err)

	switch kind {
	case MSELoss:
		return NewMSELoss(), nil
	}

	return nil, fmt.Errorf("unknown loss: %s", kind)
}
