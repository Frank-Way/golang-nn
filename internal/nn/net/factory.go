package net

import (
	"fmt"
	"nn/internal/nn"
	"nn/internal/nn/layer"
	"nn/internal/nn/loss"
	"nn/pkg/wraperr"
)

func Create(kind nn.Kind, args ...interface{}) (n INetwork, err error) {
	defer logger.CatchErr(&err)
	defer wraperr.WrapError(ErrCreate, &err)

	switch kind {
	case FFNetwork:
		if len(args) < 2 {
			return nil, fmt.Errorf("not enough arguments to create %q, required at least %d, provided %d",
				FFNetwork, 2, len(args))
		} else if l, ok := args[0].(loss.ILoss); !ok {
			return nil, fmt.Errorf("first argument is not loss.ILoss: %T", args[0])
		} else {
			layers := make([]layer.ILayer, len(args)-1)
			for i, arg := range args[1:] {
				if la, ok := arg.(layer.ILayer); !ok {
					return nil, fmt.Errorf("%d'th argument is not layer.ILayer: %T", i+1, arg)
				} else {
					layers[i] = la
				}
			}
			return NewFFNetwork(l, layers...)
		}
	}

	return nil, fmt.Errorf("unknown network: %s", kind)
}
