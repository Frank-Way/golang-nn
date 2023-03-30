package net

import (
	"fmt"
	"nn/internal/nn"
	"nn/internal/nn/layer"
	"nn/internal/nn/loss"
	"nn/pkg/wraperr"
)

const (
	FFNetwork nn.Kind = "feed forward neural network"
)

func NewFFNetwork(l loss.ILoss, layers ...layer.ILayer) (n INetwork, err error) {
	defer logger.CatchErr(&err)
	defer wraperr.WrapError(ErrCreate, &err)

	logger.Debug("create " + string(FFNetwork))

	if l == nil {
		return nil, fmt.Errorf("no loss provided: %v", l)
	} else if len(layers) < 1 {
		return nil, fmt.Errorf("no layers provided: %v", layers)
	}

	clayers := make([]layer.ILayer, len(layers))
	for i, la := range layers {
		if la == nil {
			return nil, fmt.Errorf("missing %d'th layer: %v", i, la)
		}
		if i > 0 {
			if la.InputsCount() != layers[i-1].Size() {
				return nil, fmt.Errorf("%d'th layer's size mismatch %d'th layer's inputs count: %d != %d",
					i-1, i, la.InputsCount(), layers[i-1].Size())
			}
		}
		clayers[i] = la.Copy().(layer.ILayer)
	}

	return &Network{
		kind:   FFNetwork,
		layers: clayers,
		loss:   l.Copy().(loss.ILoss),
	}, nil
}
