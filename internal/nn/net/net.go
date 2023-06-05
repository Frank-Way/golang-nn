package net

import (
	"fmt"
	"nn/internal/nn"
	"nn/internal/nn/layer"
	"nn/internal/nn/loss"
	"nn/internal/nn/operation"
	"nn/internal/utils"
	"nn/pkg/mmath/matrix"
	"nn/pkg/wraperr"
)

var _ INetwork = (*Network)(nil)

type Network struct {
	kind   nn.Kind
	layers []layer.ILayer
	loss   loss.ILoss
}

func (n *Network) Forward(x *matrix.Matrix) (y *matrix.Matrix, err error) {
	defer logger.CatchErr(&err)
	defer wraperr.WrapError(ErrExec, &err)
	defer wraperr.WrapError(fmt.Errorf("error during Forward propagation"), &err)

	if n == nil {
		return nil, ErrNil
	} else if x == nil {
		return nil, fmt.Errorf("no input provided: %v", x)
	}

	y = x.Copy()
	for i, l := range n.layers {
		y, err = l.Forward(y)
		if err != nil {
			return nil, fmt.Errorf("error processing %d'th layer: %w", i, err)
		}
	}
	return y, nil
}

func (n *Network) Loss(t *matrix.Matrix) (l float64, err error) {
	defer logger.CatchErr(&err)
	defer wraperr.WrapError(ErrExec, &err)
	defer wraperr.WrapError(fmt.Errorf("error during Loss calculation"), &err)

	if n == nil {
		return 0, ErrNil
	} else if t == nil {
		return 0, fmt.Errorf("no targets provided: %v", t)
	}

	return n.loss.Forward(t, n.layers[len(n.layers)-1].Output())
}

func (n *Network) Backward() (dx *matrix.Matrix, err error) {
	defer logger.CatchErr(&err)
	defer wraperr.WrapError(ErrExec, &err)
	defer wraperr.WrapError(fmt.Errorf("error during Backward propagation"), &err)

	if n == nil {
		return nil, ErrNil
	}

	dx, err = n.loss.Backward()
	if err != nil {
		return nil, fmt.Errorf("error during calculating loss gradient")
	}

	for i := len(n.layers) - 1; i >= 0; i-- {
		dx, err = n.layers[i].Backward(dx)
		if err != nil {
			return nil, fmt.Errorf("error during calculating gradient on %d'th layer: %w", i, err)
		}
	}

	return dx, nil
}

func (n *Network) ApplyOptim(optimizer operation.Optimizer) (err error) {
	defer logger.CatchErr(&err)
	defer wraperr.WrapError(ErrExec, &err)
	defer wraperr.WrapError(fmt.Errorf("error during optimization"), &err)

	if n == nil {
		return ErrNil
	} else if optimizer == nil {
		return fmt.Errorf("no optimizer provided")
	}

	for i, l := range n.layers {
		err = l.ApplyOptim(optimizer)
		if err != nil {
			return fmt.Errorf("error optimizing %d'th layer: %w", i, err)
		}
	}
	return nil
}

func (n *Network) Is(kind nn.Kind) bool {
	if n == nil {
		return false
	}
	return n.kind == kind
}

func (n *Network) Kind() nn.Kind {
	return n.kind
}

func (n *Network) Copy() nn.IModule {
	if n == nil {
		return nil
	}
	network := &Network{kind: n.kind}
	if n.loss != nil {
		network.loss = n.loss.Copy().(loss.ILoss)
	}
	if len(n.layers) > 0 {
		layers := make([]layer.ILayer, len(n.layers))
		for i, l := range n.layers {
			layers[i] = l.Copy().(layer.ILayer)
		}
		network.layers = layers
	}
	return network
}

func (n *Network) Equal(network nn.IModule) bool {
	if n == nil || network == nil {
		if (n != nil && network == nil) || (n == nil && network != nil) {
			return false // non-nil != nil and nil != non-nil
		} else {
			return true // nil == nil
		}
	}

	if ne, ok := network.(*Network); !ok {
		return false
	} else if ne.kind != n.kind || !n.loss.Equal(ne.loss) {
		return false
	} else if len(ne.layers) != len(n.layers) {
		return false
	} else {
		for i, l := range n.layers {
			if !l.Equal(ne.layers[i]) {
				return false
			}
		}
	}

	return true
}

func (n *Network) EqualApprox(network nn.IModule) bool {
	if n == nil || network == nil {
		if (n != nil && network == nil) || (n == nil && network != nil) {
			return false // non-nil != nil and nil != non-nil
		} else {
			return true // nil == nil
		}
	}

	if ne, ok := network.(*Network); !ok {
		return false
	} else if ne.kind != n.kind || !n.loss.EqualApprox(ne.loss) {
		return false
	} else if len(ne.layers) != len(n.layers) {
		return false
	} else {
		for i, l := range n.layers {
			if !l.EqualApprox(ne.layers[i]) {
				return false
			}
		}
	}

	return true
}

func (n *Network) layersAsSPStringers() []utils.SPStringer {
	res := make([]utils.SPStringer, len(n.layers))
	for i, l := range n.layers {
		res[i] = l
	}
	return res
}

func (n *Network) toMap(stringer func(s utils.SPStringer) string, stringers func(s []utils.SPStringer) string) map[string]string {
	return map[string]string{
		"kind":       string(n.kind),
		"loss":       stringer(n.loss),
		"operations": stringers(n.layersAsSPStringers()),
	}
}

func (n *Network) String() string {
	if n == nil {
		return "<nil>"
	}
	return utils.FormatObject(n.toMap(utils.String, utils.Strings), utils.BaseFormat)
}

func (n *Network) PrettyString() string {
	if n == nil {
		return "<nil>"
	}
	return utils.FormatObject(n.toMap(utils.PrettyString, utils.PrettyStrings), utils.PrettyFormat)
}

func (n *Network) ShortString() string {
	if n == nil {
		return "<nil>"
	}
	return utils.FormatObject(n.toMap(utils.ShortString, utils.ShortStrings), utils.ShortFormat)
}
