package net

import (
	"fmt"
	"nn/internal/nn"
	"nn/internal/nn/layer"
	"nn/internal/nn/loss"
	"nn/internal/nn/operation"
	"nn/pkg/mmath/vector"
	"nn/pkg/percent"
	"nn/pkg/wraperr"
)

type Builder struct {
	kind   nn.Kind
	layers []layer.ILayer
	loss   loss.ILoss

	layerBuilders []*layer.Builder
	lossBuilder   *loss.Builder
}

func NewBuilder(kind nn.Kind) (b *Builder, err error) {
	defer logger.CatchErr(&err)
	defer wraperr.WrapError(ErrBuilder, &err)
	defer wraperr.WrapError(fmt.Errorf("error creating Builder"), &err)

	logger.Debugf("create new builder for %s", kind)

	if _, ok := networks[kind]; !ok {
		return nil, fmt.Errorf("not a network: %s", kind)
	}

	return &Builder{
		kind: kind,
	}, nil
}

func (b *Builder) Build() (l INetwork, err error) {
	defer logger.CatchErr(&err)
	defer wraperr.WrapError(ErrBuilder, &err)
	defer wraperr.WrapError(fmt.Errorf("error building"), &err)

	if b == nil {
		return nil, ErrNil
	}

	logger.Debugf("build network %s", b.kind)
	err = b.prepare()
	if err != nil {
		return nil, err
	}

	return NewFFNetwork(b.loss, b.layers...)
}

func (b *Builder) Loss(l loss.ILoss) *Builder {
	b.loss = l
	return b
}

func (b *Builder) LossKind(kind nn.Kind) *Builder {
	if loss.IsLoss(kind) {
		builder, err := loss.NewBuilder(kind)
		if err != nil {
			panic(err)
		}
		b.lossBuilder = builder
	}
	return b
}

func (b *Builder) AddLayer(l layer.ILayer) *Builder {
	b.layers = append(b.layers, l)
	return b
}

func (b *Builder) AddLayerKind(kind nn.Kind) *Builder {
	if layer.IsLayer(kind) {
		builder, err := layer.NewBuilder(kind)
		if err != nil {
			panic(err)
		}
		b.layerBuilders = append(b.layerBuilders, builder)
	}
	return b
}

func (b *Builder) AddWeight(weight operation.IOperation) *Builder {
	if len(b.layerBuilders) > 0 {
		b.layerBuilders[len(b.layerBuilders)-1].Weight(weight)
	}
	return b
}

func (b *Builder) AddBias(bias operation.IOperation) *Builder {
	if len(b.layerBuilders) > 0 {
		b.layerBuilders[len(b.layerBuilders)-1].Bias(bias)
	}
	return b
}

func (b *Builder) AddActivation(activation operation.IOperation) *Builder {
	if len(b.layerBuilders) > 0 {
		b.layerBuilders[len(b.layerBuilders)-1].Activation(activation)
	}
	return b
}

func (b *Builder) AddDropout(dropout operation.IOperation) *Builder {
	if len(b.layerBuilders) > 0 {
		b.layerBuilders[len(b.layerBuilders)-1].Dropout(dropout)
	}
	return b
}

func (b *Builder) AddInputsCount(inputsCount int) *Builder {
	if len(b.layerBuilders) > 0 {
		b.layerBuilders[len(b.layerBuilders)-1].InputsCount(inputsCount)
	}
	return b
}

func (b *Builder) AddNeuronsCount(neuronsCount int) *Builder {
	if len(b.layerBuilders) > 0 {
		b.layerBuilders[len(b.layerBuilders)-1].NeuronsCount(neuronsCount)
	}
	return b
}

func (b *Builder) AddActivationKind(kind nn.Kind) *Builder {
	if len(b.layerBuilders) > 0 {
		b.layerBuilders[len(b.layerBuilders)-1].ActivationKind(kind)
	}
	return b
}

func (b *Builder) AddSigmoidCoeffs(sigmoidCoeffs *vector.Vector) *Builder {
	if len(b.layerBuilders) > 0 {
		b.layerBuilders[len(b.layerBuilders)-1].SigmoidCoeffs(sigmoidCoeffs)
	}
	return b
}

func (b *Builder) AddSigmoidCoeffsRange(sigmoidCoeffsRange *operation.SigmoidCoeffsRange) *Builder {
	if len(b.layerBuilders) > 0 {
		b.layerBuilders[len(b.layerBuilders)-1].SigmoidCoeffsRange(sigmoidCoeffsRange)
	}
	return b
}

func (b *Builder) AddParamInitType(paramInitType operation.ParamInitType) *Builder {
	if len(b.layerBuilders) > 0 {
		b.layerBuilders[len(b.layerBuilders)-1].ParamInitType(paramInitType)
	}
	return b
}

func (b *Builder) AddKeepProbability(probability percent.Percent) *Builder {
	if len(b.layerBuilders) > 0 {
		b.layerBuilders[len(b.layerBuilders)-1].KeepProbability(probability)
	}
	return b
}

func (b *Builder) Layer(index int, l layer.ILayer) *Builder {
	if index < 0 {
		return b
	}
	for len(b.layers) <= index {
		b.layers = append(b.layers, nil)
	}
	b.layers[index] = l
	return b
}

func (b *Builder) LayerKind(index int, kind nn.Kind) *Builder {
	if index < 0 {
		return b
	}
	for len(b.layerBuilders) <= index {
		b.layerBuilders = append(b.layerBuilders, nil)
	}
	if layer.IsLayer(kind) {
		builder, err := layer.NewBuilder(kind)
		if err != nil {
			panic(err)
		}
		b.layerBuilders[index] = builder
	}
	return b
}

func (b *Builder) Weight(index int, weight operation.IOperation) *Builder {
	if index < 0 {
		return b
	}
	for len(b.layerBuilders) <= index {
		b.layerBuilders = append(b.layerBuilders, nil)
	}
	b.layerBuilders[index].Weight(weight)
	return b
}

func (b *Builder) Bias(index int, bias operation.IOperation) *Builder {
	if index < 0 {
		return b
	}
	for len(b.layerBuilders) <= index {
		b.layerBuilders = append(b.layerBuilders, nil)
	}
	b.layerBuilders[index].Bias(bias)
	return b
}

func (b *Builder) Activation(index int, activation operation.IOperation) *Builder {
	if index < 0 {
		return b
	}
	for len(b.layerBuilders) <= index {
		b.layerBuilders = append(b.layerBuilders, nil)
	}
	b.layerBuilders[index].Activation(activation)
	return b
}

func (b *Builder) Dropout(index int, dropout operation.IOperation) *Builder {
	if index < 0 {
		return b
	}
	for len(b.layerBuilders) <= index {
		b.layerBuilders = append(b.layerBuilders, nil)
	}
	b.layerBuilders[index].Dropout(dropout)
	return b
}

func (b *Builder) InputsCount(index int, inputsCount int) *Builder {
	if index < 0 {
		return b
	}
	for len(b.layerBuilders) <= index {
		b.layerBuilders = append(b.layerBuilders, nil)
	}
	b.layerBuilders[index].InputsCount(inputsCount)
	return b
}

func (b *Builder) NeuronsCount(index int, neuronsCount int) *Builder {
	if index < 0 {
		return b
	}
	for len(b.layerBuilders) <= index {
		b.layerBuilders = append(b.layerBuilders, nil)
	}
	b.layerBuilders[index].NeuronsCount(neuronsCount)
	return b
}

func (b *Builder) ActivationKind(index int, kind nn.Kind) *Builder {
	if index < 0 {
		return b
	}
	for len(b.layerBuilders) <= index {
		b.layerBuilders = append(b.layerBuilders, nil)
	}
	b.layerBuilders[index].ActivationKind(kind)
	return b
}

func (b *Builder) SigmoidCoeffs(index int, sigmoidCoeffs *vector.Vector) *Builder {
	if index < 0 {
		return b
	}
	for len(b.layerBuilders) <= index {
		b.layerBuilders = append(b.layerBuilders, nil)
	}
	b.layerBuilders[index].SigmoidCoeffs(sigmoidCoeffs)
	return b
}

func (b *Builder) SigmoidCoeffsRange(index int, sigmoidCoeffsRange *operation.SigmoidCoeffsRange) *Builder {
	if index < 0 {
		return b
	}
	for len(b.layerBuilders) <= index {
		b.layerBuilders = append(b.layerBuilders, nil)
	}
	b.layerBuilders[index].SigmoidCoeffsRange(sigmoidCoeffsRange)
	return b
}

func (b *Builder) ParamInitType(index int, paramInitType operation.ParamInitType) *Builder {
	if index < 0 {
		return b
	}
	for len(b.layerBuilders) <= index {
		b.layerBuilders = append(b.layerBuilders, nil)
	}
	b.layerBuilders[index].ParamInitType(paramInitType)
	return b
}

func (b *Builder) KeepProbability(index int, probability percent.Percent) *Builder {
	if index < 0 {
		return b
	}
	for len(b.layerBuilders) <= index {
		b.layerBuilders = append(b.layerBuilders, nil)
	}
	b.layerBuilders[index].KeepProbability(probability)
	return b
}

func (b *Builder) prepare() (err error) {
	b.loss, err = b.getLoss()
	if err != nil {
		return err
	}
	b.layers, err = b.getLayers()
	if err != nil {
		return err
	}
	return nil
}

func (b *Builder) getLoss() (l loss.ILoss, err error) {
	if b.loss == nil {
		logger.Tracef("no loss provided")
		if b.lossBuilder == nil {
			return nil, fmt.Errorf("no loss provided and no loss builder configured")
		}
		lo, err := b.lossBuilder.Build()
		if err != nil {
			return nil, err
		}
		return lo, nil
	}
	return b.loss, nil
}

func (b *Builder) getLayers() (ls []layer.ILayer, err error) {
	if len(b.layers) < 1 {
		logger.Tracef("no layers provided")
		if len(b.layerBuilders) < 1 {
			return nil, fmt.Errorf("no layers provided and no layer builders configured")
		}
		b.layers = make([]layer.ILayer, len(b.layerBuilders))
		for i, layerBuilder := range b.layerBuilders {
			b.layers[i], err = layerBuilder.Build()
			if err != nil {
				return nil, fmt.Errorf("error building %d'th layer: %w", i, err)
			}
		}
	}
	return b.layers, nil
}
