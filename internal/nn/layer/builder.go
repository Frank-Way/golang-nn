package layer

import (
	"fmt"
	"nn/internal/nn"
	"nn/internal/nn/operation"
	"nn/pkg/mmath/vector"
	"nn/pkg/percent"
	"nn/pkg/wraperr"
)

type Builder struct {
	kind       nn.Kind
	weight     operation.IOperation
	bias       operation.IOperation
	activation operation.IOperation
	dropout    operation.IOperation

	weightBuilder     *operation.Builder
	biasBuilder       *operation.Builder
	activationBuilder *operation.Builder
	dropoutBuilder    *operation.Builder

	resetAfterBuild bool
}

func NewBuilder(kind nn.Kind) (b *Builder, err error) {
	defer logger.CatchErr(&err)
	defer wraperr.WrapError(ErrBuilder, &err)
	defer wraperr.WrapError(fmt.Errorf("error creating Builder"), &err)

	logger.Debugf("create new builder for %s", kind)

	if _, ok := layers[kind]; !ok {
		return nil, fmt.Errorf("not a layer: %s", kind)
	}
	wb, err := operation.NewBuilder(operation.WeightMultiply)
	if err != nil {
		return nil, err
	}
	bb, err := operation.NewBuilder(operation.BiasAdd)
	if err != nil {
		return nil, err
	}
	db, err := operation.NewBuilder(operation.Dropout)
	if err != nil {
		return nil, err
	}

	return &Builder{
		kind:           kind,
		weightBuilder:  wb,
		biasBuilder:    bb,
		dropoutBuilder: db,
	}, nil
}

func (b *Builder) Build() (l ILayer, err error) {
	defer logger.CatchErr(&err)
	defer wraperr.WrapError(ErrBuilder, &err)
	defer wraperr.WrapError(fmt.Errorf("error building"), &err)
	defer func() {
		if b.resetAfterBuild {
			b.weight = nil
			b.bias = nil
			b.activation = nil
			b.dropout = nil
		}
	}()

	if b == nil {
		return nil, ErrNil
	}

	logger.Debugf("build layer %s", b.kind)
	err = b.prepare()
	if err != nil {
		return nil, err
	}
	inputs, neurons, err := b.getSizes()
	if err != nil {
		return nil, err
	}
	var operations []operation.IOperation
	switch b.kind {
	case DenseLayer:
		operations = []operation.IOperation{b.weight, b.bias, b.activation}
	case DenseDropLayer:
		operations = []operation.IOperation{b.weight, b.bias, b.activation, b.dropout}
	default:
		return nil, fmt.Errorf("unknown layer: %s", b.kind)
	}
	return &Layer{
		kind:        b.kind,
		operations:  operations,
		inputsCount: inputs,
		size:        neurons,
	}, nil
}

func (b *Builder) Weight(weight operation.IOperation) *Builder {
	b.weight = weight
	return b
}

func (b *Builder) Bias(bias operation.IOperation) *Builder {
	b.bias = bias
	return b
}

func (b *Builder) Activation(activation operation.IOperation) *Builder {
	b.activation = activation
	return b
}

func (b *Builder) Dropout(dropout operation.IOperation) *Builder {
	b.dropout = dropout
	return b
}

func (b *Builder) InputsCount(inputsCount int) *Builder {
	b.weightBuilder.InputsCount(inputsCount)
	b.biasBuilder.InputsCount(inputsCount)
	if b.activationBuilder != nil {
		b.activationBuilder.InputsCount(inputsCount)
	}
	return b
}

func (b *Builder) NeuronsCount(neuronsCount int) *Builder {
	b.weightBuilder.NeuronsCount(neuronsCount)
	b.biasBuilder.NeuronsCount(neuronsCount)
	if b.activationBuilder != nil {
		b.activationBuilder.NeuronsCount(neuronsCount)
	}
	return b
}

func (b *Builder) ActivationKind(kind nn.Kind) *Builder {
	if operation.IsActivation(kind) {
		builder, err := operation.NewBuilder(kind)
		if err != nil {
			panic(err)
		}
		b.activationBuilder = builder.SetResetAfterBuild(b.resetAfterBuild)
	}
	return b
}

func (b *Builder) SigmoidCoeffs(sigmoidCoeffs *vector.Vector) *Builder {
	if b.activationBuilder != nil {
		b.activationBuilder.SigmoidCoeffs(sigmoidCoeffs)
	}
	return b
}

func (b *Builder) SigmoidCoeffsRange(sigmoidCoeffsRange *operation.SigmoidCoeffsRange) *Builder {
	if b.activationBuilder != nil {
		b.activationBuilder.SigmoidCoeffsRange(sigmoidCoeffsRange)
	}
	return b
}

func (b *Builder) ParamInitType(paramInitType operation.ParamInitType) *Builder {
	b.weightBuilder.ParamInitType(paramInitType)
	b.biasBuilder.ParamInitType(paramInitType)
	return b
}

func (b *Builder) KeepProbability(probability percent.Percent) *Builder {
	b.dropoutBuilder.KeepProbability(probability)
	return b
}

func (b *Builder) SetResetAfterBuild(value bool) *Builder {
	b.resetAfterBuild = value
	b.weightBuilder.SetResetAfterBuild(value)
	b.biasBuilder.SetResetAfterBuild(value)
	b.dropoutBuilder.SetResetAfterBuild(value)
	return b
}

func (b *Builder) prepare() (err error) {
	switch b.kind {
	case DenseLayer:
		return b.prepareWBA()
	case DenseDropLayer:
		err = b.prepareWBA()
		if err != nil {
			return err
		}
		b.dropout, err = b.getDropout()
		if err != nil {
			return err
		}
	}
	return nil
}

func (b *Builder) prepareWBA() (err error) {
	b.weight, err = b.getWeight()
	if err != nil {
		return err
	}
	b.bias, err = b.getBias()
	if err != nil {
		return err
	}
	b.activation, err = b.getActivation()
	if err != nil {
		return err
	}
	err = b.checkSizes()
	if err != nil {
		return err
	}
	return nil
}

func (b *Builder) getWeight() (operation.IOperation, error) {
	if b.weight == nil || !b.weight.Is(operation.WeightMultiply) {
		logger.Tracef("no weight provided or provided weight is not %s", operation.WeightMultiply)
		weight, err := b.weightBuilder.Build()
		if err != nil {
			return nil, err
		}
		if !weight.Is(operation.WeightMultiply) {
			return nil, fmt.Errorf("built operation is not %s", operation.WeightMultiply)
		}
		return weight, nil
	}
	return b.weight, nil
}

func (b *Builder) getBias() (operation.IOperation, error) {
	if b.bias == nil || !b.bias.Is(operation.BiasAdd) {
		logger.Tracef("no bias provided or provided bias is not %s", operation.BiasAdd)
		bias, err := b.biasBuilder.Build()
		if err != nil {
			return nil, err
		}
		if !bias.Is(operation.BiasAdd) {
			return nil, fmt.Errorf("built operation is not %s", operation.BiasAdd)
		}
		return bias, nil
	}
	return b.bias, nil
}

func (b *Builder) getActivation() (operation.IOperation, error) {
	if b.activation == nil || !b.activation.IsActivation() {
		logger.Trace("no activation provided or provided activation is not actual activation")
		if b.activationBuilder == nil {
			return nil, fmt.Errorf("no activation provided and no activation builder configured " +
				"(perhaps configured activation kind was not actual activation)")
		}
		activation, err := b.activationBuilder.Build()
		if err != nil {
			return nil, err
		}
		if !activation.IsActivation() {
			return nil, fmt.Errorf("built operation is not activation too")
		}
		return activation, nil
	}
	return b.activation, nil
}

func (b *Builder) getSizes() (inputsCount int, neuronsCount int, err error) {
	weight, ok := b.weight.(*operation.ParamOperation)
	if !ok {
		return 0, 0, fmt.Errorf("error downcasting weight to *operation.ParamOperation")
	}
	inputs, neurons := weight.Parameter().Size()

	return inputs, neurons, nil
}

func (b *Builder) checkSizes() (err error) {
	weight, ok := b.weight.(*operation.ParamOperation)
	if !ok {
		return fmt.Errorf("error casting weight to *operation.ParamOperation")
	}
	bias, ok := b.bias.(*operation.ParamOperation)
	if !ok {
		return fmt.Errorf("error casting bias to *operation.ParamOperation")
	}
	neurons := weight.Parameter().Cols()
	biasNeurons := bias.Parameter().Cols()
	if biasNeurons != neurons {
		return fmt.Errorf("bias size must match weight cols count: %d != %d", biasNeurons, neurons)
	}
	if b.activation.Is(operation.SigmoidParamActivation) {
		activation, ok := b.activation.(*operation.ConstOperation)
		if !ok {
			return fmt.Errorf("error casting bias to *operation.ParamOperation")
		}
		coeffs := activation.Parameters()[0]
		activationNeurons := coeffs.Cols()
		if activationNeurons != neurons {
			return fmt.Errorf("sigmoid coefficients count must match weight cols count: %d != %d", activationNeurons, neurons)
		}
	}
	return nil
}

func (b *Builder) getDropout() (operation.IOperation, error) {
	if b.dropout == nil || !b.dropout.Is(operation.Dropout) {
		logger.Tracef("no dropout provided or provided operation is not %s", operation.Dropout)
		dropout, err := b.dropoutBuilder.Build()
		if err != nil || !dropout.Is(operation.Dropout) {
			logger.Warnf("error building %s or built operation is not %s too, using noop %s",
				operation.Dropout, operation.Dropout, operation.Dropout)
			dropout, err = operation.NewDropout(percent.Percent100)
			if err != nil {
				return nil, err
			}
		}
		return dropout, nil
	}
	return b.dropout, nil
}
