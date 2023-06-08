package operation

import (
	"fmt"
	"nn/internal/nn"
	"nn/internal/utils"
	"nn/pkg/mmath/matrix"
	"nn/pkg/mmath/vector"
	"nn/pkg/percent"
	"nn/pkg/wraperr"
)

type SigmoidCoeffsRange struct {
	Left, Right float64
}

type ParamInitType uint8

const (
	DefaultInit ParamInitType = iota
	GlorotInit
)

type Builder struct {
	kind               nn.Kind
	paramInitType      ParamInitType
	keepProbability    percent.Percent
	sigmoidCoeffs      *vector.Vector
	sigmoidCoeffsRange *SigmoidCoeffsRange
	weight             *matrix.Matrix
	bias               *vector.Vector
	inputsCount        int
	neuronsCount       int

	resetAfterBuild bool
}

func NewBuilder(kind nn.Kind) (b *Builder, err error) {
	defer logger.CatchErr(&err)
	defer wraperr.WrapError(ErrBuilder, &err)

	logger.Debugf("create builder for %s", kind)
	if _, ok := operations[kind]; !ok {
		return nil, fmt.Errorf("error creating builder: not a operation: %s", kind)
	}

	return &Builder{kind: kind}, nil
}

func (b *Builder) Build() (o IOperation, err error) {
	defer logger.CatchErr(&err)
	defer wraperr.WrapError(ErrBuilder, &err)
	defer wraperr.WrapError(fmt.Errorf("error building"), &err)
	defer func() {
		if b.resetAfterBuild {
			b.weight = nil
			b.bias = nil
		}
	}()

	if b == nil {
		return nil, ErrNil
	}

	logger.Debugf("build operation %s", b.kind)
	err = b.prepare()
	if err != nil {
		return nil, err
	}

	switch b.kind {
	case Dropout:
		return Create(b.kind, b.keepProbability)
	case SigmoidParamActivation:
		return Create(b.kind, b.sigmoidCoeffs)
	case WeightMultiply:
		return Create(b.kind, b.weight)
	case BiasAdd:
		return Create(b.kind, b.bias)
	}
	return Create(b.kind)
}

func (b *Builder) ParamInitType(paramInitType ParamInitType) *Builder {
	b.paramInitType = paramInitType
	return b
}

func (b *Builder) KeepProbability(probability percent.Percent) *Builder {
	b.keepProbability = probability
	return b
}

func (b *Builder) SigmoidCoeffs(sigmoidCoeffs *vector.Vector) *Builder {
	b.sigmoidCoeffs = sigmoidCoeffs
	return b
}

func (b *Builder) SigmoidCoeffsRange(sigmoidCoeffsRange *SigmoidCoeffsRange) *Builder {
	b.sigmoidCoeffsRange = sigmoidCoeffsRange
	return b
}

func (b *Builder) Weight(weight *matrix.Matrix) *Builder {
	b.weight = weight
	return b
}

func (b *Builder) Bias(bias *vector.Vector) *Builder {
	b.bias = bias
	return b
}

func (b *Builder) InputsCount(inputsCount int) *Builder {
	b.inputsCount = inputsCount
	return b
}

func (b *Builder) NeuronsCount(neuronsCount int) *Builder {
	b.neuronsCount = neuronsCount
	return b
}

func (b *Builder) SetResetAfterBuild(value bool) *Builder {
	b.resetAfterBuild = value
	return b
}

func (b *Builder) prepare() (err error) {
	switch b.kind {
	case Dropout:
		if b.keepProbability == 0 {
			b.keepProbability = percent.Percent100
		}
	case SigmoidParamActivation:
		if b.sigmoidCoeffs == nil {
			if b.neuronsCount < 1 {
				return fmt.Errorf("no neurons count provided: %d", b.neuronsCount)
			}

			if b.sigmoidCoeffsRange == nil {
				b.sigmoidCoeffsRange = &SigmoidCoeffsRange{Left: 1, Right: 4}
			}

			b.sigmoidCoeffs, err = vector.LinSpace(b.sigmoidCoeffsRange.Left, b.sigmoidCoeffsRange.Right, b.neuronsCount)
			if err != nil {
				return fmt.Errorf("error computing sigmoid coefficients: %w", err)
			}
		}
	case WeightMultiply:
		if b.weight == nil {
			if b.inputsCount < 1 || b.neuronsCount < 1 {
				return fmt.Errorf("no inputs/neurons count provided: %d, %d", b.inputsCount, b.neuronsCount)
			}
			weights := utils.RandNormArray(b.inputsCount*b.neuronsCount, 0, b.scale())

			b.weight, err = matrix.NewMatrixRawFlat(b.inputsCount, b.neuronsCount, weights)
			if err != nil {
				return fmt.Errorf("error creating weights: %w", err)
			}
		}
	case BiasAdd:
		if b.bias == nil {
			if b.neuronsCount < 1 {
				return fmt.Errorf("no neurons count provided: %d", b.neuronsCount)
			}
			if b.paramInitType == GlorotInit && b.inputsCount < 1 {
				return fmt.Errorf("no inputs count provided for glorot init: %d", b.inputsCount)
			}
			biases := utils.RandNormArray(b.neuronsCount, 0, b.scale())

			b.bias, err = vector.NewVector(biases)
			if err != nil {
				return fmt.Errorf("error creating biases: %w", err)
			}
		}
	}

	return nil
}

func (b *Builder) scale() float64 {
	if b.paramInitType == GlorotInit {
		return 2.0 / float64(b.inputsCount+b.neuronsCount)
	}
	return 1.0
}
