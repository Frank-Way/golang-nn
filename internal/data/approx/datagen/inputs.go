package datagen

import (
	"fmt"
	"nn/internal/utils"
	"nn/pkg/mmath/vector"
	"nn/pkg/percent"
)

// InputRange represent down (include) and up (include) borders of range to generate input data for. Inputs generation
// relies on vector.LinSpace implementation.
//
// For example, from range {Left: 0, Right: 1, Count: 6} will be generated inputs: [0, 0.2, 0.4, 0.6, 0.8, 1]
type InputRange struct {
	Left            float64
	Right           float64
	TrainParameters *InputsGenerationParameters
	TestsParameters *InputsGenerationParameters
	ValidParameters *InputsGenerationParameters
}

type InputsGenerationParameters struct {
	Count     int
	Extension *ExtendParameters
}

type ExtendParameters struct {
	Left  percent.Percent
	Right percent.Percent
}

func checkInputRange(ir *InputRange) error {
	if ir == nil {
		return fmt.Errorf("no input range provided")
	} else if ir.Left >= ir.Right {
		return fmt.Errorf("left >= right: %f, %f", ir.Left, ir.Right)
	} else if ir.TrainParameters == nil {
		return fmt.Errorf("no train data parameters provided")
	} else if ir.TrainParameters.Count < 2 {
		return fmt.Errorf("invalid train inputs count provided: %d", ir.TrainParameters.Count)
	}
	return nil
}

func (r *InputRange) trainInputs() (*vector.Vector, error) {
	logger.Tracef("generate train inputs for %s", r.ShortString())
	return r.inputs(r.TrainParameters, r.TrainParameters)
}

func (r *InputRange) testsInputs() (*vector.Vector, error) {
	logger.Tracef("generate tests inputs for %s", r.ShortString())
	return r.inputs(r.TestsParameters, r.TrainParameters)
}

func (r *InputRange) validInputs() (*vector.Vector, error) {
	logger.Tracef("generate valid inputs for %s", r.ShortString())
	return r.inputs(r.ValidParameters, r.TrainParameters)
}

func (r *InputRange) inputs(
	mainParameters *InputsGenerationParameters,
	additionalParameters *InputsGenerationParameters,
) (*vector.Vector, error) {
	left, right := r.Left, r.Right
	length := right - left
	cnt := mainParameters.Count
	if mainParameters != nil {
		if cnt < 2 {
			cnt = additionalParameters.Count
		}
		if mainParameters.Extension != nil {
			left = left - mainParameters.Extension.Left.GetF(length)
			right = right + mainParameters.Extension.Right.GetF(length)
		}
	} else {
		cnt = additionalParameters.Count
	}
	return vector.LinSpace(left, right, cnt)
}

func (r *InputRange) toMap(stringer func(spStringer utils.SPStringer) string) map[string]string {
	return map[string]string{
		"Left":            fmt.Sprintf("%f", r.Left),
		"Right":           fmt.Sprintf("%f", r.Right),
		"TrainParameters": stringer(r.TrainParameters),
		"TestsParameters": stringer(r.TestsParameters),
		"ValidParameters": stringer(r.ValidParameters),
	}
}

func (r *InputRange) String() string {
	if r == nil {
		return "<nil>"
	}
	return utils.FormatObject(r.toMap(utils.String), utils.BaseFormat)
}

func (r *InputRange) PrettyString() string {
	if r == nil {
		return "<nil>"
	}
	return utils.FormatObject(r.toMap(utils.PrettyString), utils.PrettyFormat)
}

func (r *InputRange) ShortString() string {
	if r == nil {
		return "<nil>"
	}
	return utils.FormatObject(r.toMap(utils.ShortString), utils.ShortFormat)
}

func (p *ExtendParameters) toMap() map[string]string {
	return map[string]string{
		"Left":  fmt.Sprintf("%d %%", p.Left.GetI(100)),
		"Right": fmt.Sprintf("%d %%", p.Right.GetI(100)),
	}
}

func (p *ExtendParameters) String() string {
	if p == nil {
		return "<nil>"
	}
	return utils.FormatObject(p.toMap(), utils.BaseFormat)
}

func (p *ExtendParameters) PrettyString() string {
	if p == nil {
		return "<nil>"
	}
	return utils.FormatObject(p.toMap(), utils.PrettyFormat)
}

func (p *ExtendParameters) ShortString() string {
	if p == nil {
		return "<nil>"
	}
	return utils.FormatObject(p.toMap(), utils.ShortFormat)
}

func (p *InputsGenerationParameters) toMap(stringer func(spStringer utils.SPStringer) string) map[string]string {
	return map[string]string{
		"Count":     fmt.Sprintf("%d", p.Count),
		"Extension": stringer(p.Extension),
	}
}

func (p *InputsGenerationParameters) String() string {
	if p == nil {
		return "<nil>"
	}
	return utils.FormatObject(p.toMap(utils.String), utils.BaseFormat)
}

func (p *InputsGenerationParameters) PrettyString() string {
	if p == nil {
		return "<nil>"
	}
	return utils.FormatObject(p.toMap(utils.PrettyString), utils.PrettyFormat)
}

func (p *InputsGenerationParameters) ShortString() string {
	if p == nil {
		return "<nil>"
	}
	return utils.FormatObject(p.toMap(utils.ShortString), utils.ShortFormat)
}
