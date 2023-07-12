package estimate

import (
	"nn/pkg/mmath/matrix"
	"nn/pkg/wraperr"
)

type Result struct {
	MaxAbsoluteError         float64
	AvgAbsoluteError         float64
	MaxRelativeError         float64
	MaxRelativeErrorPercents float64

	Deltas         *matrix.Matrix
	AbsoluteDeltas *matrix.Matrix
	RelativeDeltas *matrix.Matrix
}

func Estimate(outputs *matrix.Matrix, targets *matrix.Matrix) (r *Result, err error) {
	defer logger.CatchErr(&err)
	defer wraperr.WrapError(ErrEstimate, &err)
	r = &Result{}
	logger.Debugf("estimating: outputs [%s], targets [%s]", outputs.ShortString(), targets.ShortString())

	r.Deltas, err = outputs.Sub(targets)
	if err != nil {
		return nil, err
	}

	r.AbsoluteDeltas = r.Deltas.Abs()
	r.MaxAbsoluteError = r.AbsoluteDeltas.Max()
	r.AvgAbsoluteError = r.AbsoluteDeltas.Avg()

	max := targets.Max()
	min := targets.Min()
	interval := max - min

	r.RelativeDeltas = r.AbsoluteDeltas.DivNum(interval)

	r.MaxRelativeError = r.RelativeDeltas.Max()
	r.MaxRelativeErrorPercents = r.MaxRelativeError * 100

	return r, nil
}
