package fabrics

import (
	"nn/internal/nn/loss"
	"testing"
)

type LossType uint8

const (
	MSELoss LossType = iota
)

type LossParameters struct {
}

func NewLoss(t *testing.T, lossType LossType, parameters LossParameters) *loss.Loss {
	switch lossType {
	case MSELoss:
		return loss.NewMSELoss()
	}
	t.Errorf("unknown lossType: %d", lossType)
	return nil
}
