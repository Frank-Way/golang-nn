package test

import (
	"github.com/stretchr/testify/require"
	"math"
	"nn/internal/nn/operation"
	"nn/internal/test/utils/fabrics"
	"nn/pkg/mmath/matrix"
	"nn/pkg/percent"
	"testing"
)

func TestNewDropoutOperation(t *testing.T) {
	_, err := operation.NewDropout(percent.Percent10)
	require.NoError(t, err)
}

func TestDropout_Forward(t *testing.T) {
	prob := percent.Percent30
	dropout := fabrics.NewDropout(t, fabrics.DropoutParameters{Percent: prob})
	in, err := matrix.NewMatrixOf(10, 10, 1)
	require.NoError(t, err)
	inSum := in.Sum()
	outSum := prob.GetF(inSum)
	epsilon := percent.Percent10.GetF(outSum)
	tries := 5 // results are random, so it needs to take several tries
	outs := make([]*matrix.Matrix, tries)
	for try := 0; try < tries; try++ {
		outs[try], err = dropout.Forward(in)
		require.NoError(t, err)
	}
	result := false
	for i, out := range outs {
		// ensure regenerating new mask by every call
		for j, out2 := range outs {
			if i == j {
				continue
			}
			require.False(t, out.EqualApprox(out2))
		}
		sum := out.Sum()
		delta := math.Abs(outSum - sum)
		if delta < epsilon {
			result = true
		}
	}
	require.True(t, result)
}

func TestDropout_Backward(t *testing.T) {
	prob := percent.Percent30
	dropout := fabrics.NewDropout(t, fabrics.DropoutParameters{Percent: prob})
	in, err := matrix.NewMatrixOf(10, 10, 1)
	require.NoError(t, err)
	outGrad := in.Copy()
	tries := 5
	for try := 0; try < tries; try++ {
		out, err := dropout.Forward(in)
		require.NoError(t, err)
		inGrad, err := dropout.Backward(outGrad)
		require.NoError(t, err)
		// ensure dropping out the same neurons (mask keeps the same until new Forward call)
		require.True(t, out.Equal(inGrad))
	}
}
