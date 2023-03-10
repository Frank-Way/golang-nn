package test

import (
	"github.com/stretchr/testify/require"
	"math"
	"nn/internal/data/approx/datagen"
	"nn/internal/data/dataset"
	"nn/internal/test/utils"
	"nn/internal/test/utils/fabrics"
	"testing"
)

func TestNewParameters(t *testing.T) {
	tests := []struct {
		utils.Base
		expr   string
		inputs []*fabrics.InputRangeParameters
		split  *dataset.DataSplitParameters
	}{
		{
			Base:   utils.Base{Name: "sin(x0) for x0 from 0 to 1 (11 values)"},
			expr:   "(sin x0)",
			inputs: []*fabrics.InputRangeParameters{{InputRange: &datagen.InputRange{Left: 0, Right: 1, Count: 11}}},
			split:  dataset.DefaultDataSplitParameters,
		},
	}

	for i := range tests {
		t.Run(tests[i].Name, func(t *testing.T) {
			inputs := make([]*datagen.InputRange, len(tests[i].inputs))
			for i, input := range tests[i].inputs {
				inputs[i] = fabrics.NewInputRange(t, input)
			}

			_, err := datagen.NewParameters(tests[i].expr, inputs, tests[i].split)
			if tests[i].Err == nil {
				require.NoError(t, err)
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, tests[i].Err)
			}
		})
	}
}

func TestParameters_Generate(t *testing.T) {
	tests := []struct {
		utils.Base
		params   fabrics.ParametersParameters
		dsParams fabrics.DatasetParameters
	}{
		{
			Base: utils.Base{
				Name: "sin(x0), 11 values from 0 to 1",
				Err:  nil,
			},
			params: fabrics.ParametersParameters{
				Expression: "(sin x0)",
				Ranges: []*datagen.InputRange{fabrics.NewInputRange(t, &fabrics.InputRangeParameters{
					InputRange: &datagen.InputRange{
						Left:  1,
						Right: 2,
						Count: 11,
					},
				})},
				Split: nil,
			},
			dsParams: fabrics.DatasetParameters{
				Single: &fabrics.DataParameters{
					X: fabrics.MatrixParameters{
						Rows:   11,
						Cols:   1,
						Values: []float64{1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2},
					},
					Y: fabrics.MatrixParameters{
						Rows:   11,
						Cols:   1,
						Values: []float64{math.Sin(1), math.Sin(1.1), math.Sin(1.2), math.Sin(1.3), math.Sin(1.4), math.Sin(1.5), math.Sin(1.6), math.Sin(1.7), math.Sin(1.8), math.Sin(1.9), math.Sin(2)},
					},
				},
			},
		},
	}

	for i := range tests {
		t.Run(tests[i].Name, func(t *testing.T) {
			params := fabrics.NewParameters(t, tests[i].params)
			ds, err := params.Generate()
			if tests[i].Err == nil {
				require.NoError(t, err)
				expected := fabrics.NewDataset(t, tests[i].dsParams)
				require.True(t, expected.EqualApprox(ds))
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, tests[i].Err)
			}
		})
	}
}
