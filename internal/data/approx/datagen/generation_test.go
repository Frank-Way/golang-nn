package datagen

import (
	"github.com/stretchr/testify/require"
	"math"
	"nn/internal/data/dataset"
	"nn/internal/data/dataset/datasettestutils"
	"nn/internal/testutils"
	"nn/internal/testutils/testfactories"
	"nn/pkg/percent"
	"testing"
)

type ParametersParameters struct {
	Expression string
	Ranges     []*InputRange
	Split      *dataset.DataSplitParameters
}

func newParameters(t *testing.T, parameters ParametersParameters) *Parameters {
	if parameters.Expression == "" {
		parameters.Expression = "(sin x0)"
		parameters.Ranges = []*InputRange{newInputRange(t, InputRangeParameters{})}
		parameters.Split = dataset.DefaultDataSplitParameters
	}
	params, err := NewParameters(parameters.Expression, parameters.Ranges, parameters.Split)
	require.NoError(t, err)

	return params
}

func TestNewParameters(t *testing.T) {
	tests := []struct {
		testutils.Base
		expr   string
		inputs []InputRangeParameters
		split  *dataset.DataSplitParameters
	}{
		{
			Base:   testutils.Base{Name: "sin(x0) for x0 from 0 to 1 (11 values)"},
			expr:   "(sin x0)",
			inputs: []InputRangeParameters{{InputRange: &InputRange{Left: 0, Right: 1, Count: 11}}},
			split:  dataset.DefaultDataSplitParameters,
		},
	}

	for _, test := range tests {
		t.Run(test.Name, func(t *testing.T) {
			inputs := make([]*InputRange, len(test.inputs))
			for i, input := range test.inputs {
				inputs[i] = newInputRange(t, input)
			}

			_, err := NewParameters(test.expr, inputs, test.split)
			if test.Err == nil {
				require.NoError(t, err)
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, test.Err)
			}
		})
	}
}

func TestParameters_Generate(t *testing.T) {
	tests := []struct {
		testutils.Base
		params   ParametersParameters
		dsParams datasettestutils.DatasetParameters
	}{
		{
			Base: testutils.Base{
				Name: "sin(x0), 11 values from 0 to 1",
				Err:  nil,
			},
			params: ParametersParameters{
				Expression: "(sin x0)",
				Ranges: []*InputRange{newInputRange(t, InputRangeParameters{
					InputRange: &InputRange{
						Left:  1,
						Right: 2,
						Count: 11,
					},
				})},
				Split: nil,
			},
			dsParams: datasettestutils.DatasetParameters{
				Single: &datasettestutils.DataParameters{
					X: testfactories.MatrixParameters{
						Rows:   11,
						Cols:   1,
						Values: []float64{1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2},
					},
					Y: testfactories.MatrixParameters{
						Rows:   11,
						Cols:   1,
						Values: []float64{math.Sin(1), math.Sin(1.1), math.Sin(1.2), math.Sin(1.3), math.Sin(1.4), math.Sin(1.5), math.Sin(1.6), math.Sin(1.7), math.Sin(1.8), math.Sin(1.9), math.Sin(2)},
					},
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.Name, func(t *testing.T) {
			params := newParameters(t, test.params)
			ds, err := params.Generate()
			if test.Err == nil {
				require.NoError(t, err)
				expected := datasettestutils.NewDataset(t, test.dsParams)
				require.True(t, expected.EqualApprox(ds))
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, test.Err)
			}
		})
	}
}

func TestParameters_Strings(t *testing.T) {
	parameters := newParameters(t, ParametersParameters{
		Expression: "(x0 + x1)",
		Ranges: []*InputRange{
			{Left: 0, Right: 1, Count: 11},
			{Left: 0, Right: 10, Count: 101},
		},
		Split: &dataset.DataSplitParameters{
			TrainPercent: percent.Percent60,
			TestsPercent: percent.Percent30,
			ValidPercent: percent.Percent10,
		},
	})

	t.Log("ShortString\n" + parameters.ShortString())
	t.Log("String\n" + parameters.String())
	t.Log("PrettyString\n" + parameters.PrettyString())
}
