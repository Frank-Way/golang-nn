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

func TestNewParameters(t *testing.T) {
	testcases := []struct {
		testutils.Base
		rawExpression string
		ranges        []*InputRange
	}{
		{
			Base:          testutils.Base{Name: "basic creation"},
			rawExpression: "(sin x0)",
			ranges: []*InputRange{
				{
					Left:            0,
					Right:           1,
					TrainParameters: &InputsGenerationParameters{Count: 11},
				},
			},
		},
		{
			Base:          testutils.Base{Name: "wrong expression", Err: ErrCreate},
			rawExpression: "(sin x0 1)",
		},
		{
			Base:          testutils.Base{Name: "no input ranges", Err: ErrCreate},
			rawExpression: "(sin x0)",
		},
		{
			Base:          testutils.Base{Name: "empty input ranges"},
			rawExpression: "(sin x0)",
			ranges:        []*InputRange{},
		},
		{
			Base:          testutils.Base{Name: "complex creation"},
			rawExpression: "(sin x0)",
			ranges: []*InputRange{
				{
					Left:  0,
					Right: 1,
					TrainParameters: &InputsGenerationParameters{
						Count: 11,
						Extension: &ExtendParameters{
							Left:  percent.Percent10,
							Right: percent.Percent10,
						},
					},
					TestsParameters: &InputsGenerationParameters{
						Count:     6,
						Extension: nil,
					},
					ValidParameters: &InputsGenerationParameters{
						Count:     3,
						Extension: nil,
					},
				},
			},
		},
	}

	for _, tc := range testcases {
		t.Run(tc.Name, func(t *testing.T) {
			var err error
			if tc.ranges == nil {
				_, err = NewParameters(tc.rawExpression)
			} else {
				_, err = NewParameters(tc.rawExpression, tc.ranges...)
			}
			if tc.Err == nil {
				require.NoError(t, err)
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, tc.Err)
			}
		})
	}
}

func TestGenerate(t *testing.T) {
	newInputRange := func(left float64, right float64,
		trainCount int, trainLeft percent.Percent, trainRight percent.Percent,
		testsCount int, testsLeft percent.Percent, testsRight percent.Percent,
		validCount int, validLeft percent.Percent, validRight percent.Percent) *InputRange {
		return &InputRange{
			Left:            left,
			Right:           right,
			TrainParameters: &InputsGenerationParameters{Count: trainCount, Extension: &ExtendParameters{Left: trainLeft, Right: trainRight}},
			TestsParameters: &InputsGenerationParameters{Count: testsCount, Extension: &ExtendParameters{Left: testsLeft, Right: testsRight}},
			ValidParameters: &InputsGenerationParameters{Count: validCount, Extension: &ExtendParameters{Left: validLeft, Right: validRight}},
		}
	}

	newParameters := func(rawExpression string, ranges ...*InputRange) *Parameters {
		parameters, err := NewParameters(rawExpression, ranges...)
		require.NoError(t, err)
		return parameters
	}

	testcases := []struct {
		testutils.Base
		parameters *Parameters
		expected   *dataset.Dataset
	}{
		{
			Base: testutils.Base{
				Name: "basic generation",
				Err:  nil,
			},
			parameters: newParameters("(sin x0)",
				newInputRange(0, 1,
					11, percent.Percent10, percent.Percent10,
					6, percent.Percent0, percent.Percent0,
					3, percent.Percent0, percent.Percent0)),
			expected: datasettestutils.NewDataset(t, datasettestutils.DatasetParameters{
				Train: datasettestutils.DataParameters{
					X: testfactories.MatrixParameters{Rows: 11, Cols: 1,
						Values: []float64{-0.1,
							-0.1 + 1.2/10*1,
							-0.1 + 1.2/10*2,
							-0.1 + 1.2/10*3,
							-0.1 + 1.2/10*4,
							-0.1 + 1.2/10*5,
							-0.1 + 1.2/10*6,
							-0.1 + 1.2/10*7,
							-0.1 + 1.2/10*8,
							-0.1 + 1.2/10*9,
							1.1},
					},
					Y: testfactories.MatrixParameters{Rows: 11, Cols: 1,
						Values: []float64{math.Sin(-0.1),
							math.Sin(-0.1 + 1.2/10*1),
							math.Sin(-0.1 + 1.2/10*2),
							math.Sin(-0.1 + 1.2/10*3),
							math.Sin(-0.1 + 1.2/10*4),
							math.Sin(-0.1 + 1.2/10*5),
							math.Sin(-0.1 + 1.2/10*6),
							math.Sin(-0.1 + 1.2/10*7),
							math.Sin(-0.1 + 1.2/10*8),
							math.Sin(-0.1 + 1.2/10*9),
							math.Sin(1.1)},
					},
				},
				Tests: datasettestutils.DataParameters{
					X: testfactories.MatrixParameters{Rows: 6, Cols: 1,
						Values: []float64{0,
							1.0 / 5 * 1,
							1.0 / 5 * 2,
							1.0 / 5 * 3,
							1.0 / 5 * 4,
							1},
					},
					Y: testfactories.MatrixParameters{Rows: 6, Cols: 1,
						Values: []float64{math.Sin(0),
							math.Sin(1.0 / 5 * 1),
							math.Sin(1.0 / 5 * 2),
							math.Sin(1.0 / 5 * 3),
							math.Sin(1.0 / 5 * 4),
							math.Sin(1)},
					},
				},
				Valid: datasettestutils.DataParameters{
					X: testfactories.MatrixParameters{Rows: 3, Cols: 1,
						Values: []float64{0,
							0.5,
							1},
					},
					Y: testfactories.MatrixParameters{Rows: 3, Cols: 1,
						Values: []float64{math.Sin(0),
							math.Sin(0.5),
							math.Sin(1)},
					},
				},
			}),
		},
	}

	for _, tc := range testcases {
		t.Run(tc.Name, func(t *testing.T) {
			actual, err := Generate(tc.parameters)
			if tc.Err == nil {
				require.NoError(t, err)
				require.True(t, tc.expected.EqualApprox(actual), "expected dataset: %s\nactual dataset:   %s",
					tc.expected.String(), actual.String())
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, tc.Err)
			}
		})
	}
}
