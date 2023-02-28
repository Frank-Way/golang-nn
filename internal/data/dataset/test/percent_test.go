package test

import (
	"github.com/stretchr/testify/require"
	"nn/internal/data/dataset"
	"nn/internal/test/utils"
	"testing"
)

func TestPercent_GetI(t *testing.T) {
	tests := []struct {
		utils.Base
		percent  dataset.Percent
		value    int
		expected int
	}{
		{Base: utils.Base{Name: "10% of 100"}, percent: dataset.Percent10, value: 100, expected: 10},
		{Base: utils.Base{Name: "20% of 4"}, percent: dataset.Percent20, value: 4, expected: 0},
		{Base: utils.Base{Name: "30% of 5"}, percent: dataset.Percent30, value: 5, expected: 1},
		{Base: utils.Base{Name: "100% of 1"}, percent: dataset.Percent100, value: 1, expected: 1},
	}

	for i := range tests {
		t.Run(tests[i].Name, func(t *testing.T) {
			actual := tests[i].percent.GetI(tests[i].value)
			require.Equal(t, tests[i].expected, actual)
		})
	}
}

func TestPercent_GetF(t *testing.T) {
	tests := []struct {
		utils.Base
		percent  dataset.Percent
		value    float64
		expected float64
	}{
		{Base: utils.Base{Name: "10% of 100"}, percent: dataset.Percent10, value: 100, expected: 10},
		{Base: utils.Base{Name: "20% of 4"}, percent: dataset.Percent20, value: 4, expected: 4.0 / 5.0},
		{Base: utils.Base{Name: "30% of 5"}, percent: dataset.Percent30, value: 5, expected: 5.0 * 3.0 / 10.0},
		{Base: utils.Base{Name: "100% of 1"}, percent: dataset.Percent100, value: 1, expected: 1},
	}

	for i := range tests {
		t.Run(tests[i].Name, func(t *testing.T) {
			actual := tests[i].percent.GetF(tests[i].value)
			require.Equal(t, tests[i].expected, actual)
		})
	}
}
