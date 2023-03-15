package percent

import (
	"github.com/stretchr/testify/require"
	"nn/internal/testutils"
	"testing"
)

func TestPercent_GetI(t *testing.T) {
	tests := []struct {
		testutils.Base
		percent  Percent
		value    int
		expected int
	}{
		{Base: testutils.Base{Name: "10% of 100"}, percent: Percent10, value: 100, expected: 10},
		{Base: testutils.Base{Name: "20% of 4"}, percent: Percent20, value: 4, expected: 0},
		{Base: testutils.Base{Name: "30% of 5"}, percent: Percent30, value: 5, expected: 1},
		{Base: testutils.Base{Name: "100% of 1"}, percent: Percent100, value: 1, expected: 1},
	}

	for _, test := range tests {
		t.Run(test.Name, func(t *testing.T) {
			actual := test.percent.GetI(test.value)
			require.Equal(t, test.expected, actual)
		})
	}
}

func TestPercent_GetF(t *testing.T) {
	tests := []struct {
		testutils.Base
		percent  Percent
		value    float64
		expected float64
	}{
		{Base: testutils.Base{Name: "10% of 100"}, percent: Percent10, value: 100, expected: 10},
		{Base: testutils.Base{Name: "20% of 4"}, percent: Percent20, value: 4, expected: 4.0 / 5.0},
		{Base: testutils.Base{Name: "30% of 5"}, percent: Percent30, value: 5, expected: 5.0 * 3.0 / 10.0},
		{Base: testutils.Base{Name: "100% of 1"}, percent: Percent100, value: 1, expected: 1},
	}

	for _, test := range tests {
		t.Run(test.Name, func(t *testing.T) {
			actual := test.percent.GetF(test.value)
			require.Equal(t, test.expected, actual)
		})
	}
}
