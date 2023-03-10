package test

import (
	"github.com/stretchr/testify/require"
	"nn/internal/data/approx/datagen"
	"nn/internal/test/utils"
	"testing"
)

func TestNewInputRange(t *testing.T) {
	tests := []struct {
		utils.Base
		left  float64
		right float64
		count int
	}{
		{Base: utils.Base{Name: "0 to 1, 5 values"}, left: 0, right: 1, count: 5},
		{Base: utils.Base{Name: "0 to 1, 50000 values"}, left: 0, right: 1, count: 50000},
		{Base: utils.Base{Name: "2 to 1, 5 values", Err: datagen.ErrCreate}, left: 2, right: 1, count: 5},
		{Base: utils.Base{Name: "0 to 1, 1 values", Err: datagen.ErrCreate}, left: 1, right: 1, count: 1},
	}

	for i := range tests {
		t.Run(tests[i].Name, func(t *testing.T) {
			_, err := datagen.NewInputRange(tests[i].left, tests[i].right, tests[i].count)
			if tests[i].Err == nil {
				require.NoError(t, err)
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, tests[i].Err)
			}
		})
	}
}
