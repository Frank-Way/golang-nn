package prettytable

import (
	"github.com/stretchr/testify/require"
	"math/rand"
	"strings"
	"testing"
	"time"
)

var (
	rng      = rand.New(rand.NewSource(time.Now().Unix()))
	alphabet = "qwertyuiopasdfghjklzxcvbnm1234567890QWERTYUIOPASDFGHJKLZXCVBNM"
	length   = len(alphabet)
)

func randomString(size int) string {
	var sb strings.Builder
	if size < 1 {
		size = 5 + rng.Intn(15)
	}
	for i := 0; i < size; i++ {
		idx := rng.Intn(length)
		char := rune(alphabet[idx])
		sb.WriteRune(char)
	}
	return sb.String()
}

func randomStrings(count, size int) []string {
	result := make([]string, count)
	for i := 0; i < count; i++ {
		result[i] = randomString(size)
	}
	return result
}

type randomGroupParams struct {
	headerSize   int
	columnsCount int
	columnsSize  int
}

func randomGroup(params randomGroupParams) Group {
	if params.headerSize < 1 {
		params.headerSize = 5 + rng.Intn(15)
	}
	if params.columnsCount < 0 {
		params.columnsCount = 1 + rng.Intn(4)
	}
	if params.columnsSize < 0 {
		params.columnsSize = 5 + rng.Intn(15)
	}

	columns := make([]Column, params.columnsCount)
	for i := 0; i < params.columnsCount; i++ {
		columns[i] = Column{
			Header: randomString(params.headerSize),
			Values: randomStrings(params.columnsSize, 0),
		}
	}

	return Group{Columns: columns}

}

func TestBuild(t *testing.T) {
	tests := []struct {
		name   string
		err    bool
		fill   bool
		params []randomGroupParams
	}{
		{
			name: "[2,1,3] groups, same sizes, no fill",
			params: []randomGroupParams{
				{columnsCount: 2, columnsSize: 3},
				{columnsCount: 1, columnsSize: 3},
				{columnsCount: 3, columnsSize: 3},
			},
		},
		{
			name: "[2,1,3] groups, different sizes, no fill, error",
			err:  true,
			params: []randomGroupParams{
				{columnsCount: 2, columnsSize: 3},
				{columnsCount: 1, columnsSize: 2},
				{columnsCount: 3, columnsSize: 3},
			},
		},
		{
			name: "[2,1,3] groups, different sizes, fill",
			fill: true,
			params: []randomGroupParams{
				{columnsCount: 2, columnsSize: 2},
				{columnsCount: 1, columnsSize: 4},
				{columnsCount: 3, columnsSize: 8},
			},
		},
		{
			name: "[1] groups, 0 sizes, no fill",
			params: []randomGroupParams{
				{columnsCount: 1, columnsSize: 0},
			},
		},
		{
			name:   "[] groups, 0 sizes, no fill, error",
			err:    true,
			params: []randomGroupParams{},
		},
		{
			name: "[1,0] groups, 0 sizes, no fill, error",
			err:  true,
			params: []randomGroupParams{
				{columnsCount: 1, columnsSize: 0},
				{columnsCount: 0, columnsSize: 0},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			groups := make([]Group, len(test.params))
			for i := 0; i < len(test.params); i++ {
				groups[i] = randomGroup(test.params[i])
			}
			actual, err := Build(groups, test.fill)
			if test.err {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
				t.Log("\n" + actual)
			}
		})
	}
}
