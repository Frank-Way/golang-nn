package expression

import (
	"fmt"
	"regexp"
	"strconv"
)

var symbolPattern, _ = regexp.Compile("^x\\d+$")

type symbol struct {
	index int
}

func newSymbol(input string) (*symbol, error) {
	if !symbolPattern.Match([]byte(input)) {
		return nil, fmt.Errorf("given %q is not a symbol", input)
	}
	index, err := strconv.ParseInt(input[1:], 10, 64)
	if err != nil {
		return nil, err
	}

	return &symbol{index: int(index)}, nil
}

func (s *symbol) exec(x []float64) (float64, error) {
	if x == nil {
		return 0, fmt.Errorf("no arguments provided for symbol x%d: %v", s.index, x)
	} else if len(x) <= s.index {
		return 0, fmt.Errorf("can not get symbol with index %d from given values %v", s.index, x)
	}

	return x[s.index], nil
}
