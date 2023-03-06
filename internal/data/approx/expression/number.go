package expression

import (
	"fmt"
	"regexp"
	"strconv"
)

var numberPattern, _ = regexp.Compile("^-?\\d+(\\.\\d+)?$")

type number struct {
	value float64
}

func newNumber(input string) (*number, error) {
	if !numberPattern.Match([]byte(input)) {
		return nil, fmt.Errorf("given %q is not a number", input)
	}
	value, err := strconv.ParseFloat(input, 64)
	if err != nil {
		return nil, err
	}

	return &number{value: value}, nil
}

func (n *number) exec(x []float64) (float64, error) {
	return n.value, nil
}
