// Package percent provides functionality for Percent. There is some constants for 0, 10, ..., 100 percents.
package percent

import (
	"fmt"
	"math/rand"
)

// Percent meant to represent float value in [0.0; 1.0]
type Percent uint8

const (
	Percent0 Percent = 10 * iota
	Percent10
	Percent20
	Percent30
	Percent40
	Percent50
	Percent60
	Percent70
	Percent80
	Percent90
	Percent100
)

// value unwraps under-laying uint
func (p Percent) value() uint8 {
	switch p {
	case Percent0:
		return 0
	case Percent10:
		return 10
	case Percent20:
		return 20
	case Percent30:
		return 30
	case Percent40:
		return 40
	case Percent50:
		return 50
	case Percent60:
		return 60
	case Percent70:
		return 70
	case Percent80:
		return 80
	case Percent90:
		return 90
	case Percent100:
		return 100
	}
	return 100
}

func GetApproximate(value float64) Percent {
	if value < 0.05 {
		return Percent0
	} else if value < 0.15 {
		return Percent10
	} else if value < 0.25 {
		return Percent20
	} else if value < 0.35 {
		return Percent30
	} else if value < 0.45 {
		return Percent40
	} else if value < 0.55 {
		return Percent50
	} else if value < 0.65 {
		return Percent60
	} else if value < 0.75 {
		return Percent70
	} else if value < 0.85 {
		return Percent80
	} else if value < 0.95 {
		return Percent90
	} else {
		return Percent100
	}
}

// GetI returns percent of given value as int
//
// Example:
//     Percent30.GetI(5) // 1
func (p Percent) GetI(value int) int {
	return value * int(p.value()) / 100
}

// GetF returns percent of given value as float
//
// Example:
//     Percent30.GetF(5) // 1.5
func (p Percent) GetF(value float64) float64 {
	return value * float64(p.value()) / 100.0
}

// Hit return true if randomly generated number less or equal to percent value, otherwise return false.
func (p Percent) Hit() bool {
	return rand.Float64() <= p.GetF(1)
}

// Hit return true if randomly generated number less or equal to percent value, otherwise return false.
func (p Percent) Reverse() Percent {
	return Percent(100 - p.GetI(100))
}

func (p Percent) String() string {
	return fmt.Sprintf("%d %%", p.GetI(100))
}

func (p Percent) ShortString() string {
	return fmt.Sprintf("%d %%", p.GetI(100))
}

func (p Percent) PrettyString() string {
	return fmt.Sprintf("%d %%", p.GetI(100))
}
