package percent

import (
	"fmt"
	"math/rand"
	"time"
)

type Percent uint8

var rng = rand.New(rand.NewSource(time.Now().Unix()))

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

func (p Percent) GetI(value int) int {
	return value * int(p.value()) / 100
}

func (p Percent) GetF(value float64) float64 {
	return value * float64(p.value()) / 100.0
}

func (p Percent) Hit() bool {
	return rng.Float64() <= p.GetF(1)
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
