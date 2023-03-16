package mylog

type Level uint8

const (
	_ Level = iota
	Error
	Warn
	Info
	Debug
	Trace
)

type Logger interface {
	Log(level Level, message string)
	Error(message string)
	Warn(message string)
	Info(message string)
	Debug(message string)
	Trace(message string)
}
