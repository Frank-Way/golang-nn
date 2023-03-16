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
	Logf(level Level, format string, args ...interface{})
	CatchErr(err *error)
	Error(message string)
	Warn(message string)
	Info(message string)
	Debug(message string)
	Trace(message string)
	Errorf(format string, args ...interface{})
	Warnf(format string, args ...interface{})
	Infof(format string, args ...interface{})
	Debugf(format string, args ...interface{})
	Tracef(format string, args ...interface{})
}
