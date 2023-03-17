// Package mylog provides functionality for interface Logger and implementation of it using log.Logger from stdlib
package mylog

// Level represents logging level: Error, Warn, Info, Debug, Trace. Each next level includes previous level.
type Level uint8

const (
	_ Level = iota
	Error
	Warn
	Info
	Debug
	Trace
)

// Logger provides interface to wrap an log implementation
type Logger interface {
	// IsEnabled returns true if specified level is enabled by logger configuration
	IsEnabled(level Level) bool

	// Log logs message with given level
	Log(level Level, message string)

	// Logf is similar to Log, but message is being formatted by fmt.Sprintf
	Logf(level Level, format string, args ...interface{})

	// CatchErr logs err with Error if *err is not nil
	CatchErr(err *error)

	// Error logs message with Error Level
	Error(message string)

	// Warn logs message with Warn Level
	Warn(message string)

	// Info logs message with Info Level
	Info(message string)

	// Debug logs message with Debug Level
	Debug(message string)

	// Trace logs message with Trace Level
	Trace(message string)

	// Errorf is similar to Error, but message is being formatted by fmt.Sprintf
	Errorf(format string, args ...interface{})

	// Warnf is similar to Warn, but message is being formatted by fmt.Sprintf
	Warnf(format string, args ...interface{})

	// Infof is similar to Info, but message is being formatted by fmt.Sprintf
	Infof(format string, args ...interface{})

	// Debugf is similar to Debug, but message is being formatted by fmt.Sprintf
	Debugf(format string, args ...interface{})

	// Tracef is similar to Trace, but message is being formatted by fmt.Sprintf
	Tracef(format string, args ...interface{})
}
