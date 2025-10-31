#pragma once

#include <string>
#include <iostream>
#include <fstream>
#include <map>

#define LOG_THRES LogType::Debug

//// filename
#ifdef __APPLE__
#define __FILENAME__ (strrchr(__FILE__, '/') + 1)
#else
#define __FILENAME__ (strrchr(__FILE__, '\\') + 1)
#endif

/**
@brief A description of the function or class.

@param paramName Description of the parameter.
@return Description of the return value.
*/
char* getTimeString();
/**
@brief Enumeration describing the log types.

The log types include Debug, Info, Warn, Error, and Fatal.
*/
enum class LogType2{
    Debug=0,
    Info,
    Warn,
    Error,
    Fatal
};
/**
@brief Array of log colors.

The array stores log colors for different log types.
*/
extern std::string LOG_COLORS[5];
/**
@brief A string representing the color reset.

The string represents the color reset code.
*/
extern std::string LOG_COLOR_RESET;


/**
 * @brief This macro prints log messages to the ostream based on the log type
 *
 * The LOG_OSTREAM macro is used to print log messages to the ostream based on the log type. 
 * It takes the log type as an argument and checks if the log type is below the log threshold. 
 * If the log type is below the threshold, nothing is printed. 
 * Otherwise, the log message is printed to the ostream, including the current time, filename, and line number. 
 * The log message is printed in the corresponding color based on the log type.
 *
 * @param lt The log type
 */
#define LOG_OSTREAM(lt) \
    if constexpr (lt < LOG_THRES) ; \
    else std::cout<<LOG_COLORS[(int)lt] << '[' << getTimeString() << "][" << __FILENAME__ << ":" << __LINE__ << "] " << LOG_COLOR_RESET
    
#define LOG_OSTREAM_DEBUG LOG_OSTREAM(LogType::Debug)
#define LOG_OSTREAM_INFO LOG_OSTREAM(LogType::Info)
#define LOG_OSTREAM_WARN LOG_OSTREAM(LogType::Warn)
#define LOG_OSTREAM_ERROR LOG_OSTREAM(LogType::Error)

/**
 * @brief This macro prints log messages to the standard output using printf
 *
 * The LOG_PRINT macro is used to print log messages to the standard output using printf. 
 * It takes the log type, format string, and variable arguments as arguments. 
 * If the log type is below the log threshold, nothing is printed. 
 * Otherwise, the log message is printed to the standard output, including the log type, current time, filename, line number, and the formatted log message. 
 * The log message is printed in the corresponding color based on the log type.
 *
 * @param lt The log type
 * @param format The format string for the log message
 * @param ... The variable arguments for the format string
 */
#define LOG_PRINT(lt, format, ...) \
    if(lt < LOG_THRES) ; \
    else { printf("%s[%s][%s:%d]%s "  format, LOG_COLORS[(int)lt], getTimeString(), __FILENAME__, __LINE__, LOG_COLOR_RESET, ##__VA_ARGS__); }
    
/**
@brief Prints a log message with the Debug log type.

@param format The format string for the log message.
@param ... The variable arguments for the format string.
*/
#define LOG_PRINT_DEBUG(format, ...) LOG_PRINT(LogType::Debug)

/**
@brief Prints a log message with the Info log type.

@param format The format string for the log message.
@param ... The variable arguments for the format string.
*/
#define LOG_PRINT_INFO(format, ...) LOG_PRINT(LogType::Info)

/**
@brief Prints a log message with the Warn log type.

@param format The format string for the log message.
@param ... The variable arguments for the format string.
*/
#define LOG_PRINT_WARN(format, ...) LOG_PRINT(LogType::Warn)

/**
@brief Prints a log message with the Error log type.

@param format The format string for the log message.
@param ... The variable arguments for the format string.
*/
#define LOG_PRINT_ERROR(format, ...) LOG_PRINT(LogType::Error)

/**
 * @brief The Logger class is responsible for logging messages to a file.
 *
 * The Logger class provides functionality to log messages of different log types to a file.
 * It allows specifying the log type (such as Debug, Info, Warn, or Error) and the message to be logged.
 * The Logger class also supports setting a maximum file size for the log file.
 */
class Logger {
public:
    /**
    @brief Constructs a Logger object.

    @param logFileName The name of the log file.
    @param maxFileSize The maximum size of the log file.
    */
    Logger(const std::string& logFileName, std::size_t maxFileSize);

    /**
    @brief Destroys the Logger object.
    */
    ~Logger();

    /**
    @brief Logs a message with a specified log type.

    @param type The log type.
    @param message The log message.
    */
    void Log(const LogType2& type, const std::string& message);

private:
    std::string logFileName; /**< The name of the log file. */
    std::ofstream logFile; /**< The log file stream. */
    std::size_t maxFileSize; /**< The maximum size of the log file. */
    std::size_t currentFileSize; /**< The current size of the log file. */

    std::map<LogType2, std::string> logTypeToString = {
        {LogType2::Error, "Error"},
        {LogType2::Warn, "Waring"},
        {LogType2::Info, "Info"}
    };

    /**
    @brief Opens the log file.
    */
    void OpenLogFile();

    /**
    @brief Gets the size of a file.

    @param fileName The name of the file.
    @return The size of the file.
    */
    std::size_t GetFileSize(const std::string& fileName);

    /**
    @brief Checks the size of the log file and rotates it if necessary.
    */
    void CheckFileSize();

    /**
    @brief Rotates the log file by renaming it with a new name.
    */
    void RotateLogFile();

    /**
    @brief Generates a new file name for the rotated log file.

    @return The new file name.
    */
    std::string GenerateNewFileName();

    /**
    @brief Logs a message with a specified log type.

    @param type The log type.
    @param message The log message.
    */
    void LogMessage(const LogType2& type, const std::string& message);
};

extern Logger logger;