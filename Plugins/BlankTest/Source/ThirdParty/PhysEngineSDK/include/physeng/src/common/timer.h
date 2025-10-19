#pragma once

#include <chrono>
#include <list>
#include <map>
#include <memory>
#include <string>

#include "general.h"
#include "logger.h"


#ifdef PE_USE_CUDA
#include <helper_timer.h>
#endif

class Timer {
  public:
    Timer(){};
    virtual ~Timer(){};
    virtual void tik(std::ostream& os, std::string) = 0;
    virtual double tok(std::ostream& os, std::string) = 0;
};

#ifdef PE_USE_OMP
#include <omp.h>

class OmpTimer : public Timer {
  public:
    OmpTimer(){};
    virtual ~OmpTimer(){};

    /**
     * @brief Start a timer with the given label.
     *
     * @param os The output stream to write the timer start message to.
     * @param label The label for the timer.
     */
    virtual void tik(std::ostream& os, std::string label) {
        // Find the timer in the map
        auto iter = m_map.find(label);

        // If the timer already exists, update its start time
        if (iter != m_map.end()) {
            iter->second = (double)omp_get_wtime();
            // LOG_OSTREAM_INFO << "Timer ["<< label << "] start" << std::endl;
            os << "[" << label << "] TIK" << std::endl;
        } 
        // If the timer doesn't exist, create a new one
        else {
            double ts = (double)omp_get_wtime();
            m_map.insert(std::pair<std::string, double>(label, ts));
            // LOG_OSTREAM_INFO << "Timer ["<< label << "] start" << std ::endl;
            os << "[" << label << "] TIK" << std ::endl;
        }
    };

    /**
     * @brief Calculates the time elapsed since the start of the timer with the given label.
     * 
     * @param os The output stream to write the timer end message to.
     * @param label The label for the timer.
     * @return The time elapsed in seconds, or -1 if the timer with the given label is not found.
     */
    virtual double tok(std::ostream& os, std::string label) {
        // Find the timer in the map
        auto iter = m_map.find(label);

        // If the timer is found, update the timestamp
        if (iter != m_map.end()) {
            double ts_old = iter->second;
            iter->second = (double)omp_get_wtime();
            // LOG_OSTREAM_INFO << "Timer ["<< label << "] end interval = " <<
            // std::showpoint << iter->second - ts_old << "s" << std::endl;
            os << "[" << label << "] TOK Dt = " << std::showpoint << iter->second - ts_old << "s" << std::endl;
            return iter->second - ts_old;
        } 
        // If the timer is not found, write a log message and return -1
        else {
            // LOG_OSTREAM_WARN << "Timer ["<< label << "] not found" <<
            // std::endl;
            os << "[" << label << "] not found" << std::endl;
            return -1;
        }
    };
    virtual void clear() { m_map.clear(); }

  protected:
    std::map<std::string, double> m_map;
};

#endif

class BasicTimer : public Timer {
  public:
    BasicTimer(){};
    virtual ~BasicTimer(){};
    /**
     * @brief Updates the start time of a timer identified by its label.
     *
     * @param os The output stream to write the log message to.
     * @param label The label of the timer.
     */
    virtual void tik(std::ostream& os, std::string label) override {
        // Find the timer in the map
        auto iter = m_map.find(label);

        // If the timer already exists, update its start time
        if (iter != m_map.end()) {
            iter->second = std::chrono::steady_clock::now();
            // LOG_OSTREAM_INFO << "Timer ["<< label << "] start" << std::endl;
            os << "[" << label << "] TIK" << std::endl;
        } 
        // If the timer doesn't exist, create a new one
        else {
            auto ts = std::chrono::steady_clock::now();
            m_map.insert(std::make_pair(label, ts));
            // LOG_OSTREAM_INFO << "Timer ["<< label << "] start" << std ::endl;
            os << "[" << label << "] TIK" << std ::endl;
        }
    }

    /**
     * @brief Updates the timestamp for a given label in the map and calculates the time interval since the last update.
     *
     * @param os The output stream to write the log messages to.
     * @param label The label to identify the timer.
     * @return The time interval in seconds, or -1 if the label is not found.
     */
    virtual double tok(std::ostream& os, std::string label) override {
        // Find the timer in the map
        auto iter = m_map.find(label);

        // If the timer is found, update the timestamp
        if (iter != m_map.end()) {
            auto ts_old = iter->second;
            iter->second = std::chrono::steady_clock::now();
            // LOG_OSTREAM_INFO << "Timer ["<< label << "] end interval = " <<
            // std::showpoint << iter->second - ts_old << "s" << std::endl;
            auto duration_in_seconds = (iter->second - ts_old).count() / 1000000000.0;
            os << "[" << label << "] TOK Dt = " << std::showpoint << duration_in_seconds << "s" << std::endl;
            return duration_in_seconds;
        } 
        // If the timer is not found, write a log message and return -1
        else {
            // LOG_OSTREAM_WARN << "Timer ["<< label << "] not found" <<
            // std::endl;
            os << "[" << label << "] not found" << std::endl;
            return -1;
        }
    }
    virtual void clear() { m_map.clear(); }

  protected:
    std::map<std::string, std::chrono::time_point<std::chrono::steady_clock>> m_map;
};

// ---------- Benchmark Timer -------------- //

class BenchmarkTimer {
  private:
    struct _Timer {
        bool running;
        double time;
        int invocations;

        #ifdef PE_USE_CUDA
        StopWatchInterface *timer;
        _Timer() : running(false), time(0), invocations(0) { sdkCreateTimer(&timer); start(); }
        
        /**
         * @brief Calculates the duration of the timer in seconds.
         *
         * @return The duration of the timer in seconds.
         */
        // double duration() { return timer.(std::chrono::steady_clock::now() - startTime).count() / 1000000000.0; }
        double duration() { return sdkGetTimerValue(&timer) / 1000.0; }
        
        /**
         * @brief Gets the elapsed time, even if the timer is currently running.
         *
         * @return The elapsed time.
         */
        double elapsed() { return running ? time + duration() : time; }
        
        /**
         * @brief Stops the timer and updates the time duration.
         */
        void stop() {
            // Ensure that the timer is currently running
            assert(running);
            // Stop the timer
            sdkStopTimer(&timer);
            // Update the time duration
            time += duration();
            // Set running flag to false
            running = false;
            // Reset the timer for future use
            sdkResetTimer(&timer);
        }

        /**
         * @brief Starts the timer.
         */
        void start() {
            if (running) {
                // Print error message
                std::cerr << "ERROR: timer already running. Reported timings will be inaccurate." << std::endl;
                // Stop the timer
                stop();
            }
            // Assert that the timer is not already running
            assert(!running);
            // Set the running flag to true
            running = true;
            // Increment the invocations counter
            ++invocations;
            // Start the timer
            // startTime = std::chrono::steady_clock::now();
            sdkStartTimer(&timer);
        }

        #else
        std::chrono::time_point<std::chrono::steady_clock> startTime;
        _Timer() : running(false), time(0), invocations(0) { start(); }
        
        /**
         * @brief Calculates the duration of the timer in seconds.
         *
         * @return The duration of the timer in seconds.
         */
        double duration() { return (std::chrono::steady_clock::now() - startTime).count() / 1000000000.0; }
        
        /**
         * @brief Gets the elapsed time, even if the timer is currently running.
         *
         * @return The elapsed time.
         */
        double elapsed() { return running ? time + duration() : time; }
        
        /**
         * @brief Stops the timer and updates the time duration.
         */
        void stop() {
            // Ensure that the timer is currently running
            assert(running);
            // Update the time duration
            time += duration();
            // Set running flag to false
            running = false;
        }

        /**
         * @brief Starts the timer.
         */
        void start() {
            if (running) {
                // Print error message
                std::cerr << "ERROR: timer already running. Reported timings will be inaccurate." << std::endl;
                // Stop the timer
                stop();
            }
            // Assert that the timer is not already running
            assert(!running);
            // Set the running flag to true
            running = true;
            // Increment the invocations counter
            ++invocations;
            // Start the timer
            startTime = std::chrono::steady_clock::now();
        }
        #endif
    };

    typedef std::map<std::string, _Timer> TimerMap;

    struct _Section : public _Timer {
        TimerMap timers;
        _Section() : _Timer() {}

        /**
         * @brief Starts a timer with the specified name.
         * 
         * @param name The name of the timer.
         */
        void startTimer(std::string name) {
            auto it = timers.find(name);
            // Check if the timer already exists
            if (it != timers.end()) {
                // Check if the timer is already running
                if (it->second.running) {
                    std::cerr << "ERROR: timer " << name << " already started. " << std::endl;
                }
                // Start the timer
                it->second.start();
            } else {
                // Create a new timer
                timers[name] = _Timer();
            }
        }

        using _Timer::duration;
        using _Timer::start;

        /**
         * @brief Stops the timer and all sub-timers.
         */
        void stop() {
            _Timer::stop();

            // Stop all sub-timers
            for (auto& entry : timers) {
                if (entry.second.running) {
                    // Print a warning message
                    std::cerr << "WARNING: stopping timer " << entry.first
                              << " implicitly in enclosing section's stop()" << std::endl;
                    // Stop the sub-timer
                    entry.second.stop();
                }
            }
        }

        /**
         * @brief Starts a timer with the given name.
         * 
         * @param name The name of the timer.
         */
        void start(const std::string& name) {
            // Find the timer with the given name or the position where it should be inserted
            auto lb = timers.lower_bound(name);

            // If the timer does not exist, insert a new timer with the given name and start it
            if ((lb == timers.end()) || (lb->first != name))
                timers.emplace_hint(lb, name, _Timer());
            // If the timer already exists, start it
            else
                lb->second.start();  // The full section timer must be started too...
        }

        /**
         * @brief Stops the given timer and all sub-timers.
         */
        void stop(const std::string& name) { timers.at(name).stop(); }

        /**
         * @brief Reports the timers to the specified output stream.
         * 
         * @param os The output stream to write the report to.
         */
        void report(std::ostream& os) {
            for (auto& entry : timers)
                os << displayName(entry.first) << '\t' << entry.second.elapsed() << '\t' << entry.second.invocations
                   << std::endl;
        }
    };

    typedef std::map<std::string, _Section> SectionMap;
    typedef SectionMap::iterator SectionIterator;
    typedef SectionMap::const_iterator SectionConstIterator;

    SectionMap m_sections;
    std::list<std::string> m_sectionStack;

    /**
     * @brief Generates a display name for the given name.
     * 
     * @param name The name to generate the display name for.
     * @return The display name.
     */
    static std::string displayName(std::string name) {
        // Counter for the number of levels in the name
        size_t levels = 0;

        // Increment the counter for each ':' character found
        for (char c : name)
            if (c == ':') ++levels;

        // If there are no ':' characters in the name, prepend "-> " to the name and return it
        if (levels == 0) return "-> " + name;

        // Create a string with spaces to represent the indentation level based on the number of levels
        std::string result(4 * levels, ' ');

        // Append "|-> " to the result string
        result += "|-> ";

        // Append the substring of the name after the last ':' character to the result string
        result.append(name, name.rfind(':') + 1, std::string::npos);

        return result;
    }

  public:
    BenchmarkTimer() { reset(); }

    /**
     * @brief Starts a new section with the given name.
     * 
     * @param name The name of the section.
     * @param verbose Whether to print verbose output.
     */
    void startSection(const std::string& name, bool verbose = false) {
        // Generate the label for the section
        std::string label;
        if (!m_sectionStack.empty())
            label = m_sectionStack.back() + ':' + name;
        else
            label = name;

        // Add the label to the section stack
        m_sectionStack.push_back(label);

        // Find or insert the section in the map
        auto lb = m_sections.lower_bound(label);
        if ((lb == m_sections.end()) || (lb->first != label))
            m_sections.emplace_hint(lb, label, _Section());
        else
            lb->second.start();  // The full section timer must be started too...

        // Print verbose output if enabled
        if (verbose) std::cout << "[" << name << "] TIK" << std ::endl;
    }

    /**
     * @brief Stops a section with the given name.
     * 
     * @param name The name of the section.
     * @param verbose Whether to print verbose output.
     */
    void stopSection(const std::string& name, bool verbose = false) {
        // Ensure that the section stack is not empty
        assert(!m_sectionStack.empty());
        
        // Get the current label from the top of the section stack
        std::string currentLabel = m_sectionStack.back();
        
        // Remove the current label from the section stack
        m_sectionStack.pop_back();

        // Generate the label for the section to be stopped
        std::string label;
        if (!m_sectionStack.empty())
            label = m_sectionStack.back() + ':' + name;
        else
            label = name;

        // Check if the label matches the expected current label
        if (label != currentLabel) {
            std::cerr << "ERROR: sections must be stopped in the reverse of "
                         "the order they were started."
                      << std::endl;
            std::cerr << "(Expected " << currentLabel << ", but got " << label << ")" << std::endl;
        }

        // Print verbose output if requested
        if (verbose)
            std::cout << "[" << name << "] TOK Dt = " << std::showpoint << m_sections.at(label).duration() << "s"
                      << std::endl;
        
        // Stop the section
        m_sections.at(label).stop();
    }

    /**
     * @brief Start a timer for a specific section.
     * 
     * @param sectionName The name of the section.
     * @param timer The name of the timer.
     */
    void start(std::string timer) {
        std::string sectionName;

        // Check if there is a section in the stack
        if (!m_sectionStack.empty()) {
            sectionName = m_sectionStack.back();
            timer = sectionName + ':' + timer;
        }

        // Start the timer for the specified section and timer
        m_sections.at(sectionName).start(timer);
    }

    /**
     * @brief Stop a timer.
     * 
     * @param timer The name of the timer.
     */
    void stop(std::string timer) {
        std::string sectionName;

        // Check if there is a section in the stack
        if (!m_sectionStack.empty()) {
            sectionName = m_sectionStack.back();
            timer = sectionName + ':' + timer;
        }

        // Stop the timer for the specified section and timer
        m_sections.at(sectionName).stop(timer);
    }

    /**
     * @brief Remove all timers and restart the global section.
     */
    void reset() {
        // Clear the section map
        m_sections.clear();

        // Clear the section stack
        m_sectionStack.clear();

        // Restart the global section
        m_sections.insert(std::make_pair("", _Section()));
    }

    /**
     * @brief Reports the timers to the specified output stream.
     * 
     * @param os The output stream to write the report to.
     */
    void report(std::ostream& os) {
        // Iterate over the sections
        for (SectionIterator it = m_sections.begin(); it != m_sections.end(); ++it) {
            // Skip the global section, as it is reported at the end
            if (it->first != "") {  
                // Print the display name, elapsed time, and invocations for the section
                os << displayName(it->first) << "\t" << it->second.elapsed() << "\t" << it->second.invocations
                   << std::endl;

                // Recursively report the sub-timers in the section
                it->second.report(os);
            }
        }
        // TODO add percentage

        // Report the global section
        m_sections.at("").report(os);

        // Print the elapsed time for the global section
        os << "Full time\t" << m_sections.at("").elapsed() << std::endl;
        os << "========" << std::endl;
    }
};

extern std::unique_ptr<Timer> g_timer;

#ifdef GLOBALBENCHMARK
extern std::unique_ptr<BenchmarkTimer> g_benchmarkTimer;
#define BENCHMARK_START_TIMER_SECTION(label)         \
    {                                                \
        LOG_OSTREAM_DEBUG;                           \
        g_benchmarkTimer->startSection(label, true); \
    }
#define BENCHMARK_STOP_TIMER_SECTION(label)         \
    {                                               \
        LOG_OSTREAM_DEBUG;                          \
        g_benchmarkTimer->stopSection(label, true); \
    }
inline void BENCHMARK_RESET() { g_benchmarkTimer->reset(); }
inline void BENCHMARK_REPORT() { g_benchmarkTimer->report(std::cout); }
#else
#define BENCHMARK_START_TIMER_SECTION(label) ;
#define BENCHMARK_STOP_TIMER_SECTION(label) ;
inline void BENCHMARK_RESET() {}
inline void BENCHMARK_REPORT() {}
#endif

#define TIMER_TIK(label)                \
    {                                   \
        LOG_OSTREAM_DEBUG;              \
        g_timer->tik(std::cout, label); \
    }
#define TIMER_TOK(label)                \
    {                                   \
        LOG_OSTREAM_DEBUG;              \
        g_timer->tok(std::cout, label); \
    }
#define TIMER_CLEAR() g_timer->clear()

struct ScopedBenchmarkTimer {
    ScopedBenchmarkTimer(const std::string& name) : m_name(name) {
        g_benchmarkTimer->startSection(name, false);
    }

    ~ScopedBenchmarkTimer() { g_benchmarkTimer->stopSection(m_name, false); }

  private:
    std::string m_name;
};

#ifdef GLOBALBENCHMARK
#define PHY_PROFILE(label) ScopedBenchmarkTimer __profile(label)
#else
#define PHY_PROFILE(label)
#endif