#pragma once
#include <taskflow/taskflow.hpp>

#include "common/general.h"
#include "common/array.h"
#include "common/timer.h"

PHYS_NAMESPACE_BEGIN

// TODO
// 1. provide more interface: depend on what we need.

/**
 * @brief ThreadPool class provide some useful interfaces based on TaskFlow. 
 */
class ThreadPool {
  public:
    // static tf::Taskflow taskflow;
    static tf::Executor executor;

    /**
   * @brief Provide the basic interface for loop-based parallel
   *
   * @tparam F The type of the function object to be executed in parallel.
   * @param start The starting index of the loop.
   * @param end The ending index of the loop.
   * @param func The function object to be executed in parallel.
   *
   * @details This function creates a task flow and uses it to parallelize the execution of
   * the provided function object over the specified range. The number of threads used for
   * parallelization is determined by the hardware concurrency. The function blocks until
   * the execution of all tasks in the task flow is completed.
   */
    template <typename F>
    static void parallelFor(int start, int end, const F &func) {
        // Create a task flow
        tf::Taskflow taskflow;

        // Determine the number of threads to use
        size_t w = std::max(unsigned{1}, std::thread::hardware_concurrency());
        
        // Determine the grain size (number of iterations per task)
        size_t g = std::max((end - start + w - 1) / w, size_t{1});
        
        // Add parallel_for task to the task flow
        taskflow.parallel_for(start, end, 1, func, g);

        // Execute the task flow and wait for completion
        executor.run(taskflow).get();
        // taskflow.dump(std::cout);
    }

    /**
     * @brief Executes a parallel for loop using multiple threads.
     *
     * @tparam F The type of the function object to be called in each iteration.
     * @param start The starting index of the loop.
     * @param end The ending index of the loop.
     * @param workers The number of worker threads to be used.
     * @param func The function object to be called in each iteration.
     */
    template <typename F>
    static void parallelForWithThreads(int start, int end, unsigned int workers, const F &func) {
        // Create a taskflow object
        tf::Taskflow taskflow;

        // Determine the number of workers and granularity
        size_t w = std::max(unsigned{1}, workers);
        size_t g = std::max((end - start + w - 1) / w, size_t{2});
      
        // Add a parallel_for task to the taskflow
        taskflow.parallel_for(start, end, 1, func, g);
        
        // Run the taskflow using the executor and wait for completion
        executor.run(taskflow).get();
        // taskflow.dump(std::cout);
    }

    /**
     * @brief Provide an interface for loop-based parallel where the result of each iterator are stacked into the result_array.
     * 
     * @param beg is the begin index
     * @param end is the end index
     * @param result_array is the array of objects T from add_to_array_op
     * @param add_to_array_op (ObjectArray<T>& res_arr, I i) is the function runing on the index i and push the result object back to the res_arr.
     */
    template <typename I, typename T, typename G>
    static void reduceToArray(I beg, I end, ObjectArray<T>& result_array, G&& add_to_array_op) {
        // Create a taskflow object
        tf::Taskflow taskflow;

        // Calculate the size of the range
        size_t d = end-beg;

        // Determine the number of workers and group size
        size_t w = std::max(unsigned{1}, std::thread::hardware_concurrency());
        size_t g = std::max((d + w - 1) / w, size_t{2});
        
        // Create an array of result arrays for each worker
        auto g_arrays = new ObjectArray<T>[w];

        // Call the internal function to perform the parallel reduction
        reduceToArrayInternal(taskflow, beg, end, g_arrays, result_array, add_to_array_op);

        // Run the taskflow using the executor and wait for completion
        executor.run(taskflow).get();
        executor.wait_for_all();
        
        // Clean up the memory allocated for the result arrays
        delete [] g_arrays;
    }

  protected:
    /**
     * @brief Reduces a range of elements to an array using parallel tasks.
     * 
     * @param taskflow The taskflow object.
     * @param beg The beginning iterator.
     * @param end The ending iterator.
     * @param g_arrays The array of temporary result arrays.
     * @param result_array The final result array.
     * @param add_to_array_op The operation to add elements to the array.
     * @return A pair of tasks representing the source and target tasks.
     */
    template <typename I, typename T, typename G>
    static std::pair<tf::Task, tf::Task> reduceToArrayInternal(tf::Taskflow& taskflow, I beg, I end, ObjectArray<T>* g_arrays, ObjectArray<T>& result_array, G&& add_to_array_op) {
            
      size_t d = end-beg;

      auto source = taskflow.placeholder();
      auto target = taskflow.placeholder();

      // Determine the number of workers and group size
      size_t w = std::max(unsigned{1}, std::thread::hardware_concurrency());
      size_t g = std::max((d + w - 1) / w, size_t{2});

      
      size_t id = 0;
      size_t remain = d;

      //// map
      while(beg != end) {

        auto e = beg;
        
        size_t x = std::min(remain, g);
        e += x;
        remain -= x;
        
        //// create a task for indices between [beg, e)
        auto task = taskflow.emplace([beg, e, add_to_array_op, res_arr = &g_arrays[id]] () mutable {
          I i = beg;
          for(; i != e; ++i) {
            // res_arr->emplace_back(add_to_array_op(i));
            add_to_array_op(res_arr, i);
          }
        });
        source.precede(task);
        task.precede(target);

        beg = e;
        id ++;
      }

      //// reduce
      target.work([w=id, g_arrays, &result_array] () {
        for(auto i=0u; i<w; i++) {
          result_array.insert(result_array.end(), g_arrays[i].begin(), g_arrays[i].end());
        }
      });

      return std::make_pair(source, target); 
    }


};

PHYS_NAMESPACE_END