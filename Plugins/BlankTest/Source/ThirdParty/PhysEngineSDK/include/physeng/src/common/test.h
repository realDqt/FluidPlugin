#pragma once

#include "math/math.h"
#include "common/logger.h"


ATTRIBUTE_ALIGNED16(class)
TestCase{
public:

    virtual std::string getName(){ return "DefaultCase"; }

    virtual bool judge(){ return true;}

    virtual void failCallback(){}

    /**
     * @brief Runs a test case and checks if it passes or fails.
     * 
     * @details This function logs the beginning of the test case, runs the `judge()` function
     * to determine if the test case passes or fails, and logs the result. If the test
     * case fails, it calls the `failCallback()` function and exits the program with
     * a status code of -1.
     */
    virtual void test(){
        // Log the beginning of the test case
        LOG_OSTREAM_INFO << "TestCase " << getName() << " begin:" << std::endl; 
        
        // Check if the test case passes or fails
        if(judge()) {
            // Log the pass result
            LOG_OSTREAM_INFO << "==== pass ====" << std::endl; 
        } else {
            // Log the failed result
            LOG_OSTREAM_INFO << "==== failed ====" << std::endl; 

            // Call the failCallback() function
            failCallback(); 
            
            // Exit the program with status code -1
            exit(-1);
        }
    }
};
