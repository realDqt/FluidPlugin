#pragma once
#include <iostream>
#include <fstream>
#include<string>
#include <map>
#include "common/real.h"
#define GET_PARAM(c,p) c.getParam(#p,p)

// Config class for handling configuration parameters
class Config
{
private:
    // Map to store configuration parameter key-value pairs
    std::map<std::string, std::string> paramMap; 

    /**
     * @brief Private helper function to check if a parameter exists.
     */
    bool checkParam(std::string);

public:
    /**
     * @brief Constructor that takes a configuration file path as input.
     */
    Config(std::string configFile);

    /**
     * @brief Function to retrieve an unsigned integer parameter from the configuration.
     */
    void getParam(std::string, unsigned int&);

    /**
     * @brief Function to retrieve an integer parameter from the configuration.
     */
    void getParam(std::string, int&);

    /**
     * @brief Function to retrieve a floating-point parameter from the configuration.
     */
    void getParam(std::string, float&);

    /**
     * @brief Function to retrieve a boolean parameter from the configuration.
     */
    void getParam(std::string, bool&);

    /**
     * @brief Function to retrieve a float3 (a custom type) parameter from the configuration.
     */
    void getParam(std::string, float3&);

    /**
     * @brief Destructor to clean up resources.
     */
    ~Config();
};
