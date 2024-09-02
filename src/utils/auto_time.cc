/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : utils/auto_time.cc
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-20:17:22:13
 * Description:
 * 
 */

#include <cstdlib>
#include <cstring>

#include <sys/time.h>
#include <utils/auto_time.h>
#include <utils/mariana_define.h>

namespace mariana {

Timer::Timer() {
    reset();
}

Timer::~Timer() {
    // do nothing
}

void Timer::reset() {
    struct timeval current;
    gettimeofday(&current, nullptr);
    last_reset_time_ = current.tv_sec * 1000000 + current.tv_usec;
}

uint64_t Timer::duration_in_us() {
    struct timeval current;
    gettimeofday(&current, nullptr);
    auto last_time = current.tv_sec * 1000000 + current.tv_usec;
    return last_time - last_reset_time_;   
}

AutoTime::AutoTime(int line, const char* func) : Timer() {
    name_ = ::strdup(func);
    line_ = line;
}

AutoTime::~AutoTime() {
    auto timeInUs = duration_in_us();
    MLOG(INFO)<<name_<<" cost time:"<<(float)timeInUs / 1000.0f<<" ms";
    free(name_);
}

} // namespace mariana
