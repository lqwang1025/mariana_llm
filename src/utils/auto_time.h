/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : utils/auto_time.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-20:17:21:59
 * Description:
 *
 */

#ifndef __UTILS_AUTO_TIME_H__
#define __UTILS_AUTO_TIME_H__

#include <cstdint>

namespace mariana {

class Timer {
public:
    Timer();
    ~Timer();
    Timer(const Timer&)  = delete;
    Timer(const Timer&&) = delete;
    Timer& operator=(const Timer&)  = delete;
    Timer& operator=(const Timer&&) = delete;
    void reset();
    uint64_t duration_in_us();
    uint64_t current() const {
        return last_reset_time_;
    }
protected:
    uint64_t last_reset_time_;
};

class AutoTime : public Timer {
public:
    AutoTime(int line, const char* func);
    ~AutoTime();
    AutoTime(const AutoTime&)  = delete;
    AutoTime(const AutoTime&&) = delete;
    AutoTime& operator=(const AutoTime&) = delete;
    AutoTime& operator=(const AutoTime&&) = delete;
private:
    uint32_t line_;
    char* name_;
};

} // namespace mariana

#endif /* __UTILS_AUTO_TIME_H__ */

