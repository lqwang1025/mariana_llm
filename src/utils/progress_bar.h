/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : utils/progress_bar.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-09-09:12:34:11
 * Description:
 *
 */

#ifndef __UTILS_PROGRESS_BAR_H__
#define __UTILS_PROGRESS_BAR_H__

#include <iostream>
#include <string>

#include <utils/auto_time.h>

namespace mariana {

class ProgressBar {
public:
    ProgressBar(const std::string tips = "", const char finish = '#', const char unfini = '.')
        : _flags("-\\|/"),
          _finish(finish),
          _progress_str(100, unfini),
          _cur_progress(0),
          _tips(tips) {}
    void print_bar(const std::string& flag, const ushort n) {
        for (ushort i = _cur_progress; i < n; i++) {
            _progress_str[i] = _finish;
        }
        _cur_progress = n;
        std::string f, p, t;
        float time_ms = static_cast<float>(_timer.duration_in_us())/1000000.f;
        t = "\e[1;31m"+std::to_string(time_ms)+" s\e[m";
        if (n == 100) {    
            f = "\e[1;32mOK\e[m";
            p = "\e[1;32m100%\e[m";
        } else {
            f = _flags[n % 4];
            p = std::to_string(n) + '%';
        }
        std::cout << std::unitbuf
                  << '[' << _tips+f << ']'
                  << '[' << _progress_str << ']'
                  << '[' << p << "] " <<t<<'\r';
        if (n >= 100) {
            std::cout << std::endl;
        }
    }
private:
    std::string _flags;
    char _finish;
    std::string _progress_str;
    ushort _cur_progress;
    std::string _tips;
    Timer _timer;
};

} // namespace mariana

#endif /* __UTILS_PROGRESS_BAR_H__ */

