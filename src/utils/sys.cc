/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : sys.cc
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-12:15:51:17
 * Description:
 * 
 */

#include <cstring>
#include <cstdlib>
#include <unistd.h>
#include <sched.h>

#include <utils/sys.h>
#include <absl/strings/match.h>
#include <iostream>
namespace mariana {

bool create_folders(const char *dir) {
    char order[100] = "mkdir -p ";
    strcat(order, dir);
    int ret = system(order);
    if (ret==-1) {
        return false;
    }
    return true;
}

std::string os_path_join(const std::string& a, const std::string& b) {
    std::string path_a = a;
    std::string path_b = b;
    if (absl::EndsWith(path_a, "/")) {
        path_a = a.substr(0, a.size()-1);
    }
    if (absl::StartsWith(path_b, "/")) {
        path_b = b.substr(1);
    }
    return path_a+"/"+path_b;
}

bool file_exist(const std::string& filename) {
    int ok = access(filename.c_str(), F_OK);
    if (ok == -1) return false;
    return true;
}

bool set_cpu_affinity(uint32_t cpuid, uint64_t tid) {
    cpu_set_t mask;
    CPU_ZERO(&mask);    //置空
    CPU_SET(cpuid, &mask);   // 将当前线程和CPU绑定
    if(sched_setaffinity(tid, sizeof(mask), &mask) == 0) {
        return true;
    } else {
        return false;
    }
}

int32_t current_thread_on_cpu() {
    return sched_getcpu();
}

std::string get_env(const std::string& name) {
    char *p = getenv(name.c_str());
    if (p == nullptr) {
        return std::string{};
    }
    return std::string(p);
}

} // namespace mariana
