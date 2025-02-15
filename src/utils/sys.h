/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : utils/sys.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-12:15:51:13
 * Description:
 *
 */

#ifndef __SYS_H__
#define __SYS_H__

#include <string>
#include <cstdint>

namespace mariana {

bool create_folders(const char *dir);

std::string os_path_join(const std::string& a, const std::string& b);

bool file_exist(const std::string& filename);

bool set_cpu_affinity(uint32_t cpuid, uint64_t tid=0);

// returns the number of the CPU on which the calling
// thread is currently executing.
int32_t current_thread_on_cpu();

std::string get_env(const std::string& name);

} // namespace mariana

#endif /* __SYS_H__ */

