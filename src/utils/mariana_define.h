/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : utils/mariana_define.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-12:15:38:09
 * Description:
 *
 */

#ifndef __MAR_DEFINE_H__
#define __MAR_DEFINE_H__

#include <utils/sys.h>
#include <glog/logging.h>
#include <utils/auto_time.h>

namespace mariana {
    
class GolgLogger final {
public:
    GolgLogger() {
        create_folders(LOG_DIR);
        init();
    }
    void init() {
        google::InitGoogleLogging("mariana");
        FLAGS_log_dir = LOG_DIR;
        FLAGS_colorlogtostderr = true;
        google::SetLogFilenameExtension(".log");
        google::EnableLogCleaner(3); // Keep the log alive 3 days.
    }
    ~GolgLogger() {
        google::ShutdownGoogleLogging();
    }
private:
    const char LOG_DIR[16] = "/tmp/mar.logs";
};

} // namespace mariana

using google::WARNING;
using google::ERROR;
using google::FATAL;
using google::INFO;

#define MLOG(severity)                          \
    LOG(severity)

#define MVLOG(verboselevel)                     \
    VLOG(verboselevel)

#ifdef DEBUG
#define MCHECK(condition)                       \
    CHECK(condition)
#else
#define MCHECK(condition)
#endif

#define MLOG_IF(severity, condition)            \
    LOG_IF(severity, condition)

#define MLOG_EVERY_N(severity, n)               \
    LOG_EVERY_N(severity, n)

#define MLOG_IF_EVERY_N(severity, condition, n) \
    LOG_IF_EVERY_N(severity, condition, n)

#define MLOG_EVERY_T(severity, T)               \
    LOG_EVERY_T(severity, T)

#define MCHECK_EQ(val1, val2)                   \
    CHECK_EQ(val1, val2)
#define MCHECK_NE(val1, val2)                   \
    CHECK_NE(val1, val2)
#define MCHECK_LE(val1, val2)                   \
    CHECK_LE(val1, val2)
#define MCHECK_LT(val1, val2)                   \
    CHECK_LT(val1, val2)
#define MCHECK_GE(val1, val2)                   \
    CHECK_GE(val1, val2)
#define MCHECK_GT(val1, val2)                   \
    CHECK_GT(val1, val2)

#define MCHECK_NOTNULL(val)                     \
    CHECK_NOTNULL(val)

#define MCHECK_STREQ(s1, s2)                    \
    CHECK_STREQ(s1, s2)

#define MCHECK_STRNE(s1, s2)                    \
    CHECK_STRNE(s1, s2)

#define MCHECK_STRCASEEQ(s1, s2)                \
    CHECK_STRCASEEQ(s1, s2)

#define MCHECK_STRCASENE(s1, s2)                \
    CHECK_STRCASENE(s1, s2)
    
#define MCHECK_INDEX(I,A)                       \
    CHECK_INDEX(I,A)

#define MCHECK_BOUND(B,A)                       \
    CHECK_BOUND(B,A)

#define MCHECK_DOUBLE_EQ(val1, val2)            \
    CHECK_DOUBLE_EQ(val1, val2)

#define MCHECK_NEAR(val1, val2, margin)         \
    CHECK_NEAR(val1, val2, margin)

#define STR_IMP(x) #x
#define STR(x) STR_IMP(x)

#ifdef DEBUG
#define TRACE(msg)                                                  \
    MVLOG(3)<<"code trace: "<<__PRETTY_FUNCTION__<<" "<<STR(msg);
#else
#define TRACE(condition)
#endif

#define JSON_ARRAY_HANDLE(item, func, ctrl)     \
    if (item.IsArray()) {                       \
        func;                                   \
        ctrl;                                   \
    }

#define JSON_DOUBLE_HANDLE(item, func, ctrl)    \
    if (item.IsDouble()) {                      \
        func;                                   \
        ctrl;                                   \
    }

#define JSON_INT_HANDLE(item, func, ctrl)       \
    if (item.IsInt()) {                         \
        func;                                   \
        ctrl;                                   \
    }

#define JSON_BOOL_HANDLE(item, func, ctrl)      \
    if (item.IsBool()) {                        \
        func;                                   \
        ctrl;                                   \
    }

#define JSON_STRING_HANDLE(item, func, ctrl)    \
    if (item.IsString()) {                      \
        func;                                   \
        ctrl;                                   \
    }

#define JSON_OBJECT_HANDLE(item, func, ctrl)    \
    if (item.IsObject()) {                      \
        func;                                   \
        ctrl;                                   \
    }

#define DISABLE_COPY_AND_ASSIGN(classname)      \
    private:                                    \
    classname(const classname&);                \
    classname& operator=(const classname&)

#define TRY_ANY_CAST(reciver, any, ctrl)                                \
    try {                                                               \
        reciver = ::absl::any_cast<decltype(reciver)>(any);             \
    } catch(const absl::bad_any_cast& e) {                              \
        MLOG(ERROR)<<e.what()<<" for "<<STR(reciver);                   \
        ctrl;                                                           \
    }

#define TRY_STL(scentence, ctrl)                                        \
    try {                                                               \
        scentence;                                                      \
    } catch(const std::exception& e) {                                  \
        MLOG(ERROR)<<e.what()<<" for "<<STR(scentence);                 \
        ctrl;                                                           \
    }

#ifdef OPEN_TIME_TRACE
#define AUTOTIME(msg) mariana::AutoTime ___t(__LINE__, msg)
#else
#define AUTOTIME
#endif

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define DECLARE_SOMETHING_HOLDER(classname, identify, maker)            \
    class classname##Holder final {                                     \
public:                                                                 \
typedef std::unordered_map<identify, maker> classname##Map;             \
static classname##Map& get_##classname##Map() {                         \
    static classname##Map* func_map = new classname##Map;               \
    return *func_map;                                                   \
}                                                                       \
                                                                        \
static void add_##classname(const identify& category, maker func) {     \
    classname##Map& func_map = get_##classname##Map();                  \
    if (func_map.count(category) == 1) {                                \
        MLOG(WARNING)<<STR_IMP(classname)<<" "<<STR_IMP(category)       \
                     <<" had been registred.";                          \
        return;                                                         \
    }                                                                   \
    func_map[category] = func;                                          \
}                                                                       \
                                                                        \
static maker search(const identify& category) {                         \
    classname##Map& func_map = get_##classname##Map();                  \
    if (func_map.size() == 0 || func_map.count(category) == 0) {        \
        MLOG(ERROR)<<"There is no func in registry: "                   \
                   <<STR_IMP(classname)<<" "<<STR_IMP(category);        \
            return nullptr;                                             \
    }                                                                   \
    return func_map[category];                                          \
}                                                                       \
                                                                        \
static void release() {                                                 \
    classname##Map& func_map = get_##classname##Map();                  \
    func_map.clear();                                                   \
}                                                                       \
private:                                                                \
classname##Holder()=delete;                                             \
DISABLE_COPY_AND_ASSIGN(classname##Holder);                             \
}

#endif /* __MAR_DEFINE_H__ */

