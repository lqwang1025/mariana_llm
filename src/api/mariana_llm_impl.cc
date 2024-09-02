/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : api/mariana_llm_impl.cc
 * Authors    : lqwang@inspur
 * Create Time: 2024-08-28:09:05:48
 * Description:
 * 
 */

#include <cmath>

#include <mariana_llm/mariana_llm_impl.h>

namespace mariana {

float Rect2D::area() const {
    float h = fabs(br.y - tl.y);
    float w = fabs(br.x - tl.x);
    return h*w;
}

float Rect2D::height() const {
    return fabs(br.y - tl.y);
}

float Rect2D::width() const {
    return fabs(br.x - tl.x);
}

Point2D Rect2D::cxy() const {
    Point2D cxy;
    cxy.x = tl.x + 0.5f*width();
    cxy.y = tl.y + 0.5f*height();
    return cxy;
}

} // namespace mariana

