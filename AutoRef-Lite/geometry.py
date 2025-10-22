import numpy as np


def point_side_of_line(p, a, b):
    px, py = p
    ax, ay = a
    bx, by = b
    # wektorowe 2D (cross)
    cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
    if cross > 0:
        return 1
    if cross < 0:
        return -1
    return 0


def segments_intersect(p1, p2, q1, q2):
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    o1 = orient(p1, p2, q1)
    o2 = orient(p1, p2, q2)
    o3 = orient(q1, q2, p1)
    o4 = orient(q1, q2, p2)
    return (o1 * o2 < 0) and (o3 * o4 < 0)

