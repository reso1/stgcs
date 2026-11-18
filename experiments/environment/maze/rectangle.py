from __future__ import annotations
from typing import List

import numpy as np
from experiments.environment.interval import Interval, AABB

HORIZONTAL = 0 
VERTICAL   = 1
ARBITRARY  = 2


class Rectangle:

    def __init__(self, lowerleft: tuple, upperright: tuple) -> None:
        ll = (lowerleft[0], lowerleft[1])
        ur = (upperright[0], upperright[1])
        self.lowerleft = ll
        self.upperright = ur
        self.width = ur[0] - ll[0]
        self.height = ur[1] - ll[1]
        self.lowerright = (ur[0], ll[1])
        self.upperleft = (ll[0], ur[1])

        self.top_rect: Rectangle = None
        self.bot_rect: Rectangle = None
        self.left_rect: Rectangle = None
        self.right_rect: Rectangle = None

    def __hash__(self) -> int:
        return (*self.lowerleft, self.width, self.height).__hash__()

    @property
    def local_optimal_orientation(self) -> int:
        onehot = tuple([1 if x else 0 for x in [self.top_rect, self.left_rect, self.bot_rect, self.right_rect]])
        a_case, f_case = (0, 0, 0, 0), (1, 1, 1, 1)
        c_case = [tuple(np.roll([1, 1, 0, 0], i)) for i in range(4)]
        for case in [a_case, f_case] + c_case:
            if onehot == case:
                return ARBITRARY
        
        if onehot == (1, 0, 0, 0) or onehot == (0, 0, 1, 0) or onehot == (1, 0, 1, 0) or \
           onehot == (1, 1, 1, 0) or onehot == (1, 0, 1, 1):
            return VERTICAL
        
        if onehot == (0, 1, 0, 0) or onehot == (0, 0, 0, 1) or onehot == (0, 1, 0, 1) or \
           onehot == (1, 1, 0, 1) or onehot == (0, 1, 1, 1):
            return HORIZONTAL

    @property
    def neighbor_rects(self) -> List[Rectangle]:
        return [x for x in [self.top_rect, self.left_rect, self.bot_rect, self.right_rect] if x]

    @property
    def area(self) -> float:
        return self.width * self.height

    def boustrophedon_path(self, interval:float, alt=1) -> list:
        ret = []
        if self.width > self.height:
            x_min, x_max = self.lowerleft[0]+interval/2, self.upperright[0]-interval/2
            for y in np.arange(self.lowerleft[1]+interval/2, self.upperright[1], interval):
                ret.extend([(x_min, y), (x_max, y)] if alt else [(x_max, y), (x_min, y)])
                alt = not alt
        else:
            y_min, y_max = self.lowerleft[1]+interval/2, self.upperright[1]-interval/2
            for x in np.arange(self.lowerleft[0]+interval/2, self.upperright[0], interval):
                ret.extend([(x, y_min), (x, y_max)] if alt else [(x, y_max), (x, y_min)])
                alt = not alt
        return ret

    def copy(self) -> Rectangle:
        return Rectangle(self.lowerleft, self.upperright)
    
    def merge(self, other:Rectangle) -> Rectangle|None:
        ret = None
        if self.lowerleft[0] == other.lowerleft[0] and \
           self.upperright[0] == other.upperright[0]:
            if self.lowerleft[1] + self.height == other.lowerleft[1]:
                ret = Rectangle(self.lowerleft, other.upperright)
            elif self.lowerleft[1] == other.lowerleft[1] + self.height:
                ret = Rectangle(other.lowerleft, self.upperright)

        elif self.lowerleft[1] == other.lowerleft[1] and \
           self.upperright[1] == other.upperright[1]:
            if self.lowerleft[0] + self.width == other.lowerleft[0]:
                ret = Rectangle(self.lowerleft, other.upperright)
            elif self.lowerleft[0] == other.lowerleft[0] + self.width:
                ret = Rectangle(other.lowerleft, self.upperright)
        
        return ret

    @staticmethod
    def merge_all(rects:List[Rectangle]) -> Rectangle:
        # check if all rectangles are vertically aligned
        if all([rects[0].lowerleft[0] == r.lowerleft[0] and rects[0].upperright[0] == r.upperright[0] 
                for r in rects[1:]]):
            sorted_rects = sorted(rects, key=lambda r: r.lowerleft[1])
            return Rectangle(
                (sorted_rects[0].lowerleft[0], sorted_rects[0].lowerleft[1]),
                (sorted_rects[0].upperright[0], sorted_rects[-1].upperright[1])
            )
        # check if all rectangles are horizontally aligned
        if all([rects[0].lowerleft[1] == r.lowerleft[1] and rects[0].upperright[1] == r.upperright[1] 
                for r in rects[1:]]):
            sorted_rects = sorted(rects, key=lambda r: r.lowerleft[0])
            return Rectangle(
                (sorted_rects[0].lowerleft[0], sorted_rects[0].lowerleft[1]),
                (sorted_rects[-1].upperright[0], sorted_rects[0].upperright[1])
            )

        raise ValueError("Rectangles are not aligned for merging.")

    @property
    def center(self) -> tuple:
        return ((self.lowerleft[0]+self.upperright[0])/2, 
                (self.lowerleft[1]+self.upperright[1])/2)

    def grids(self) -> List[Rectangle]:
        ret = []
        for x in np.arange(self.lowerleft[0], self.upperright[0]):
            for y in np.arange(self.lowerleft[1], self.upperright[1]):
                ret.append(Rectangle((x, y), (x+1, y+1)))
        return ret

    def intersects(self, other:Rectangle) -> bool:
        itvls = [
            Interval(self.lowerleft[0], self.upperright[0]),
            Interval(self.lowerleft[1], self.upperright[1])
        ]
        other_itvls = [
            Interval(other.lowerleft[0], other.upperright[0]),
            Interval(other.lowerleft[1], other.upperright[1])
        ]
        return AABB(itvls, other_itvls)

    def contains(self, other:Rectangle) -> bool:
        return self.lowerleft[0] <= other.lowerleft[0] and \
               self.lowerleft[1] <= other.lowerleft[1] and \
               self.upperright[0] >= other.upperright[0] and \
               self.upperright[1] >= other.upperright[1]

    def point_in_rect(self, point:tuple) -> bool:
        return self.lowerleft[0] <= point[0] <= self.upperright[0] and \
               self.lowerleft[1] <= point[1] <= self.upperright[1]

    @property
    def ply_ndarray(self) -> np.ndarray:
        return np.array([
            [self.lowerleft[0], self.lowerleft[1]],
            [self.upperright[0], self.lowerleft[1]],
            [self.upperright[0], self.upperright[1]],
            [self.lowerleft[0], self.upperright[1]],
        ])
