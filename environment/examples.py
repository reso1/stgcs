from environment.instance import Instance
import numpy as np

from environment.obstacle import StaticPolygon, DynamicSphere
from mrmp.interval import Interval


EMPTY2D = Instance(
    name="empty2d",
    CSpace = [
        np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
    ],
)


SIMPLE2D = Instance(
    name = "simple2d",
    CSpace = [
        np.array([[0.0, 0.0], [1.85, 0.0], [1.85, 1.85], [0.0, 1.85]]),
        np.array([[0.0, 2.15], [1.85, 2.15], [1.85, 4.0], [0.0, 4.0]]),
        np.array([[2.15, 0.0], [4.0, 0.0], [4.0, 1.85], [2.15, 1.85]]),
        np.array([[2.15, 2.15], [4.0, 2.15], [4.0, 4.0], [2.15, 4.0]]),
        np.array([[1.85, 1.6], [2.15, 1.6], [2.15, 2.4], [1.85, 2.4]]),
	],
    OStatic = [
        StaticPolygon(np.array([[1.85, 0.0], [2.15, 0.0], [2.15, 1.6], [1.85, 1.6]])),
        StaticPolygon(np.array([[1.85, 2.4], [2.15, 2.4], [2.15, 4.0], [1.85, 4.0]])),
        StaticPolygon(np.array([[0.0, 1.85], [1.85, 1.85], [1.85, 2.15], [0.0, 2.15]])),
        StaticPolygon(np.array([[2.15, 1.85], [4.0, 1.85], [4.0, 2.15], [2.15, 2.15]])),
	]
)


COMPLEX2D = Instance(
    name = "complex2d",
    CSpace = [
        np.array([[0.4, 0.0], [0.4, 5.0], [0.0, 5.0], [0.0, 0.0]]),
		np.array([[0.4, 2.4], [1.0, 2.4], [1.0, 2.6], [0.4, 2.6]]),
		np.array([[1.4, 2.2], [1.4, 4.6], [1.0, 4.6], [1.0, 2.2]]),
		np.array([[1.4, 2.2], [2.4, 2.6], [2.4, 2.8], [1.4, 2.8]]),
		np.array([[2.2, 2.8], [2.4, 2.8], [2.4, 4.6], [2.2, 4.6]]),
		np.array([[1.4, 2.2], [1.0, 2.2], [1.0, 0.0], [3.8, 0.0], [3.8, 0.2]]),
		np.array([[3.8, 4.6], [3.8, 5.0], [1.0, 5.0], [1.0, 4.6]]),
		np.array([[5.0, 0.0], [5.0, 1.2], [4.8, 1.2], [3.8, 0.2], [3.8, 0.0]]),
		np.array([[3.4, 2.6], [4.8, 1.2], [5.0, 1.2], [5.0, 2.6]]),
		np.array([[3.4, 2.6], [3.8, 2.6], [3.8, 4.6], [3.4, 4.6]]),
		np.array([[3.8, 2.8], [4.4, 2.8], [4.4, 3.0], [3.8, 3.0]]),
		np.array([[5.0, 2.8], [5.0, 5.0], [4.4, 5.0], [4.4, 2.8]]),
	],
    OStatic = [
        StaticPolygon(np.array([[3.4, 2.6], [2.4, 2.6], [1.4, 2.2], [3.8, 0.2], [4.8, 1.2]])),
		StaticPolygon(np.array([[3.4, 4.6], [2.4, 4.6], [2.4, 2.6], [3.4, 2.6]])),
		StaticPolygon(np.array([[1.4, 2.8], [2.2, 2.8], [2.2, 4.6], [1.4, 4.6]])),
		StaticPolygon(np.array([[1.0, 2.6], [1.0, 5.0], [0.4, 5.0], [0.4, 2.6]])),
		StaticPolygon(np.array([[1.0, 2.4], [1.0, 0.0], [0.4, 0.0], [0.4, 2.4]])),
		StaticPolygon(np.array([[3.8, 3.0], [3.8, 5.0], [4.4, 5.0], [4.4, 3.0]])),
		StaticPolygon(np.array([[3.8, 2.8], [3.8, 2.6], [5.0, 2.6], [5.0, 2.8]])),
	]
)


SIMPLE2D_4DynamicSPHERE = Instance(
    name = "simple2d+4dyna_spheres",
    CSpace = [
        np.array([[0.0, 0.0], [1.85, 0.0], [1.85, 1.85], [0.0, 1.85]]),
        np.array([[0.0, 2.15], [1.85, 2.15], [1.85, 4.0], [0.0, 4.0]]),
        np.array([[2.15, 0.0], [4.0, 0.0], [4.0, 1.85], [2.15, 1.85]]),
        np.array([[2.15, 2.15], [4.0, 2.15], [4.0, 4.0], [2.15, 4.0]]),
        np.array([[1.85, 1.6], [2.15, 1.6], [2.15, 2.4], [1.85, 2.4]]),
	],
    OStatic = [
        StaticPolygon(np.array([[1.85, 0.0], [2.15, 0.0], [2.15, 1.6], [1.85, 1.6]])),
        StaticPolygon(np.array([[1.85, 2.4], [2.15, 2.4], [2.15, 4.0], [1.85, 4.0]])),
        StaticPolygon(np.array([[0.0, 1.85], [1.85, 1.85], [1.85, 2.15], [0.0, 2.15]])),
        StaticPolygon(np.array([[2.15, 1.85], [4.0, 1.85], [4.0, 2.15], [2.15, 2.15]])),
	],
    ODynamic = [
        DynamicSphere(x0=np.array([1.45, 0.45]), xt=np.array([0.4, 0.45]), itvl=Interval(0, 2), radius=0.4),
        DynamicSphere(x0=np.array([0.4, 1.45]), xt=np.array([1.45, 1.45]), itvl=Interval(0, 4), radius=0.4),
        DynamicSphere(x0=np.array([2.6, 2.55]), xt=np.array([2.6, 3.6]), itvl=Interval(2, 5), radius=0.4),
        DynamicSphere(x0=np.array([3.6, 3.6]), xt=np.array([3.6, 2.55]), itvl=Interval(2, 7), radius=0.4),
    ]
)


SIMPLE2D_8DynamicSPHERE = Instance(
    name = "simple2d+8dyna_spheres",
    CSpace = [
        np.array([[0.0, 0.0], [1.85, 0.0], [1.85, 1.85], [0.0, 1.85]]),
        np.array([[0.0, 2.15], [1.85, 2.15], [1.85, 4.0], [0.0, 4.0]]),
        np.array([[2.15, 0.0], [4.0, 0.0], [4.0, 1.85], [2.15, 1.85]]),
        np.array([[2.15, 2.15], [4.0, 2.15], [4.0, 4.0], [2.15, 4.0]]),
        np.array([[1.85, 1.6], [2.15, 1.6], [2.15, 2.4], [1.85, 2.4]]),
	],
    OStatic = [
        StaticPolygon(np.array([[1.85, 0.0], [2.15, 0.0], [2.15, 1.6], [1.85, 1.6]])),
        StaticPolygon(np.array([[1.85, 2.4], [2.15, 2.4], [2.15, 4.0], [1.85, 4.0]])),
        StaticPolygon(np.array([[0.0, 1.85], [1.85, 1.85], [1.85, 2.15], [0.0, 2.15]])),
        StaticPolygon(np.array([[2.15, 1.85], [4.0, 1.85], [4.0, 2.15], [2.15, 2.15]])),
	],
    ODynamic = [
        DynamicSphere(x0=np.array([1.35, 0.5]), xt=np.array([0.5, 0.5]), itvl=Interval(0, 5), radius=0.3),
        DynamicSphere(x0=np.array([0.5, 1.35]), xt=np.array([1.35, 1.35]), itvl=Interval(0, 5), radius=0.3),
        DynamicSphere(x0=np.array([2.7, 2.45]), xt=np.array([2.7, 3.5]), itvl=Interval(0, 5), radius=0.3),
        DynamicSphere(x0=np.array([3.5, 3.5]), xt=np.array([3.5, 2.7]), itvl=Interval(0, 5), radius=0.3),
        DynamicSphere(x0=np.array([0.5, 2.7]), xt=np.array([1.35, 2.7]), itvl=Interval(0, 5), radius=0.3),
        DynamicSphere(x0=np.array([1.35, 3.5]), xt=np.array([0.5, 3.5]), itvl=Interval(0, 5), radius=0.3),
        DynamicSphere(x0=np.array([2.7, 1.35]), xt=np.array([2.7, 0.5]), itvl=Interval(0, 5), radius=0.3),
        DynamicSphere(x0=np.array([3.5, 0.5]), xt=np.array([3.5, 1.35]), itvl=Interval(0, 5), radius=0.3),
    ]
)


Complex2D_4DynamicSPHERE = Instance(
    name = "complex2d+4dyna_spheres",
        CSpace = [
        np.array([[0.4, 0.0], [0.4, 5.0], [0.0, 5.0], [0.0, 0.0]]),
		np.array([[0.4, 2.4], [1.0, 2.4], [1.0, 2.6], [0.4, 2.6]]),
		np.array([[1.4, 2.2], [1.4, 4.6], [1.0, 4.6], [1.0, 2.2]]),
		np.array([[1.4, 2.2], [2.4, 2.6], [2.4, 2.8], [1.4, 2.8]]),
		np.array([[2.2, 2.8], [2.4, 2.8], [2.4, 4.6], [2.2, 4.6]]),
		np.array([[1.4, 2.2], [1.0, 2.2], [1.0, 0.0], [3.8, 0.0], [3.8, 0.2]]),
		np.array([[3.8, 4.6], [3.8, 5.0], [1.0, 5.0], [1.0, 4.6]]),
		np.array([[5.0, 0.0], [5.0, 1.2], [4.8, 1.2], [3.8, 0.2], [3.8, 0.0]]),
		np.array([[3.4, 2.6], [4.8, 1.2], [5.0, 1.2], [5.0, 2.6]]),
		np.array([[3.4, 2.6], [3.8, 2.6], [3.8, 4.6], [3.4, 4.6]]),
		np.array([[3.8, 2.8], [4.4, 2.8], [4.4, 3.0], [3.8, 3.0]]),
		np.array([[5.0, 2.8], [5.0, 5.0], [4.4, 5.0], [4.4, 2.8]]),
	],
    OStatic = [
        StaticPolygon(np.array([[3.4, 2.6], [2.4, 2.6], [1.4, 2.2], [3.8, 0.2], [4.8, 1.2]])),
		StaticPolygon(np.array([[3.4, 4.6], [2.4, 4.6], [2.4, 2.6], [3.4, 2.6]])),
		StaticPolygon(np.array([[1.4, 2.8], [2.2, 2.8], [2.2, 4.6], [1.4, 4.6]])),
		StaticPolygon(np.array([[1.0, 2.6], [1.0, 5.0], [0.4, 5.0], [0.4, 2.6]])),
		StaticPolygon(np.array([[1.0, 2.4], [1.0, 0.0], [0.4, 0.0], [0.4, 2.4]])),
		StaticPolygon(np.array([[3.8, 3.0], [3.8, 5.0], [4.4, 5.0], [4.4, 3.0]])),
		StaticPolygon(np.array([[3.8, 2.8], [3.8, 2.6], [5.0, 2.6], [5.0, 2.8]])),
	],
    ODynamic = [
        DynamicSphere(x0=np.array([0.2, 4.5]), xt=np.array([0.2, 0.5]), itvl=Interval(0, 4), radius=0.05),
        DynamicSphere(x0=np.array([1.5, 1.5]), xt=np.array([2.5, 0.5]), itvl=Interval(2, 6), radius=0.2),
        DynamicSphere(x0=np.array([4.8, 1.7]), xt=np.array([4.0, 2.4]), itvl=Interval(3, 7), radius=0.1),
        DynamicSphere(x0=np.array([1.5, 4.8]), xt=np.array([3.5, 4.8]), itvl=Interval(2, 7), radius=0.08),
    ]
)