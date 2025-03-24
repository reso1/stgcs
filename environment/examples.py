from environment.env import Env
import numpy as np

from environment.obstacle import StaticPolygon, DynamicSphere, ConcatDynamicSphere
from mrmp.interval import Interval


EMPTY1D = Env(
    name="empty1d",
    CSpace=[np.array([[0.0], [5.0]])],
    robot_radius=0.5
)


EMPTY2D = Env(
    name="empty2d",
    CSpace=[np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])],
    robot_radius=0.01
)


EMPTY2D_4DynamicSPHERE = Env(
    name="empty2d+4dyna_spheres",
    CSpace=EMPTY2D.C_Space,
    ODynamic=[
        ConcatDynamicSphere(
            X0 = [np.array([0.1, 0.9]), np.array([0.1, 0.1]), np.array([0.9, 0.1]), np.array([0.9, 0.9])],
            Xt = [np.array([0.1, 0.1]), np.array([0.9, 0.1]), np.array([0.9, 0.9]), np.array([0.1, 0.9])],
            itvls = [Interval(0, 1), Interval(1, 2), Interval(2, 3), Interval(3, 4), Interval(4, 5)],
            radius = 0.05
        ),
        ConcatDynamicSphere(
            X0 = [np.array([0.2, 0.8]), np.array([0.2, 0.2]), np.array([0.8, 0.2]), np.array([0.8, 0.8])],
            Xt = [np.array([0.2, 0.2]), np.array([0.8, 0.2]), np.array([0.8, 0.8]), np.array([0.2, 0.8])],
            itvls = [Interval(0, 1), Interval(1, 2), Interval(2, 3), Interval(3, 4), Interval(4, 5)],
            radius = 0.05
        ),
        ConcatDynamicSphere(
            X0 = [np.array([0.3, 0.7]), np.array([0.3, 0.3]), np.array([0.7, 0.3]), np.array([0.7, 0.7])],
            Xt = [np.array([0.3, 0.3]), np.array([0.7, 0.3]), np.array([0.7, 0.7]), np.array([0.3, 0.7])],
            itvls = [Interval(0, 1), Interval(1, 2), Interval(2, 3), Interval(3, 4), Interval(4, 5)],
            radius = 0.05
        ),
        ConcatDynamicSphere(
            X0 = [np.array([0.4, 0.6]), np.array([0.4, 0.4]), np.array([0.6, 0.4]), np.array([0.6, 0.6])],
            Xt = [np.array([0.4, 0.4]), np.array([0.6, 0.4]), np.array([0.6, 0.6]), np.array([0.4, 0.6])],
            itvls = [Interval(0, 1), Interval(1, 2), Interval(2, 3), Interval(3, 4), Interval(4, 5)],
            radius = 0.05
        ),
    ],
    robot_radius=0.05
)


SIMPLE2D = Env(
    name="simple2d",
    CSpace = [
        np.array([[0.0, 0.0], [1.85, 0.0], [1.85, 1.85], [0.0, 1.85]]),
        np.array([[0.0, 2.15], [1.85, 2.15], [1.85, 4.0], [0.0, 4.0]]),
        np.array([[2.15, 0.0], [4.0, 0.0], [4.0, 1.85], [2.15, 1.85]]),
        np.array([[2.15, 2.15], [4.0, 2.15], [4.0, 4.0], [2.15, 4.0]]),
        np.array([[1.85, 1.6], [2.15, 1.6], [2.15, 2.4], [1.85, 2.4]]),
	],
    OStatic = [
        StaticPolygon(np.array([[1.9, -0.05], [2.1, -0.05], [2.1, 1.55], [1.9, 1.55]])),
        StaticPolygon(np.array([[1.9, 2.45], [2.1, 2.45], [2.1, 4.05], [1.9, 4.05]])), 
        StaticPolygon(np.array([[-0.05, 1.9], [1.8, 1.9], [1.8, 2.1], [-0.05, 2.1]])), 
        StaticPolygon(np.array([[2.2, 1.9], [4.05, 1.9], [4.05, 2.1], [2.2, 2.1]])),
	],
    robot_radius=0.05
)
    

SIMPLE2D_4DynamicSPHERE = Env(
    name = "simple2d+4dyna_spheres",
    CSpace = SIMPLE2D.C_Space,
    OStatic = SIMPLE2D.O_Static,
    robot_radius = SIMPLE2D.robot_radius,
    ODynamic = [
        DynamicSphere(x0=np.array([1.45, 0.45]), xt=np.array([0.4, 0.45]), itvl=Interval(0, 2), radius=0.4),
        DynamicSphere(x0=np.array([0.4, 1.45]), xt=np.array([1.45, 1.45]), itvl=Interval(0, 4), radius=0.4),
        DynamicSphere(x0=np.array([2.6, 2.55]), xt=np.array([2.6, 3.6]), itvl=Interval(2, 5), radius=0.4),
        DynamicSphere(x0=np.array([3.6, 3.6]), xt=np.array([3.6, 2.55]), itvl=Interval(2, 7), radius=0.4),
    ], 
)


SIMPLE2D_8DynamicSPHERE = Env(
    name = "simple2d+8dyna_spheres",
    CSpace = SIMPLE2D.C_Space,
    OStatic = SIMPLE2D.O_Static,
    robot_radius = SIMPLE2D.robot_radius,
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


COMPLEX2D = Env(
    name="complex2d",
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
        StaticPolygon(np.array([[3.3793, 2.55], [2.4096, 2.55], [1.4964, 2.1847], [3.7969, 0.2676], [4.7293, 1.2]])),
		StaticPolygon(np.array([[3.35, 4.55], [2.45, 4.55], [2.45, 2.55], [3.35, 2.55]])),
		StaticPolygon(np.array([[1.45, 2.85], [2.15, 2.85], [2.15, 4.55], [1.45, 4.55]])),
		StaticPolygon(np.array([[0.95, 2.65], [0.95, 5.05], [0.45, 5.05], [0.45, 2.65]])),
		StaticPolygon(np.array([[0.95, 2.35], [0.95, -0.05], [0.45, -0.05], [0.45, 2.35]])),
		StaticPolygon(np.array([[3.85, 3.05], [3.85, 5.05], [4.35, 5.05], [4.35, 3.05]])),
		StaticPolygon(np.array([[3.85, 2.75], [3.85, 2.65], [5.05, 2.65], [5.05, 2.75]])),
	],
    robot_radius=0.05
)


COMPLEX2D_4DynamicSPHERE = Env(
    name = "complex2d+4dyna_spheres",
    CSpace = COMPLEX2D.C_Space,
    OStatic = COMPLEX2D.O_Static,
    robot_radius = COMPLEX2D.robot_radius,
    ODynamic = [
        DynamicSphere(x0=np.array([0.2, 4.5]), xt=np.array([0.2, 0.5]), itvl=Interval(0, 4), radius=0.05),
        DynamicSphere(x0=np.array([1.5, 1.5]), xt=np.array([2.5, 0.5]), itvl=Interval(2, 6), radius=0.2),
        DynamicSphere(x0=np.array([4.8, 1.7]), xt=np.array([4.0, 2.4]), itvl=Interval(3, 7), radius=0.1),
        DynamicSphere(x0=np.array([1.5, 4.8]), xt=np.array([3.5, 4.8]), itvl=Interval(2, 7), radius=0.08),
    ]
)
