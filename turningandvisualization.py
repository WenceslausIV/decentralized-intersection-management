import numpy as np
from world import World
from agents import Car, RectangleBuilding, Pedestrian, Painting, CircleBuilding
from geometry import Point
import time
import cvxopt
from cvxopt import matrix, sparse
from cvxopt.solvers import qp
from scipy.special import comb
cvxopt.solvers.options['show_progress'] = False
N = 3
x = np.zeros((2, N))
t_r_path = []
r_b_path = []
b_l_path = []
l_t_path = []

class PIDController:
    def __init__(self, kp, ki, kd, target_velocity):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target_velocity = target_velocity
        self.integral_error = 0
        self.previous_error = 0

    def control(self, current_velocity, dt):
        error = self.target_velocity - current_velocity
        self.integral_error += error * dt
        derivative_error = (error - self.previous_error) / dt
        self.previous_error = error

        control_signal = (self.kp * error + 
                          self.ki * self.integral_error + 
                          self.kd * derivative_error)
        return control_signal


human_controller = False

dt = 0.05 # time steps in terms of seconds. In other words, 1/dt is the FPS.
w = World(dt, width = 120, height = 120, ppm = 6) # The world is 120 meters by 120 meters. ppm is the pixels per meter.

w.add(Painting(Point(125.5, 116.5), Point(113, 75), 'gray80')) 
w.add(Painting(Point(125.5, 116.5), Point(97, 89), 'gray80')) 
w.add(RectangleBuilding(Point(126.5, 117.5), Point(105, 83))) 
# Let's repeat this for 4 different RectangleBuildings.
w.add(Painting(Point(-5.5, 116.5), Point(97, 89), 'gray80'))
w.add(Painting(Point(-5.5, 116.5), Point(113, 75), 'gray80'))
w.add(RectangleBuilding(Point(-6.5, 117.5), Point(105, 83)))

w.add(Painting(Point(-5.5, 10), Point(97, 88), 'gray80'))
w.add(Painting(Point(-5.5, 10), Point(113, 75), 'gray80'))
w.add(RectangleBuilding(Point(-6.5, 9), Point(95, 73)))

w.add(Painting(Point(125.5, 10.5), Point(97, 87), 'gray80'))
w.add(Painting(Point(125.5, 10.5), Point(113, 75), 'gray80'))
w.add(RectangleBuilding(Point(126, 9), Point(95, 73)))

#bottom middle lines
w.add(Painting(Point(60, 45), Point(2, 8), 'gray80'))
w.add(Painting(Point(60, 25), Point(2, 8), 'gray80'))
w.add(Painting(Point(60, 5), Point(2, 8), 'gray80'))
#top middle lines
w.add(Painting(Point(60, 80), Point(2, 8), 'gray80'))
w.add(Painting(Point(60, 100), Point(2, 8), 'gray80'))
w.add(Painting(Point(60, 120), Point(2, 8), 'gray80'))
#right middle lines
w.add(Painting(Point(82.5, 63), Point(8, 2), 'gray80'))
w.add(Painting(Point(102.5, 63), Point(8, 2), 'gray80'))
w.add(Painting(Point(122.5, 63), Point(8, 2), 'gray80'))
#left middle lines
w.add(Painting(Point(38, 63), Point(8, 2), 'gray80'))
w.add(Painting(Point(18, 63), Point(8, 2), 'gray80'))
w.add(Painting(Point(-2, 63), Point(8, 2), 'gray80'))
#cb = CircleBuilding(Point(world_width/2, world_height/2), inner_building_radius, 'gray80')
w.add(CircleBuilding(Point(77, 80), 8, 'gray80')) # top right circle
w.add(CircleBuilding(Point(77, 46), 8, 'gray80')) # bottom right circle
w.add(CircleBuilding(Point(43, 80), 8, 'gray80')) # top left circle
w.add(CircleBuilding(Point(43, 46), 8, 'gray80')) # bottom left circle
w.add(Painting(Point(71, 74), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(70, 73), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(69, 72), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(68, 71), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(67, 70), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(66, 69), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(65, 68), Point(0.5, 0.5), 'red')) # top right conflict zone
w.add(Painting(Point(64, 67), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(63, 66), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(62, 65), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(61, 64), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(60, 63), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(59, 62), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(58, 61), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(57, 60), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(56, 59), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(55, 58), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(54, 57), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(53, 56), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(52, 55), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(51, 54), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(50, 53), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(49, 52), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(55, 58), Point(0.5, 0.5), 'red')) # bottom left conflict zone
w.add(Painting(Point(49, 74), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(50, 73), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(51, 72), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(52, 71), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(53, 70), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(54, 69), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(55, 68), Point(0.5, 0.5), 'red')) # top left conflict zone
w.add(Painting(Point(56, 67), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(57, 66), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(58, 65), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(59, 64), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(60, 63), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(61, 62), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(62, 61), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(63, 60), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(64, 59), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(65, 58), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(66, 57), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(67, 56), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(68, 55), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(69, 54), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(70, 53), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(71, 52), Point(0.5, 0.5), 'blue'))
# now vertical blue lines
#w.add(Painting(Point(60, 47), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(60, 48), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(60, 49), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(60, 50), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(60, 51), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(60, 52), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(60, 53), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(60, 54), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(60, 55), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(60, 56), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(60, 57), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(60, 58), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(60, 59), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(60, 60), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(60, 61), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(60, 62), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(60, 63), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(60, 64), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(60, 65), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(60, 66), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(60, 67), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(60, 68), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(60, 69), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(60, 70), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(60, 71), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(60, 72), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(60, 73), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(60, 74), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(60, 75), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(60, 76), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(60, 77), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(60, 78), Point(0.5, 0.5), 'blue'))
# now horizontal blue lines
w.add(Painting(Point(45, 63), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(46, 63), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(47, 63), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(48, 63), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(49, 63), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(50, 63), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(51, 63), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(52, 63), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(53, 63), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(54, 63), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(55, 63), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(56, 63), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(57, 63), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(58, 63), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(59, 63), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(60, 63), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(61, 63), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(62, 63), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(63, 63), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(64, 63), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(65, 63), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(66, 63), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(67, 63), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(68, 63), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(69, 63), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(70, 63), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(71, 63), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(72, 63), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(73, 63), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(74, 63), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(75, 63), Point(0.5, 0.5), 'blue'))

#rightest side vertical blue line
w.add(Painting(Point(77, 55), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(77, 56), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(77, 57), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(77, 58), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(77, 59), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(77, 60), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(77, 61), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(77, 62), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(77, 63), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(77, 64), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(77, 65), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(77, 66), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(77, 67), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(77, 68), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(77, 69), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(77, 70), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(77, 71), Point(0.5, 0.5), 'blue'))
#leftest side vertical blue line
w.add(Painting(Point(43, 55), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(43, 56), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(43, 57), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(43, 58), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(43, 59), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(43, 60), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(43, 61), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(43, 62), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(43, 63), Point(0.5, 0.5), 'blue')) #middle
w.add(Painting(Point(43, 64), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(43, 65), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(43, 66), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(43, 67), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(43, 68), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(43, 69), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(43, 70), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(43, 71), Point(0.5, 0.5), 'blue'))

#very top horizontal
w.add(Painting(Point(52, 80), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(53, 80), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(54, 80), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(55, 80), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(56, 80), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(57, 80), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(58, 80), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(59, 80), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(60, 80), Point(0.5, 0.5), 'blue')) # middle
w.add(Painting(Point(61, 80), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(62, 80), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(63, 80), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(64, 80), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(65, 80), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(66, 80), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(67, 80), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(68, 80), Point(0.5, 0.5), 'blue'))
# very bottom horizontal
w.add(Painting(Point(52, 46), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(53, 46), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(54, 46), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(55, 46), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(56, 46), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(57, 46), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(58, 46), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(59, 46), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(60, 46), Point(0.5, 0.5), 'blue')) # middle
w.add(Painting(Point(61, 46), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(62, 46), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(63, 46), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(64, 46), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(65, 46), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(66, 46), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(67, 46), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(68, 46), Point(0.5, 0.5), 'blue'))
w.add(Painting(Point(60, 68), Point(0.5, 0.5), 'red')) # top middle conflict zone
w.add(Painting(Point(55, 63), Point(0.5, 0.5), 'red')) # middle left conflict zone
w.add(Painting(Point(60, 58), Point(0.5, 0.5), 'red')) # bottom middle conflict zone
w.add(Painting(Point(65, 58), Point(0.5, 0.5), 'red')) # bottom right conflict zone
w.add(Painting(Point(65, 63), Point(0.5, 0.5), 'pink')) #middle right lane conflict point

def draw_curved_line(center, radius, start_angle, end_angle, num_points, color):
    angles = np.linspace(start_angle, end_angle, num_points)
    for angle in angles:
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        w.add(Painting(Point(x, y), Point(0.5, 0.5), color))

def draw_curved_line_list(center, radius, start_angle, end_angle, num_points, color):
    angles = np.linspace(start_angle, end_angle, num_points)
    for angle in angles:
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        w.add(Painting(Point(x, y), Point(0.5, 0.5), color))
        if center == (77, 46):
            r_b_path.append((x,y))
        elif center == (77, 80):
            t_r_path.append((x,y))

draw_curved_line(center=(77, 80), radius=16, start_angle=np.pi, end_angle=3*np.pi/2, num_points=20, color='blue')
draw_curved_line(center=(43, 80), radius=16, start_angle=3*np.pi/2, end_angle=2*np.pi, num_points=20, color='blue')
draw_curved_line(center=(77, 46), radius=16, start_angle=np.pi/2, end_angle=np.pi, num_points=20, color='blue')
draw_curved_line(center=(43, 46), radius=16, start_angle=0, end_angle=np.pi/2, num_points=20, color='blue')
draw_curved_line(center=(77, 80), radius=8, start_angle=np.pi, end_angle=3*np.pi/2, num_points=20, color='blue')
draw_curved_line(center=(43, 80), radius=8, start_angle=3*np.pi/2, end_angle=2*np.pi, num_points=20, color='blue')
draw_curved_line(center=(77, 46), radius=8, start_angle=np.pi/2, end_angle=np.pi, num_points=20, color='blue')
draw_curved_line(center=(43, 46), radius=8, start_angle=0, end_angle=np.pi/2, num_points=20, color='blue')

draw_curved_line_list(center=(77, 80), radius=21, start_angle=np.pi, end_angle=3*np.pi/2, num_points=20, color='white') # right and top curved path
draw_curved_line_list(center=(43, 80), radius=21, start_angle=3*np.pi/2, end_angle=2*np.pi, num_points=20, color='white') # left and top curved path
draw_curved_line_list(center=(77, 46), radius=21, start_angle=np.pi/2, end_angle=np.pi, num_points=20, color='green')# right and bottom curved path
r_b_path.append((55,0))
draw_curved_line_list(center=(43, 46), radius=21, start_angle=0, end_angle=np.pi/2, num_points=20, color='white')# left and bottom curved path

# A Car object is a dynamic object -- it can move. We construct it using its center location and heading angle.
#c1 = Car(Point(70,20), np.pi/2)
c1 = Car(Point(110,68), np.pi, 'black')
c1.velocity = Point(0,0)
w.add(c1)
current_heading = c1.heading
print("Current heading:", current_heading)


c2 = Car(Point(130,68), np.pi, 'blue')
c2.velocity = Point(0,0) 
w.add(c2)

c3 = Car(Point(65, 40), np.pi/2, 'green')
c3.velocity = Point(0,0) 
w.add(c3)


clist = []
clist.append(c1)
clist.append(c2)
clist.append(c3)
lane_list = ['rd', 'r', 'd']

w.render() # This visualizes the world we just constructed.


def cbf(x, u_ref):
    N = 3

    H = sparse(matrix(2*np.identity(N)))
    f = -2*np.reshape(u_ref, N, order='F')
    num_constraints = 2 * int(comb(N, 2)) + 2 + 6
    A = np.zeros((num_constraints, N))
    b = np.zeros(num_constraints)
    phi = 0.1
    lambda_1 = 0.1
    lambda_2 = 0.1
    lambda_3 = 0.1
    lambda_4 = 0.2
    gamma = 10 #standstill_distance
    gamma2 = 1
    car_width_halved = 1.5
    count = 0
    for i in range(N):
        for j in range(i+1, N):
            v_i = np.sqrt(clist[i].velocity.x**2 + clist[i].velocity.y**2)
            v_j = np.sqrt(clist[j].velocity.x**2 + clist[j].velocity.y**2)

            if lane_list[i] == lane_list[j]:
                constraint_value1 = 0

                F_rm_i = 0.001


                if lane_list[i] == 'r' and (x[0,i] > x[0,j]) and abs(x[1,i] - x[1,j]) < car_width_halved: #i and j are on 'r' and i is behind and i and j's y location difference is small

                    constraint_value1 = (1 / phi) * (lambda_3 * (-x[0,j] + x[0,i] - (gamma + phi * v_i)) + v_j - v_i) + F_rm_i
                    A[count, i] = 1.0
                    b[count] = constraint_value1
                    count += 1

                elif lane_list[i] == 'r' and (x[0,i] < x[0,j]):

                    constraint_value1 = (1 / phi) * (lambda_3 * (-x[0,i] + x[0,j] - (gamma + phi * v_j)) + v_i - v_j) + F_rm_i
                    A[count, j] = 1.0
                    b[count] = constraint_value1
                    count += 1

                elif lane_list[i] == 'rd': #and (x[0,i] > x[0,j]):
                    if (x[0,i] > x[0,j]) or x[1,i] > x[1,j]: #if i is behind j or if i is above j
                        constraint_value1 = (1 / phi) * (lambda_3 * (np.linalg.norm(-x[:,j] + x[:,i]) - (gamma + phi * v_i)) + v_j - v_i) + F_rm_i
                        A[count, i] = 1.0
                        b[count] = constraint_value1
                        count += 1
                    elif (x[0,i] < x[0,j]) or x[1,i] > x[1,j]:
                        constraint_value1 = (1 / phi) * (lambda_3 * (np.linalg.norm(-x[:,i] + x[:,j]) - (gamma + phi * v_j)) + v_i - v_j) + F_rm_i
                        A[count, j] = 1.0
                        b[count] = constraint_value1
                        count += 1
                elif lane_list[i] == 'd':
                    if (x[1,i] < x[1,j]):
                        constraint_value1 = (1 / phi) * (lambda_3 * (np.linalg.norm(-x[:,j] + x[:,i]) - (gamma + phi * v_i)) + v_j - v_i) + F_rm_i
                        A[count, i] = 1.0
                        b[count] = constraint_value1
                        count += 1
                    elif (x[1,i] > x[1,j]):
                        constraint_value1 = (1 / phi) * (lambda_3 * (np.linalg.norm(-x[:,i] + x[:,j]) - (gamma + phi * v_j)) + v_i - v_j) + F_rm_i
                        A[count, j] = 1.0
                        b[count] = constraint_value1
                        count += 1



            else:

                '''
                if lane_list[j] == 'd' and x[0,i] > 68: 
                    #print('down car exists as j')
                    constraint_value3 = (1 / phi) * (lambda_4 * ((68.0-x[1,j]) + ( -(65.0-x[0,i]) ) - (gamma2 + phi * v_j)) - (v_j + v_i) ) + F_rm_i
                    A[count, j] = 1.0
                    b[count] = constraint_value3
                    count += 1
                '''
                '''
                elif lane_list[i] == 'd' and lane_list[j] == 'rd': #여기도 커브 들어갓으니까 점사이 거리 다시구하기
                    constraint_value4 = (1 / phi) * (lambda_4 * ((63.0-x[1,i]) + ( -(65.0-x[0,j]) ) - (gamma2 + phi * v_i)) - (v_i + v_j) ) + F_rm_i
                    A[count, j] = 1.0
                    b[count] = constraint_value4
                    count += 1   
                '''
                '''
                elif lane_list[i] == 'rd' and x[0,i] > 68:         
                    constraint_value3 = (1 / phi) * (lambda_4 * ((65.0-x[1,j]) + ( -(63.0-x[0,i]) ) - (gamma2 + phi * v_j)) - (v_j + v_i) ) + F_rm_i
                    A[count, j] = 1.0
                    b[count] = constraint_value4
                    count += 1
                '''

    
    for i in range(N):
        vmax = 100
        vmin = 0.001
        v_i = np.sqrt(clist[i].velocity.x**2 + clist[i].velocity.y**2)
        constraint_value4 = F_rm_i + lambda_1 * (vmax - v_i)
        A[count, j] = 1.0
        b[count] = constraint_value4 
        count += 1

        constraint_value5 = -(F_rm_i + lambda_2 * (v_i - vmin))
        A[count, j] = -1.0
        b[count] = constraint_value4 
        count += 1

    # Convert to cvxopt matrix
    H = matrix(H)
    f = matrix(f)
    A = matrix(A)
    b = matrix(b)
    # Solve the quadratic program
    solution = qp(H, f, A, b)
        
    # Get the optimal control inputs
    u_optimal = np.array(solution['x']).flatten()
        
    return u_optimal


def calculate_steering_angle(car, target_point):
    car_to_target = Point(target_point[0] - car.center.x, target_point[1] - car.center.y)
    angle_to_target = np.arctan2(car_to_target.y, car_to_target.x)
    steering_angle = angle_to_target - car.heading
    return steering_angle

pid_controller = PIDController(kp=1.0, ki=0.5, kd=0.1, target_velocity=6.0)
pid_controller2 = PIDController(kp=1.0, ki=0.5, kd=0.1, target_velocity = 5.0)
pid_controller3 = PIDController(kp=1.0, ki=0.5, kd=0.1, target_velocity=4.0)

if not human_controller:
    time_steps = int(400 * dt)  # Adjust for correct time steps based on dt
    time_passed = 0.0
    current_milestone_index = 0
    
    while time_passed < 800.0:  # Run simulation for 40 seconds


        # Get current velocity and calculate throttle
        velocity_magnitude = np.sqrt(c1.velocity.x**2 + c1.velocity.y**2)
        acceleration = pid_controller.control(velocity_magnitude, dt)
        throttle1 = acceleration
        velocity_magnitude = np.sqrt(c2.velocity.x**2 + c2.velocity.y**2)
        acceleration = pid_controller2.control(velocity_magnitude, dt)
        throttle2 = acceleration
        velocity_magnitude = np.sqrt(c3.velocity.x**2 + c3.velocity.y**2)
        acceleration = pid_controller3.control(velocity_magnitude, dt)
        throttle3 = acceleration
        u_ref = np.array([throttle1 ,throttle2, throttle3])
        
        # Update world
        w.tick() # Tick the world for one time step
        w.render()
        time.sleep(dt/4) # Watch it at 4x slower speed
        time_passed += dt

        # Update car states
        for i in range(N):
            x[0,i] = clist[i].center.x
            x[1,i] = clist[i].center.y
            if x[0,i] > 43 and x[1,i] > 63:
                lane_list[i] = 'rightout'

        th = cbf(x, u_ref)

        # Calculate steering angle for c1 to follow the path
        if current_milestone_index < len(r_b_path):
            target_point = r_b_path[current_milestone_index]
            distance_to_target = np.sqrt((c1.center.x - target_point[0])**2 + (c1.center.y - target_point[1])**2)
            if distance_to_target < 1.0:  # If close enough to the target point, move to the next one
                current_milestone_index += 1
            steering_angle = calculate_steering_angle(c1, target_point)
        else:
            steering_angle = 0
        if current_milestone_index == len(r_b_path):
            #target_point = (55,0)
            current_milestone_index = 0

        c1.set_control(steering_angle, th[0])
        c2.set_control(0, th[1])
        c3.set_control(0, th[2])
        #c1.set_control(steering_angle, u_ref[0])
        #c2.set_control(0, u_ref[1])
        #c3.set_control(0, u_ref[2])

        for i in range(N):
            if x[0,i] >= 141:
                clist[i].center = Point(140, 68)
            
            elif x[0,i] < 0:
                clist[i].center = Point(130,68)
                clist[i].heading = np.pi
            elif x[1,i] < 0:
                clist[i].center = Point(121,68)
                clist[i].heading = np.pi
            elif x[1,i] >= 141:
                clist[i].center = Point(65, 0)
                clist[i].heading = np.pi/2
    w.close()
