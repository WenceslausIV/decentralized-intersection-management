import numpy as np
from world import World
from agents import Car, RectangleBuilding, Pedestrian, Painting, CircleBuilding
from geometry import Point
import time
import cvxopt
from cvxopt import matrix, sparse
from cvxopt.solvers import qp
from scipy.special import comb
from scipy.spatial import Voronoi
from scipy.optimize import fsolve
from scipy.optimize import root
from sympy import symbols, Eq, solve
from collections import Counter
from scipy.optimize import brentq
import math
cvxopt.solvers.options['show_progress'] = False
N = 4
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

dt = 0.1 # time steps in terms of seconds. In other words, 1/dt is the FPS.
w = World(dt, width = 120, height = 120, ppm = 6) # The world is 120 meters by 120 meters. ppm is the pixels per meter.
w.add(Painting(Point(125.5, 116.5), Point(113, 75), 'gray80')) 
w.add(Painting(Point(125.5, 116.5), Point(97, 89), 'gray80')) 
w.add(RectangleBuilding(Point(126, 117.5), Point(99, 76))) 
# Let's repeat this for 4 different RectangleBuildings.
w.add(Painting(Point(-5.5, 116.5), Point(97, 89), 'gray80'))
w.add(Painting(Point(-5.5, 116.5), Point(113, 75), 'gray80'))
w.add(RectangleBuilding(Point(-6.5, 117.5), Point(99, 76)))

w.add(Painting(Point(-5.5, 10), Point(97, 88), 'gray80'))
w.add(Painting(Point(-5.5, 10), Point(113, 75), 'gray80'))
w.add(RectangleBuilding(Point(-6.5, 9), Point(99, 76)))

w.add(Painting(Point(125.5, 10.5), Point(97, 87), 'gray80'))
w.add(Painting(Point(125.5, 10.5), Point(113, 75), 'gray80'))
w.add(RectangleBuilding(Point(126, 9), Point(99, 76)))

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
w.add(RectangleBuilding(Point(126, 117.5), Point(99, 76))) 
w.add(RectangleBuilding(Point(-6.5, 117.5), Point(99, 76)))
w.add(RectangleBuilding(Point(-6.5, 9), Point(99, 76)))
w.add(RectangleBuilding(Point(126, 9), Point(99, 76)))
#hi

def paint_x_mark(center_x, center_y, size, color, world, point_spacing=0.5):
    half_size = size / 2
    num_points = int(size / point_spacing) + 1
    for i in range(num_points):
        x_offset = i * point_spacing
        world.add(Painting(Point(center_x - half_size + x_offset, center_y - half_size + x_offset), Point(0.4, 0.4), color))
        world.add(Painting(Point(center_x - half_size + x_offset, center_y + half_size - x_offset), Point(0.4, 0.4), color))

# Example usage:
paint_x_mark(60, 63, 22, 'blue', w, point_spacing=0.3)

# now vertical blue lines
w.add(Painting(Point(60, 55), Point(0.3, 18), 'blue'))
w.add(Painting(Point(60, 70), Point(0.3, 20), 'blue'))

# now horizontal blue lines
w.add(Painting(Point(57, 63), Point(28, 0.3), 'blue'))
w.add(Painting(Point(63, 63), Point(28, 0.3), 'blue'))

# rightest side vertical blue line
w.add(Painting(Point(77, 63), Point(0.3, 18), 'blue'))

# leftest side vertical blue line
w.add(Painting(Point(43, 63), Point(0.3, 18), 'blue')) # middle

# very top horizontal
w.add(Painting(Point(60, 80), Point(18, 0.3), 'blue')) # middle

# very bottom horizontal
w.add(Painting(Point(60, 46), Point(18, 0.3), 'blue')) # middle

w.add(Painting(Point(65, 68), Point(1, 1), 'red')) # top right conflict zone
w.add(Painting(Point(55, 58), Point(1, 1), 'red')) # bottom left conflict zone
w.add(Painting(Point(55, 68), Point(1, 1), 'red')) # top left conflict zone
w.add(Painting(Point(60, 68), Point(1, 1), 'red')) # top middle conflict zone
w.add(Painting(Point(55, 63), Point(1, 1), 'red')) # middle left conflict zone
w.add(Painting(Point(60, 58), Point(1, 1), 'red')) # bottom middle conflict zone
w.add(Painting(Point(65, 58), Point(1, 1), 'red')) # bottom right conflict zone
w.add(Painting(Point(65, 63), Point(1, 1), 'red')) # middle right lane conflict point

def draw_curved_line(center, radius, start_angle, end_angle, num_points, color):
    angles = np.linspace(start_angle, end_angle, num_points)
    for angle in angles:
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        w.add(Painting(Point(x, y), Point(0.4, 0.4), color))

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

draw_curved_line(center=(77, 80), radius=16, start_angle=np.pi, end_angle=3*np.pi/2, num_points=70, color='blue')
draw_curved_line(center=(43, 80), radius=16, start_angle=3*np.pi/2, end_angle=2*np.pi, num_points=70, color='blue')
draw_curved_line(center=(77, 46), radius=16, start_angle=np.pi/2, end_angle=np.pi, num_points=70, color='blue')
draw_curved_line(center=(43, 46), radius=16, start_angle=0, end_angle=np.pi/2, num_points=70, color='blue')
draw_curved_line(center=(77, 80), radius=8, start_angle=np.pi, end_angle=3*np.pi/2, num_points=70, color='blue')
draw_curved_line(center=(43, 80), radius=8, start_angle=3*np.pi/2, end_angle=2*np.pi, num_points=70, color='blue')
draw_curved_line(center=(77, 46), radius=8, start_angle=np.pi/2, end_angle=np.pi, num_points=70, color='blue')
draw_curved_line(center=(43, 46), radius=8, start_angle=0, end_angle=np.pi/2, num_points=70, color='blue')

draw_curved_line_list(center=(77, 80), radius=21, start_angle=np.pi, end_angle=3*np.pi/2, num_points=10, color='white') # right and top curved path
draw_curved_line_list(center=(43, 80), radius=21, start_angle=3*np.pi/2, end_angle=2*np.pi, num_points=10, color='white') # left and top curved path
draw_curved_line_list(center=(77, 46), radius=21, start_angle=np.pi/2, end_angle=np.pi, num_points=10, color='white')# right and bottom curved path
r_b_path.append((55,0))
draw_curved_line_list(center=(43, 46), radius=21, start_angle=0, end_angle=np.pi/2, num_points=10, color='white')# left and bottom curved path
#w.add(Painting(Point(60, 63), Point(1, 1), 'red')) # bottom middle conflict zone
# A Car object is a dynamic object -- it can move. We construct it using its center location and heading angle.
c1 = Car(Point(51,98), -np.pi/2, 'black')
c1.velocity = Point(0,0)
w.add(c1)
current_heading = c1.heading
#print("Current heading:", current_heading)

c2 = Car(Point(110,68), np.pi, 'blue')
c2.velocity = Point(0,0) 
w.add(c2)

c3 = Car(Point(68, 20), np.pi/2, 'green')
c3.velocity = Point(0,0) 
w.add(c3)

c4 = Car(Point(20, 55), -np.pi, 'red')
c4.velocity = Point(0,0) 
w.add(c4)
'''
c5 = Car(Point(0, 58), 0, 'yellow')
c5.velocity = Point(0,0) 

w.add(c5)

c6= Car(Point(0, 58), 0, 'brown')
c6.velocity = Point(0,0) 

w.add(c6)
'''
clist = []
clist.append(c1)
clist.append(c2)

clist.append(c3)
clist.append(c4)
#clist.append(c5)
lane_list = ['rd', 'r', 'd', 'l']


#########################################################################
import numpy as np
import matplotlib.pyplot as plt

prange = 0
# Generate random points
plist = []
for i in range(5000):
    plist.append(Pedestrian(Point(-1,-1), np.pi))
    w.add(plist[i])

w.render() # This visualizes the world we just constructed.

pointlist_r = []
pointlist_r_clone = []
def wv():
    global pointlist_r_clone, pointlist_r
    todeleteindex = []
    termlistcar = []
    termlistcar2 = []
    termlistboundary = []
    x_clone = x.copy()
    velocitylist = [] #try this~~~
    #u_ref = np.array([0 ,0, 0])
    gamma = 1
    safety_radius = 2
    c = 1

    risk = np.zeros(N)
    for i in range(1):
        i = 0 
        for j in range(N):
            if i != j:
                risk[i] = risk[i] + (-( 2*(x[0,i] - x[0,j]) * (clist[i].velocity.x - clist[j].velocity.x) + 2 *(x[1,i] - x[1,j]) * (clist[i].velocity.y - clist[j].velocity.y) ) - gamma* (( x[0,i] - x[0,j])**2 + (x[1,i] - x[1,j])**2 - safety_radius**2) - c)
                #print("risk", i, risk[i])
    for i in range(1):
        i = 1 
        for j in range(N):
            if i != j:
                risk[i] = risk[i] + (-( 2*(x[0,i] - x[0,j]) * (clist[i].velocity.x - clist[j].velocity.x) + 2 *(x[1,i] - x[1,j]) * (clist[i].velocity.y - clist[j].velocity.y) ) - gamma* (( x[0,i] - x[0,j])**2 + (x[1,i] - x[1,j])**2 - safety_radius**2) - c)
   
    for i in range(1):
        i = 2 
        for j in range(N):
            if i != j:
                risk[i] = risk[i] + (-( 2*(x[0,i] - x[0,j]) * (clist[i].velocity.x - clist[j].velocity.x) + 2 *(x[1,i] - x[1,j]) * (clist[i].velocity.y - clist[j].velocity.y) ) - gamma* (( x[0,i] - x[0,j])**2 + (x[1,i] - x[1,j])**2 - safety_radius**2) - c)

    for i in range(1):
        i = 3
        for j in range(N):
            if i != j:
                risk[i] = risk[i] + (-( 2*(x[0,i] - x[0,j]) * (clist[i].velocity.x - clist[j].velocity.x) + 2 *(x[1,i] - x[1,j]) * (clist[i].velocity.y - clist[j].velocity.y) ) - gamma* (( x[0,i] - x[0,j])**2 + (x[1,i] - x[1,j])**2 - safety_radius**2) - c)
    
    risk = risk / 1000
    print(risk)

    for i in range(N-1):
        for j in range(i+1, N):

            global prange

            phi = 2 # Example value

            v_i = np.sqrt(clist[i].velocity.x**2 + clist[i].velocity.y**2)
            v_j = np.sqrt(clist[j].velocity.x**2 + clist[j].velocity.y**2)
            ci  = phi * v_i + v_i + phi * u_ref[i]
            cj = phi * v_j + v_j + phi * u_ref[j]

            def equation1(b):
                term1 = np.sqrt((- x_clone[0, i])**2 + (b - x_clone[1, i])**2) - ci - risk[i]
                term2 = np.sqrt((- x_clone[0, j])**2 + (b - x_clone[1, j])**2) - cj - risk[j]
                return term1 - term2

            pointlist = []
            b_initial_guess = 0.0
            b_solution, = fsolve(equation1, b_initial_guess)

            pointlist.append((0.0, b_solution))
            def equation2(b):
                term1 = np.sqrt((120- x_clone[0, i])**2 + (b - x_clone[1, i])**2) - ci - risk[i]
                term2 = np.sqrt((120- x_clone[0, j])**2 + (b - x_clone[1, j])**2) - cj - risk[j]
                return term1 - term2

            b_initial_guess = 0.0
            b_solution2, = fsolve(equation2, b_initial_guess)

            pointlist.append((120.0, b_solution2))

            def equation3(a):
                term1 = np.sqrt((a- x_clone[0, i])**2 + (0 - x_clone[1, i])**2) - ci - risk[i]
                term2 = np.sqrt((a- x_clone[0, j])**2 + (0 - x_clone[1, j])**2) - cj - risk[j]
                return term1 - term2
            a_initial_guess = 0.0
            a_solution, = fsolve(equation3, a_initial_guess)

            pointlist.append((a_solution, 0.0))
            def equation4(a):
                term1 = np.sqrt((a- x_clone[0, i])**2 + (120 - x_clone[1, i])**2) - ci - risk[i]
                term2 = np.sqrt((a- x_clone[0, j])**2 + (120 - x_clone[1, j])**2) - cj - risk[j]
                return term1 - term2
            a_initial_guess = 0.0
            a_solution2, = fsolve(equation4, a_initial_guess)
            pointlist.append((a_solution2,120.0))

            indices = []
            biga = 0
            smalla = 0
            bigb = 0
            smallb = 0
            adifference =0
            bdifference =0
            # Iterate through the pointlist
            for index, point in enumerate(pointlist):
                if 0 <= point[0] <= 120 and 0 <= point[1] <= 120:
                    indices.append(index)
                    
            if len(indices) == 2:
                if pointlist[indices[0]][0] > pointlist[indices[1]][0]:
                    biga = pointlist[indices[0]][0]
                    smalla = pointlist[indices[1]][0]
                    adifference = biga - smalla
                else:
                    biga = pointlist[indices[1]][0]
                    smalla = pointlist[indices[0]][0]
                    adifference = biga - smalla        
                if pointlist[indices[0]][1] > pointlist[indices[1]][1]:
                    bigb = pointlist[indices[0]][1]
                    smallb = pointlist[indices[1]][1]
                    bdifference = bigb - smallb
                else:
                    bigb = pointlist[indices[1]][1]
                    smallb = pointlist[indices[0]][1]
                    bdifference = bigb - smallb


            elif len(indices) == 1:
                smalla = 0
                biga = 60
                smallb = 0
                bigb = 60

            term3 = np.sqrt((x_clone[0, i] - x_clone[0, i])**2 + (x_clone[1, i] - x_clone[1, i])**2) - ci - risk[i]
            term4 = np.sqrt((x_clone[0, i] - x_clone[0, j])**2 + (x_clone[1, i] - x_clone[1, j])**2) - cj - risk[j]
            term5 = np.sqrt((x_clone[0, j] - x_clone[0, i])**2 + (x_clone[1, j] - x_clone[1, i])**2) - ci - risk[i]
            term6 = np.sqrt((x_clone[0, j] - x_clone[0, j])**2 + (x_clone[1, j] - x_clone[1, j])**2) - cj - risk[j]

            termlistcar.append(term3 - term4)
            termlistcar2.append(term5 - term6)

            if len(indices) == 2: # 여기가 문제였음 >= 로 바꾸면 점이 촘촘해짐
                for k in range(100):

                    steplength = abs(120 - 0) / 100
                    step_a = 0 + steplength * k
                    def equation5(b):
                        term1 = np.sqrt((step_a- x_clone[0, i])**2 + (b - x_clone[1, i])**2) - ci - risk[i]
                        term2 = np.sqrt((step_a- x_clone[0, j])**2 + (b - x_clone[1, j])**2) - cj - risk[j]

                        return term1 - term2
                    try:
                        b_solution = brentq(equation5, -3, 123)
                        pointlist_r.append((step_a, b_solution))
                    except ValueError:
                        # 해가 없는 경우 예외 처리
                        continue

                    #pointlist_r.append((step_a, b_solution))
        
                for k in range(100):

                    steplength = abs(120 - 0) / 100
                    step_b = 0 + (steplength * k)
                    def equation6(a):
                        term1 = np.sqrt((a- x_clone[0, i])**2 + (step_b - x_clone[1, i])**2) - ci - risk[i]
                        term2 = np.sqrt((a- x_clone[0, j])**2 + (step_b - x_clone[1, j])**2) - cj - risk[j]
                        return term1 - term2

                    try:
                        # 해를 찾을 구간을 설정합니다. 여기서는 domain_min과 domain_max로 설정합니다.
                        a_solution = brentq(equation6, -3, 123)
                        pointlist_r.append((a_solution, step_b))
                    except ValueError:
                        # 해가 없는 경우 예외 처리
                        continue

            elif len(indices) == 1:
                for k in range(100):

                    steplength = abs(120 - 0) / 100
                    step_a = 0+ steplength * k
                    def equation5(b):
                        term1 = np.sqrt((step_a- x_clone[0, i])**2 + (b - x_clone[1, i])**2) - ci - risk[i]
                        term2 = np.sqrt((step_a- x_clone[0, j])**2 + (b - x_clone[1, j])**2) - cj - risk[j]

                        return term1 - term2

                    try:
                        b_solution = brentq(equation5, -3, 123)
                        pointlist_r.append((step_a, b_solution))
                    except ValueError:
                        # 해가 없는 경우 예외 처리
                        continue

                for k in range(100):
                    steplength = abs(120 - 0) / 100
                    step_a = 120 - steplength * k
                    def equation5(b):
                        term1 = np.sqrt((step_a- x_clone[0, i])**2 + (b - x_clone[1, i])**2) - ci - risk[i]
                        term2 = np.sqrt((step_a- x_clone[0, j])**2 + (b - x_clone[1, j])**2) - cj - risk[j]

                        return term1 - term2

                    try:
                        b_solution = brentq(equation5, -3, 123)
                        pointlist_r.append((step_a, b_solution))
                    except ValueError:
                        # 해가 없는 경우 예외 처리
                        continue

                for k in range(100):

                    steplength = abs(120 - 0) / 100
                    step_b = 0 + (steplength * k)
                    def equation6(a):
                        term1 = np.sqrt((a- x_clone[0, i])**2 + (step_b - x_clone[1, i])**2) - ci - risk[i]
                        term2 = np.sqrt((a- x_clone[0, j])**2 + (step_b - x_clone[1, j])**2) - cj - risk[j]
                        return term1 - term2

                    try:
                        # 해를 찾을 구간을 설정합니다. 여기서는 domain_min과 domain_max로 설정합니다.
                        a_solution = brentq(equation6, -3, 123)
                        pointlist_r.append((a_solution, step_b))
                    except ValueError:
                        # 해가 없는 경우 예외 처리
                        continue

                for k in range(100):

                    steplength = abs(120 - 0) / 100
                    step_b = 120 - (steplength * k)
                    def equation6(a):
                        term1 = np.sqrt((a- x_clone[0, i])**2 + (step_b - x_clone[1, i])**2) - ci - risk[i]
                        term2 = np.sqrt((a- x_clone[0, j])**2 + (step_b - x_clone[1, j])**2) - cj - risk[j]
                        return term1 - term2

                    try:
                        # 해를 찾을 구간을 설정합니다. 여기서는 domain_min과 domain_max로 설정합니다.
                        a_solution = brentq(equation6, -3, 123)
                        pointlist_r.append((a_solution, step_b))
                    except ValueError:
                        # 해가 없는 경우 예외 처리
                        continue





    print(len(pointlist_r))
    for i in range(N):
        i = 0
        todeleteindex = []
        for j in range(N):
            if i != j:

                phi = 2
                v_i = np.sqrt(clist[i].velocity.x**2 + clist[i].velocity.y**2)
                v_j = np.sqrt(clist[j].velocity.x**2 + clist[j].velocity.y**2)
                ci  = phi * v_i + v_i + phi * u_ref[i]
                cj = phi * v_j + v_j + phi * u_ref[j]
                term3 = np.sqrt((x_clone[0, i] - x_clone[0, i])**2 + (x_clone[1, i] - x_clone[1, i])**2) - ci - risk[i]
                term4 = np.sqrt((x_clone[0, i] - x_clone[0, j])**2 + (x_clone[1, i] - x_clone[1, j])**2) - cj - risk[j]

                if (term3 - term4) > 0:
                    if i == 1 and j == 2:

                        pass
                    for v in range(len(pointlist_r)):
                        term3 = np.sqrt((pointlist_r[v][0] - x_clone[0, i])**2 + (pointlist_r[v][1] - x_clone[1, i])**2) - ci - risk[i]
                        term4 = np.sqrt((pointlist_r[v][0] - x_clone[0, j])**2 + (pointlist_r[v][1] - x_clone[1, j])**2) - cj - risk[j]
                        if term3 - term4 > 0.0001:
                            todeleteindex.append(v)

                elif (term3 - term4) < 0:
                    if i == 1 and j == 2:

                        pass
                    for v in range(len(pointlist_r)):
                        term3 = np.sqrt((pointlist_r[v][0] - x_clone[0, i])**2 + (pointlist_r[v][1] - x_clone[1, i])**2) - ci - risk[i]
                        term4 = np.sqrt((pointlist_r[v][0] - x_clone[0, j])**2 + (pointlist_r[v][1] - x_clone[1, j])**2) - cj - risk[j]
                        if term3 - term4 < -0.0001:
                            todeleteindex.append(v)

            
            element_counts = Counter(todeleteindex)
            duplicates = {element: count for element, count in element_counts.items() if count >= N-1}
            sorted_list = sorted(duplicates, reverse=True)
            
            for index in sorted_list:
                #print(index)
                pointlist_r.pop(index)
        todeleteindex = []


    
    for i in range(1):
        i = 1
        todeleteindex = []
        for j in range(N):
            if i != j:

                phi = 2  # Example value
                v_i = np.sqrt(clist[i].velocity.x**2 + clist[i].velocity.y**2)
                v_j = np.sqrt(clist[j].velocity.x**2 + clist[j].velocity.y**2)
                ci  = phi * v_i + v_i + phi * u_ref[i]
                cj = phi * v_j + v_j + phi * u_ref[j]
                term5 = np.sqrt((x_clone[0, i] - x_clone[0, i])**2 + (x_clone[1, i] - x_clone[1, i])**2) - ci - risk[i]
                term6 = np.sqrt((x_clone[0, i] - x_clone[0, j])**2 + (x_clone[1, i] - x_clone[1, j])**2) - cj - risk[j]
                if (term5 - term6) > 0:
                    if i == 1 and j == 0:
                        pass
                    for v in range(len(pointlist_r)):
                        term5 = np.sqrt((pointlist_r[v][0] - x_clone[0, i])**2 + (pointlist_r[v][1] - x_clone[1, i])**2) - ci - risk[i]
                        term6 = np.sqrt((pointlist_r[v][0] - x_clone[0, j])**2 + (pointlist_r[v][1] - x_clone[1, j])**2) - cj - risk[j]
                        if term5 - term6 > 0.0001:
                            todeleteindex.append(v)

                elif (term5 - term6) < 0:
                    if i == 1 and j == 0:
                        pass
                    for v in range(len(pointlist_r)):
                        term5 = np.sqrt((pointlist_r[v][0] - x_clone[0, i])**2 + (pointlist_r[v][1] - x_clone[1, i])**2) - ci - risk[i]
                        term6 = np.sqrt((pointlist_r[v][0] - x_clone[0, j])**2 + (pointlist_r[v][1] - x_clone[1, j])**2) - cj - risk[j]
                        if term5 - term6 < -0.0001:
                            todeleteindex.append(v)

                element_counts = Counter(todeleteindex)
                duplicates = {element: count for element, count in element_counts.items() if count >= N-1}
                sorted_list = sorted(duplicates, reverse=True)
                
                for index in sorted_list:

                    pointlist_r.pop(index)   
                    #pass
        todeleteindex = []
    
    for i in range(1):
        i = 2
        todeleteindex = []
        for j in range(N):
            if i != j:

                phi = 2  # Example value
                v_i = np.sqrt(clist[i].velocity.x**2 + clist[i].velocity.y**2)
                v_j = np.sqrt(clist[j].velocity.x**2 + clist[j].velocity.y**2)
                ci  = phi * v_i + v_i + phi * u_ref[i]
                cj = phi * v_j + v_j + phi * u_ref[j]
                term5 = np.sqrt((x_clone[0, i] - x_clone[0, i])**2 + (x_clone[1, i] - x_clone[1, i])**2) - ci - risk[i]
                term6 = np.sqrt((x_clone[0, i] - x_clone[0, j])**2 + (x_clone[1, i] - x_clone[1, j])**2) - cj - risk[j]
                if (term5 - term6) > 0:
                    if i == 1 and j == 0:
                        pass
                    for v in range(len(pointlist_r)):
                        term5 = np.sqrt((pointlist_r[v][0] - x_clone[0, i])**2 + (pointlist_r[v][1] - x_clone[1, i])**2) - ci - risk[i]
                        term6 = np.sqrt((pointlist_r[v][0] - x_clone[0, j])**2 + (pointlist_r[v][1] - x_clone[1, j])**2) - cj - risk[j]
                        if term5 - term6 > 0.0001:
                            todeleteindex.append(v)

                elif (term5 - term6) < 0:
                    if i == 1 and j == 0:
                        pass
                    for v in range(len(pointlist_r)):
                        term5 = np.sqrt((pointlist_r[v][0] - x_clone[0, i])**2 + (pointlist_r[v][1] - x_clone[1, i])**2) - ci - risk[i]
                        term6 = np.sqrt((pointlist_r[v][0] - x_clone[0, j])**2 + (pointlist_r[v][1] - x_clone[1, j])**2) - cj - risk[j]
                        if term5 - term6 < -0.0001:
                            todeleteindex.append(v)

                element_counts = Counter(todeleteindex)
                duplicates = {element: count for element, count in element_counts.items() if count >= N-1}
                sorted_list = sorted(duplicates, reverse=True)
                
                for index in sorted_list:

                    pointlist_r.pop(index)   
                    #pass
        todeleteindex = []

    for i in range(1):
        i = 3
        todeleteindex = []
        for j in range(N):
            if i != j:

                phi = 2  # Example value
                v_i = np.sqrt(clist[i].velocity.x**2 + clist[i].velocity.y**2)
                v_j = np.sqrt(clist[j].velocity.x**2 + clist[j].velocity.y**2)
                ci  = phi * v_i + v_i + phi * u_ref[i]
                cj = phi * v_j + v_j + phi * u_ref[j]
                term5 = np.sqrt((x_clone[0, i] - x_clone[0, i])**2 + (x_clone[1, i] - x_clone[1, i])**2) - ci - risk[i]
                term6 = np.sqrt((x_clone[0, i] - x_clone[0, j])**2 + (x_clone[1, i] - x_clone[1, j])**2) - cj - risk[j]
                if (term5 - term6) > 0:
                    if i == 1 and j == 0:
                        pass
                    for v in range(len(pointlist_r)):
                        term5 = np.sqrt((pointlist_r[v][0] - x_clone[0, i])**2 + (pointlist_r[v][1] - x_clone[1, i])**2) - ci - risk[i]
                        term6 = np.sqrt((pointlist_r[v][0] - x_clone[0, j])**2 + (pointlist_r[v][1] - x_clone[1, j])**2) - cj - risk[j]
                        if term5 - term6 > 0.0001:
                            todeleteindex.append(v)

                elif (term5 - term6) < 0:
                    if i == 1 and j == 0:
                        pass
                    for v in range(len(pointlist_r)):
                        term5 = np.sqrt((pointlist_r[v][0] - x_clone[0, i])**2 + (pointlist_r[v][1] - x_clone[1, i])**2) - ci - risk[i]
                        term6 = np.sqrt((pointlist_r[v][0] - x_clone[0, j])**2 + (pointlist_r[v][1] - x_clone[1, j])**2) - cj - risk[j]
                        if term5 - term6 < -0.0001:
                            todeleteindex.append(v)

                element_counts = Counter(todeleteindex)
                duplicates = {element: count for element, count in element_counts.items() if count >= N-1}
                sorted_list = sorted(duplicates, reverse=True)
                
                for index in sorted_list:

                    pointlist_r.pop(index)   
                    #pass
        todeleteindex = []




    pointlist_r_clone = pointlist_r.copy()

    



def wvcontrolr2l(pl, caridx):
    closest_point = None
    min_distance = float('inf')
    
    for v in range(len(pl)):
        x_val = pl[v][0]
        y_val = pl[v][1]
        
        if x_val < x[0, caridx] and 65 <= y_val <= 71:
            distance = abs(x[0, caridx] - x_val)
            if distance < min_distance:
                min_distance = distance
                closest_point = pl[v]
    

    return closest_point

def wvcontrolb2u(pl, caridx):
    closest_point = None
    min_distance = float('inf')
    
    for v in range(len(pl)):
        x_val = pl[v][0]
        y_val = pl[v][1]
        
        if y_val > x[1, caridx] and 65 <= x_val <= 71:
            distance = abs(x[1, caridx] - y_val)
            if distance < min_distance:
                min_distance = distance
                closest_point = pl[v]
    
    if closest_point == None:
        print("NONE!")
        pass
    else:
        #c6.center.x = closest_point[0]
        #c6.center.y = closest_point[1]
        pass
    return closest_point

def wvcontrolu2d(pl, caridx):
    closest_point = None
    min_distance = float('inf')
    
    for v in range(len(pl)):
        x_val = pl[v][0]
        y_val = pl[v][1]
        
        if y_val < x[1, caridx] and 48 <= x_val <= 54:
            distance = abs(x[1, caridx] - y_val)
            if distance < min_distance:
                min_distance = distance
                closest_point = pl[v]
    
    return closest_point

def wvcontroll2r(pl, caridx):
    closest_point = None
    min_distance = float('inf')
    
    for v in range(len(pl)):
        x_val = pl[v][0]
        y_val = pl[v][1]
        
        if x_val > x[0, caridx] and 56 <= y_val <= 60:
            distance = abs(x[0, caridx] - x_val)
            if distance < min_distance:
                min_distance = distance
                closest_point = pl[v]
    

    return closest_point






pid_controller = PIDController(kp=1.0, ki=0.5, kd=0.1, target_velocity=0.0)
pid_controller2 = PIDController(kp=1.0, ki=0.5, kd=0.1, target_velocity = 0.0)
pid_controller3 = PIDController(kp=1.0, ki=0.5, kd=0.1, target_velocity= 0.0)
#pid_controller4 = PIDController(kp=1.0, ki=0.5, kd=0.1, target_velocity=4.0)
#pid_controller5 = PIDController(kp=1.0, ki=0.5, kd=0.1, target_velocity=4.0)
#############################################################################


if not human_controller:
    time_steps = int(400 * dt)  # Adjust for correct time steps based on dt
    time_passed = 0.0
    current_milestone_index = 0


    while time_passed < 800.0:  # Run simulation for 40 seconds
        # Get current velocity and calculate throttle

            #print("hello")


        #u_ref = np.array([throttle1 ,throttle2, throttle3, throttle4,throttle5])
        #u_ref = np.array([throttle1 ,throttle2, throttle3])
        global u_ref
        u_ref = np.array([0,0,0,0])
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

        #th = cbf(x, u_ref)

        # Calculate Voronoi edges
        #car_positions = np.array([[car.center.x, car.center.y] for car in clist])
        #draw_voronoi_edges(car_positions, w)

        # Calculate steering angle for c1 to follow the path




        if time_passed < 710:

            pointlist_r = []
            pointlist_r_clone = []
            for v in range(3000):
                plist[v].center = Point(-1, -1)

            wv()
            cp1 = wvcontrolu2d(pointlist_r, 0)
            if cp1 == None:
                gp1 = 40
            else:
                gp1 = np.linalg.norm( math.sqrt((x[0,0]-cp1[0])**2 + (x[1,0]    -cp1[1])**2 ))
            k1 = 0.1

            cp2 = wvcontrolr2l(pointlist_r, 1)
            if cp2 == None:
                gp2 = 40
            else:
                gp2 = np.linalg.norm( math.sqrt((x[0,1]  -cp2[0])**2 + (x[1,1]    -cp2[1])**2 ))
                #np.linalg.norm(x[:,1] - np.array([cp2[0], cp2[1]]))
            k2 = 0.1

            cp3 = wvcontrolb2u(pointlist_r, 2)
            if cp3 == None:
                gp3 = 40
            else:
                gp3 = np.linalg.norm( math.sqrt((x[0,2]-cp3[0])**2 + (x[1,2]   -cp3[1])**2 ))

            k3 = 0.1
            pid_controller = PIDController(kp=1.0, ki=0.5, kd=0.1, target_velocity=k1 * gp1)
            velocity_magnitude = np.sqrt(c1.velocity.x**2 + c1.velocity.y**2)
            acceleration = pid_controller.control(velocity_magnitude, dt)
            throttle1 = acceleration

            pid_controller2 = PIDController(kp=1.0, ki=0.5, kd=0.1, target_velocity=k2 * gp2)
            velocity_magnitude = np.sqrt(c2.velocity.x**2 + c2.velocity.y**2)
            acceleration = pid_controller2.control(velocity_magnitude, dt)
            throttle2 = acceleration

            pid_controller3 = PIDController(kp=1.0, ki=0.5, kd=0.1, target_velocity=4)
            velocity_magnitude = np.sqrt(c3.velocity.x**2 + c3.velocity.y**2)
            acceleration = pid_controller3.control(velocity_magnitude, dt)
            throttle3 = acceleration
            u_ref = np.array([throttle1 ,throttle2, throttle3, 0])
            print(u_ref)

            c1.set_control(0, u_ref[0])
            c2.set_control(0, u_ref[1])
            c3.set_control(0, u_ref[2])
            
            for v in range(len(pointlist_r_clone)):

                plist[v].center = Point(pointlist_r_clone[v][0], pointlist_r_clone[v][1])





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
                #clist[i].color = 'white'

    w.close()
