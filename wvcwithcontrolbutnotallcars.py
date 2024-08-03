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
cvxopt.solvers.options['show_progress'] = False
N = 5
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
c1 = Car(Point(40,68), np.pi, 'black')
c1.velocity = Point(0,0)
w.add(c1)
current_heading = c1.heading
#print("Current heading:", current_heading)

c2 = Car(Point(90,68), np.pi, 'blue')
c2.velocity = Point(0,0) 
w.add(c2)

c3 = Car(Point(67, 20), np.pi/2, 'green')
c3.velocity = Point(0,0) 
w.add(c3)

c4 = Car(Point(54, 90), -np.pi/2, 'red')
c4.velocity = Point(0,0) 
w.add(c4)

c5 = Car(Point(0, 58), 0, 'yellow')
c5.velocity = Point(0,0) 

w.add(c5)

c6 = Car(Point(0, 0), 0, 'pink')
c6.velocity = Point(0,0) 

w.add(c6)

clist = []
clist.append(c1)
clist.append(c2)

clist.append(c3)
clist.append(c4)
clist.append(c5)
lane_list = ['rd', 'r', 'd', 'u', 'u']


#########################################################################
import numpy as np
import matplotlib.pyplot as plt

prange = 0
# Generate random points
plist = []
for i in range(3000):
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
    velocitylist = [] #try this~~~
    #u_ref = np.array([0 ,0, 0])
    for i in range(N-1):
        for j in range(i+1, N):
            #print(i, j)
            global prange

            phi = 1  # Example value

            v_i = np.sqrt(clist[i].velocity.x**2 + clist[i].velocity.y**2)
            v_j = np.sqrt(clist[j].velocity.x**2 + clist[j].velocity.y**2)
            ci  = phi * v_i + v_i + phi * u_ref[i]
            cj = phi * v_j + v_j + phi * u_ref[j]

            def equation1(b):
                term1 = np.sqrt((- x[0, i])**2 + (b - x[1, i])**2) - ci
                term2 = np.sqrt((- x[0, j])**2 + (b - x[1, j])**2) - cj
                return term1 - term2

            pointlist = []
            b_initial_guess = 60.0
            b_solution, = fsolve(equation1, b_initial_guess)
            #print(b_solution)
            pointlist.append((0.0, b_solution))
            def equation2(b):
                term1 = np.sqrt((120- x[0, i])**2 + (b - x[1, i])**2) - ci
                term2 = np.sqrt((120- x[0, j])**2 + (b - x[1, j])**2) - cj
                return term1 - term2

            b_initial_guess = 60.0
            b_solution2, = fsolve(equation2, b_initial_guess)
            #print(b_solution2)
            pointlist.append((120.0, b_solution2))

            def equation3(a):
                term1 = np.sqrt((a- x[0, i])**2 + (0 - x[1, i])**2) - ci
                term2 = np.sqrt((a- x[0, j])**2 + (0 - x[1, j])**2) - cj
                return term1 - term2
            a_initial_guess = 60.0
            a_solution, = fsolve(equation3, a_initial_guess)
            #print(a_solution)
            pointlist.append((a_solution, 0.0))
            def equation4(a):
                term1 = np.sqrt((a- x[0, i])**2 + (120 - x[1, i])**2) - ci
                term2 = np.sqrt((a- x[0, j])**2 + (120 - x[1, j])**2) - cj
                return term1 - term2
            a_initial_guess = 60.0
            a_solution2, = fsolve(equation4, a_initial_guess)
            pointlist.append((a_solution2,120.0))
            #print(a_solution2)
            #print(pointlist)
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

            #this is to see if the graph will longer in x axis or y axis -> to decide which axis to use to draw dots following the line 
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
                biga = 120
                smallb = 0
                bigb = 120

            term3 = np.sqrt((x[0, i] - x[0, i])**2 + (x[1, i] - x[1, i])**2) - ci
            term4 = np.sqrt((x[0, i] - x[0, j])**2 + (x[1, i] - x[1, j])**2) - cj
            term5 = np.sqrt((x[0, j] - x[0, i])**2 + (x[1, j] - x[1, i])**2) - ci
            term6 = np.sqrt((x[0, j] - x[0, j])**2 + (x[1, j] - x[1, j])**2) - cj

            termlistcar.append(term3 - term4)
            termlistcar2.append(term5 - term6)

            if adifference > bdifference:
                for k in range(100):
                    steplength = abs(123 - (-3)) / 100
                    step_a = 0 + steplength * k

                    def equation5(b):
                        term1 = np.sqrt((step_a - x[0, i])**2 + (b - x[1, i])**2) - ci
                        term2 = np.sqrt((step_a - x[0, j])**2 + (b - x[1, j])**2) - cj
                        return term1 - term2

                    try:
                        b_solution = brentq(equation5, -3, 123)
                        pointlist_r.append((step_a, b_solution))
                    except ValueError:
                        # 해가 없는 경우 예외 처리
                        continue
            else:
                for k in range(100):
                    steplength = abs(123 - (-3)) / 100
                    step_b = 0 + (steplength * k)

                    def equation6(a):
                        term1 = np.sqrt((a - x[0, i])**2 + (step_b - x[1, i])**2) - ci
                        term2 = np.sqrt((a - x[0, j])**2 + (step_b - x[1, j])**2) - cj
                        return term1 - term2

                    try:
                        # 해를 찾을 구간을 설정합니다. 여기서는 domain_min과 domain_max로 설정합니다.
                        a_solution = brentq(equation6, -3, 123)
                        pointlist_r.append((a_solution, step_b))
                    except ValueError:
                        # 해가 없는 경우 예외 처리
                        continue

    for i in range(1):
        i = 0
        todeleteindex = []
        for j in range(N):
            if i != j:
                #print(i, j)
                phi = 1  # Example value
                v_i = np.sqrt(clist[i].velocity.x**2 + clist[i].velocity.y**2)
                v_j = np.sqrt(clist[j].velocity.x**2 + clist[j].velocity.y**2)
                ci  = phi * v_i + v_i + phi * u_ref[i]
                cj = phi * v_j + v_j + phi * u_ref[j]
                term3 = np.sqrt((x[0, i] - x[0, i])**2 + (x[1, i] - x[1, i])**2) - ci
                term4 = np.sqrt((x[0, i] - x[0, j])**2 + (x[1, i] - x[1, j])**2) - cj
                term5 = np.sqrt((x[0, j] - x[0, i])**2 + (x[1, j] - x[1, i])**2) - ci
                term6 = np.sqrt((x[0, j] - x[0, j])**2 + (x[1, j] - x[1, j])**2) - cj

                #print("i and j", i , j)
                if (term3 - term4) > 0:
                    if i == 1 and j == 2:
                        #print("12 if")
                        pass
                    for v in range(len(pointlist_r)):
                        term3 = np.sqrt((pointlist_r[v][0] - x[0, i])**2 + (pointlist_r[v][1] - x[1, i])**2) - ci
                        term4 = np.sqrt((pointlist_r[v][0] - x[0, j])**2 + (pointlist_r[v][1] - x[1, j])**2) - cj
                        if term3 - term4 > 0.001:
                            todeleteindex.append(v)

                elif (term3 - term4) < 0:
                    if i == 1 and j == 2:
                        #print("12 elif")
                        pass
                    #print(len(pointlist_r))
                    for v in range(len(pointlist_r)):
                        term3 = np.sqrt((pointlist_r[v][0] - x[0, i])**2 + (pointlist_r[v][1] - x[1, i])**2) - ci
                        term4 = np.sqrt((pointlist_r[v][0] - x[0, j])**2 + (pointlist_r[v][1] - x[1, j])**2) - cj
                        if term3 - term4 < -0.001:
                            todeleteindex.append(v)

    element_counts = Counter(todeleteindex)
    duplicates = {element: count for element, count in element_counts.items() if count >= N-1}
    sorted_list = sorted(duplicates, reverse=True)

    for index in sorted_list:
        #print("pointlist_r length", len(pointlist_r))
        #print("sorted list length", len(sorted_list))
        pointlist_r.pop(index)
    todeleteindex = []


    for i in range(1):
        i = 1
        todeleteindex = []
        for j in range(N):
            if i != j:
                #print(i, j)
                phi = 1  # Example value
                v_i = np.sqrt(clist[i].velocity.x**2 + clist[i].velocity.y**2)
                v_j = np.sqrt(clist[j].velocity.x**2 + clist[j].velocity.y**2)
                ci  = phi * v_i + v_i + phi * u_ref[i]
                cj = phi * v_j + v_j + phi * u_ref[j]
                term3 = np.sqrt((x[0, i] - x[0, i])**2 + (x[1, i] - x[1, i])**2) - ci
                term4 = np.sqrt((x[0, i] - x[0, j])**2 + (x[1, i] - x[1, j])**2) - cj
                term5 = np.sqrt((x[0, j] - x[0, i])**2 + (x[1, j] - x[1, i])**2) - ci
                term6 = np.sqrt((x[0, j] - x[0, j])**2 + (x[1, j] - x[1, j])**2) - cj

                #print("i and j", i , j)
                if (term3 - term4) > 0:
                    if i == 1 and j == 2:
                        #print("12 if")
                        pass
                    for v in range(len(pointlist_r)):
                        term3 = np.sqrt((pointlist_r[v][0] - x[0, i])**2 + (pointlist_r[v][1] - x[1, i])**2) - ci
                        term4 = np.sqrt((pointlist_r[v][0] - x[0, j])**2 + (pointlist_r[v][1] - x[1, j])**2) - cj
                        if term3 - term4 > 0.001:
                            todeleteindex.append(v)

                elif (term3 - term4) < 0:
                    if i == 1 and j == 2:
                        #print("12 elif")
                        pass
                    #print(len(pointlist_r))
                    for v in range(len(pointlist_r)):
                        term3 = np.sqrt((pointlist_r[v][0] - x[0, i])**2 + (pointlist_r[v][1] - x[1, i])**2) - ci
                        term4 = np.sqrt((pointlist_r[v][0] - x[0, j])**2 + (pointlist_r[v][1] - x[1, j])**2) - cj
                        if term3 - term4 < -0.001:
                            todeleteindex.append(v)

    element_counts = Counter(todeleteindex)
    duplicates = {element: count for element, count in element_counts.items() if count >= N-1}
    sorted_list = sorted(duplicates, reverse=True)

    for index in sorted_list:
        #print("pointlist_r length", len(pointlist_r))
        #print("sorted list length", len(sorted_list))
        pointlist_r.pop(index)
    todeleteindex = []


    for i in range(1):
        i = 2
        todeleteindex = []
        for j in range(N):
            if i != j:
                #print(i, j)
                phi = 1  # Example value
                v_i = np.sqrt(clist[i].velocity.x**2 + clist[i].velocity.y**2)
                v_j = np.sqrt(clist[j].velocity.x**2 + clist[j].velocity.y**2)
                ci  = phi * v_i + v_i + phi * u_ref[i]
                cj = phi * v_j + v_j + phi * u_ref[j]
                term3 = np.sqrt((x[0, i] - x[0, i])**2 + (x[1, i] - x[1, i])**2) - ci
                term4 = np.sqrt((x[0, i] - x[0, j])**2 + (x[1, i] - x[1, j])**2) - cj
                term5 = np.sqrt((x[0, j] - x[0, i])**2 + (x[1, j] - x[1, i])**2) - ci
                term6 = np.sqrt((x[0, j] - x[0, j])**2 + (x[1, j] - x[1, j])**2) - cj

                #print("i and j", i , j)
                if (term3 - term4) > 0:
                    if i == 1 and j == 2:
                        #print("12 if")
                        pass
                    for v in range(len(pointlist_r)):
                        term3 = np.sqrt((pointlist_r[v][0] - x[0, i])**2 + (pointlist_r[v][1] - x[1, i])**2) - ci
                        term4 = np.sqrt((pointlist_r[v][0] - x[0, j])**2 + (pointlist_r[v][1] - x[1, j])**2) - cj
                        if term3 - term4 > 0.001:
                            todeleteindex.append(v)

                elif (term3 - term4) < 0:
                    if i == 1 and j == 2:
                        #print("12 elif")
                        pass
                    #print(len(pointlist_r))
                    for v in range(len(pointlist_r)):
                        term3 = np.sqrt((pointlist_r[v][0] - x[0, i])**2 + (pointlist_r[v][1] - x[1, i])**2) - ci
                        term4 = np.sqrt((pointlist_r[v][0] - x[0, j])**2 + (pointlist_r[v][1] - x[1, j])**2) - cj
                        if term3 - term4 < -0.001:
                            todeleteindex.append(v)

    element_counts = Counter(todeleteindex)
    duplicates = {element: count for element, count in element_counts.items() if count >= N-1}
    sorted_list = sorted(duplicates, reverse=True)

    for index in sorted_list:
        #print("pointlist_r length", len(pointlist_r))
        #print("sorted list length", len(sorted_list))
        pointlist_r.pop(index)
    todeleteindex = []
               
    for i in range(1):
        i = 3
        todeleteindex = []
        for j in range(N):
            if i != j:
                #print(i, j)
                phi = 1  # Example value
                v_i = np.sqrt(clist[i].velocity.x**2 + clist[i].velocity.y**2)
                v_j = np.sqrt(clist[j].velocity.x**2 + clist[j].velocity.y**2)
                ci  = phi * v_i + v_i + phi * u_ref[i]
                cj = phi * v_j + v_j + phi * u_ref[j]
                term3 = np.sqrt((x[0, i] - x[0, i])**2 + (x[1, i] - x[1, i])**2) - ci
                term4 = np.sqrt((x[0, i] - x[0, j])**2 + (x[1, i] - x[1, j])**2) - cj
                term5 = np.sqrt((x[0, j] - x[0, i])**2 + (x[1, j] - x[1, i])**2) - ci
                term6 = np.sqrt((x[0, j] - x[0, j])**2 + (x[1, j] - x[1, j])**2) - cj

                #print("i and j", i , j)
                if (term3 - term4) > 0:
                    if i == 1 and j == 2:
                        #print("12 if")
                        pass
                    for v in range(len(pointlist_r)):
                        term3 = np.sqrt((pointlist_r[v][0] - x[0, i])**2 + (pointlist_r[v][1] - x[1, i])**2) - ci
                        term4 = np.sqrt((pointlist_r[v][0] - x[0, j])**2 + (pointlist_r[v][1] - x[1, j])**2) - cj
                        if term3 - term4 > 0.001:
                            todeleteindex.append(v)

                elif (term3 - term4) < 0:
                    if i == 1 and j == 2:
                        #print("12 elif")
                        pass
                    #print(len(pointlist_r))
                    for v in range(len(pointlist_r)):
                        term3 = np.sqrt((pointlist_r[v][0] - x[0, i])**2 + (pointlist_r[v][1] - x[1, i])**2) - ci
                        term4 = np.sqrt((pointlist_r[v][0] - x[0, j])**2 + (pointlist_r[v][1] - x[1, j])**2) - cj
                        if term3 - term4 < -0.001:
                            todeleteindex.append(v)

    element_counts = Counter(todeleteindex)
    duplicates = {element: count for element, count in element_counts.items() if count >= N-1}
    sorted_list = sorted(duplicates, reverse=True)

    for index in sorted_list:
        #print("pointlist_r length", len(pointlist_r))
        #print("sorted list length", len(sorted_list))
        pointlist_r.pop(index)
    todeleteindex = []

    for i in range(1):
        i = 4
        todeleteindex = []
        for j in range(N):
            if i != j:
                #print(i, j)
                phi = 1  # Example value
                v_i = np.sqrt(clist[i].velocity.x**2 + clist[i].velocity.y**2)
                v_j = np.sqrt(clist[j].velocity.x**2 + clist[j].velocity.y**2)
                ci  = phi * v_i + v_i + phi * u_ref[i]
                cj = phi * v_j + v_j + phi * u_ref[j]
                term3 = np.sqrt((x[0, i] - x[0, i])**2 + (x[1, i] - x[1, i])**2) - ci
                term4 = np.sqrt((x[0, i] - x[0, j])**2 + (x[1, i] - x[1, j])**2) - cj
                term5 = np.sqrt((x[0, j] - x[0, i])**2 + (x[1, j] - x[1, i])**2) - ci
                term6 = np.sqrt((x[0, j] - x[0, j])**2 + (x[1, j] - x[1, j])**2) - cj

                #print("i and j", i , j)
                if (term3 - term4) > 0:
                    if i == 1 and j == 2:
                        #print("12 if")
                        pass
                    for v in range(len(pointlist_r)):
                        term3 = np.sqrt((pointlist_r[v][0] - x[0, i])**2 + (pointlist_r[v][1] - x[1, i])**2) - ci
                        term4 = np.sqrt((pointlist_r[v][0] - x[0, j])**2 + (pointlist_r[v][1] - x[1, j])**2) - cj
                        if term3 - term4 > 0.01:
                            todeleteindex.append(v)

                elif (term3 - term4) < 0:
                    if i == 1 and j == 2:
                        #print("12 elif")
                        pass
                    #print(len(pointlist_r))
                    for v in range(len(pointlist_r)):
                        term3 = np.sqrt((pointlist_r[v][0] - x[0, i])**2 + (pointlist_r[v][1] - x[1, i])**2) - ci
                        term4 = np.sqrt((pointlist_r[v][0] - x[0, j])**2 + (pointlist_r[v][1] - x[1, j])**2) - cj
                        if term3 - term4 < -0.01:
                            todeleteindex.append(v)

    element_counts = Counter(todeleteindex)
    duplicates = {element: count for element, count in element_counts.items() if count >= N-1}
    sorted_list = sorted(duplicates, reverse=True)

    for index in sorted_list:
        #print("pointlist_r length", len(pointlist_r))
        #print("sorted list length", len(sorted_list))
        pointlist_r.pop(index)
    todeleteindex = []
    
    return pointlist_r

def wvcontrol2(pl, caridx):
    closest_point = None
    min_distance = float('inf')
    
    for v in range(len(pl)):
        x_val = pl[v][0]
        y_val = pl[v][1]
        
        if x_val < x[0, caridx] and 67 <= y_val <= 69:
            distance = abs(x[0, caridx] - x_val)
            if distance < min_distance:
                min_distance = distance
                closest_point = pl[v]
    

    return closest_point

def wvcontrol3(pl, caridx):
    closest_point = None
    min_distance = float('inf')
    
    for v in range(len(pl)):
        x_val = pl[v][0]
        y_val = pl[v][1]
        
        if y_val > x[1, caridx] and 65 <= x_val <= 69:
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

def wvcontrol4(pl, caridx):
    closest_point = None
    min_distance = float('inf')
    
    for v in range(len(pl)):
        x_val = pl[v][0]
        y_val = pl[v][1]
        
        if y_val < x[1, caridx] and 52 <= x_val <= 56:
            distance = abs(x[1, caridx] - y_val)
            if distance < min_distance:
                min_distance = distance
                closest_point = pl[v]
    
    return closest_point




def calculate_steering_angle(car, target_point):
    car_to_target = Point(target_point[0] - car.center.x, target_point[1] - car.center.y)
    angle_to_target = np.arctan2(car_to_target.y, car_to_target.x)
    steering_angle = angle_to_target - car.heading
    return steering_angle

pid_controller = PIDController(kp=1.0, ki=0.5, kd=0.1, target_velocity=6.0)
pid_controller2 = PIDController(kp=1.0, ki=0.5, kd=0.1, target_velocity = 6.0)
pid_controller3 = PIDController(kp=1.0, ki=0.5, kd=0.1, target_velocity= 5.0)
pid_controller4 = PIDController(kp=1.0, ki=0.5, kd=0.1, target_velocity=4.0)
pid_controller5 = PIDController(kp=1.0, ki=0.5, kd=0.1, target_velocity=0.0)
#############################################################################

if not human_controller:
    global u_ref
    time_steps = int(400 * dt)  # Adjust for correct time steps based on dt
    time_passed = 0.0
    current_milestone_index = 0

    firstflag = 0
    while time_passed < 800.0:  # Run simulation for 40 seconds
        # Get current velocity and calculate throttle
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
        if current_milestone_index < len(r_b_path):
            target_point = r_b_path[current_milestone_index]
            distance_to_target = np.sqrt((c1.center.x - target_point[0])**2 + (c1.center.y - target_point[1])**2)
            if distance_to_target < 1.0:  # If close enough to the target point, move to the next one
                current_milestone_index += 1
            steering_angle = calculate_steering_angle(c1, target_point)
        else:
            steering_angle = 0
        if current_milestone_index == len(r_b_path):
            current_milestone_index = 0



        if time_passed < 710:
            if firstflag == 0:
                firstflag += 1
                u_ref = np.array([0,0,0,0,0])
            pointlist_r = []

            for v in range(3000):
                plist[v].center = Point(-1, -1)

            wv()
            cp2 = wvcontrol2(pointlist_r, 1)
            if cp2 == None:
                gp2 = 40
            else:
                gp2 = np.linalg.norm(x[:,1] - np.array([cp2[0], cp2[1]]))- 4 #2 is car length /2
            k2 = 0.1
            #print(k * gp)
            cp3 = wvcontrol3(pointlist_r, 2)
            if cp3 == None:
                gp3 = 40
            else:
                gp3 = np.linalg.norm(x[:,3] - np.array([cp3[0], cp3[1]])) - 4#2 is car length /2
            k3 = 0.1
            #print(k3 * gp3)
            cp4 = wvcontrol4(pointlist_r, 3)
            if cp4 == None:
                gp4 = 40
                #print(k4 * gp4)
            else:
                gp4 = np.linalg.norm(x[:,4] - np.array([cp4[0], cp4[1]])) - 4#2 is car length /2
            k4 = 0.1
            print(k4 * gp4)
            
            velocity_magnitude = np.sqrt(c1.velocity.x**2 + c1.velocity.y**2)
            acceleration = pid_controller.control(velocity_magnitude, dt)
            throttle1 = acceleration

            pid_controller2 = PIDController(kp=1.0, ki=0.5, kd=0.1, target_velocity = k2 * gp2)

            velocity_magnitude = np.sqrt(c2.velocity.x**2 + c2.velocity.y**2)
            acceleration = pid_controller2.control(velocity_magnitude, dt)
            throttle2 = acceleration
            
            pid_controller3 = PIDController(kp=1.0, ki=0.5, kd=0.1, target_velocity = k3 * gp3)
           
            velocity_magnitude = np.sqrt(c3.velocity.x**2 + c3.velocity.y**2)
            acceleration = pid_controller3.control(velocity_magnitude, dt)
            throttle3 = acceleration

            pid_controller4 = PIDController(kp=1.0, ki=0.5, kd=0.1, target_velocity = k4 * gp4)
           
            velocity_magnitude = np.sqrt(c4.velocity.x**2 + c4.velocity.y**2)
            acceleration = pid_controller4.control(velocity_magnitude, dt)
            throttle4 = acceleration

            velocity_magnitude = np.sqrt(c5.velocity.x**2 + c5.velocity.y**2)
            acceleration = pid_controller5.control(velocity_magnitude, dt)
            throttle5 = acceleration
            u_ref = np.array([throttle1 ,throttle2, throttle3, throttle4,throttle5])

            c1.set_control(steering_angle, u_ref[0])
            c2.set_control(0, u_ref[1])
            c3.set_control(0, u_ref[2])
            c4.set_control(0, u_ref[3])
            c5.set_control(0, u_ref[4])

            for v in range(len(pointlist_r)):

                plist[v].center = Point(pointlist_r[v][0], pointlist_r[v][1])


        for i in range(N):
            if x[0,i] >= 141:
                clist[i].center = Point(140, 68)

            
            elif x[0,i] < 0:
                clist[i].center = Point(130,68)
                clist[i].heading = np.pi

            elif x[1,i] < 0:
                clist[i].center = Point(54,120)


            elif x[1,i] >= 141:
                clist[i].center = Point(65, 0)
                clist[i].heading = np.pi/2


    w.close()
