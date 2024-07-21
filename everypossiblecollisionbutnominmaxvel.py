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

dt = 0.15 # time steps in terms of seconds. In other words, 1/dt is the FPS.
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

w.add(Painting(Point(56, 46), Point(1, 1), 'pink')) # down lane left part conflict point
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

# A Car object is a dynamic object -- it can move. We construct it using its center location and heading angle.
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

c4 = Car(Point(56, 100), -np.pi/2, 'red')
c4.velocity = Point(0,0) 
w.add(c4)

c5 = Car(Point(28, 58), 0, 'yellow')
c5.velocity = Point(0,0) 
w.add(c5)

clist = []
clist.append(c1)
clist.append(c2)
clist.append(c3)
clist.append(c4)
clist.append(c5)
#lane_list = ['fromrighttodown', 'fromrighttoleft', 'fromdowntoup', 'fromuptodown', 'fromlefttoright']
lane_list = ['rd','r','d','u','l']
lane_decision = ['fromdowntoup','fromdowntoright','fromdowntoleft',  
               'fromrighttoleft', 'fromrighttoup','fromrighttodown',
               'fromuptodown', 'fromuptoright', 'fromuptoleft',
               'fromlefttodown', 'fromlefttoright', 'fromlefttoup'
               ]
current_lane = []

w.render() # This visualizes the world we just constructed.
######################## for distance of curved path to conflict point
def find_closest_waypoint(position, waypoints):
    distances = [np.linalg.norm(np.array(position) - np.array(point)) for point in waypoints]
    closest_index = np.argmin(distances)
    return closest_index, distances[closest_index]

def calculate_path_distance(path, start_index, end_index):
    distance = 0.0
    step = 1 if start_index < end_index else -1
    for i in range(start_index, end_index, step):
        distance += np.linalg.norm(np.array(path[i]) - np.array(path[i + step]))
    return distance

def distance_along_path(car_position, path, cp):
    # Find the closest waypoint to the car's current position and the distance to it
    car_index, dist_to_closest_waypoint = find_closest_waypoint(car_position, path)
    # Find the index of the conflict point
    cp_index = find_closest_waypoint(cp, path)[0]
    
    # Calculate the distance along the path from the closest waypoint to the conflict point
    if car_index <= cp_index:
        path_distance = calculate_path_distance(path, car_index, cp_index)
    else:
        path_distance = calculate_path_distance(path, car_index, cp_index)
    
    # Total distance is the sum of the distance to the closest waypoint and the path distance
    total_distance = dist_to_closest_waypoint + path_distance
    
    return total_distance
##########################
def cbf(x, u_ref):
    N = 5

    H = sparse(matrix(2*np.identity(N)))
    f = -2*np.reshape(u_ref, N, order='F')
    num_constraints = 2 * int(comb(N, 2)) + 2 + 6 + 5
    A = np.zeros((num_constraints, N))
    b = np.zeros(num_constraints)
    phi = 0.1
    lambda_1 = 0.1
    lambda_2 = 0.1
    lambda_3 = 5
    lambda_4 = 0.2
    gamma = 10 #standstill_distance
    gamma2 = 10
    car_width_halved = 1.5
    count = 0
    F_rm_i = 0.001
    for i in range(N):
        for j in range(i+1, N):
            v_i = np.sqrt(clist[i].velocity.x**2 + clist[i].velocity.y**2)
            v_j = np.sqrt(clist[j].velocity.x**2 + clist[j].velocity.y**2)

            if lane_list[i] == lane_list[j]:
                constraint_value1 = 0


                if lane_list[i] == 'r': 
                    if (x[0,i] > x[0,j]):# and abs(x[1,i] - x[1,j]) < car_width_halved: #i and j are on 'r' and i is behind and i and j's y location difference is small
                        constraint_value1 = (1 / phi) * (lambda_3 * (np.linalg.norm(-x[:,j] + x[:,i]) - (gamma + phi * v_i)) + v_j - v_i) + F_rm_i
                        A[count, i] = 1.0
                        b[count] = constraint_value1
                        count += 1
                    elif (x[0,i] < x[0,j]):
                        constraint_value1 = (1 / phi) * (lambda_3 * (np.linalg.norm(-x[:,i] + x[:,j]) - (gamma + phi * v_j)) + v_i - v_j) + F_rm_i
                        A[count, j] = 1.0
                        b[count] = constraint_value1
                        count += 1

                if lane_list[i] == 'l': 
                    if (x[0,i] < x[0,j]):
                        constraint_value1 = (1 / phi) * (lambda_3 * (np.linalg.norm(x[:,j] - x[:,i]) - (gamma + phi * v_i)) + v_j - v_i) + F_rm_i
                        A[count, i] = 1.0
                        b[count] = constraint_value1
                        count += 1
                    elif (x[0,i] > x[0,j]):
                        constraint_value1 = (1 / phi) * (lambda_3 * (np.linalg.norm(x[:,i] - x[:,j]) - (gamma + phi * v_j)) + v_i - v_j) + F_rm_i
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

                elif lane_list[i] == 'u':
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

            elif lane_list[i] != lane_list[j]:
                if lane_list[i] == 'rd' and lane_list[j] == 'r':
                    if x[0,i] > 65 and (x[0,i] < x[0,j]): # rd is preceeding r and rd has not exited the safe point
                        constraint_value1 = (1 / phi) * (lambda_3 * (np.linalg.norm(-x[:,i] + x[:,j]) - (gamma + phi * v_j)) + v_i - v_j) + F_rm_i
                        A[count, j] = 1.0
                        b[count] = constraint_value1
                        count += 1
                elif lane_list[j] == 'rd' and lane_list[i] == 'r':
                    if x[0,j] > 65 and (x[0,i] > x[0,j]):
                        constraint_value1 = (1 / phi) * (lambda_3 * (np.linalg.norm(-x[:,j] + x[:,i]) - (gamma + phi * v_i)) + v_j - v_i) + F_rm_i
                        A[count, i] = 1.0
                        b[count] = constraint_value1
                        count += 1


                #55, 58
                elif lane_list[i] == 'rd' and lane_list[j] == 'u':
                    if (x[1,i] < 58 or x[1,j] < 58) and (x[1,i] < x[1,j]):
                        constraint_value1 = (1 / phi) * (lambda_3 * (np.linalg.norm(-x[:,i] + x[:,j]) - (gamma + phi * v_j)) + v_i - v_j) + F_rm_i
                        A[count, j] = 1.0
                        b[count] = constraint_value1
                        count += 1

                    elif (x[1,i] < 58 or x[1,j] < 58) and (x[1,i] >= x[1,j]):
                        constraint_value1 = (1 / phi) * (lambda_3 * (np.linalg.norm(-x[:,i] + x[:,j]) - (gamma + phi * v_j)) + v_i - v_j) + F_rm_i
                        A[count, i] = 1.0
                        b[count] = constraint_value1
                        count += 1

                elif lane_list[j] == 'rd' and lane_list[i] == 'u':
                    if (x[1,i] < 58 or x[1,j] < 58) and (x[1,i] < x[1,j]):
                        constraint_value1 = (1 / phi) * (lambda_3 * (np.linalg.norm(-x[:,i] + x[:,j]) - (gamma + phi * v_j)) + v_i - v_j) + F_rm_i
                        A[count, j] = 1.0
                        b[count] = constraint_value1
                        count += 1

                    elif (x[1,i] < 58 or x[1,j] < 58) and (x[1,i] >= x[1,j]):
                        constraint_value1 = (1 / phi) * (lambda_3 * (np.linalg.norm(-x[:,i] + x[:,j]) - (gamma + phi * v_j)) + v_i - v_j) + F_rm_i
                        A[count, i] = 1.0
                        b[count] = constraint_value1
                        count += 1

                elif lane_list[i] == 'r' and lane_list[j] == 'u':
                    
                    delta_i = 15
                    s_n_i =  np.linalg.norm(x[:,i] - np.array([55, 68]))
                    s_n_j = np.linalg.norm(x[:,j] - np.array([55, 68]))
                    if delta_i >= s_n_i  + s_n_j:

                        if x[0,i] > 55 and x[1,j] > 66 and (s_n_i > s_n_j):

                            constraint_value4= 1 / phi * (lambda_4 * (s_n_i + s_n_j - delta_i) - (v_i + v_j)) + F_rm_i
                            A[count, i] = 1.0
                            b[count] = constraint_value4
                            count += 1
                        elif x[0,i] > 55 and x[1,j] > 66 and (s_n_i <= s_n_j):

                            constraint_value4= 1 / phi * (lambda_4 * (s_n_i + s_n_j - delta_i) - (v_i + v_j)) + F_rm_i
                            A[count, j] = 1.0
                            b[count] = constraint_value4
                            count += 1
                elif lane_list[j] == 'r' and lane_list[i] == 'u':
                    delta_i = 15
                    s_n_i =  np.linalg.norm(x[:,i] - np.array([55, 68]))
                    s_n_j = np.linalg.norm(x[:,j] - np.array([55, 68]))
                    if delta_i >= s_n_i  + s_n_j:

                        if x[0,j] > 55 and x[1,i] > 66 and (s_n_i > s_n_j):

                            constraint_value4= 1 / phi * (lambda_4 * (s_n_i + s_n_j - delta_i) - (v_i + v_j)) + F_rm_i
                            A[count, i] = 1.0
                            b[count] = constraint_value4
                            count += 1
                        elif x[0,j] > 55 and x[1,i] > 66 and (s_n_i <= s_n_j):

                            constraint_value4= 1 / phi * (lambda_4 * (s_n_i + s_n_j - delta_i) - (v_i + v_j)) + F_rm_i
                            A[count, j] = 1.0
                            b[count] = constraint_value4
                            count += 1
                elif lane_list[i] == 'r' and lane_list[j] == 'd':
                    #65, 68
                    delta_i = 20
                    s_n_i =  np.linalg.norm(x[:,i] - np.array([65, 68]))
                    s_n_j = np.linalg.norm(x[:,j] - np.array([65, 68]))
                    if delta_i >= s_n_i  + s_n_j:
                        if x[0,i] > 65 and x[1,j] < 68 and (s_n_i > s_n_j):

                            constraint_value4= 1 / phi * (lambda_4 * (s_n_i + s_n_j - delta_i) - (v_i + v_j)) + F_rm_i
                            A[count, i] = 1.0
                            b[count] = constraint_value4
                            count += 1
                        elif x[0,i] > 65 and x[1,j] < 68 and (s_n_i <= s_n_j):

                            constraint_value4= 1 / phi * (lambda_4 * (s_n_i + s_n_j - delta_i) - (v_i + v_j)) + F_rm_i
                            A[count, j] = 1.0
                            b[count] = constraint_value4
                            count += 1
                elif lane_list[j] == 'r' and lane_list[i] == 'd':
                    #65, 68
                    delta_i = 20
                    s_n_i =  np.linalg.norm(x[:,i] - np.array([65, 68]))
                    s_n_j = np.linalg.norm(x[:,j] - np.array([65, 68]))
                    if delta_i >= s_n_i  + s_n_j:
                        if x[0,j] > 65 and x[1,i] < 68 and (s_n_i > s_n_j):

                            constraint_value4= 1 / phi * (lambda_4 * (s_n_i + s_n_j - delta_i) - (v_i + v_j)) + F_rm_i
                            A[count, i] = 1.0
                            b[count] = constraint_value4
                            count += 1
                        elif x[0,j] > 65 and x[1,i] < 68 and (s_n_i <= s_n_j):

                            constraint_value4= 1 / phi * (lambda_4 * (s_n_i + s_n_j - delta_i) - (v_i + v_j)) + F_rm_i
                            A[count, j] = 1.0
                            b[count] = constraint_value4
                            count += 1

                elif lane_list[i] == 'd' and lane_list[j] == 'l':
                    #65, 58

                    delta_i = 30
                    lambda_4 = 10
                    phi = 0.1
                    s_n_i =  np.linalg.norm(x[:,i] - np.array([65, 58]))
                    s_n_j = np.linalg.norm(x[:,j] - np.array([65, 58]))
                    if delta_i >= s_n_i  + s_n_j:
                        if x[0,j] < 65 and x[1,i] < 58 and (s_n_i > s_n_j):
 
                            constraint_value4= 1 / phi * (lambda_4 * (s_n_i + s_n_j - delta_i) - (v_i + v_j)) + F_rm_i
                            A[count, i] = 1.0
                            b[count] = constraint_value4
                            count += 1
                        elif x[0,j] < 65 and x[1,i] < 58 and (s_n_i <= s_n_j):

                            constraint_value4 = 1 / phi * (lambda_4 * (s_n_i + s_n_j - delta_i) - (v_i + v_j)) + F_rm_i
                            A[count, j] = 1.0
                            b[count] = constraint_value4
                            count += 1
                    
                elif lane_list[j] == 'd' and lane_list[i] == 'l':
                    #65, 58

                    delta_i = 20
                    s_n_i =  np.linalg.norm(x[:,i] - np.array([65, 58]))
                    s_n_j = np.linalg.norm(x[:,j] - np.array([65, 58]))
                    if delta_i >= s_n_i  + s_n_j:
                        if x[0,i] < 65 and x[1,j] < 58 and (s_n_i > s_n_j):

                            constraint_value4= 1 / phi * (lambda_4 * (s_n_i + s_n_j - delta_i) - (v_i + v_j)) + F_rm_i
                            A[count, i] = 1.0
                            b[count] = constraint_value4
                            count += 1
                        elif x[0,i] < 65 and x[1,j] < 58 and (s_n_i <= s_n_j):

                            constraint_value4= 1 / phi * (lambda_4 * (s_n_i + s_n_j - delta_i) - (v_i + v_j)) + F_rm_i
                            A[count, j] = 1.0
                            b[count] = constraint_value4
                            count += 1       

                elif lane_list[i] == 'l' and lane_list[j] == 'u':
                    #55, 58

                    delta_i = 30
                    lambda_4 = 10
                    phi = 0.1
                    s_n_i =  np.linalg.norm(x[:,i] - np.array([55, 58]))
                    s_n_j = np.linalg.norm(x[:,j] - np.array([55, 58]))
                    if delta_i >= s_n_i  + s_n_j:
                        if x[0,i] < 55 and x[1,j] > 58 and (s_n_i > s_n_j):
   
                            constraint_value4= 1 / phi * (lambda_4 * (s_n_i + s_n_j - delta_i) - (v_i + v_j)) + F_rm_i
                            A[count, i] = 1.0
                            b[count] = constraint_value4
                            count += 1
                        elif x[0,i] < 55 and x[1,j] > 58 and (s_n_i <= s_n_j):

                            constraint_value4 = 1 / phi * (lambda_4 * (s_n_i + s_n_j - delta_i) - (v_i + v_j)) + F_rm_i
                            A[count, j] = 1.0
                            b[count] = constraint_value4
                            count += 1
                    
                elif lane_list[j] == 'l' and lane_list[i] == 'u':
                    #55, 58

                    delta_i = 20
                    s_n_i =  np.linalg.norm(x[:,i] - np.array([55, 58]))
                    s_n_j = np.linalg.norm(x[:,j] - np.array([55, 58]))
                    if delta_i >= s_n_i  + s_n_j:
                        if x[0,j] < 55 and x[1,i] > 58 and (s_n_i > s_n_j):

                            constraint_value4= 1 / phi * (lambda_4 * (s_n_i + s_n_j - delta_i) - (v_i + v_j)) + F_rm_i
                            A[count, i] = 1.0
                            b[count] = constraint_value4
                            count += 1
                        elif x[0,j] < 55 and x[1,i] > 58 and (s_n_i <= s_n_j):

                            constraint_value4= 1 / phi * (lambda_4 * (s_n_i + s_n_j - delta_i) - (v_i + v_j)) + F_rm_i
                            A[count, j] = 1.0
                            b[count] = constraint_value4
                            count += 1                  



                elif (lane_list[i] == 'rd' and lane_list[j] == 'u') or (lane_list[j] == 'rd' and lane_list[i] == 'u'):
                    delta = 40.0
                    cp = np.array([56, 46])
                    di = distance_along_path(x[:,i], r_b_path, cp)
                    dj = distance_along_path(x[:,j], r_b_path, cp)
                    if delta >= di + dj:

                        if lane_list[i] == 'rd' and lane_list[j] == 'u' and x[1,j] > 46 and x[1,i] > 46 and (di <= dj):

                            s_n_i =  distance_along_path(x[:,i], r_b_path, cp)

                            s_n_j = np.linalg.norm(x[:,j] - np.array([56, 46]))
                            phi = 0.1
                            lambda_4 = 10.0
                            delta_i = 40.0

                            #constraint_value4 = (1 / phi) * (lambda_4 * ((46.0-x[1,i]) + ( -(56.0-x[0,j]) ) - (gamma2 + phi * v_i)) - (v_i + v_j) ) + F_rm_i
                            constraint_value4= 1 / phi * (lambda_4 * (s_n_i + s_n_j - delta_i) - (v_i + v_j)) + F_rm_i

                            A[count, j] = 1.0
                            b[count] = constraint_value4
                            count += 1

                        elif lane_list[i] == 'rd' and lane_list[j] == 'u' and x[1,j] > 46 and x[1,i] > 46 and (di >= dj):   

                            s_n_i =  distance_along_path(x[:,i], r_b_path, cp)

                            s_n_j = np.linalg.norm(x[:,j] - np.array([56, 46]))
                            phi = 0.1
                            lambda_4 = 10.0
                            delta_i = 40.0

                            #constraint_value4 = (1 / phi) * (lambda_4 * ((46.0-x[1,i]) + ( -(56.0-x[0,j]) ) - (gamma2 + phi * v_i)) - (v_i + v_j) ) + F_rm_i
                            constraint_value4= 1 / phi * (lambda_4 * (s_n_i + s_n_j - delta_i) - (v_i + v_j)) + F_rm_i

                            A[count, i] = 1.0
                            b[count] = constraint_value4
                            count += 1
                        elif lane_list[j] == 'rd' and lane_list[i] == 'u' and x[1,j] > 46 and x[1,i] > 46 and (di <= dj):
                            #Point(56, 46)
                            #s_n_i = np.linalg.norm(x[:,i] - np.array([56, 46]))
                            cp = np.array([56, 46])
                            s_n_i =  np.linalg.norm(x[:,i] - np.array([56, 46]))

                            s_n_j =  distance_along_path(x[:,j], r_b_path, cp)
                            phi = 0.1
                            lambda_4 = 10.0
                            delta_i = 40.0

                            #constraint_value4 = (1 / phi) * (lambda_4 * ((46.0-x[1,i]) + ( -(56.0-x[0,j]) ) - (gamma2 + phi * v_i)) - (v_i + v_j) ) + F_rm_i
                            constraint_value4= 1 / phi * (lambda_4 * (s_n_i + s_n_j - delta_i) - (v_i + v_j)) + F_rm_i

                            A[count, j] = 1.0
                            b[count] = constraint_value4
                            count += 1   
                        elif lane_list[j] == 'rd' and lane_list[i] == 'u' and x[1,j] > 46 and x[1,i] > 46 and (di >= dj):
                            #Point(56, 46)
                            #s_n_i = np.linalg.norm(x[:,i] - np.array([56, 46]))
                            cp = np.array([56, 46])
                            s_n_i =  np.linalg.norm(x[:,i] - np.array([56, 46]))

                            s_n_j =  distance_along_path(x[:,j], r_b_path, cp)
                            phi = 0.1
                            lambda_4 = 10.0
                            delta_i = 40.0

                            #constraint_value4 = (1 / phi) * (lambda_4 * ((46.0-x[1,i]) + ( -(56.0-x[0,j]) ) - (gamma2 + phi * v_i)) - (v_i + v_j) ) + F_rm_i
                            constraint_value4= 1 / phi * (lambda_4 * (s_n_i + s_n_j - delta_i) - (v_i + v_j)) + F_rm_i

                            A[count, i] = 1.0
                            b[count] = constraint_value4
                            count += 1   

                elif (lane_list[i] == 'rd' and lane_list[j] == 'd') or (lane_list[j] == 'rd' and lane_list[i] == 'd'):
                    delta = 30.0

                    cp = np.array([65, 63])
                    di = distance_along_path(x[:,i], r_b_path, cp)
                    dj = distance_along_path(x[:,j], r_b_path, cp)
                    if delta >= di + dj:

                        if lane_list[i] == 'rd' and lane_list[j] == 'd' and x[1,j] < 63 and x[0,i] > 65 and (di <= dj):

                            s_n_i =  distance_along_path(x[:,i], r_b_path, cp)
                            #print(s_n_i)
                            s_n_j = np.linalg.norm(x[:,j] - np.array([65, 63]))
                            phi = 0.1
                            lambda_4 = 10.0
                            delta_i = 30.0

                            #constraint_value4 = (1 / phi) * (lambda_4 * ((46.0-x[1,i]) + ( -(56.0-x[0,j]) ) - (gamma2 + phi * v_i)) - (v_i + v_j) ) + F_rm_i
                            constraint_value4= 1 / phi * (lambda_4 * (s_n_i + s_n_j - delta_i) - (v_i + v_j)) + F_rm_i

                            A[count, j] = 1.0
                            b[count] = constraint_value4
                            count += 1

                        elif lane_list[i] == 'rd' and lane_list[j] == 'd' and x[1,j] < 63 and x[0,i] > 65 and (di >= dj):   

                            s_n_i =  distance_along_path(x[:,i], r_b_path, cp)

                            s_n_j = np.linalg.norm(x[:,j] - np.array([65, 63]))
                            phi = 0.1
                            lambda_4 = 10.0
                            delta_i = 30.0

                            #constraint_value4 = (1 / phi) * (lambda_4 * ((46.0-x[1,i]) + ( -(56.0-x[0,j]) ) - (gamma2 + phi * v_i)) - (v_i + v_j) ) + F_rm_i
                            constraint_value4= 1 / phi * (lambda_4 * (s_n_i + s_n_j - delta_i) - (v_i + v_j)) + F_rm_i

                            A[count, i] = 1.0
                            b[count] = constraint_value4
                            count += 1
                        elif lane_list[j] == 'rd' and lane_list[i] == 'd' and x[1,i] < 63 and x[0,j] > 65 and (di <= dj):
                            #Point(56, 46)

                            #s_n_i = np.linalg.norm(x[:,i] - np.array([56, 46]))
                            cp = np.array([65, 63])
                            s_n_i =  np.linalg.norm(x[:,i] - np.array([56, 46]))

                            s_n_j =  distance_along_path(x[:,j], r_b_path, cp)
                            phi = 0.1
                            lambda_4 = 10.0
                            delta_i = 30.0

                            #constraint_value4 = (1 / phi) * (lambda_4 * ((46.0-x[1,i]) + ( -(56.0-x[0,j]) ) - (gamma2 + phi * v_i)) - (v_i + v_j) ) + F_rm_i
                            constraint_value4= 1 / phi * (lambda_4 * (s_n_i + s_n_j - delta_i) - (v_i + v_j)) + F_rm_i

                            A[count, j] = 1.0
                            b[count] = constraint_value4
                            count += 1   
                        elif lane_list[j] == 'rd' and lane_list[i] == 'd' and x[1,i] < 63 and x[0,j] > 65 and (di >= dj):
                            #Point(56, 46)

                            #s_n_i = np.linalg.norm(x[:,i] - np.array([56, 46]))
                            cp = np.array([65, 63])
                            s_n_i =  np.linalg.norm(x[:,i] - np.array([56, 46]))

                            s_n_j =  distance_along_path(x[:,j], r_b_path, cp)
                            phi = 0.1
                            lambda_4 = 10.0
                            delta_i = 30.0

                            #constraint_value4 = (1 / phi) * (lambda_4 * ((46.0-x[1,i]) + ( -(56.0-x[0,j]) ) - (gamma2 + phi * v_i)) - (v_i + v_j) ) + F_rm_i
                            constraint_value4= 1 / phi * (lambda_4 * (s_n_i + s_n_j - delta_i) - (v_i + v_j)) + F_rm_i

                            A[count, i] = 1.0
                            b[count] = constraint_value4
                            count += 1
                elif (lane_list[i] == 'rd' and lane_list[j] == 'l') or (lane_list[j] == 'rd' and lane_list[i] == 'l'):
                    delta = 30.0
                    #print("this is the problem0")
                    cp = np.array([60, 58])
                    di = distance_along_path(x[:,i], r_b_path, cp)
                    dj = distance_along_path(x[:,j], r_b_path, cp)
                    if delta >= di + dj:
                        print("this is the problem0")
                        if lane_list[i] == 'rd' and lane_list[j] == 'l' and x[1,i] > 58 and x[0,j] < 60 and (di <= dj):
                            print("this is the problem1")
                            s_n_i =  distance_along_path(x[:,i], r_b_path, cp)
                            #print(s_n_i)
                            s_n_j = np.linalg.norm(x[:,j] - np.array([60, 58]))
                            phi = 0.1
                            lambda_4 = 10.0
                            delta_i = 30.0

                            #constraint_value4 = (1 / phi) * (lambda_4 * ((46.0-x[1,i]) + ( -(56.0-x[0,j]) ) - (gamma2 + phi * v_i)) - (v_i + v_j) ) + F_rm_i
                            constraint_value4= 1 / phi * (lambda_4 * (s_n_i + s_n_j - delta_i) - (v_i + v_j)) + F_rm_i

                            A[count, j] = 1.0
                            b[count] = constraint_value4
                            count += 1

                        elif lane_list[i] == 'rd' and lane_list[j] == 'l' and x[1,i] > 58 and x[0,j] < 60 and (di >= dj):   
                            print("this is the problem2")
                            s_n_i =  distance_along_path(x[:,i], r_b_path, cp)

                            s_n_j = np.linalg.norm(x[:,j] - np.array([60, 58]))
                            phi = 0.1
                            lambda_4 = 10.0
                            delta_i = 30.0

                            #constraint_value4 = (1 / phi) * (lambda_4 * ((46.0-x[1,i]) + ( -(56.0-x[0,j]) ) - (gamma2 + phi * v_i)) - (v_i + v_j) ) + F_rm_i
                            constraint_value4= 1 / phi * (lambda_4 * (s_n_i + s_n_j - delta_i) - (v_i + v_j)) + F_rm_i

                            A[count, i] = 1.0
                            b[count] = constraint_value4
                            count += 1
                        elif lane_list[j] == 'rd' and lane_list[i] == 'l' and x[0,i] < 60 and x[1,j] > 58 and (di <= dj):
                            #Point(56, 46)
                            print("this is the problem3")
                            #s_n_i = np.linalg.norm(x[:,i] - np.array([56, 46]))
                            cp = np.array([60, 58])
                            s_n_i =  np.linalg.norm(x[:,i] - np.array([60, 58]))

                            s_n_j =  distance_along_path(x[:,j], r_b_path, cp)
                            phi = 0.1
                            lambda_4 = 10.0
                            delta_i = 30.0

                            #constraint_value4 = (1 / phi) * (lambda_4 * ((46.0-x[1,i]) + ( -(56.0-x[0,j]) ) - (gamma2 + phi * v_i)) - (v_i + v_j) ) + F_rm_i
                            constraint_value4= 1 / phi * (lambda_4 * (s_n_i + s_n_j - delta_i) - (v_i + v_j)) + F_rm_i

                            A[count, j] = 1.0
                            b[count] = constraint_value4
                            count += 1   
                        elif lane_list[j] == 'rd' and lane_list[i] == 'l' and x[0,i] < 60 and x[1,j] > 58 and (di >= dj):
                            #Point(56, 46)
                            print("this is the problem4")
                            #s_n_i = np.linalg.norm(x[:,i] - np.array([56, 46]))
                            cp = np.array([60, 58])
                            s_n_i =  np.linalg.norm(x[:,i] - np.array([60, 58]))

                            s_n_j =  distance_along_path(x[:,j], r_b_path, cp)
                            phi = 0.1
                            lambda_4 = 10.0
                            delta_i = 30.0

                            #constraint_value4 = (1 / phi) * (lambda_4 * ((46.0-x[1,i]) + ( -(56.0-x[0,j]) ) - (gamma2 + phi * v_i)) - (v_i + v_j) ) + F_rm_i
                            constraint_value4= 1 / phi * (lambda_4 * (s_n_i + s_n_j - delta_i) - (v_i + v_j)) + F_rm_i

                            A[count, i] = 1.0
                            b[count] = constraint_value4
                            count += 1



    # Convert to cvxopt matrix
    H = matrix(H)
    f = matrix(f)
    A = matrix(A)
    b = matrix(b)
    # Solve the quadratic program
    solution = qp(H, f, A, b)
    #print("cbf")
    # Get the optimal control inputs
    u_optimal = np.array(solution['x']).flatten()
        
    return u_optimal

def calculate_steering_angle(car, target_point):
    car_to_target = Point(target_point[0] - car.center.x, target_point[1] - car.center.y)
    angle_to_target = np.arctan2(car_to_target.y, car_to_target.x)
    steering_angle = angle_to_target - car.heading
    return steering_angle

pid_controller = PIDController(kp=1.0, ki=0.5, kd=0.1, target_velocity=6.0)
pid_controller2 = PIDController(kp=1.0, ki=0.5, kd=0.1, target_velocity = 8.0)
pid_controller3 = PIDController(kp=1.0, ki=0.5, kd=0.1, target_velocity=4.0)
pid_controller4 = PIDController(kp=1.0, ki=0.5, kd=0.1, target_velocity=4.0)
pid_controller5 = PIDController(kp=1.0, ki=0.5, kd=0.1, target_velocity=4.0)
#############################################################################






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
        velocity_magnitude = np.sqrt(c4.velocity.x**2 + c4.velocity.y**2)
        acceleration = pid_controller4.control(velocity_magnitude, dt)
        throttle4 = acceleration
        velocity_magnitude = np.sqrt(c5.velocity.x**2 + c5.velocity.y**2)
        acceleration = pid_controller5.control(velocity_magnitude, dt)
        throttle5 = acceleration
        u_ref = np.array([throttle1 ,throttle2, throttle3, throttle4,throttle5])
        
        # Update world
        w.tick() # Tick the world for one time step
        w.render()
        time.sleep(dt/4) # Watch it at 4x slower speed
        time_passed += dt

        # Update car states
        for i in range(N):
            x[0,i] = clist[i].center.x
            x[1,i] = clist[i].center.y
            #if x[0,i] > 43 and x[1,i] > 63:
            #   lane_list[i] = 'rightout'


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

        th = cbf(x, u_ref)
        '''
        c1.set_control(steering_angle, u_ref[0])
        c2.set_control(0, u_ref[1])
        c3.set_control(0, u_ref[2])
        c4.set_control(0, u_ref[3])
        c5.set_control(0, u_ref[4])
        '''
        c1.set_control(steering_angle, th[0])
        c2.set_control(0, th[1])
        c3.set_control(0, th[2])
        c4.set_control(0, th[3])
        c5.set_control(0, th[4])

        for i in range(N):
            if x[0,i] >= 141:
                clist[i].center = Point(0, 58)
            
            elif x[0,i] < 0:
                clist[i].center = Point(130,68)
                clist[i].heading = np.pi
            elif x[1,i] < 0:
                if clist[i] == c4:
                    clist[i].center = Point(56, 130)
                    #clist[i].heading = np.pi
                else:
                    clist[i].center = Point(121,68)
                    clist[i].heading = np.pi
            elif x[1,i] >= 141:
                clist[i].center = Point(65, 0)
                clist[i].heading = np.pi/2

    w.close()
