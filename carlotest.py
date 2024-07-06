import numpy as np
from world import World
from agents import Car, RectangleBuilding, Pedestrian, Painting
from geometry import Point
import time
import numpy as np
from cvxopt import matrix, sparse
from cvxopt.solvers import qp
from scipy.special import comb
N = 2
x = np.zeros((2, N))

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

w.add(Painting(Point(125.5, 116.5), Point(97, 82), 'gray80')) 
w.add(RectangleBuilding(Point(126.5, 117.5), Point(95, 80))) 
# Let's repeat this for 4 different RectangleBuildings.
w.add(Painting(Point(-5.5, 116.5), Point(97, 82), 'gray80'))
w.add(RectangleBuilding(Point(-6.5, 117.5), Point(95, 80)))

w.add(Painting(Point(-5.5, 10), Point(97, 82), 'gray80'))
w.add(RectangleBuilding(Point(-6.5, 9), Point(95, 80)))

w.add(Painting(Point(125.5, 10.5), Point(97, 82), 'gray80'))
w.add(RectangleBuilding(Point(126, 9), Point(95, 80)))

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

# A Car object is a dynamic object -- it can move. We construct it using its center location and heading angle.
c1 = Car(Point(70,20), np.pi/2)
w.add(c1)

c2 = Car(Point(118,70), np.pi, 'blue')
c2.velocity = Point(3.0,0) 
w.add(c2)

clist = []
clist.append(c1)
clist.append(c2)
lane_list = ['d', 'r']

w.render() # This visualizes the world we just constructed.


def cbf(x, u_ref):
    N = len(u_ref_i)  # Number of vehicles

    # Convert inputs to numpy arrays if they are not already
    u_ref_i = np.array(u_ref_i)
    v_i = np.array(v_i)
    v_k = np.array(v_k)
    p_i = np.array(p_i)
    p_k = np.array(p_k)
    F_r = np.array(F_r)
    m_i = np.array(m_i)
        
    # Construct the quadratic programming matrices
    H = 2 * np.identity(N)
    f = -2 * u_ref_i
    num_constraints = int(comb(N, 2))    
    # Construct the inequality constraint matrices
    A = np.zeros((num_constraints, N))
    b = np.zeros(num_constraints)
    
    count = 0
    for i in range(N):
        for j in range(i+1, N):
            phi = 0.05
            lambda_3 = 10
            gamma = 0.2 #standstill_distance
            v_i = np.sqrt(clist[i].velocity.x**2 + clist[i].velocity.y**2)
            v_j = np.sqrt(clist[j].velocity.x**2 + clist[j].velocity.y**2)
            if lane_list[i] == lane_list[j]:
                if lane_list[i] == 'r' or lane_list[i] == 'l':  
                    
                    constraint_value1 = (1 / phi) * (lambda_3 * (x[0,j] - x[0,i] - (gamma + phi * v_i)) + v_j - v_i) + F_r[i] / m_i[i]
                    constraint_value2 = (1 / phi) * (lambda_3 * (x[0,i] - x[0,j] - (gamma + phi * v_j)) + v_i - v_j) + F_r[i] / m_i[i]
                elif lane_list[i] == 'd' or lane_list[i] == 'u':
                    constraint_value1 = (1 / phi) * (lambda_3 * (x[1,j] - x[1,i] - (gamma + phi * v_i)) + v_j - v_i) + F_r[i] / m_i[i]
                    constraint_value2 = (1 / phi) * (lambda_3 * (x[1,i] - x[1,j] - (gamma + phi * v_j)) + v_i - v_j) + F_r[i] / m_i[i]               
            A[count, i] = 1.0
            b[count] = constraint_value1
            count += 1
            A[count, i] = 1.0
            b[count] = constraint_value2
            count += 1

    # Convert numpy matrices to cvxopt matrices
    H = matrix(H)
    f = matrix(f)
    A = matrix(A)
    b = matrix(b)
        
    # Solve the quadratic program
    solution = qp(H, f, A, b)
        
    # Get the optimal control inputs
    u_optimal = np.array(solution['x']).flatten()
        
    return u_optimal



target_velocity = 1.5 # Target speed in m/s
pid_controller = PIDController(kp=1.0, ki=0.5, kd=0.1, target_velocity=target_velocity)

if not human_controller:
    time_steps = int(400 * dt)  # Adjust for correct time steps based on dt
    time_passed = 0.0
    while time_passed < 80.0:  # Run simulation for 40 seconds
        velocity_magnitude = np.sqrt(c1.velocity.x**2 + c1.velocity.y**2)
        acceleration = pid_controller.control(velocity_magnitude, dt)
        throttle1 = np.clip(acceleration, -1.0, 1.0)
        velocity_magnitude = np.sqrt(c2.velocity.x**2 + c2.velocity.y**2)
        acceleration = pid_controller.control(velocity_magnitude, dt)
        throttle2 = np.clip(acceleration, -1.0, 1.0)
        u_ref = [throttle1, throttle2]
        
        
        
        w.tick() # Tick the world for one time step
        w.render()
        time.sleep(dt/4) # Watch it at 4x slower speed
        time_passed += dt
        #print(f"Time: {time_passed:.2f}s, c1 Location: ({c1.center.x:.2f}, {c1.center.y:.2f})")
        for i in range(N):
            x[0,i] = clist[i].center.x
            x[1,i] = clist[i].center.y
        print(x)

        c1.set_control(0, throttle1)
        c2.set_control(0, throttle2)
        cbf(x, u_ref)
        

    w.close()
