import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import numpy as np
import sys
π = np.pi

L = 10.668 # length between wheels (in meters)
W = 3.048 # width between wheels (in meters)
r = 18 # goal circle radius (in meters)
T = 30 # time of termination in seconds
if len(sys.argv) > 1:
    T = int(sys.argv[1])

x = 0 # m | 0 = center
y = 0 # m | 0 = center
θ = 0 # radians | 0 = east; + = counter-clockwise; - = clockwise

# Steering angle
#r_current = 0 # m | 0 = rotate in place
#direction = 0 # +/- | 0 = straight; 1 = counter-clockwise; -1 = clockwise

# Steering factor
v_left = 0
v_right = 0

# Update speed
min_dt = 0.5 # seconds
dt = min_dt
print("dt: " + str(dt))

class SkidSteer:
    def __init__(self, W, min_dt):
        self.v = 0
        self.W = W
        self.r = None
        self.d = None
        self.θ = None
        self.min_dt = min_dt
        self.clockwise = False

    def set_v(self, v):
        self.v_goal = v

    def set_instructions(self, r, d=None, θ=None, clockwise=False):
        if θ is not None and d is not None:
            raise ValueError("Rotation and distance cannot both be specified, because their result may not match (θ != None, d != None)")
        elif r is None and θ is not None:
            raise ValueError("Cannot rotate a certain amount while going in a straight line (r == None, θ != None)")
        elif r == 0 and d is not None:
            raise ValueError("Cannot go a distance while rotating on the spot (r == 0, d != None)")
        self.r = r
        self.d = d
        self.θ = θ
        self.clockwise = clockwise

    # Returns (Velocity Left, Velocity Right, Time)
    def get_velocities(self):
        self.v = self.v_goal
        if self.d is not None: # θ is None
            _t = self.d / self.v
            
            if _t is not None and _t < self.min_dt:
                _t = self.min_dt
                self.v = self.d / _t
        elif self.θ is not None: # r is not None, d is None
            _r = self.r if self.r != 0 else (self.W / 2)
            ω = self.v / _r
            _t = self.θ / ω

            if _t is not None and _t < self.min_dt:
                _t = self.min_dt
                ω = self.θ / _t
                self.v = ω * _r
        else:
            _t = None

        if self.r is None:
            return (self.v, self.v, _t)
        elif self.r == 0:
            if self.clockwise:
                return (self.v, -self.v, _t)
            else:
                return (-self.v, self.v, _t)
        else:
            r_inner = self.r - self.W/2
            r_outer = self.r + self.W/2
            ω = self.v / self.r
            v_inner = ω * r_inner
            v_outer = ω * r_outer
            if self.clockwise:
                return (v_outer, v_inner, _t)
            else:
                return (v_inner, v_outer, _t)

ss = SkidSteer(W, min_dt)
ss.set_v(8)

reach_flag = 0
def update_commands():
    global reach_flag
    global v_left
    global v_right
    global dt

    if reach_flag == 0:
        r_current = np.sqrt(x**2 + y**2)
        if r_current - r >= -0.1:
            reach_flag = 1
            ss.set_instructions(0, θ=π/2)
            v_left, v_right, _t = ss.get_velocities()
            dt = _t
        else:
            ss.set_instructions(None, d=r - r_current)
    elif reach_flag == 1:
        reach_flag = 2

    if reach_flag == 2:
        ss.set_instructions(r)
    
    if reach_flag != 1:
        v_left, v_right, _t = ss.get_velocities()
        dt = _t if _t is not None else min_dt

    # sanity check
    if dt < min_dt:
        print("dt too low")
        dt = min_dt

dt_list = set()
x_hist = [(dt, x)]
y_hist = [(dt, y)]
θ_hist = [(dt, θ)]
t_hist = [0]
dt_list.add(dt)

update_commands()
vt_hist = [0]
v_left_hist = [v_left]
v_right_hist = [v_right]

t = 0
while t < T:
    t += dt
    if ss.r is not None:
        if ss.r == 0:
            θ += (-1 if ss.clockwise else 1) * (ss.v / (ss.W / 2)) * dt
        else:
            arc_length = ss.v * dt
            arc_radians = arc_length / ss.r
            dθ = (-1 if ss.clockwise else 1) * arc_length / ss.r
            h = 2*ss.r*np.sin(abs(dθ)/2)
            θ += dθ/2
            x += h * np.cos(θ)
            y += h * np.sin(θ)
            θ += dθ/2
    else:
        x_vel = ss.v*np.cos(θ) # x_dot
        y_vel = ss.v*np.sin(θ) # y_dot
        #θ_vel = 0 # angular velocity

        x += x_vel * dt
        y += y_vel * dt
        #θ += θ_vel * dt
    
    t_hist.append(t)
    x_hist.append((dt, x))
    y_hist.append((dt, y))
    θ_hist.append((dt, θ))
    dt_list.add(dt)
    
    vt_hist.append(t)
    v_left_hist.append(v_left)
    v_right_hist.append(v_right)
    update_commands()
    vt_hist.append(t)
    v_left_hist.append(v_left)
    v_right_hist.append(v_right)

#print(θ_hist)
#for _dt in dt_list:
#    plt.plot(list(map(lambda x: x[1], filter(lambda x: x[0] == _dt, x_hist))), list(map(lambda y: y[1], filter(lambda y: y[0] == _dt, y_hist))), label=str(1/_dt) + "Hz")
fig, axs = plt.subplots(3)
axs[0].plot(list(map(lambda x: x[1], x_hist)), list(map(lambda x: x[1], y_hist)), label="<= " + str(1/min_dt) + "Hz")
axs[0].plot(0, 0, '--bo')
axs[0].plot(x, y, '--ro')
axs[0].xaxis.set_major_formatter(mticker.FormatStrFormatter('%g m'))
axs[0].yaxis.set_major_formatter(mticker.FormatStrFormatter('%g m'))
axs[0].legend()

axs[1].plot(vt_hist, v_left_hist, label="Left side")
axs[1].plot(vt_hist, v_right_hist, label="Right side")
axs[1].xaxis.set_major_formatter(mticker.FormatStrFormatter('%g sec'))
axs[1].yaxis.set_major_formatter(mticker.FormatStrFormatter('%g m/s'))
axs[1].set_ylabel("Speed")
axs[1].grid()
axs[1].legend()

axs[2].plot(t_hist, list(map(lambda x: 180*x[1]/π, θ_hist)), label="θ")
axs[2].xaxis.set_major_formatter(mticker.FormatStrFormatter('%g sec'))
axs[2].yaxis.set_major_formatter(mticker.FormatStrFormatter('%g°'))
axs[2].set_xlabel("Time")
axs[2].set_ylabel("Heading")
axs[2].grid()
axs[2].legend()

#ax = plt.gca()
axs[0].add_patch(plt.Circle((0,0), radius=r, color='pink'))
axs[0].add_patch(plt.Circle((0,0), radius=(r-0.5), color='white'))

plt.show()
