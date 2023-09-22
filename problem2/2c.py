import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import numpy as np
import sys
π = np.pi

L = 10.668 # length between wheels (in meters)
W = 3.048 # width between wheels (in meters)
r = 9 # goal circle radius (in meters)
T = 30 # time of termination in seconds
if len(sys.argv) > 1:
    T = int(sys.argv[1])

x = 0 # m | 0 = start
y = 0 # m | 0 = start
x_euler = 0 # m | 0 = start
y_euler = 0 # m | 0 = start
θ = 0 # radians | 0 = east; + = counter-clockwise; - = clockwise
θ_euler = 0 # radians | 0 = east; + = counter-clockwise; - = clockwise

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

def update_commands():
    global v_left
    global v_right
    #global dt

    ss.set_instructions(r)
    v_left, v_right, _t = ss.get_velocities()
    #dt = _t if _t is not None else min_dt

    # sanity check
    #if dt < min_dt:
    #    print("dt too low")
    #    dt = min_dt

x_hist = [x]
y_hist = [y]
θ_hist = [θ]
x_euler_hist = {}
y_euler_hist = {}
θ_euler_hist = {}
inaccuracy_hist = {}
t_hist = {}

update_commands()
vt_hist = [0]
v_left_hist = [v_left]
v_right_hist = [v_right]

dt_list = [1, 0.1, 0.01]
for dt in dt_list:
    x = 0
    y = 0
    θ = 0

    t = 0

    x_euler = 0
    y_euler = 0
    θ_euler = 0

    x_euler_hist[dt] = [x_euler]
    y_euler_hist[dt] = [y_euler]
    θ_euler_hist[dt] = [θ_euler]
    inaccuracy_hist[dt] = [0]
    t_hist[dt] = [0]

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

        θ_euler += (v_right-v_left)/W * dt / 2
        x_euler += ss.v*np.cos(θ_euler)*dt
        y_euler += ss.v*np.sin(θ_euler)*dt
        θ_euler += (v_right-v_left)/W * dt / 2
        
        x_euler_hist[dt].append(x_euler)
        y_euler_hist[dt].append(y_euler)
        θ_euler_hist[dt].append(θ_euler)
        
        t_hist[dt].append(t)
        inaccuracy_hist[dt].append(np.sqrt((x_euler-x)**2+(y_euler-y)**2))

        update_commands()


fig, axs = plt.subplots(1, 2)
for _dt in dt_list:
    axs[0].plot(
            x_euler_hist[_dt],
            y_euler_hist[_dt],
            label="Euler @ " + str(1/_dt) + "Hz")
axs[0].plot(0, 0, '--bo')
axs[0].xaxis.set_major_formatter(mticker.FormatStrFormatter('%g m'))
axs[0].yaxis.set_major_formatter(mticker.FormatStrFormatter('%g m'))
axs[0].legend()

for _dt in dt_list:
    axs[1].plot(
            t_hist[_dt],
            inaccuracy_hist[_dt],
            label="Inaccuracy @ " + str(1/_dt) + "Hz")
axs[1].xaxis.set_major_formatter(mticker.FormatStrFormatter('%g sec'))
axs[1].yaxis.set_major_formatter(mticker.FormatStrFormatter('%g m'))
axs[1].set_xlabel("Time")
axs[1].set_ylabel("Distance")
axs[1].grid()
axs[1].legend()

#ax = plt.gca()
axs[0].add_patch(plt.Circle((0,r), radius=r, color='pink'))
axs[0].add_patch(plt.Circle((0,r), radius=(r-0.5), color='white'))

plt.show()
