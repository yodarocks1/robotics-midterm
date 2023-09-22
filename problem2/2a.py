import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import numpy as np
import sys
π = np.pi

α_max = 75 * π/180 # maximum turning angle (in radians)
L = 10.668 # length between wheels (in meters)
r = 18 # goal circle radius (in meters)
T = 30 # time of termination in seconds
if len(sys.argv) > 1:
    T = int(sys.argv[1])
r_min = L/np.tan(α_max)

x = 0 # m | 0 = center
y = 0 # m | 0 = center
θ = 0 # radians | 0 = east; + = counter-clockwise; - = clockwise

# Velocity
v = 8 # m/s | 0 = stopped; + = forward; - = reverse

# Steering angle
α_start = np.arctan(L/(r/2))
α_circle = np.arctan(L/r)
α = α_start/2 # radians | 0 = steering is straight; + = counter-clockwise; - = clockwise

α_factor = 0*π

# Update speed
min_dt = 0.5 # seconds
dt = (π*r)/(v*6)
print("dt: " + str(dt))


reach_flag = False
def update_commands():
    global α
    global v
    global reach_flag
    global dt
    if not reach_flag:
        if x**2 + y**2 >= r**2 - 0.1:
            reach_flag = True
        else:
            α = α_start
    if reach_flag:
        α = α_circle
        dt = min_dt

    # sanity check
    if α > α_max or α < -α_max:
        print("α exceeded limit")
        α = np.clip(α, -α_max, α_max)
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
αt_hist = [0]
α_hist = [α]

t = 0
while t < T:
    t += dt
    
    if α != 0:
        r_current = L/np.tan(α)
        arc_length = v * dt
        arc_radians = arc_length / r_current
        dθ = arc_length / r_current
        h = 2*r_current*np.sin(dθ/2)
        θ += dθ/2
        x += h * np.cos(θ)
        y += h * np.sin(θ)
        θ += dθ/2
    else:
        x_vel = v*np.cos(θ_hist[-1]) # x_dot
        y_vel = v*np.sin(θ_hist[-1]) # y_dot
        θ_vel = v/L * np.tan(α) # angular velocity

        x += x_vel * dt
        y += y_vel * dt
        θ += θ_vel * dt

    x_hist.append((dt, x))
    y_hist.append((dt, y))
    θ_hist.append((dt, θ))
    t_hist.append(t)
    dt_list.add(dt)
    
    αt_hist.append(t)
    α_hist.append(α)
    update_commands()
    αt_hist.append(t)
    α_hist.append(α)

fig, axs = plt.subplots(3)

#print(θ_hist)
for _dt in dt_list:
    axs[0].plot(list(map(lambda x: x[1], filter(lambda x: x[0] == _dt, x_hist))), list(map(lambda y: y[1], filter(lambda y: y[0] == _dt, y_hist))), label=str(1/_dt) + "Hz")
axs[0].plot(0, 0, '--bo')
axs[0].plot(x, y, '--ro')
axs[0].xaxis.set_major_formatter(mticker.FormatStrFormatter('%g m'))
axs[0].yaxis.set_major_formatter(mticker.FormatStrFormatter('%g m'))
axs[0].legend()

axs[1].axhline(180 * α_max / π, color='red', label="$α_{max}$")
axs[1].axhline(-180 * α_max / π, color='red')
axs[1].plot(αt_hist, list(map(lambda x: 180*x/π, α_hist)), label="$α$")
axs[1].xaxis.set_major_formatter(mticker.FormatStrFormatter('%g sec'))
axs[1].yaxis.set_major_formatter(mticker.FormatStrFormatter('%g°'))
axs[1].set_ylabel("Steering Angle")
axs[1].grid()
axs[1].legend()

axs[2].plot(t_hist, list(map(lambda x: 180*x[1]/π, θ_hist)), label="θ")
axs[2].xaxis.set_major_formatter(mticker.FormatStrFormatter('%g sec'))
axs[2].yaxis.set_major_formatter(mticker.FormatStrFormatter('%g°'))
axs[2].set_xlabel("Time")
axs[2].set_ylabel("Heading")
axs[2].grid()
axs[2].legend()

axs[0].add_patch(plt.Circle((0,0), radius=r, color='pink'))
axs[0].add_patch(plt.Circle((0,0), radius=(r-0.5), color='white'))
axs[0].add_patch(plt.Circle((0,r/2), radius=(r/2), color='pink'))

plt.show()
