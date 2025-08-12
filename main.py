import tkinter as tk
import numpy as np

#Implement your own solve_ivp. The problem should be written in a file named "solve.py"
#from scipy.integrate import solve_ivp
from solve import solve_ivp
import random

# --- Configuration -------------------------------------------------------
m0 = 1200.0
mf = 400.0
ve = 2500.0
g = 9.81

Kp = 1e-2
Kd = 5e-2

sensor_range = 35000.0      # keep wide so guidance always engages
launch_alt = 25000.0        # matches max_y in launch()

t_max = 300.0
intercept_radius = 50.0

# Target motion params
target_speed = 150.0
circle_radius = 5000.0
circle_omega = 2 * np.pi / 200.0
sine_amp = 2000.0
sine_freq = 2 * np.pi / 50.0
spiral_rate = 50.0
spiral_omega = 2 * np.pi / 100.0
ellipse_a = 8000.0
ellipse_b = 4000.0
ellipse_omega = 2 * np.pi / 60.0
fig8_a = 5000.0
fig8_b = 5000.0
fig8_omega_x = 2 * np.pi / 80.0
fig8_omega_y = 2 * np.pi / 160.0
zigzag_speed = 200.0
zigzag_amp = 5000.0
zigzag_period = 20.0
lissajous_A = 6000.0
lissajous_B = 3000.0
lissajous_omega_x = 2 * np.pi / 70.0
lissajous_omega_y = 2 * np.pi / 90.0

NUM_CLOUDS = 12

# --- Shapes --------------------------------------------------------------
# More airplane-like silhouette (top-down), nose pointing up when angle=pi/2
# Tunable dimensions
JET_L = 56  # length
JET_W = 48  # max wingspan
FUS_W = 10  # fuselage width

# Outline path going clockwise from nose
def aircraft_shape(length=JET_L, wingspan=JET_W, fuselage=FUS_W):
    L = length; W = wingspan; F = fuselage
    return [
        (0, -L/2),              # 0 nose
        (-F/2, -L/3),           # 1 left fuselage shoulder
        (-W/2, -L/6),           # 2 left wing tip (leading)
        (-W/3, -L/12),          # 3 left wing trailing root
        (-F/2, L/12),           # 4 left mid fuselage
        (-W/3, L/3),            # 5 left tailplane tip
        (-F/4, L/2 - 6),        # 6 left of tail end
        (0, L/2),               # 7 tail end
        (F/4, L/2 - 6),         # 8 right of tail end
        (W/3, L/3),             # 9 right tailplane tip
        (F/2, L/12),            # 10 right mid fuselage
        (W/3, -L/12),           # 11 right wing trailing root
        (W/2, -L/6),            # 12 right wing tip (leading)
        (F/2, -L/3),            # 13 right fuselage shoulder
    ]

MISSILE_SHAPE = aircraft_shape(length=68, wingspan=50, fuselage=10)
TARGET_SHAPE  = aircraft_shape(length=56, wingspan=52, fuselage=11)

# --- Clouds --------------------------------------------------------------
def draw_clouds(canvas, W, H):
    ids = []
    for _ in range(NUM_CLOUDS):
        x = random.randint(30, W-120)
        y = random.randint(30, H-250)
        w = random.randint(80, 160)
        h = random.randint(30, 70)
        # multi-blob cloud
        base = canvas.create_oval(x, y, x+w, y+h, fill='#FFFFFF', outline='')
        ids.append(base)
        if random.random() < 0.8:
            ids.append(canvas.create_oval(x+int(w*0.2), y-15, x+int(w*0.8), y+int(h*0.9), fill='#F8F8F8', outline=''))
        if random.random() < 0.6:
            ids.append(canvas.create_oval(x-15, y+10, x+int(w*0.6), y+int(h*0.95), fill='#F2F2F2', outline=''))
    return ids

def clear_clouds(canvas, ids):
    for i in ids:
        canvas.delete(i)

# --- Dynamics ------------------------------------------------------------

def compute_trajectories(mode, max_x, max_y):
    center_x = max_x / 2
    center_y = max_y / 2

    def dynamics(t, state):
        x, y, vx, vy, m = state
        pos = np.array([x, y])
        vel = np.array([vx, vy])
        # Target motion + velocity
        if mode == 'horizontal':
            r_t = np.array([center_x + target_speed * t, center_y])
            v_t = np.array([target_speed, 0.0])
        elif mode == 'circular':
            ang = circle_omega * t
            r_t = np.array([center_x + circle_radius*np.cos(ang), center_y + circle_radius*np.sin(ang)])
            v_t = np.array([-circle_radius*circle_omega*np.sin(ang), circle_radius*circle_omega*np.cos(ang)])
        elif mode == 'sine':
            r_t = np.array([center_x + target_speed * t, center_y + sine_amp*np.sin(sine_freq*t)])
            v_t = np.array([target_speed, sine_amp*sine_freq*np.cos(sine_freq*t)])
        elif mode == 'vertical':
            r_t = np.array([center_x, center_y + target_speed * t])
            v_t = np.array([0.0, target_speed])
        elif mode == 'spiral':
            ang = spiral_omega * t
            r_t = np.array([center_x + spiral_rate*t*np.cos(ang), center_y + spiral_rate*t*np.sin(ang)])
            v_t = spiral_rate * np.array([np.cos(ang), np.sin(ang)]) + spiral_rate*t*spiral_omega*np.array([-np.sin(ang), np.cos(ang)])
        elif mode == 'ellipse':
            ang = ellipse_omega * t
            r_t = np.array([center_x + ellipse_a*np.cos(ang), center_y + ellipse_b*np.sin(ang)])
            v_t = np.array([-ellipse_a*ellipse_omega*np.sin(ang), ellipse_b*ellipse_omega*np.cos(ang)])
        elif mode == 'figure8':
            r_t = np.array([center_x + fig8_a*np.sin(fig8_omega_x*t), center_y + fig8_b*np.sin(fig8_omega_y*t)])
            v_t = np.array([fig8_a*fig8_omega_x*np.cos(fig8_omega_x*t), fig8_b*fig8_omega_y*np.cos(fig8_omega_y*t)])
        elif mode == 'zigzag':
            leg = int(t // zigzag_period)
            y_off = zigzag_amp if leg % 2 == 0 else -zigzag_amp
            r_t = np.array([center_x + zigzag_speed*t, center_y + y_off])
            v_t = np.array([zigzag_speed, 0.0])
        elif mode == 'lissajous':
            r_t = np.array([center_x + lissajous_A*np.sin(lissajous_omega_x*t), center_y + lissajous_B*np.sin(lissajous_omega_y*t)])
            v_t = np.array([lissajous_A*lissajous_omega_x*np.cos(lissajous_omega_x*t), lissajous_B*lissajous_omega_y*np.cos(lissajous_omega_y*t)])
        else:
            r_t = np.array([center_x, center_y])
            v_t = np.zeros(2)

        # Guidance law (always on)
        e  = r_t - pos
        ed = v_t - vel
        a_cmd = Kp*e + Kd*ed + np.array([0.0, g])
        T = max(0.0, m * np.linalg.norm(a_cmd))
        u = T / ve
        dm = -u if m > mf else 0.0
        return [vx, vy, a_cmd[0], a_cmd[1], dm]

    # Missile starts center-top
    state0 = [center_x, max_y, 0.0, 0.0, m0]

    sol = solve_ivp(dynamics, [0, t_max], state0, max_step=0.5, rtol=1e-6, atol=1e-8)
    t = sol.t; x = sol.y[0]; y = sol.y[1]

    # Precompute target arrays for visualization
    if mode == 'horizontal':
        xt = center_x + target_speed * t; yt = np.full_like(t, center_y)
    elif mode == 'circular':
        ang = circle_omega * t; xt = center_x + circle_radius*np.cos(ang); yt = center_y + circle_radius*np.sin(ang)
    elif mode == 'sine':
        xt = center_x + target_speed * t; yt = center_y + sine_amp*np.sin(sine_freq*t)
    elif mode == 'vertical':
        xt = np.full_like(t, center_x); yt = center_y + target_speed * t
    elif mode == 'spiral':
        ang = spiral_omega * t; xt = center_x + spiral_rate*t*np.cos(ang); yt = center_y + spiral_rate*t*np.sin(ang)
    elif mode == 'ellipse':
        ang = ellipse_omega * t; xt = center_x + ellipse_a*np.cos(ang); yt = center_y + ellipse_b*np.sin(ang)
    elif mode == 'figure8':
        xt = center_x + fig8_a*np.sin(fig8_omega_x*t); yt = center_y + fig8_b*np.sin(fig8_omega_y*t)
    elif mode == 'zigzag':
        legs = (t // zigzag_period).astype(int); xt = center_x + zigzag_speed*t; yt = np.where(legs % 2 == 0, center_y + zigzag_amp, center_y - zigzag_amp)
    elif mode == 'lissajous':
        xt = center_x + lissajous_A*np.sin(lissajous_omega_x*t); yt = center_y + lissajous_B*np.sin(lissajous_omega_y*t)
    else:
        xt = np.full_like(t, center_x); yt = np.full_like(t, center_y)

    vx = np.gradient(x, t); vy = np.gradient(y, t)
    return t, x, y, xt, yt, vx, vy

# --- Mapping -------------------------------------------------------------

def map_to_canvas(x, y, max_x, max_y, w, h, pad=40):
    # Clamp to world first
    x = min(max(x, 0), max_x); y = min(max(y, 0), max_y)
    x_px = pad + (x / max_x) * (w - 2*pad)
    y_px = h - pad - (y / max_y) * (h - 2*pad)
    return x_px, y_px

# --- GUI -----------------------------------------------------------------

def main():
    root = tk.Tk(); root.title("2D Missile Intercept Simulation")
    W, H = 900, 650
    canvas = tk.Canvas(root, width=W, height=H, bg='#9fd3ff'); canvas.pack()

    cloud_ids = draw_clouds(canvas, W, H)

    modes = ['horizontal', 'circular', 'sine', 'vertical', 'spiral', 'ellipse', 'figure8', 'zigzag', 'lissajous']
    mode_var = tk.StringVar(value=modes[0])
    fm = tk.Frame(root); tk.Label(fm, text="Target Path:").pack(side='left')
    for m in modes:
        tk.Radiobutton(fm, text=m.capitalize(), variable=mode_var, value=m).pack(side='left')
    fm.pack(pady=5)

    slow_var = tk.DoubleVar(value=200)
    tk.Scale(root, from_=1, to=500, label="Slowdown Factor",
             orient=tk.HORIZONTAL, variable=slow_var).pack(fill='x', padx=10)

    ctrl = tk.Frame(root); ctrl.pack(pady=5)
    launch_btn = tk.Button(ctrl, text="Launch"); launch_btn.pack(side='left', padx=5)
    reset_btn = tk.Button(ctrl, text="Reset", state='disabled'); reset_btn.pack(side='left', padx=5)

    missile, target = None, None
    data = {}; anim_id = None
    acquired_label = [None]
    touch_time = [0.0]

    # Static ruler placement
    NUM_TICKS = 11; RULER_X = 24; RULER_PAD = 40
    ruler_line_ids = []; ruler_text_ids = []

    def rotate_and_translate(pts, ang, tx, ty):
        c, s = np.cos(ang), np.sin(ang); out=[]
        for px, py in pts:
            rx = px*c - py*s + tx; ry = px*s + py*c + ty; out.extend([rx, ry])
        return out

    def clear_ruler():
        for iid in ruler_line_ids + ruler_text_ids:
            canvas.delete(iid)
        ruler_line_ids.clear(); ruler_text_ids.clear()

    def init_static_ruler():
        clear_ruler()
        for j in range(NUM_TICKS):
            y_pix = RULER_PAD + (H - 2*RULER_PAD) * j / (NUM_TICKS - 1)
            l = canvas.create_line(RULER_X, y_pix, RULER_X+12, y_pix, width=2, fill='#1b3a57')
            t = canvas.create_text(RULER_X-6, y_pix, text="", anchor='e', fill='#0b2239', font=('Arial', 9))
            ruler_line_ids.append(l); ruler_text_ids.append(t)

    def update_ruler_numbers(y_focus, max_y):
        # Update tick labels only (ruler graphics stay put)
        for j in range(NUM_TICKS):
            y_pix = RULER_PAD + (H - 2*RULER_PAD) * j / (NUM_TICKS - 1)
            frac = 1 - (y_pix - RULER_PAD) / (H - 2*RULER_PAD)
            y_value = y_focus - max_y/2 + frac*max_y
            y_value = min(max(y_value, 0), max_y)
            canvas.itemconfig(ruler_text_ids[j], text=f"{y_value:.0f}")

    def launch():
        nonlocal data, touch_time, missile, target, acquired_label, cloud_ids
        max_x = 40000; max_y = 25000
        mode = mode_var.get()
        launch_btn.config(state='disabled'); reset_btn.config(state='normal')
        # refresh clouds
        clear_clouds(canvas, cloud_ids); cloud_ids = draw_clouds(canvas, W, H)
        # integrate
        data['t'], data['x'], data['y'], data['xt'], data['yt'], data['vx'], data['vy'] = compute_trajectories(mode, max_x, max_y)
        data['max_x'] = max_x; data['max_y'] = max_y
        touch_time[0] = 0.0
        if missile: canvas.delete(missile)
        if target: canvas.delete(target)
        if acquired_label[0]: acquired_label[0].destroy(); acquired_label[0] = None
        init_static_ruler()
        # place sprites
        sx, sy = map_to_canvas(data['x'][0],  data['y'][0],  data['max_x'], data['max_y'], W, H)
        tx, ty = map_to_canvas(data['xt'][0], data['yt'][0], data['max_x'], data['max_y'], W, H)
        missile = canvas.create_polygon(*rotate_and_translate(MISSILE_SHAPE, np.pi/2, sx, sy), fill='#8c939b', outline='#2a2f33', width=2)
        # target always upright (vertical)
        target  = canvas.create_polygon(*rotate_and_translate(TARGET_SHAPE,  np.pi/2, tx, ty), fill='#cc2b2b', outline='#5b1010', width=2)
        data['missile'], data['target'] = missile, target
        animate(0)

    def animate(i):
        nonlocal anim_id, missile, target, acquired_label
        t_arr, x_arr, y_arr = data['t'], data['x'], data['y']
        xt_arr, yt_arr, vx_arr, vy_arr = data['xt'], data['yt'], data['vx'], data['vy']
        if i < len(t_arr):
            sx, sy = map_to_canvas(x_arr[i],  y_arr[i],  data['max_x'], data['max_y'], W, H)
            tx, ty = map_to_canvas(xt_arr[i], yt_arr[i], data['max_x'], data['max_y'], W, H)
            ang = np.arctan2(vy_arr[i], vx_arr[i]) + np.pi/2
            sx = min(max(sx, 0), W); sy = min(max(sy, 0), H)
            tx = min(max(tx, 0), W); ty = min(max(ty, 0), H)
            canvas.coords(missile, *rotate_and_translate(MISSILE_SHAPE, ang,    sx, sy))
            canvas.coords(target,  *rotate_and_translate(TARGET_SHAPE,  np.pi/2, tx, ty))  # red target stays vertical

            dist = np.hypot(x_arr[i] - xt_arr[i], y_arr[i] - yt_arr[i])
            dt = t_arr[1] - t_arr[0] if i > 0 else 0.05
            update_ruler_numbers(y_arr[i], data['max_y'])

            if dist < intercept_radius:
                touch_time[0] += dt
            else:
                touch_time[0] = 0.0

            if touch_time[0] >= 3.0:
                print("Target acquired!")
                launch_btn.config(state='normal'); reset_btn.config(state='disabled')
                if acquired_label[0]: acquired_label[0].destroy()
                acquired_label[0] = tk.Label(root, text="Target acquired!", font=("Arial", 30, "bold"), fg="green", bg="yellow")
                acquired_label[0].place(relx=0.5, rely=0.08, anchor='center')
                return

            delay = min(max(int(dt * 1000 * slow_var.get()), 1), 200)  # cap 200ms
            anim_id = root.after(delay, lambda idx=i+1: animate(idx))
        else:
            launch_btn.config(state='normal'); reset_btn.config(state='disabled')

    def reset():
        nonlocal anim_id, missile, target, acquired_label, cloud_ids
        if anim_id: root.after_cancel(anim_id)
        if missile: canvas.delete(missile)
        if target: canvas.delete(target)
        if acquired_label[0]: acquired_label[0].destroy(); acquired_label[0] = None
        missile = None; target = None
        clear_ruler(); launch_btn.config(state='normal'); reset_btn.config(state='disabled')
        clear_clouds(canvas, cloud_ids); cloud_ids = draw_clouds(canvas, W, H)

    launch_btn.config(command=launch); reset_btn.config(command=reset)
    root.mainloop()

if __name__ == '__main__':
    main()

