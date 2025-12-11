import numpy as np 
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ==========================================
# 1. SYSTEM PARAMETERS
# ==========================================
g = 9.81
R = 0.02
L = 1.0        # Beam length (m), from -0.5 to 0.5
alpha = g / (1 + (2 * R**2) / (5 * L**2))

# ==========================================
# 2. CONTROLLER (Full-State Compensator)
# ==========================================
# Controller gains (K) -> poles roughly at -2, -3
k2 = -5.0 / alpha
k1 = -6.0 / alpha
K = np.array([k1, k2])

# Observer gains (L) -> fast poles at -10, -11
l1 = 21.0
l2 = 110.0
L_gain = np.array([l1, l2])

# Observer state (estimate of [x, x_dot])
x_hat = np.array([0.0, 0.0])

# ==========================================
# 3. SIMULATION SETUP & CRITERIA
# ==========================================
dt = 0.02                 # 50 Hz
TOTAL_TIME = 20.0         # satu siklus simulasi 20 s

SURVIVE_TIME_REQ = 15.0   # dipakai untuk info "on beam"
CENTER_TIME_REQ = 5.0     # dipakai untuk info "center"

SAFE_TOL   = L / 2        # "on beam" = |x| <= 0.5
CENTER_TOL = 0.03         # "center zone" = |x| <= 3 cm
STILL_VEL_TOL  = 0.02     # dianggap "berhenti" jika |x_dot| <= 0.02 m/s
LAUNCH_VEL_TOL = 0.005    # dianggap mulai meluncur jika |x_dot| > 0.005 m/s

# Window waktu bola mencapai target (center & berhenti)
TARGET_WINDOW_MIN = 15.0  # 15 detik
TARGET_WINDOW_MAX = 16.0  # 16 detik

# --- POSISI AWAL PERTAMA (DIPAKAI TERUS SETIAP RESET) ---
initial_position_first = np.random.uniform(-L / 2.1, L / 2.1)

current_state = [initial_position_first, 0.0]   # [position, velocity]
beam_angle = [0.0]                              # sudut beam (radian)

history_t = [0.0]
history_pos = [initial_position_first]

# Counters info
time_on_beam = 0.0
center_time = 0.0

# --------- ANGLE / TIME INFOS ----------
first_reach_time = None          # waktu pertama kali benar-benar di center
angle_at_target_deg = None       # sudut beam saat pertama kali di center
reach_status = "Not reached yet" # status waktu target

first_launch_time = None         # waktu pertama kali bola mulai meluncur
first_launch_angle_deg = None    # sudut beam saat pertama kali bola meluncur

# Flag & timer sukses
success_active = False
stabilized_at_time = None   # waktu global ketika sukses pertama kali
stable_T = 0.0              # counter T yang naik dari 0
T_target = 0.0              # T maksimum = 20 - time_stabilized

# ==========================================
# 4. CONTROL & DYNAMICS
# ==========================================
def student_control_system(x_measured, x_dot_measured):
    global x_hat

    u_prev = beam_angle[0]

    # Predict (linear)
    x_hat_dot_pred_0 = x_hat[1]
    x_hat_dot_pred_1 = -alpha * u_prev

    # Measurement y = x
    y = x_measured
    error = y - x_hat[0]

    correction_0 = L_gain[0] * error
    correction_1 = L_gain[1] * error

    x_hat[0] += (x_hat_dot_pred_0 + correction_0) * dt
    x_hat[1] += (x_hat_dot_pred_1 + correction_1) * dt

    u = -(K[0] * x_hat[0] + K[1] * x_hat[1])
    return np.clip(u, -0.2, 0.2)

def ball_and_beam_dynamic(t, state):
    x, x_dot = state
    u = student_control_system(x, x_dot)
    beam_angle[0] = u
    x_ddot = -alpha * np.sin(u)
    return [x_dot, x_ddot]

# ==========================================
# 5. FIGURE & AXES
# ==========================================
fig, (ax_anim, ax_graph) = plt.subplots(
    2, 1, figsize=(9, 9),
    gridspec_kw={'height_ratios': [1, 1]}
)
plt.subplots_adjust(hspace=0.3)

# --- Animation axis ---
ax_anim.set_xlim(-L/2 - 0.2, L/2 + 0.2)
ax_anim.set_ylim(-0.3, 0.4)
ax_anim.set_title("Challenge 1: Full-State Compensator (Ball and Beam)")
ax_anim.set_xlabel("Beam Length (m)")
ax_anim.set_ylabel("Height (m)")

ax_anim.axvspan(-L/2, L/2, color='green', alpha=0.15, label='Safe Zone (on beam)')
ax_anim.axvspan(-CENTER_TOL, CENTER_TOL, color='yellow', alpha=0.4, label='Center Zone')
ax_anim.axvline(0.0, color='black', linestyle=':', alpha=0.8, label='Center (x = 0)')

beam_line, = ax_anim.plot([], [], 'k-', lw=5, label='Beam')
ball, = ax_anim.plot([], [], 'ro', markersize=12, zorder=5, label='Ball')

timer_text   = ax_anim.text(0.02, 0.95, '', transform=ax_anim.transAxes,
                            fontsize=11, fontweight='bold', va='top')
info_text    = ax_anim.text(0.02, 0.88, '', transform=ax_anim.transAxes,
                            fontsize=9, va='top')
success_text = ax_anim.text(0.02, 0.72, '', transform=ax_anim.transAxes,
                            fontsize=10, color='green', fontweight='bold', va='top')

ax_anim.legend(loc='upper right', fontsize='small')

# --- Response axis ---
ax_graph.set_title("Position Response vs Time")
ax_graph.set_xlabel("Time (s)")
ax_graph.set_ylabel("Position (m) [0 = Center]")
ax_graph.set_xlim(0, TOTAL_TIME)
ax_graph.set_ylim(-0.6, 0.6)
ax_graph.grid(True, linestyle=':', alpha=0.6)

ax_graph.axhline(L/2,   color='red',   linestyle='--', alpha=0.5, label='Beam Edge (+0.5)')
ax_graph.axhline(-L/2,  color='red',   linestyle='--', alpha=0.5, label='Beam Edge (-0.5)')
ax_graph.axhline(0.0,   color='green', linestyle='-',  alpha=0.3, label='Center (0.0)')

line_pos, = ax_graph.plot([], [], 'b-', lw=1.5, label='Ball Position')
ax_graph.legend(loc='upper right', fontsize='small')

# ==========================================
# 6. RESET FUNCTION
# ==========================================
def reset_simulation():
    global current_state, x_hat, beam_angle
    global history_t, history_pos
    global time_on_beam, center_time
    global success_active, stabilized_at_time, stable_T, T_target
    global first_reach_time, angle_at_target_deg, reach_status
    global first_launch_time, first_launch_angle_deg
    global initial_position_first

    # Kembali ke posisi & kemiringan awal pertama
    current_state = [initial_position_first, 0.0]
    x_hat[:] = 0.0
    beam_angle[0] = 0.0

    history_t = [0.0]
    history_pos = [initial_position_first]
    ax_graph.set_xlim(0, TOTAL_TIME)

    time_on_beam = 0.0
    center_time = 0.0

    success_active = False
    stabilized_at_time = None
    stable_T = 0.0
    T_target = 0.0

    first_reach_time = None
    angle_at_target_deg = None
    reach_status = "Not reached yet"

    first_launch_time = None
    first_launch_angle_deg = None

    timer_text.set_text('')
    # info_text tidak perlu di-clear, karena akan di-overwrite di update()
    success_text.set_text('')

# ==========================================
# 7. ANIMATION FUNCTIONS
# ==========================================
def init():
    beam_line.set_data([], [])
    ball.set_data([], [])
    line_pos.set_data([], [])
    timer_text.set_text('')
    info_text.set_text('')
    success_text.set_text('')
    return beam_line, ball, line_pos, timer_text, info_text, success_text

def update(frame):
    global current_state, time_on_beam, center_time
    global success_active, stabilized_at_time, stable_T, T_target
    global first_reach_time, angle_at_target_deg, reach_status
    global first_launch_time, first_launch_angle_deg

    current_time = history_t[-1]

    # Auto-reset tiap 20 detik
    if current_time >= TOTAL_TIME:
        reset_simulation()
        current_time = history_t[-1]

    # Integrasi dinamika dt
    sol = solve_ivp(ball_and_beam_dynamic, [0, dt], current_state, t_eval=[dt])
    current_state[:] = sol.y[:, -1]
    pos_val = current_state[0]
    vel_val = current_state[1]

    # Cegah keluar batang
    if pos_val > L/2:
        pos_val = L/2 - 1e-4
        current_state[0] = pos_val
        current_state[1] = 0.0
        vel_val = 0.0
    elif pos_val < -L/2:
        pos_val = -L/2 + 1e-4
        current_state[0] = pos_val
        current_state[1] = 0.0
        vel_val = 0.0

    current_time = history_t[-1] + dt
    history_t.append(current_time)
    history_pos.append(pos_val)
    line_pos.set_data(history_t, history_pos)

    # Info counters
    if abs(pos_val) <= SAFE_TOL:
        time_on_beam += dt
    else:
        time_on_beam = 0.0

    if abs(pos_val) <= CENTER_TOL:
        center_time += dt
    else:
        center_time = 0.0

    # -------------------------------
    #  Sudut saat pertama bola meluncur
    # -------------------------------
    if first_launch_time is None and abs(vel_val) > LAUNCH_VEL_TOL:
        first_launch_time = current_time
        first_launch_angle_deg = np.degrees(beam_angle[0])

    # -------------------------------
    #  Target time & beam angle di target
    # -------------------------------
    still_at_center = (abs(pos_val) <= CENTER_TOL) and (abs(vel_val) <= STILL_VEL_TOL)

    if first_reach_time is None and still_at_center:
        # Pertama kali BENAR-BENAR di center
        first_reach_time = current_time
        angle_at_target_deg = np.degrees(beam_angle[0])

        if TARGET_WINDOW_MIN <= first_reach_time <= TARGET_WINDOW_MAX:
            reach_status = "OK (within 15–16s)"
        elif first_reach_time < TARGET_WINDOW_MIN:
            reach_status = "Too fast (<15s)"
        else:
            reach_status = "Too slow (>16s)"

    # -------------------------------
    #  Logika sukses:
    #  - hanya aktif jika target time ada & berada dalam window 15–16s
    #  - dan masih benar2 di tengah & diam
    # -------------------------------
    if (not success_active) \
       and (first_reach_time is not None) \
       and (TARGET_WINDOW_MIN <= first_reach_time <= TARGET_WINDOW_MAX) \
       and still_at_center:
        success_active = True
        stabilized_at_time = first_reach_time
        T_target = TOTAL_TIME - stabilized_at_time
        if T_target < 0:
            T_target = 0.0
        stable_T = 0.0  # mulai dari 0

    if success_active:
        stable_T += dt
        if stable_T > T_target:
            stable_T = T_target

    # -------------------------------
    #  Teks
    # -------------------------------
    timer_text.set_text(f"Time: {current_time:.1f}s / {TOTAL_TIME:.1f}s")

    # strings
    reached_str = "-" if first_reach_time is None else f"{first_reach_time:.2f}s"
    angle_target_str = "-" if angle_at_target_deg is None else f"{angle_at_target_deg:.2f}°"
    angle_launch_str = "-" if first_launch_angle_deg is None else f"{first_launch_angle_deg:.2f}°"

    info_text.set_text(
        f"On beam: {time_on_beam:4.1f}s / {SURVIVE_TIME_REQ:.1f}s\n"
        f"Center:  {center_time:4.1f}s / {CENTER_TIME_REQ:.1f}s\n"
        f"Reached at center: {reached_str}\n"
        f"Target time status: {reach_status}\n"
        f"Beam angle at first slide: {angle_launch_str}\n"
        f"Beam angle at target:      {angle_target_str}"
    )

    if success_active:
        success_text.set_text(
            f"Sukses! Stable (T = {stable_T:.1f}s)   |   "
            f"Has stabilized at: {stabilized_at_time:.2f}s"
        )
    else:
        success_text.set_text("")

    # Animasi beam & bola
    angle = beam_angle[0]

    beam_x = [-L/2 * np.cos(angle), L/2 * np.cos(angle)]
    beam_y = [-L/2 * np.sin(angle), L/2 * np.sin(angle)]
    beam_line.set_data(beam_x, beam_y)

    ball_x = beam_x[0] + (pos_val + L/2) * np.cos(angle)
    ball_y = beam_y[0] + (pos_val + L/2) * np.sin(angle)
    ball.set_data([ball_x], [ball_y])

    return beam_line, ball, line_pos, timer_text, info_text, success_text

# ==========================================
# 8. RUN ANIMATION
# ==========================================
ani = animation.FuncAnimation(
    fig,
    update,
    init_func=init,
    blit=True,
    interval=dt * 1000,
    frames=100000
)

plt.show()
