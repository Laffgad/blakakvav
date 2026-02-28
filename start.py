import os
import numpy as np
import scipy.linalg
import casadi as ca
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver


# =============================================================================
# CAR TRAJECTORY GENERATOR
# =============================================================================
def generate_car_trajectory(t_eval):
    V_STRAIGHT = 2.5
    V_CORNER = 0.5
    TRACK_W = 4.0
    TRACK_H = 6.0
    CORNER_R = 3.0
    res = 100
    points, vels = [], []

    def add_straight(p1, p2):
        points.extend(np.linspace(p1, p2, res))
        vels.extend([V_STRAIGHT] * res)

    def add_arc(center, start_angle, end_angle):
        angles = np.linspace(start_angle, end_angle, res)
        points.extend(np.column_stack((center[0] + CORNER_R * np.cos(angles),
                                       center[1] + CORNER_R * np.sin(angles))))
        vels.extend([V_CORNER] * res)

    for _ in range(3):
        add_straight([0, -TRACK_H - CORNER_R], [TRACK_W, -TRACK_H - CORNER_R])
        add_arc([TRACK_W, -TRACK_H], -np.pi / 2, 0)
        add_straight([TRACK_W + CORNER_R, -TRACK_H], [TRACK_W + CORNER_R, TRACK_H])
        add_arc([TRACK_W, TRACK_H], 0, np.pi / 2)
        add_straight([TRACK_W, TRACK_H + CORNER_R], [-TRACK_W, TRACK_H + CORNER_R])
        add_arc([-TRACK_W, TRACK_H], np.pi / 2, np.pi)
        add_straight([-TRACK_W - CORNER_R, TRACK_H], [-TRACK_W - CORNER_R, -TRACK_H])
        add_arc([-TRACK_W, -TRACK_H], np.pi, 3 * np.pi / 2)
        add_straight([-TRACK_W, -TRACK_H - CORNER_R], [0, -TRACK_H - CORNER_R])

    raw_points = np.array(points)
    raw_vels = np.array(vels)

    window_size = 150
    window = np.ones(window_size) / window_size
    vels_padded = np.concatenate((raw_vels[-window_size:], raw_vels, raw_vels[:window_size]))
    vels_smooth = np.convolve(vels_padded, window, mode='same')[window_size:-window_size]

    dx = np.diff(raw_points[:, 0], prepend=raw_points[0, 0])
    dy = np.diff(raw_points[:, 1], prepend=raw_points[0, 1])
    ds = np.sqrt(dx ** 2 + dy ** 2)
    ds[0] = 0
    dt_arr = ds / vels_smooth
    dt_arr[0] = 0
    cumulative_time = np.cumsum(dt_arr)

    x = np.interp(t_eval, cumulative_time, raw_points[:, 0])
    y = np.interp(t_eval, cumulative_time, raw_points[:, 1])
    z = np.full_like(x, -0.5)  # Car moves at z = -0.5

    return np.column_stack([x, y, z])


# =============================================================================
# 1. HIGH-LEVEL KINEMATIC PLANNER (Point Mass)
# =============================================================================
def create_hl_kinematic_model():
    model = AcadosModel()
    model.name = "point_mass_nmpc"

    px = ca.SX.sym('px')
    py = ca.SX.sym('py')
    pz = ca.SX.sym('pz')
    vx = ca.SX.sym('vx')
    vy = ca.SX.sym('vy')
    vz = ca.SX.sym('vz')
    x = ca.vertcat(px, py, pz, vx, vy, vz)

    ax = ca.SX.sym('ax')
    ay = ca.SX.sym('ay')
    az = ca.SX.sym('az')
    u = ca.vertcat(ax, ay, az)

    xdot = ca.vertcat(vx, vy, vz, ax, ay, az)

    model.x = x
    model.u = u
    model.xdot = ca.SX.sym('xdot', 6)
    model.f_expl_expr = xdot
    model.f_impl_expr = model.xdot - xdot

    p_car = ca.SX.sym('p_car', 4)  # [car_x, car_y, car_z, car_yaw]
    model.p = p_car

    c_x, c_y, c_z, c_yaw = p_car[0], p_car[1], p_car[2], p_car[3]

    dx = px - c_x
    dy = py - c_y
    dz = pz - c_z

    # Drone position in Car's local frame
    x_loc = ca.cos(c_yaw) * dx + ca.sin(c_yaw) * dy
    y_loc = -ca.sin(c_yaw) * dx + ca.cos(c_yaw) * dy
    z_loc = dz

    # Obstacle Field for the 5 Closed Walls
    eps = 1e-4

    vy_dist = y_loc ** 2 - 0.25
    smax_y = 0.5 * (vy_dist + ca.sqrt(vy_dist ** 2 + eps))

    vz_dist = (z_loc + 0.75) ** 2 - 0.49
    smax_z = 0.5 * (vz_dist + ca.sqrt(vz_dist ** 2 + eps))

    d_out = smax_y + smax_z

    # Repelling Wall
    h_wall = (x_loc + 1.0) * d_out

    h_expr = ca.vertcat(x_loc, y_loc, z_loc, h_wall)
    model.con_h_expr = h_expr
    model.con_h_expr_e = h_expr

    return model


def setup_hl_nmpc(model, dt, N):
    ocp = AcadosOcp()
    ocp.model = model
    nx = model.x.size()[0]
    nu = model.u.size()[0]

    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = N * dt

    Q = np.diag([2000, 2000, 2000, 10, 10, 10])
    R = np.diag([0.1, 0.1, 0.1])
    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS'
    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    ocp.cost.W_e = Q

    ocp.cost.Vx = np.zeros((nx + nu, nx))
    ocp.cost.Vx[:nx, :] = np.eye(nx)
    ocp.cost.Vu = np.zeros((nx + nu, nu))
    ocp.cost.Vu[nx:, :] = np.eye(nu)
    ocp.cost.Vx_e = np.eye(nx)

    ocp.cost.yref = np.zeros(nx + nu)
    ocp.cost.yref_e = np.zeros(nx)

    max_acc_xy = 15.0
    max_acc_z = 8.0
    ocp.constraints.idxbu = np.array([0, 1, 2])
    ocp.constraints.lbu = np.array([-max_acc_xy, -max_acc_xy, -max_acc_z])
    ocp.constraints.ubu = np.array([max_acc_xy, max_acc_xy, max_acc_z])
    ocp.constraints.x0 = np.zeros(nx)

    # Nonlinear constraint bounds (box dimensions in car frame)
    ocp.constraints.lh = np.array([-20.0, -0.5, -1.45, -10000.0])
    ocp.constraints.uh = np.array([0.5, 0.5, -0.05, 0.02])
    ocp.constraints.lh_e = np.array([-20.0, -0.5, -1.45, -10000.0])
    ocp.constraints.uh_e = np.array([0.5, 0.5, -0.05, 0.02])

    ocp.constraints.idxsh = np.array([0, 1, 2, 3])
    ocp.constraints.idxsh_e = np.array([0, 1, 2, 3])

    pen_l1 = 2000.0
    pen_wall = 10000.0
    pen_l2 = 500.0
    pen_l2_wall = 5000.0

    ocp.cost.zl = np.array([pen_l1, pen_l1, pen_l1, pen_wall])
    ocp.cost.zu = np.array([pen_l1, pen_l1, pen_l1, pen_wall])
    ocp.cost.Zl = np.array([pen_l2, pen_l2, pen_l2, pen_l2_wall])
    ocp.cost.Zu = np.array([pen_l2, pen_l2, pen_l2, pen_l2_wall])

    ocp.cost.zl_e = np.copy(ocp.cost.zl)
    ocp.cost.zu_e = np.copy(ocp.cost.zu)
    ocp.cost.Zl_e = np.copy(ocp.cost.Zl)
    ocp.cost.Zu_e = np.copy(ocp.cost.Zu)

    ocp.parameter_values = np.zeros(4)

    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'

    return AcadosOcpSolver(ocp, json_file="acados_ocp_hl.json")


# =============================================================================
# 2. LOW-LEVEL QUADROTOR DYNAMICS (WITH HARD DYNAMIC CONSTRAINTS)
# =============================================================================
def create_quadrotor_model():
    model = AcadosModel()
    model.name = "quadrotor_nmpc"

    m = 1.2577
    g = 9.81
    Ixx = 8.131036e-3
    Iyy = 8.131036e-3
    Izz = 0.01794236

    X = ca.SX.sym('X')
    Y = ca.SX.sym('Y')
    Z = ca.SX.sym('Z')
    dX = ca.SX.sym('dX')
    dY = ca.SX.sym('dY')
    dZ = ca.SX.sym('dZ')
    phi = ca.SX.sym('phi')
    theta = ca.SX.sym('theta')
    psi = ca.SX.sym('psi')
    p = ca.SX.sym('p')
    q = ca.SX.sym('q')
    r = ca.SX.sym('r')
    x = ca.vertcat(X, Y, Z, dX, dY, dZ, phi, theta, psi, p, q, r)

    F = ca.SX.sym('F')
    Mx = ca.SX.sym('Mx')
    My = ca.SX.sym('My')
    Mz = ca.SX.sym('Mz')
    u = ca.vertcat(F, Mx, My, Mz)

    ddX = -(F / m) * (ca.sin(psi) * ca.sin(phi) + ca.cos(psi) * ca.sin(theta) * ca.cos(phi))
    ddY = -(F / m) * (-ca.cos(psi) * ca.sin(phi) + ca.sin(psi) * ca.sin(theta) * ca.cos(phi))
    ddZ = -(F / m) * (ca.cos(theta) * ca.cos(phi)) + g

    dphi = p + ca.sin(phi) * ca.tan(theta) * q + ca.cos(phi) * ca.tan(theta) * r
    dtheta = ca.cos(phi) * q - ca.sin(phi) * r
    dpsi = ca.sin(phi) / ca.cos(theta) * q + ca.cos(phi) / ca.cos(theta) * r
    dp = (Iyy - Izz) / Ixx * q * r + Mx / Ixx
    dq = (Izz - Ixx) / Iyy * r * p + My / Iyy
    dr = (Ixx - Iyy) / Izz * p * q + Mz / Izz

    xdot = ca.vertcat(dX, dY, dZ, ddX, ddY, ddZ, dphi, dtheta, dpsi, dp, dq, dr)

    model.x = x
    model.u = u
    model.xdot = ca.SX.sym('xdot', 12)
    model.f_expl_expr = xdot
    model.f_impl_expr = model.xdot - xdot

    # --- ADDING HARD CONSTRAINTS FOR CAR BOX AVOIDANCE ---
    p_car = ca.SX.sym('p_car', 4)  # Dynamic Parameter: [car_x, car_y, car_z, car_yaw]
    model.p = p_car

    c_x, c_y, c_z, c_yaw = p_car[0], p_car[1], p_car[2], p_car[3]

    dx_val = X - c_x
    dy_val = Y - c_y
    dz_val = Z - c_z

    # Drone position in Car's local frame
    x_loc = ca.cos(c_yaw) * dx_val + ca.sin(c_yaw) * dy_val
    y_loc = -ca.sin(c_yaw) * dx_val + ca.cos(c_yaw) * dy_val
    z_loc = dz_val

    # Obstacle Field for the 5 Closed Walls (Top & Bottom open)
    eps_val = 1e-4

    vy_dist = y_loc ** 2 - 0.25
    smax_y = 0.5 * (vy_dist + ca.sqrt(vy_dist ** 2 + eps_val))

    vz_dist = (z_loc + 0.75) ** 2 - 0.49
    smax_z = 0.5 * (vz_dist + ca.sqrt(vz_dist ** 2 + eps_val))

    d_out = smax_y + smax_z
    h_wall = (x_loc + 1.0) * d_out  # Repelling wall formula

    model.con_h_expr = ca.vertcat(h_wall)
    model.con_h_expr_e = ca.vertcat(h_wall)

    return model


def setup_ll_nmpc(model, dt, N):
    ocp = AcadosOcp()
    ocp.model = model
    nx = model.x.size()[0]
    nu = model.u.size()[0]

    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = N * dt

    Q = np.diag([200, 200, 200, 20, 20, 20, 10, 10, 10, 1, 1, 1])
    R = np.diag([0.1, 0.1, 0.1, 0.1])
    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS'
    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    ocp.cost.W_e = Q

    ocp.cost.Vx = np.zeros((nx + nu, nx))
    ocp.cost.Vx[:nx, :] = np.eye(nx)
    ocp.cost.Vu = np.zeros((nx + nu, nu))
    ocp.cost.Vu[nx:, :] = np.eye(nu)
    ocp.cost.Vx_e = np.eye(nx)

    ocp.cost.yref = np.zeros(nx + nu)
    ocp.cost.yref_e = np.zeros(nx)

    max_thrust = 76.7636
    max_moment_x = 4.22
    max_moment_z = 0.6

    ocp.constraints.idxbu = np.array([0, 1, 2, 3])
    ocp.constraints.lbu = np.array([0.0, -max_moment_x, -max_moment_x, -max_moment_z])
    ocp.constraints.ubu = np.array([max_thrust, max_moment_x, max_moment_x, max_moment_z])

    max_angle = np.pi / 3.0

    ocp.constraints.idxbx = np.array([2, 6, 7])
    ocp.constraints.lbx = np.array([-100.0, -max_angle, -max_angle])
    ocp.constraints.ubx = np.array([0.0, max_angle, max_angle])

    ocp.constraints.idxbx_e = np.array([2, 6, 7])
    ocp.constraints.lbx_e = np.array([-100.0, -max_angle, -max_angle])
    ocp.constraints.ubx_e = np.array([0.0, max_angle, max_angle])

    ocp.constraints.x0 = np.zeros(nx)

    # --- IMPLEMENTING HARD NONLINEAR CONSTRAINTS ---
    # We assign bounds directly to lh and uh without any slack variables (idxsh),
    # making them absolute mathematical walls for the QP solver.
    ocp.constraints.lh = np.array([-10000.0])
    ocp.constraints.uh = np.array([0.02])
    ocp.constraints.lh_e = np.array([-10000.0])
    ocp.constraints.uh_e = np.array([0.02])
    ocp.parameter_values = np.zeros(4)  # Initialize dynamic parameter vector

    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'

    return AcadosOcpSolver(ocp, json_file="acados_ocp_ll.json")


# =============================================================================
# 3. SIMULATION + VISUALIZATION
# =============================================================================
if __name__ == "__main__":
    dt_ll = 0.02
    N_ll = 25
    dt_hl = 0.1
    N_hl = 30
    t_end = 45.0
    m = 1.2577
    g = 9.81
    N_sim = int(t_end / dt_ll)

    hl_model = create_hl_kinematic_model()
    hl_solver = setup_hl_nmpc(hl_model, dt_hl, N_hl)
    ll_model = create_quadrotor_model()
    ll_solver = setup_ll_nmpc(ll_model, dt_ll, N_ll)
    hover_thrust = m * g

    t_full = np.linspace(0, t_end, N_sim)
    pts = generate_car_trajectory(t_full)
    target_x_full, target_y_full, target_z_full = pts[:, 0], pts[:, 1], pts[:, 2]

    x_current = np.zeros(12)
    x_current[0:3] = [8, 8, -1.5]  # Drone starts with a -1.0 offset from the car

    # Note: f_expl_expr does not depend on the parameter p, so we do not pass it here
    f_fun = ca.Function("f", [ll_model.x, ll_model.u], [ll_model.f_expl_expr])

    plt.ion()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-9, 9)
    ax.set_ylim(-9, 9)
    ax.set_zlim(-9, 9)

    ax.plot(target_x_full, target_y_full, -target_z_full, color='gray', linestyle='--', alpha=0.4, label='Track')
    line_traj, = ax.plot([], [], [], 'b', linewidth=2.5, label='Drone Path')

    arm1, = ax.plot([], [], [], 'g', linewidth=3)
    arm2, = ax.plot([], [], [], 'g', linewidth=3)
    motor_pts = [ax.plot([], [], [], 'ko', markersize=6)[0] for _ in range(4)]

    drone_cube_faces = Poly3DCollection([], alpha=0.3, facecolor='orange', edgecolor='black', label='Drone Cube')
    ax.add_collection3d(drone_cube_faces)

    target_plot, = ax.plot([], [], [], 'r-', linewidth=2.5, label='Car (1x0.5m)')
    hl_path_plot, = ax.plot([], [], [], 'm:', linewidth=2, label='HL Interception Path')

    box_closed_faces = Poly3DCollection([], alpha=0.5, facecolor='cyan', edgecolor='black', label='Box Walls')
    box_open_face = Poly3DCollection([], alpha=0.15, facecolor='red', edgecolor='red', label='Box Entrance')
    ax.add_collection3d(box_closed_faces)
    ax.add_collection3d(box_open_face)

    ax.legend()
    trajectory = []
    u_history = []
    arm_length = 0.25

    drone_cube_h = 0.1

    t_hl_arr = np.linspace(0, (N_hl - 1) * dt_hl, N_hl)
    t_ll_arr = np.linspace(0, (N_ll - 1) * dt_ll, N_ll)

    landing_mode = False

    for i in range(N_sim):
        curr_target_x = target_x_full[i]
        curr_target_y = target_y_full[i]
        curr_target_z = target_z_full[i]

        if i == 0:
            curr_vx, curr_vy, curr_vz = 0.0, 0.0, 0.0
            car_yaw = 0.0
            car_omega = 0.0
        else:
            curr_vx = (target_x_full[i] - target_x_full[i - 1]) / dt_ll
            curr_vy = (target_y_full[i] - target_y_full[i - 1]) / dt_ll
            curr_vz = (target_z_full[i] - target_z_full[i - 1]) / dt_ll

            if curr_vx == 0 and curr_vy == 0:
                car_yaw = 0.0
            else:
                car_yaw = np.arctan2(curr_vy, curr_vx)

            if i == 1:
                car_omega = 0.0
            else:
                prev_vx = (target_x_full[i - 1] - target_x_full[i - 2]) / dt_ll
                prev_vy = (target_y_full[i - 1] - target_y_full[i - 2]) / dt_ll
                prev_yaw = np.arctan2(prev_vy, prev_vx) if (prev_vx != 0 or prev_vy != 0) else 0.0

                d_yaw = car_yaw - prev_yaw
                d_yaw = (d_yaw + np.pi) % (2 * np.pi) - np.pi
                car_omega = d_yaw / dt_ll

        v_speed = np.sqrt(curr_vx ** 2 + curr_vy ** 2)
        eps = 1e-4

        dx_w = x_current[0] - curr_target_x
        dy_w = x_current[1] - curr_target_y
        dz_w = x_current[2] - curr_target_z

        dx_l = np.cos(car_yaw) * dx_w + np.sin(car_yaw) * dy_w
        dy_l = -np.sin(car_yaw) * dx_w + np.cos(car_yaw) * dy_w

        if not landing_mode and dx_l > -0.6 and dx_l < 0.6 and abs(dy_l) < 0.6 and abs(dz_w + 1.0) < 0.3:
            landing_mode = True
            print(f"Landing Phase Initiated at step {i}!")

        if not landing_mode:
            hl_x0 = np.array([x_current[0], x_current[1], x_current[2],
                              x_current[3], x_current[4], x_current[5]])
            hl_solver.set(0, "lbx", hl_x0)
            hl_solver.set(0, "ubx", hl_x0)

            for j in range(N_hl):
                T_pred = j * dt_hl
                if abs(car_omega) > eps:
                    pred_x = curr_target_x + (v_speed / car_omega) * (
                            np.sin(car_yaw + car_omega * T_pred) - np.sin(car_yaw))
                    pred_y = curr_target_y + (v_speed / car_omega) * (
                            -np.cos(car_yaw + car_omega * T_pred) + np.cos(car_yaw))
                    pred_yaw = car_yaw + car_omega * T_pred
                else:
                    pred_x = curr_target_x + v_speed * T_pred * np.cos(car_yaw)
                    pred_y = curr_target_y + v_speed * T_pred * np.sin(car_yaw)
                    pred_yaw = car_yaw

                pred_z = curr_target_z + curr_vz * T_pred
                pred_vx = v_speed * np.cos(pred_yaw)
                pred_vy = v_speed * np.sin(pred_yaw)

                hl_solver.set(j, "p", np.array([pred_x, pred_y, pred_z, pred_yaw]))
                yref_hl = np.zeros(9)
                yref_hl[0:3] = [pred_x, pred_y, pred_z - 1.0]
                yref_hl[3:6] = [pred_vx, pred_vy, curr_vz]
                hl_solver.set(j, "yref", yref_hl)

            T_pred_e = N_hl * dt_hl
            if abs(car_omega) > eps:
                pred_x_e = curr_target_x + (v_speed / car_omega) * (
                        np.sin(car_yaw + car_omega * T_pred_e) - np.sin(car_yaw))
                pred_y_e = curr_target_y + (v_speed / car_omega) * (
                        -np.cos(car_yaw + car_omega * T_pred_e) + np.cos(car_yaw))
                pred_yaw_e = car_yaw + car_omega * T_pred_e
            else:
                pred_x_e = curr_target_x + v_speed * T_pred_e * np.cos(car_yaw)
                pred_y_e = curr_target_y + v_speed * T_pred_e * np.sin(car_yaw)
                pred_yaw_e = car_yaw

            pred_z_e = curr_target_z + curr_vz * T_pred_e
            pred_vx_e = v_speed * np.cos(pred_yaw_e)
            pred_vy_e = v_speed * np.sin(pred_yaw_e)

            hl_solver.set(N_hl, "p", np.array([pred_x_e, pred_y_e, pred_z_e, pred_yaw_e]))
            yref_hl_e = np.zeros(6)
            yref_hl_e[0:3] = [pred_x_e, pred_y_e, pred_z_e - 1.0]
            yref_hl_e[3:6] = [pred_vx_e, pred_vy_e, curr_vz]
            hl_solver.set(N_hl, "yref", yref_hl_e)

            hl_solver.solve()
            hl_traj = np.zeros((N_hl, 6))
            for j in range(N_hl):
                hl_traj[j, :] = hl_solver.get(j, "x")
        else:
            hl_traj = np.zeros((N_hl, 6))
            p_d_curr = x_current[0:3].copy()
            v_d_curr = x_current[3:6].copy()

            Kp = np.array([2.0, 2.0])
            Kd = np.array([1.0, 1.0])
            Kpz = 1.0
            Kdz = 0.5
            r_safe = 0.5
            alpha_cbf = 2.5

            for j in range(N_hl):
                T_pred = j * dt_hl
                if abs(car_omega) > eps:
                    pred_cx = curr_target_x + (v_speed / car_omega) * (
                            np.sin(car_yaw + car_omega * T_pred) - np.sin(car_yaw))
                    pred_cy = curr_target_y + (v_speed / car_omega) * (
                            -np.cos(car_yaw + car_omega * T_pred) + np.cos(car_yaw))
                    pred_cyaw = car_yaw + car_omega * T_pred
                else:
                    pred_cx = curr_target_x + v_speed * T_pred * np.cos(car_yaw)
                    pred_cy = curr_target_y + v_speed * T_pred * np.sin(car_yaw)
                    pred_cyaw = car_yaw

                pred_cz = curr_target_z + curr_vz * T_pred
                pred_cvx = v_speed * np.cos(pred_cyaw)
                pred_cvy = v_speed * np.sin(pred_cyaw)
                pred_cvz = curr_vz

                e_x = p_d_curr[0] - pred_cx
                e_y = p_d_curr[1] - pred_cy
                e_z = p_d_curr[2] - pred_cz
                e_vx = v_d_curr[0] - pred_cvx
                e_vy = v_d_curr[1] - pred_cvy
                e_vz = v_d_curr[2] - pred_cvz

                vd_x_des = pred_cvx - Kp[0] * e_x - Kd[0] * e_vx
                vd_y_des = pred_cvy - Kp[1] * e_y - Kd[1] * e_vy

                target_z_offset = -1.0
                vd_z_des = pred_cvz - Kpz * (e_z - target_z_offset) - Kdz * e_vz
                vd_z_safe = np.clip(vd_z_des, -0.6, 0.6)

                A = np.array([2 * e_x, 2 * e_y])
                b = 2 * e_x * pred_cvx + 2 * e_y * pred_cvy + alpha_cbf * (r_safe ** 2 - e_x ** 2 - e_y ** 2)

                vd_des_2d = np.array([vd_x_des, vd_y_des])
                A_dot_v = np.dot(A, vd_des_2d)

                if A_dot_v <= b:
                    vd_safe = vd_des_2d
                else:
                    norm_A_sq = np.dot(A, A)
                    if norm_A_sq > 1e-6:
                        vd_safe = vd_des_2d - ((A_dot_v - b) / norm_A_sq) * A
                    else:
                        vd_safe = vd_des_2d

                v_norm = np.linalg.norm(vd_safe)
                if v_norm > 4.0:
                    vd_safe = (vd_safe / v_norm) * 4.0

                v_d_curr[0] = vd_safe[0]
                v_d_curr[1] = vd_safe[1]
                v_d_curr[2] = vd_z_safe

                p_d_curr += v_d_curr * dt_hl

                hl_traj[j, 0:3] = p_d_curr
                hl_traj[j, 3:6] = v_d_curr

        ll_ref_x = np.interp(t_ll_arr, t_hl_arr, hl_traj[:, 0])
        ll_ref_y = np.interp(t_ll_arr, t_hl_arr, hl_traj[:, 1])
        ll_ref_z = np.interp(t_ll_arr, t_hl_arr, hl_traj[:, 2])
        ll_ref_vx = np.interp(t_ll_arr, t_hl_arr, hl_traj[:, 3])
        ll_ref_vy = np.interp(t_ll_arr, t_hl_arr, hl_traj[:, 4])
        ll_ref_vz = np.interp(t_ll_arr, t_hl_arr, hl_traj[:, 5])

        ll_solver.set(0, "lbx", x_current)
        ll_solver.set(0, "ubx", x_current)

        # --- DYNAMICALLY SETTING THE HARD CONSTRAINT FOR LOW-LEVEL ---
        for j in range(N_ll):
            T_pred_ll = j * dt_ll
            if abs(car_omega) > eps:
                pred_cx = curr_target_x + (v_speed / car_omega) * (
                            np.sin(car_yaw + car_omega * T_pred_ll) - np.sin(car_yaw))
                pred_cy = curr_target_y + (v_speed / car_omega) * (
                            -np.cos(car_yaw + car_omega * T_pred_ll) + np.cos(car_yaw))
                pred_cyaw = car_yaw + car_omega * T_pred_ll
            else:
                pred_cx = curr_target_x + v_speed * T_pred_ll * np.cos(car_yaw)
                pred_cy = curr_target_y + v_speed * T_pred_ll * np.sin(car_yaw)
                pred_cyaw = car_yaw
            pred_cz = curr_target_z + curr_vz * T_pred_ll

            ll_solver.set(j, "p", np.array([pred_cx, pred_cy, pred_cz, pred_cyaw]))

            yref_ll = np.zeros(16)
            yref_ll[0:6] = [ll_ref_x[j], ll_ref_y[j], ll_ref_z[j],
                            ll_ref_vx[j], ll_ref_vy[j], ll_ref_vz[j]]
            yref_ll[8] = 0.0
            yref_ll[12] = hover_thrust
            ll_solver.set(j, "yref", yref_ll)

        # Terminal Node Parameter update
        T_pred_ll_e = N_ll * dt_ll
        if abs(car_omega) > eps:
            pred_cx_e = curr_target_x + (v_speed / car_omega) * (
                        np.sin(car_yaw + car_omega * T_pred_ll_e) - np.sin(car_yaw))
            pred_cy_e = curr_target_y + (v_speed / car_omega) * (
                        -np.cos(car_yaw + car_omega * T_pred_ll_e) + np.cos(car_yaw))
            pred_cyaw_e = car_yaw + car_omega * T_pred_ll_e
        else:
            pred_cx_e = curr_target_x + v_speed * T_pred_ll_e * np.cos(car_yaw)
            pred_cy_e = curr_target_y + v_speed * T_pred_ll_e * np.sin(car_yaw)
            pred_cyaw_e = car_yaw
        pred_cz_e = curr_target_z + curr_vz * T_pred_ll_e

        ll_solver.set(N_ll, "p", np.array([pred_cx_e, pred_cy_e, pred_cz_e, pred_cyaw_e]))

        yref_ll_e = np.zeros(12)
        yref_ll_e[0:6] = [ll_ref_x[N_ll - 1], ll_ref_y[N_ll - 1], ll_ref_z[N_ll - 1],
                          ll_ref_vx[N_ll - 1], ll_ref_vy[N_ll - 1], ll_ref_vz[N_ll - 1]]
        yref_ll_e[8] = 0.0
        ll_solver.set(N_ll, "yref", yref_ll_e)

        ll_solver.solve()

        u0 = ll_solver.get(0, "u")
        u_history.append(u0)

        k1 = np.array(f_fun(x_current, u0)).flatten()
        k2 = np.array(f_fun(x_current + dt_ll / 2 * k1, u0)).flatten()
        k3 = np.array(f_fun(x_current + dt_ll / 2 * k2, u0)).flatten()
        k4 = np.array(f_fun(x_current + dt_ll * k3, u0)).flatten()
        x_current = x_current + dt_ll / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        trajectory.append(x_current.copy())
        traj = np.array(trajectory)

        line_traj.set_data(traj[:, 0], traj[:, 1])
        line_traj.set_3d_properties(-traj[:, 2])
        hl_path_plot.set_data(hl_traj[:, 0], hl_traj[:, 1])
        hl_path_plot.set_3d_properties(-hl_traj[:, 2])

        X, Y, Z = x_current[0:3]
        phi, theta, psi = x_current[6:9]

        Rz = np.array([[np.cos(psi), -np.sin(psi), 0],
                       [np.sin(psi), np.cos(psi), 0],
                       [0, 0, 1]])
        Ry = np.array([[np.cos(theta), 0, np.sin(theta)],
                       [0, 1, 0],
                       [-np.sin(theta), 0, np.cos(theta)]])
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(phi), -np.sin(phi)],
                       [0, np.sin(phi), np.cos(phi)]])
        R_drone = Rz @ Ry @ Rx

        arm1_body = np.array([[-arm_length, 0, 0], [arm_length, 0, 0]]).T
        arm2_body = np.array([[0, -arm_length, 0], [0, arm_length, 0]]).T
        arm1_world = R_drone @ arm1_body
        arm2_world = R_drone @ arm2_body

        arm1.set_data(X + arm1_world[0, :], Y + arm1_world[1, :])
        arm1.set_3d_properties(-(Z + arm1_world[2, :]))
        arm2.set_data(X + arm2_world[0, :], Y + arm2_world[1, :])
        arm2.set_3d_properties(-(Z + arm2_world[2, :]))

        motors_body = np.array([[arm_length, 0, 0], [-arm_length, 0, 0], [0, arm_length, 0], [0, -arm_length, 0]]).T
        motors_world = R_drone @ motors_body
        for k in range(4):
            motor_pts[k].set_data([X + motors_world[0, k]], [Y + motors_world[1, k]])
            motor_pts[k].set_3d_properties([-(Z + motors_world[2, k])])

        d_L = 2.0 * arm_length
        d_W = 2.0 * arm_length
        d_H = drone_cube_h

        d_v = np.array([
            [-d_L / 2, -d_W / 2, -d_H / 2], [d_L / 2, -d_W / 2, -d_H / 2],
            [d_L / 2, d_W / 2, -d_H / 2], [-d_L / 2, d_W / 2, -d_H / 2],
            [-d_L / 2, -d_W / 2, d_H / 2], [d_L / 2, -d_W / 2, d_H / 2],
            [d_L / 2, d_W / 2, d_H / 2], [-d_L / 2, d_W / 2, d_H / 2]
        ]).T

        d_world = R_drone @ d_v
        d_pts_x = X + d_world[0, :]
        d_pts_y = Y + d_world[1, :]
        d_pts_z = -(Z + d_world[2, :])

        d_pts_3d = np.column_stack((d_pts_x, d_pts_y, d_pts_z))

        d_faces = [
            [d_pts_3d[0], d_pts_3d[1], d_pts_3d[2], d_pts_3d[3]],
            [d_pts_3d[4], d_pts_3d[5], d_pts_3d[6], d_pts_3d[7]],
            [d_pts_3d[0], d_pts_3d[1], d_pts_3d[5], d_pts_3d[4]],
            [d_pts_3d[1], d_pts_3d[2], d_pts_3d[6], d_pts_3d[5]],
            [d_pts_3d[2], d_pts_3d[3], d_pts_3d[7], d_pts_3d[6]],
            [d_pts_3d[3], d_pts_3d[0], d_pts_3d[4], d_pts_3d[7]]
        ]

        drone_cube_faces.set_verts(d_faces)

        car_L, car_W = 1.0, 0.5
        car_local = np.array([
            [car_L / 2, car_W / 2, 0], [-car_L / 2, car_W / 2, 0],
            [-car_L / 2, -car_W / 2, 0], [car_L / 2, -car_W / 2, 0],
            [car_L / 2, car_W / 2, 0]
        ]).T

        R_car = np.array([
            [np.cos(car_yaw), -np.sin(car_yaw), 0],
            [np.sin(car_yaw), np.cos(car_yaw), 0],
            [0, 0, 1]
        ])

        car_world = R_car @ car_local
        car_x_pts = curr_target_x + car_world[0, :]
        car_y_pts = curr_target_y + car_world[1, :]
        car_z_pts = -curr_target_z + car_world[2, :]

        target_plot.set_data(car_x_pts, car_y_pts)
        target_plot.set_3d_properties(car_z_pts)

        box_L, box_W, box_H = 1.5, 1.5, 1.5
        b_v = np.array([
            [-box_L / 2, -box_W / 2, 0], [box_L / 2, -box_W / 2, 0],
            [box_L / 2, box_W / 2, 0], [-box_L / 2, box_W / 2, 0],
            [-box_L / 2, -box_W / 2, box_H], [box_L / 2, -box_W / 2, box_H],
            [box_L / 2, box_W / 2, box_H], [-box_L / 2, box_W / 2, box_H]
        ]).T

        b_world = R_car @ b_v
        box_x = curr_target_x + b_world[0, :]
        box_y = curr_target_y + b_world[1, :]
        box_z = -curr_target_z + b_world[2, :]
        pts_3d = np.column_stack((box_x, box_y, box_z))

        closed_faces = [
            [pts_3d[0], pts_3d[1], pts_3d[2], pts_3d[3]],
            [pts_3d[4], pts_3d[5], pts_3d[6], pts_3d[7]],
            [pts_3d[0], pts_3d[1], pts_3d[5], pts_3d[4]],
            [pts_3d[3], pts_3d[2], pts_3d[6], pts_3d[7]],
            [pts_3d[1], pts_3d[2], pts_3d[6], pts_3d[5]]
        ]
        open_face = [[pts_3d[0], pts_3d[3], pts_3d[7], pts_3d[4]]]

        box_closed_faces.set_verts(closed_faces)
        box_open_face.set_verts(open_face)

        plt.draw()
        plt.pause(dt_ll)

    plt.ioff()
    plt.close()

    try:
        os.remove("acados_ocp_hl.json")
        os.remove("acados_ocp_ll.json")
    except:
        pass