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
    V_STRAIGHT = 8
    V_CORNER = 2
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
    z = np.zeros_like(x)

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

    x_loc = ca.cos(c_yaw) * dx + ca.sin(c_yaw) * dy
    y_loc = -ca.sin(c_yaw) * dx + ca.cos(c_yaw) * dy
    z_loc = dz

    eps = 1e-4

    vy_bound = y_loc ** 2 - 0.25
    smax_y = 0.5 * (vy_bound + ca.sqrt(vy_bound ** 2 + eps))

    vz_bound = (z_loc + 0.75) ** 2 - 0.49
    smax_z = 0.5 * (vz_bound + ca.sqrt(vz_bound ** 2 + eps))

    d_out = smax_y + smax_z
    h_wall = (x_loc + 1.0) * d_out

    h_expr = ca.vertcat(x_loc, y_loc, z_loc, h_wall)
    model.con_h_expr = h_expr
    model.con_h_expr_e = h_expr

    return model

# по центру открытого plane коробки и back up(обЪть коробку если high level планирует сквозь коробку)
def setup_hl_nmpc(model, dt, N):
    ocp = AcadosOcp()
    ocp.model = model
    nx = model.x.size()[0]
    nu = model.u.size()[0]

    ocp.dims.N = N
    ocp.solver_options.tf = N * dt

    Q = np.diag([200, 200, 200, 10, 10, 10])
    R = np.diag([0.1, 0.1, 0.1])
    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS'
    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    ocp.cost.W_e = np.diag([2000, 2000, 2000, 10, 10, 10])

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

    # ==========================================
    # NEW: Hard State Constraint on Z-axis (Index 2)
    # ==========================================
    ocp.constraints.idxbx = np.array([2])
    ocp.constraints.lbx = np.array([-100.0])  # Unconstrained upwards
    ocp.constraints.ubx = np.array([-0.5] ) # Strictly prevents reaching 0.5
    ocp.constraints.idxbx_e = np.array([2])
    ocp.constraints.lbx_e = np.array([-100.0])
    ocp.constraints.ubx_e = np.array([-0.5])

    ocp.constraints.x0 = np.zeros(nx)

    ocp.constraints.lh = np.array([-20.0, -0.1, -1.45, -10000.0])
    ocp.constraints.uh = np.array([-0.1, 0.1, -0.05, 0.02])
    ocp.constraints.lh_e = np.array([-20.0, -0.1, -1.45, -10000.0])
    ocp.constraints.uh_e = np.array([-0.5, 0.5, -0.05, 0.02])

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
# 2. LOW-LEVEL QUADROTOR DYNAMICS
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

    return model


def setup_ll_nmpc(model, dt, N):
    ocp = AcadosOcp()
    ocp.model = model
    nx = model.x.size()[0]
    nu = model.u.size()[0]

    ocp.dims.N = N
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
    # ==========================================
    # NEW: Include Z-axis (Index 2) alongside Angles
    # ==========================================
    ocp.constraints.idxbx = np.array([2, 6, 7])
    ocp.constraints.lbx = np.array([-100.0, -max_angle, -max_angle])
    ocp.constraints.ubx = np.array([-0.5, max_angle, max_angle])

    ocp.constraints.idxbx_e = np.array([2, 6, 7])
    ocp.constraints.lbx_e = np.array([-100.0, -max_angle, -max_angle])
    ocp.constraints.ubx_e = np.array([-0.5, max_angle, max_angle])

    ocp.constraints.x0 = np.zeros(nx)

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
    x_current[0:3] = [9, -9, -4]

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

    for i in range(N_sim):
        curr_target_x = target_x_full[i]
        curr_target_y = target_y_full[i]
        curr_target_z = target_z_full[i]

        if i == 0:
            curr_vx, curr_vy, curr_vz = 0.0, 0.0, 0.0
            car_yaw = 0.0
            curr_omega = 0.0
        elif i == 1:
            curr_vx = (target_x_full[i] - target_x_full[i - 1]) / dt_ll
            curr_vy = (target_y_full[i] - target_y_full[i - 1]) / dt_ll
            curr_vz = (target_z_full[i] - target_z_full[i - 1]) / dt_ll
            car_yaw = np.arctan2(curr_vy, curr_vx)
            curr_omega = 0.0
        else:
            curr_vx = (target_x_full[i] - target_x_full[i - 1]) / dt_ll
            curr_vy = (target_y_full[i] - target_y_full[i - 1]) / dt_ll
            curr_vz = (target_z_full[i] - target_z_full[i - 1]) / dt_ll
            car_yaw = np.arctan2(curr_vy, curr_vx)

            prev_vx = (target_x_full[i - 1] - target_x_full[i - 2]) / dt_ll
            prev_vy = (target_y_full[i - 1] - target_y_full[i - 2]) / dt_ll
            prev_yaw = np.arctan2(prev_vy, prev_vx) if (prev_vx != 0 or prev_vy != 0) else 0.0

            yaw_diff = (car_yaw - prev_yaw + np.pi) % (2 * np.pi) - np.pi
            curr_omega = yaw_diff / dt_ll

        curr_v = np.hypot(curr_vx, curr_vy)

        hl_x0 = np.array([x_current[0], x_current[1], x_current[2],
                          x_current[3], x_current[4], x_current[5]])
        hl_solver.set(0, "lbx", hl_x0)
        hl_solver.set(0, "ubx", hl_x0)

        for j in range(N_hl):
            T = j * dt_hl
            if abs(curr_omega) > 1e-4:
                pred_x = curr_target_x + (curr_v / curr_omega) * (np.sin(car_yaw + curr_omega * T) - np.sin(car_yaw))
                pred_y = curr_target_y + (curr_v / curr_omega) * (-np.cos(car_yaw + curr_omega * T) + np.cos(car_yaw))
                pred_yaw = car_yaw + curr_omega * T
            else:
                pred_x = curr_target_x + curr_v * T * np.cos(car_yaw)
                pred_y = curr_target_y + curr_v * T * np.sin(car_yaw)
                pred_yaw = car_yaw

            pred_z = curr_target_z + curr_vz * T
            pred_vx = curr_v * np.cos(pred_yaw)
            pred_vy = curr_v * np.sin(pred_yaw)

            hl_solver.set(j, "p", np.array([pred_x, pred_y, pred_z, pred_yaw]))

            yref_hl = np.zeros(9)
            yref_hl[0:3] = [pred_x, pred_y, pred_z - 0.75]
            yref_hl[3:6] = [pred_vx, pred_vy, curr_vz]
            hl_solver.set(j, "yref", yref_hl)

        T_e = N_hl * dt_hl
        if abs(curr_omega) > 1e-4:
            pred_x_e = curr_target_x + (curr_v / curr_omega) * (np.sin(car_yaw + curr_omega * T_e) - np.sin(car_yaw))
            pred_y_e = curr_target_y + (curr_v / curr_omega) * (-np.cos(car_yaw + curr_omega * T_e) + np.cos(car_yaw))
            pred_yaw_e = car_yaw + curr_omega * T_e
        else:
            pred_x_e = curr_target_x + curr_v * T_e * np.cos(car_yaw)
            pred_y_e = curr_target_y + curr_v * T_e * np.sin(car_yaw)
            pred_yaw_e = car_yaw

        pred_z_e = curr_target_z + curr_vz * T_e
        pred_vx_e = curr_v * np.cos(pred_yaw_e)
        pred_vy_e = curr_v * np.sin(pred_yaw_e)

        hl_solver.set(N_hl, "p", np.array([pred_x_e, pred_y_e, pred_z_e, pred_yaw_e]))

        yref_hl_e = np.zeros(6)
        yref_hl_e[0:3] = [pred_x_e, pred_y_e, pred_z_e - 0.75]
        yref_hl_e[3:6] = [pred_vx_e, pred_vy_e, curr_vz]
        hl_solver.set(N_hl, "yref", yref_hl_e)

        hl_solver.solve()

        hl_traj = np.zeros((N_hl, 6))
        for j in range(N_hl):
            hl_traj[j, :] = hl_solver.get(j, "x")

        ll_ref_x = np.interp(t_ll_arr, t_hl_arr, hl_traj[:, 0])
        ll_ref_y = np.interp(t_ll_arr, t_hl_arr, hl_traj[:, 1])
        ll_ref_z = np.interp(t_ll_arr, t_hl_arr, hl_traj[:, 2])
        ll_ref_vx = np.interp(t_ll_arr, t_hl_arr, hl_traj[:, 3])
        ll_ref_vy = np.interp(t_ll_arr, t_hl_arr, hl_traj[:, 4])
        ll_ref_vz = np.interp(t_ll_arr, t_hl_arr, hl_traj[:, 5])

        ll_solver.set(0, "lbx", x_current)
        ll_solver.set(0, "ubx", x_current)

        for j in range(N_ll):
            yref_ll = np.zeros(16)
            yref_ll[0:6] = [ll_ref_x[j], ll_ref_y[j], ll_ref_z[j],
                            ll_ref_vx[j], ll_ref_vy[j], ll_ref_vz[j]]
            yref_ll[8] = 0.0
            yref_ll[12] = hover_thrust
            ll_solver.set(j, "yref", yref_ll)

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