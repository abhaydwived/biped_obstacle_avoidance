
import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time
import math
import random

class BipedWalkingEnv(gym.Env):
    metadata = {"render.modes": ['human']}

    # ---------- Grid / A* parameters ----------
    GRID_X_MIN, GRID_X_MAX = -3.0, 3.0
    GRID_Y_MIN, GRID_Y_MAX =  0.0, 8.0
    GRID_RES = 0.25
    ASTAR_8_CONNECTED = True

    # ---------- Assisted gait / controller defaults ----------
    assist_cpg = True
    cpg_base_freq = 1.6
    cpg_min_freq  = 0.6
    cpg_max_freq  = 2.2
    cpg_nom_speed = 0.25
    hip_amp_deg   = 16.0
    knee_amp_deg  = 22.0
    ankle_off_deg = -6.0
    cpg_mix       = 0.55
    anti_stall_steps = 25
    anti_stall_boost = 1.5

    lookahead_dist = 0.9
    max_wp_advance = 2
    lateral_kp     = 0.7
    heading_kp     = 1.2

    default_motor_force = 65.0
    pos_gain = 0.12
    vel_gain = 0.12

    # Goal condition
    goal_radius = 0.35

    def __init__(self, render=False, waypoints=None, seed: int = 123):
        super().__init__()
        self.render = render
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        self.physics_client = p.connect(p.GUI if self.render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8, physicsClientId=self.physics_client)
        self.timestep = 1.0 / 50.0
        p.setTimeStep(self.timestep)

        # Counters / episode state
        self.progress_stuck_counter = 0
        self.kick_stall_counter = 0
        self.last_check_pos = None
        self.stuck_threshold_steps = int(0.7 / self.timestep)

        self.max_steps = 5000
        self.step_counter = 0

        # World
        self.plane_id = p.loadURDF("plane.urdf")
        p.changeDynamics(self.plane_id, -1, lateralFriction=1.0)

        # Robot
        urdf_path = os.path.join(os.path.dirname(__file__), "yaw_up/urdf/yaw_up.urdf")
        self.robot_id = p.loadURDF(
            urdf_path,
            basePosition=[0, 0, 0.38],
            useFixedBase=False
        )
        p.changeDynamics(self.robot_id, -1, linearDamping=1.0, angularDamping=1.0)

        # Mass for CoT denominator
        try:
            self.robot_mass = float(p.getDynamicsInfo(self.robot_id, -1)[0])
        except Exception:
            self.robot_mass = 10.0  # fallback

        self.num_joints = p.getNumJoints(self.robot_id)
        self.joint_indices, self.joint_limits = [], []
        self.left_foot_link, self.right_foot_link = 3, 7

        # Build joint lists and name map
        self.joint_name_to_index = {}
        for i in range(self.num_joints):
            ji = p.getJointInfo(self.robot_id, i)
            if ji[2] == p.JOINT_REVOLUTE:
                self.joint_indices.append(i)
                self.joint_limits.append((ji[8], ji[9]))
                self.joint_name_to_index[ji[1].decode('utf-8')] = i
            if b"left_foot" in ji[12]:
                self.left_foot_link = i
            if b"right_foot" in ji[12]:
                self.right_foot_link = i

        # find hip pitch joints by name
        def find_idx(substrs):
            for name, idx in self.joint_name_to_index.items():
                if all(s in name for s in substrs):
                    return idx
            return None
        self.idx_L_hip = find_idx(["left", "hip"])
        self.idx_R_hip = find_idx(["right", "hip"])

        p.changeDynamics(self.robot_id, self.left_foot_link, lateralFriction=2.0)
        p.changeDynamics(self.robot_id, self.right_foot_link, lateralFriction=2.0)

        self.n_actuated_joints = len(self.joint_indices)
        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=(self.n_actuated_joints,), dtype=np.float32)

        # obs: angles, vels, base rpy(3), ω(3), v(3), contacts(2),
        #      distance_to_wp(1), heading_err(1), cross_track(1)
        obs_dim = self.n_actuated_joints * 2 + 3 + 3 + 3 + 2 + 1 + 1 + 1
        obs_high = np.array([np.finfo(np.float32).max] * obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        # Nav / Obstacles
        self.obstacles = []
        self.waypoints = waypoints if waypoints else []
        self.current_wp_idx = 0
        self.prev_distance_to_goal = 0.0
        self.path_debug_ids = []
        self.goal_debug_ids = []

        self.start_xy = np.array([0.0, 0.0], dtype=np.float32)
        self.goal_xy  = np.array([0.0, 6.0], dtype=np.float32)

        # IMPORTANT: set obstacles > 0 so you can SEE them
        self.num_obstacles = 16
        self.obs_global_scaling = 3.0
        self.obs_clear_radius = 0.35

        # ---- CPG state ----
        self._phase = 0.0
        self._kick_until = 0

        # ---- Episode metrics accumulators ----
        self._metrics = {}
        self._reset_episode_metrics()

        self.reset()

    # ===================== Metrics helpers =====================
    def _reset_episode_metrics(self):
        self._metrics = {
            "success": False,
            "fall": False,
            "collision_any": False,
            "steps": 0,
            "L_actual": 0.0,     # actual traversed distance (m)
            "L_star": 0.0,       # planned shortest path length (m) from A*
            "energy": 0.0,       # Joules proxy (sum |tau*qdot| dt)
            "cot": float("inf"),
            "spl": 0.0,
            "dist_to_goal": float("inf"),
        }
        self._prev_xy_for_length = None

    def _compute_L_star(self):
        # Length of planned polyline: start -> waypoints -> goal
        if not self.waypoints:
            return float(np.linalg.norm(self.goal_xy - self.start_xy))
        pts = [self.start_xy.tolist()] + [wp for wp in self.waypoints]
        L = 0.0
        for a, b in zip(pts[:-1], pts[1:]):
            a = np.array(a, dtype=np.float32)
            b = np.array(b, dtype=np.float32)
            L += float(np.linalg.norm(b - a))
        return L

    def get_episode_metrics(self):
        """Call this AFTER the episode ends (terminated or truncated)."""
        return dict(self._metrics)

    # ===================== Grid / A* helpers =====================
    def _grid_shape(self):
        nx = int(round((self.GRID_X_MAX - self.GRID_X_MIN) / self.GRID_RES)) + 1
        ny = int(round((self.GRID_Y_MAX - self.GRID_Y_MIN) / self.GRID_RES)) + 1
        return nx, ny

    def _xy_to_ij(self, x, y):
        i = int(round((x - self.GRID_X_MIN) / self.GRID_RES))
        j = int(round((y - self.GRID_Y_MIN) / self.GRID_RES))
        return i, j

    def _ij_to_xy(self, i, j):
        x = self.GRID_X_MIN + i * self.GRID_RES
        y = self.GRID_Y_MIN + j * self.GRID_RES
        return x, y

    def _in_bounds(self, i, j, nx, ny):
        return 0 <= i < nx and 0 <= j < ny

    def _astar(self, occ):
        nx, ny = occ.shape
        si, sj = self._xy_to_ij(self.start_xy[0], self.start_xy[1])
        gi, gj = self._xy_to_ij(self.goal_xy[0],  self.goal_xy[1])

        si = int(np.clip(si, 0, nx - 1)); sj = int(np.clip(sj, 0, ny - 1))
        gi = int(np.clip(gi, 0, nx - 1)); gj = int(np.clip(gj, 0, ny - 1))

        if occ[si, sj] or occ[gi, gj]:
            return []

        def h(i, j):
            dx = abs(i - gi); dy = abs(j - gj)
            return max(dx, dy) if self.ASTAR_8_CONNECTED else dx + dy

        if self.ASTAR_8_CONNECTED:
            nbrs = [(-1, 0),(1, 0),(0,-1),(0, 1),(-1,-1),(-1, 1),(1,-1),(1, 1)]
            step_cost = lambda di,dj: math.sqrt(2.0) if di != 0 and dj != 0 else 1.0
        else:
            nbrs = [(-1, 0),(1, 0),(0,-1),(0, 1)]
            step_cost = lambda di,dj: 1.0

        open_set = {(si, sj)}
        came_from = {}
        g = {(si, sj): 0.0}
        f = {(si, sj): h(si, sj)}

        while open_set:
            current = min(open_set, key=lambda n: f.get(n, float('inf')))
            if current == (gi, gj):
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return list(reversed(path))

            open_set.remove(current)
            ci, cj = current
            for di, dj in nbrs:
                ni, nj = ci + di, cj + dj
                if not self._in_bounds(ni, nj, nx, ny) or occ[ni, nj]:
                    continue
                tentative_g = g[current] + step_cost(di, dj)
                if tentative_g < g.get((ni, nj), float('inf')):
                    came_from[(ni, nj)] = current
                    g[(ni, nj)] = tentative_g
                    f[(ni, nj)] = tentative_g + h(ni, nj)
                    open_set.add((ni, nj))
        return []

    def _path_to_waypoints(self, path_ij, stride=2, corner_prune=True):
        if not path_ij:
            return []
        ds = path_ij[::max(1, stride)]
        if ds[-1] != path_ij[-1]:
            ds.append(path_ij[-1])
        pts = [self._ij_to_xy(i, j) for (i, j) in ds]
        if corner_prune and len(pts) > 2:
            pruned = [pts[0]]
            for k in range(1, len(pts) - 1):
                a = np.array(pruned[-1]); b = np.array(pts[k]); c = np.array(pts[k + 1])
                v1 = b - a; v2 = c - b
                if np.linalg.norm(v1) < 1e-6 or np.linalg.norm(v2) < 1e-6:
                    continue
                cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                if cosang < 0.996:
                    pruned.append(pts[k])
            pruned.append(pts[-1])
            pts = pruned
        return [[float(x), float(y)] for (x, y) in pts]

    # ===================== World build =====================
    def _clear_debug(self):
        for i in self.path_debug_ids:
            try: p.removeUserDebugItem(i)
            except Exception: pass
        self.path_debug_ids.clear()
        for i in self.goal_debug_ids:
            try: p.removeUserDebugItem(i)
            except Exception: pass
        self.goal_debug_ids.clear()

    def _spawn_obstacles(self):
        for obs_id in self.obstacles:
            try: p.removeBody(obs_id)
            except Exception: pass
        self.obstacles.clear()

        def far_from(pnt, ref, r):
            return np.linalg.norm(np.array(pnt) - np.array(ref)) > r

        attempts = 0
        while len(self.obstacles) < self.num_obstacles and attempts < self.num_obstacles * 30:
            attempts += 1
            x = self.rng.uniform(self.GRID_X_MIN + 0.3, self.GRID_X_MAX - 0.3)
            y = self.rng.uniform(self.GRID_Y_MIN + 0.3, self.GRID_Y_MAX - 0.3)
            pos = [x, y, 0.25]
            if (far_from([x, y], self.start_xy, self.obs_clear_radius) and
                far_from([x, y], self.goal_xy,  self.obs_clear_radius)):
                cube_id = p.loadURDF("cube_small.urdf", basePosition=pos, globalScaling=self.obs_global_scaling)
                self.obstacles.append(cube_id)
        for oid in self.obstacles:
            p.changeDynamics(oid, -1, lateralFriction=0.9, rollingFriction=0.001)

    def _make_occupancy_from_bodies(self):
        nx, ny = self._grid_shape()
        occ = np.zeros((nx, ny), dtype=bool)
        inflation = max(1, int(round(0.35 / self.GRID_RES)))
        for oid in self.obstacles:
            pos, _ = p.getBasePositionAndOrientation(oid)
            i, j = self._xy_to_ij(pos[0], pos[1])
            for di in range(-inflation, inflation + 1):
                for dj in range(-inflation, inflation + 1):
                    ii, jj = i + di, j + dj
                    if 0 <= ii < nx and 0 <= jj < ny:
                        occ[ii, jj] = True
        si, sj = self._xy_to_ij(self.start_xy[0], self.start_xy[1])
        gi, gj = self._xy_to_ij(self.goal_xy[0],  self.goal_xy[1])
        if 0 <= si < nx and 0 <= sj < ny: occ[si, sj] = False
        if 0 <= gi < nx and 0 <= gj < ny: occ[gi, gj] = False
        return occ

    def _draw_path_and_goal(self):
        self._clear_debug()
        if len(self.waypoints) >= 2:
            for k in range(len(self.waypoints) - 1):
                a = self.waypoints[k]   + [0.02]
                b = self.waypoints[k+1] + [0.02]
                cid = p.addUserDebugLine(a, b, [0, 0, 1], lineWidth=2.5)
                self.path_debug_ids.append(cid)
        g = list(self.goal_xy) + [0.02]
        arm = 0.35
        g1 = [g[0] - arm, g[1] - arm, g[2]]
        g2 = [g[0] + arm, g[1] + arm, g[2]]
        g3 = [g[0] - arm, g[1] + arm, g[2]]
        g4 = [g[0] + arm, g[1] - arm, g[2]]
        self.goal_debug_ids.append(p.addUserDebugLine(g1, g2, [1, 0, 0], lineWidth=6))
        self.goal_debug_ids.append(p.addUserDebugLine(g3, g4, [1, 0, 0], lineWidth=6))

    def _plan_path(self):
        self._spawn_obstacles()
        occ = self._make_occupancy_from_bodies()
        path_ij = self._astar(occ)

        retries = 0
        while not path_ij and retries < 3:
            # if unsolvable, slightly reduce obstacles
            self.num_obstacles = max(8, int(self.num_obstacles * 0.75))
            self._spawn_obstacles()
            occ = self._make_occupancy_from_bodies()
            path_ij = self._astar(occ)
            retries += 1

        if not path_ij:
            self.waypoints = [[0.0, 1.5], [0.0, 3.0], [0.0, 4.5], [0.0, 6.0]]
        else:
            self.waypoints = self._path_to_waypoints(path_ij, stride=2, corner_prune=True)

        self._draw_path_and_goal()

        # Update L_star after planning
        self._metrics["L_star"] = self._compute_L_star()

    # ===================== Path geometry =====================
    def _active_target_and_tangent(self, pos_xy):
        if not self.waypoints:
            return self.goal_xy, np.array([0.0, 1.0])

        progressed = True
        while progressed and self.current_wp_idx < len(self.waypoints):
            progressed = False
            wp = np.array(self.waypoints[self.current_wp_idx])
            if np.linalg.norm(wp - pos_xy) < 0.35 and self.current_wp_idx < len(self.waypoints) - 1:
                self.current_wp_idx += 1
                progressed = True

        pts = [pos_xy.tolist()] + self.waypoints[self.current_wp_idx:]
        acc = 0.0
        for a, b in zip(pts[:-1], pts[1:]):
            a = np.array(a); b = np.array(b)
            seg = b - a; L = np.linalg.norm(seg)
            if L < 1e-6:
                continue
            if acc + L >= self.lookahead_dist:
                t = (self.lookahead_dist - acc) / L
                target = a + t * seg
                tangent = seg / L
                return target, tangent
            acc += L

        if self.waypoints:
            if self.current_wp_idx < len(self.waypoints):
                tail = np.array(self.waypoints[-1]) - np.array(self.waypoints[-2]) if len(self.waypoints) > 1 else np.array([0.0,1.0])
                tangent = tail / (np.linalg.norm(tail) + 1e-9)
            else:
                tangent = np.array([0.0, 1.0])
        else:
            tangent = np.array([0.0, 1.0])
        return self.goal_xy, tangent

    def _cross_track_error(self, pos_xy):
        if self.current_wp_idx >= len(self.waypoints):
            return 0.0
        a = pos_xy
        b0 = np.array(self.waypoints[self.current_wp_idx - 1]) if self.current_wp_idx > 0 else a
        b1 = np.array(self.waypoints[self.current_wp_idx])
        seg = b1 - b0
        L2 = np.dot(seg, seg) + 1e-9
        t = np.dot(a - b0, seg) / L2
        t = np.clip(t, 0.0, 1.0)
        proj = b0 + t * seg
        d = a - proj
        n = np.array([-seg[1], seg[0]])
        sign = np.sign(np.dot(d, n))
        return float(sign * np.linalg.norm(d))

    def _apf_bias(self, pos_xy, d0=0.9, k=0.25):
        acc = np.zeros(2, dtype=np.float32)
        for oid in self.obstacles:
            pos, _ = p.getBasePositionAndOrientation(oid)
            o = np.array(pos[:2])
            r = pos_xy - o
            d = np.linalg.norm(r)
            if 1e-6 < d < d0:
                acc += k * ((1.0/d - 1.0/d0) / (d*d)) * (r / (d+1e-9))
        n = np.linalg.norm(acc)
        return acc / n if n > 1e-6 else acc

    # ===================== Obs / checks =====================
    def _get_obs(self):
        joint_angles, joint_vels = [], []
        for i in self.joint_indices:
            s = p.getJointState(self.robot_id, i)
            joint_angles.append(s[0]); joint_vels.append(s[1])

        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        roll, pitch, yaw = p.getEulerFromQuaternion(orn)
        linear_vel, angular_vel = p.getBaseVelocity(self.robot_id)

        target, tangent = self._active_target_and_tangent(np.array(pos[:2]))
        goal_vec = np.array(target) - np.array(pos[:2])
        goal_angle = np.arctan2(goal_vec[1], goal_vec[0])
        heading_error = np.arctan2(np.sin(goal_angle - yaw), np.cos(goal_angle - yaw))
        distance_to_wp = float(np.linalg.norm(goal_vec))
        cross_track = self._cross_track_error(np.array(pos[:2]))

        l_contact = int(len(p.getContactPoints(self.robot_id, self.plane_id, linkIndexA=self.left_foot_link)) > 0)
        r_contact = int(len(p.getContactPoints(self.robot_id, self.plane_id, linkIndexA=self.right_foot_link)) > 0)

        return np.array(
            joint_angles + joint_vels + [roll, pitch, yaw] +
            list(angular_vel) + list(linear_vel) + [l_contact, r_contact] +
            [distance_to_wp, heading_error, cross_track], dtype=np.float32
        )

    def _check_fall(self):
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        roll, pitch, _ = p.getEulerFromQuaternion(orn)
        torso_z = pos[2]
        return (torso_z < 0.15 or abs(pitch) > 1.0 or abs(roll) > 1.0)

    def _check_collision(self):
        for obs_id in self.obstacles:
            if len(p.getContactPoints(bodyA=self.robot_id, bodyB=obs_id)) > 0:
                return True
        return False

    # ===================== CPG =====================
    def _cpg_targets_deg(self, v_des):
        freq = self.cpg_base_freq * np.clip(
            v_des / (self.cpg_nom_speed + 1e-6),
            self.cpg_min_freq / self.cpg_base_freq,
            self.cpg_max_freq / self.cpg_base_freq
        )
        self._phase = (self._phase + 2*np.pi*freq*self.timestep) % (2*np.pi)
        amp_scale = self.anti_stall_boost if self.step_counter < self._kick_until else 1.0

        phiL = self._phase
        phiR = (self._phase + np.pi) % (2*np.pi)

        hipL  =  self.hip_amp_deg  * amp_scale * np.sin(phiL)
        kneeL =  self.knee_amp_deg * amp_scale * np.maximum(0.0, np.sin(phiL))
        ankleL=  self.ankle_off_deg

        hipR  =  self.hip_amp_deg  * amp_scale * np.sin(phiR)
        kneeR =  self.knee_amp_deg * amp_scale * np.maximum(0.0, np.sin(phiR))
        ankleR=  self.ankle_off_deg

        pattern = [hipL, kneeL, ankleL, hipR, kneeR, ankleR]
        out = []
        k = 0
        for _ in self.joint_indices:
            out.append(pattern[k % len(pattern)])
            k += 1
        return np.array(out, dtype=np.float32)

    def _blend_action_with_cpg(self, action, v_des, heading_error, cross_track):
        cpg_deg = self._cpg_targets_deg(v_des)
        cpg_norm = []
        for idx, _j in enumerate(self.joint_indices):
            low, high = self.joint_limits[idx]
            target_rad = np.deg2rad(cpg_deg[idx])
            mapped = (2.0*(target_rad - low)/(high - low)) - 1.0
            cpg_norm.append(np.clip(mapped, -1.0, 1.0))
        cpg_norm = np.array(cpg_norm, dtype=np.float32)

        steer = np.clip(-self.lateral_kp * cross_track - 0.5*heading_error, -0.5, 0.5)
        if self.idx_L_hip in self.joint_indices:
            k = self.joint_indices.index(self.idx_L_hip)
            cpg_norm[k] = np.clip(cpg_norm[k] + steer, -1.0, 1.0)
        if self.idx_R_hip in self.joint_indices:
            k = self.joint_indices.index(self.idx_R_hip)
            cpg_norm[k] = np.clip(cpg_norm[k] - steer, -1.0, 1.0)

        return (self.cpg_mix * cpg_norm + (1.0 - self.cpg_mix) * action).astype(np.float32)

    # ===================== Reward (unchanged from your v2) =====================
    def _compute_reward(self, target, tangent, cross_track):
        current_pos, current_orn = p.getBasePositionAndOrientation(self.robot_id)
        roll, pitch, yaw = p.getEulerFromQuaternion(current_orn)
        linear_vel, _angular_vel = p.getBaseVelocity(self.robot_id)
        torso_z = current_pos[2]

        to_target = np.array(target) - np.array(current_pos[:2])
        distance_to_goal = float(np.linalg.norm(to_target))
        distance_delta = self.prev_distance_to_goal - distance_to_goal
        distance_progress_reward = 7.0 * distance_delta
        self.prev_distance_to_goal = distance_to_goal

        t = tangent / (np.linalg.norm(tangent) + 1e-9)
        n = np.array([-t[1], t[0]])
        v = np.array([linear_vel[0], linear_vel[1]])
        v_along = float(np.dot(v, t))
        v_lat   = float(np.dot(v, n))

        along_speed_reward = 6.0 * v_along
        facing_dir = np.array([np.cos(yaw), np.sin(yaw)])
        orientation_reward = 2.5 * float(np.dot(facing_dir, t))

        cross_penalty = -3.5 * (cross_track ** 2)
        upright_penalty = -1.8 * (abs(roll) + abs(pitch))
        height_penalty  = -0.8 * abs(torso_z - 0.45)
        lateral_penalty = -1.2 * (v_lat ** 2)
        energy_penalty = -0.004 * sum(abs(p.getJointState(self.robot_id, i)[3]) for i in self.joint_indices)
        collision_penalty = -50.0 if self._check_collision() else 0.0
        goal_shaping_reward = 2.2 * (1.0 / (distance_to_goal + 1e-5))
        time_penalty = -0.08

        if np.linalg.norm(np.array(current_pos) - np.array(self.last_check_pos)) < 0.01:
            self.progress_stuck_counter += 1
        else:
            self.progress_stuck_counter = 0
            self.last_check_pos = current_pos
        stuck_penalty = -18.0 if self.progress_stuck_counter > self.stuck_threshold_steps else 0.0

        low_velocity_penalty = -4.0 if v_along < 0.02 else 0.0

        lf_z = p.getLinkState(self.robot_id, self.left_foot_link)[0][2]
        rf_z = p.getLinkState(self.robot_id, self.right_foot_link)[0][2]
        lc = len(p.getContactPoints(self.robot_id, self.plane_id, linkIndexA=self.left_foot_link)) > 0
        rc = len(p.getContactPoints(self.robot_id, self.plane_id, linkIndexA=self.right_foot_link)) > 0
        swing_bonus = 0.0
        if not lc: swing_bonus += 0.4 * np.clip(lf_z - 0.03, 0.0, 0.03) / 0.03
        if not rc: swing_bonus += 0.4 * np.clip(rf_z - 0.03, 0.0, 0.03) / 0.03

        reward = (
            distance_progress_reward + along_speed_reward + orientation_reward +
            cross_penalty + upright_penalty + height_penalty + lateral_penalty +
            energy_penalty + collision_penalty + goal_shaping_reward +
            stuck_penalty + time_penalty + low_velocity_penalty + swing_bonus
        )
        return max(reward, -15.0)

    # ===================== Reset =====================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        for i in self.joint_indices:
            p.resetJointState(self.robot_id, i, targetValue=0.0, targetVelocity=0.0)

        self.step_counter = 0
        self.progress_stuck_counter = 0
        self.kick_stall_counter = 0
        self.current_wp_idx = 0
        self._phase = 0.0
        self._kick_until = 0

        # Reset metrics
        self._reset_episode_metrics()

        # Remove obstacles, replan
        for obs_id in self.obstacles:
            try: p.removeBody(obs_id)
            except Exception: pass
        self.obstacles.clear()
        self._clear_debug()
        self._plan_path()

        # Reset base
        p.resetBasePositionAndOrientation(
            self.robot_id,
            [float(self.start_xy[0]), float(self.start_xy[1]), 0.45],
            p.getQuaternionFromEuler([0, 0, 0])
        )
        p.resetBaseVelocity(self.robot_id, [0, 0, 0], [0, 0, 0])

        pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        self.last_check_pos = pos
        self.prev_distance_to_goal = float(np.linalg.norm(self.goal_xy - np.array(pos[:2])))

        # For L_actual integration
        self._prev_xy_for_length = np.array(pos[:2], dtype=np.float32)

        return self._get_obs(), {}

    # ===================== Step =====================
    def step(self, action):
        # Base pose
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
        roll, pitch, yaw = p.getEulerFromQuaternion(base_orn)
        pos_xy = np.array(base_pos[:2], dtype=np.float32)

        # Look-ahead
        target, tangent = self._active_target_and_tangent(pos_xy)

        # APF micro-bias
        bias = self._apf_bias(pos_xy)
        if np.linalg.norm(bias) > 0:
            tangent = (tangent + 0.6 * bias)
            tangent = tangent / (np.linalg.norm(tangent) + 1e-9)

        cross_track = self._cross_track_error(pos_xy)

        # Desired forward speed
        path_yaw = math.atan2(tangent[1], tangent[0])
        head_err = math.atan2(math.sin(path_yaw - yaw), math.cos(path_yaw - yaw))
        heading_quality = np.clip(1.0 - 0.6*abs(head_err), 0.25, 1.0)
        v_des = self.heading_kp * self.cpg_nom_speed * heading_quality
        v_des = float(np.clip(v_des, 0.08, 0.45))

        # Dynamic lookahead
        self.lookahead_dist = float(np.clip(0.6 + 1.2 * v_des / 0.45, 0.6, 1.6))

        # Anti-stall kick schedule
        lin_vel, _ = p.getBaseVelocity(self.robot_id)
        speed = math.hypot(lin_vel[0], lin_vel[1])
        if speed < 0.03:
            self.kick_stall_counter += 1
        else:
            self.kick_stall_counter = 0
        if self.kick_stall_counter > self.anti_stall_steps:
            self._kick_until = self.step_counter + int(0.6 / self.timestep)
            self.kick_stall_counter = 0

        # Blend action with CPG
        if self.assist_cpg:
            action = self._blend_action_with_cpg(action, v_des, head_err, cross_track)

        # Apply joint control
        for idx, joint_id in enumerate(self.joint_indices):
            low, high = self.joint_limits[idx]
            a = float(np.clip(action[idx], -1.0, 1.0))
            mapped_angle = low + (a + 1.0) * (high - low) / 2.0
            p.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=joint_id,
                controlMode=p.POSITION_CONTROL,
                targetPosition=mapped_angle,
                force=self.default_motor_force,
                positionGain=self.pos_gain,
                velocityGain=self.vel_gain,
            )

        # Torso stabilizer
        kp, kd = 0.28, 0.03
        desired_rpy = np.array([0.0, 0.0, 0.0])
        _, ang_vel = p.getBaseVelocity(self.robot_id)
        error = desired_rpy[:2] - np.array([roll, pitch])
        d_error = -np.array(ang_vel[:2])
        torque = np.array([kp * error[0] + kd * d_error[0],
                           kp * error[1] + kd * d_error[1], 0.0])
        p.applyExternalTorque(self.robot_id, -1, torqueObj=torque.tolist(), flags=p.WORLD_FRAME)

        # Sim step
        p.stepSimulation()
        if self.render:
            time.sleep(self.timestep)

        # ===================== Update episode metrics each step =====================
        self._metrics["steps"] += 1

        # L_actual integrate
        cur_xy = np.array(p.getBasePositionAndOrientation(self.robot_id)[0][:2], dtype=np.float32)
        if self._prev_xy_for_length is not None:
            self._metrics["L_actual"] += float(np.linalg.norm(cur_xy - self._prev_xy_for_length))
        self._prev_xy_for_length = cur_xy

        # Energy proxy integrate: sum |tau*qdot| dt
        power = 0.0
        for jid in self.joint_indices:
            js = p.getJointState(self.robot_id, jid)
            qdot = float(js[1])
            tau = float(js[3])  # applied motor torque
            power += abs(tau * qdot)
        self._metrics["energy"] += power * float(self.timestep)

        # Collision-any flag
        if not self._metrics["collision_any"]:
            if self._check_collision():
                self._metrics["collision_any"] = True

        # Distance to goal (true goal, not lookahead)
        cur_pos = p.getBasePositionAndOrientation(self.robot_id)[0]
        dist_goal = float(np.linalg.norm(self.goal_xy - np.array(cur_pos[:2], dtype=np.float32)))
        self._metrics["dist_to_goal"] = dist_goal

        # ===================== Termination logic =====================
        terminated = False
        truncated = False

        if self._check_fall():
            self._metrics["fall"] = True
            reward = -50.0
            terminated = True
        else:
            reward = self._compute_reward(target, tangent, cross_track)
            if dist_goal < self.goal_radius:
                self._metrics["success"] = True
                reward += 30.0
                terminated = True
            truncated = self.step_counter >= self.max_steps

        self.step_counter += 1

        # ===================== Finalize metrics when episode ends =====================
        if terminated or truncated:
            L_star = float(self._metrics.get("L_star", 0.0))
            L_actual = float(self._metrics.get("L_actual", 0.0))
            success = bool(self._metrics.get("success", False))

            # SPL definition (standard): success * (L_star / max(L_star, L_actual))
            denom = max(L_star, L_actual, 1e-9)
            spl = (1.0 if success else 0.0) * (L_star / denom)
            self._metrics["spl"] = float(np.clip(spl, 0.0, 1.0))

            # CoT definition: energy / (m*g*distance). If no progress => inf.
            g = 9.81
            if L_actual > 1e-6:
                self._metrics["cot"] = float(self._metrics["energy"] / (self.robot_mass * g * L_actual))
            else:
                self._metrics["cot"] = float("inf")

        # Build outputs
        state = self._get_obs()
        info = {
            "collision": self._check_collision(),
            "waypoint_idx": self.current_wp_idx,
            "num_waypoints": len(self.waypoints),
            "cross_track": cross_track,
            "assist": self.assist_cpg,
            "lookahead": self.lookahead_dist,
        }

        # Attach episode metrics on final step so evaluators can read it directly
        if terminated or truncated:
            info["episode_metrics"] = self.get_episode_metrics()

        return state, float(reward), bool(terminated), bool(truncated), info

    def close(self):
        p.disconnect(self.physics_client)
