import random
from typing import Tuple, Optional, Any, Dict, List, Union

import gymnasium as gym
import pygame

from gymnasium import spaces
from gymnasium.core import ObsType
from gymnasium.spaces.utils import flatten_space

import numpy as np

from utils import plane_sailing_position
from vessels import OwnShip, Target, StaticObject


class MarineEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    OWN_SHIP_PARAMS: List[str] = []
    WP_PARAMS: List[str] = []
    TARGET_PARAMS: List[str] = []

    # env properties
    INITIAL_LAT: float = 0.0
    INITIAL_LON: float = 0.0
    ENV_RANGE: int = 20  # defines the size of the field

    # own ship properties
    MAX_TURN_ANGLE: int = 20  # rate of turn defaults to 20 deg per min
    MAX_SPEED_CHANGE: float = 0.5  # rate of speed change knots / min
    WP_REACH_THRESHOLD: float = 0.2  # in nautical miles, terminates an episode

    # target limits collision avoidance settings
    CPA_THRESHOLD: float = 1.0  # in nautical miles
    TCPA_THRESHOLD: float = 15  # in minutes
    # limits at witch the onw ship should act
    CPA_LIMIT: float = 0.1
    TCPA_LIMIT: float = 1

    ASPECTS = [
        'static',
        'head-on',
        'crossing',
    ]

    # Constants for rewards and penalties
    CPA_AVOIDANCE_THRESHOLD = 1.2  # CPA threshold for excessive avoidance
    AVOIDANCE_PENALTY_FACTOR = 5.0  # Scaling factor for excessive avoidance penalty
    COLLISION_PENALTY: int = -300  # Large penalty for collision
    # CPA_VIOLATION_PENALTY: int = -75  # Penalty for violating CPA threshold
    # CPA_SAFE_REWARD: int = 15  # Reward for maintaining safe CPA
    # STARBOARD_TURN_REWARD: int = 2  # Reward for correct starboard turn
    # TURN_PENALTY: int = -10  # Penalty for incorrect port turn
    # TURN_REWARD: int = 2
    # SPEED_PENALTY: int = -5
    # SPEED_REWARD: int = 10
    # UNNECESSARY_MOVEMENT_PENALTY: int = -1  # Penalty for unnecessary maneuvers
    # COLREG_VIOLATION: int = -50
    # # WP rewards and penalties
    WP_REACH_REWARD: int = 100  # reward for reaching the waypoint
    # ETA_REWARD: int = 2
    # ETA_PENALTY: int = -5
    ETA_VIOLATION_PENALTY: int = -50

    def __init__(
            self,
            # environment properties
            render_mode=None,
            continuous: bool = False,
            timescale: int = 1,  # defines the step size, defaults to 1 min step
            training_stage: int = 1,  # defines different training stages, 0 for no training
            total_targets: int = 1,  # minimum targets should be one
            training: bool = True,
            seed: Optional[int] = None,
    ):
        super(MarineEnv, self).__init__()

        self.training = training
        self.step_counter = 0
        self.training_stage = training_stage
        self.total_targets = total_targets
        self.seed = self._set_global_seed(seed)

        # initialize the environment bounds
        self.lat_bounds: Tuple[float, float] = (self.INITIAL_LAT, self.INITIAL_LAT + self.ENV_RANGE / 60)
        self.lon_bounds: Tuple[float, float] = (self.INITIAL_LON, self.INITIAL_LON + self.ENV_RANGE / 60)

        # define the timescale and scaling of real word time to simulation time
        self.timescale = timescale

        # initialize action space
        if continuous:
            # the agent can continuously adjust course and speed
            self.MAX_TURN_ANGLE = self.MAX_TURN_ANGLE * timescale  # scaling the turn
            self.MAX_SPEED_CHANGE = self.MAX_SPEED_CHANGE * timescale  # scaling the speed change
            self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(5)

        # define the observation space
        self.observation_space = flatten_space(self._define_observation_space())

        # initialize the state
        self.observation = None

        # initialize own ship
        self.own_ship = OwnShip()
        self.own_ship_trace: list = []
        self.target_traces: dict = {}

        # initialize the wp
        self.waypoint = StaticObject()  # 2 elements, latitude and longitude
        self.waypoints = []

        # extract params
        self.OWN_SHIP_PARAMS = [key for key in self._define_observation_space()['own_ship'].spaces.keys()][:2]
        self.WP_PARAMS = [key for key in self._define_observation_space()['own_ship'].spaces.keys()][2:]
        self.TARGET_PARAMS = [key for key in self._define_observation_space()['targets'][0].spaces.keys()]

        # pygame setup
        assert render_mode is None or render_mode in self.metadata["render_modes"], \
            f"Invalid render_mode: {render_mode}. Available modes: {self.metadata['render_modes']}"
        self.render_mode = render_mode

        self.window_size = 600  # pixels for visualization
        self.scale = self.window_size / ((self.lat_bounds[1] - self.lat_bounds[0]) * 60)  # pixels per NM
        self.window = None
        self.clock = None
        self.vessel_size = 5  # vessel radius in pixels

    def _set_global_seed(self, seed=None) -> Union[int, None]:
        """Sets the global random seed for reproducibility."""
        if seed is not None:
            import torch

            self.seed_value = seed
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            print(f'Seed set to {seed}')
        return seed

    def _define_observation_space(self) -> spaces.Dict:
        return spaces.Dict({
            'own_ship': spaces.Dict({
                'course': spaces.Box(low=0, high=360, shape=(), dtype=np.float32),
                'speed': spaces.Box(low=-10, high=20, shape=(), dtype=np.float32),
                'wp_distance': spaces.Box(low=0, high=50, shape=(), dtype=np.float32),
                'wp_eta': spaces.Box(low=0, high=300, shape=(), dtype=np.float32),
                'wp_relative_bearing': spaces.Box(low=-180, high=180, shape=(), dtype=np.float32),
                'wp_target_eta': spaces.Box(low=0, high=300, shape=(), dtype=np.float32),
            }),
            'targets': spaces.Tuple([
                spaces.Dict({
                    'bcr': spaces.Box(low=-50, high=50, shape=(), dtype=np.float32),
                    'cpa': spaces.Box(low=0, high=50, shape=(), dtype=np.float32),
                    'target_course': spaces.Box(low=0, high=360, shape=(), dtype=np.float32),
                    'target_distance': spaces.Box(low=0, high=50, shape=(), dtype=np.float32),
                    'target_relative_bearing': spaces.Box(low=-180, high=180, shape=(), dtype=np.float32),
                    'target_relative_course': spaces.Box(low=0, high=360, shape=(), dtype=np.float32),
                    'target_relative_speed': spaces.Box(low=0, high=50, shape=(), dtype=np.float32),
                    'target_speed': spaces.Box(low=-10, high=20, shape=(), dtype=np.float32),
                    'tbc': spaces.Box(low=-100, high=100, shape=(), dtype=np.float32),
                    'tcpa': spaces.Box(low=0, high=100, shape=(), dtype=np.float32),
                }) for _ in range(self.total_targets)  # assuming targets are considered dangerous
            ])
        })

    @property
    def wp_distance(self) -> float:
        return self.own_ship.calculate_distance(self.waypoint)

    @property
    def wp_eta(self) -> float:
        # will be recalculated when calling reset()
        return 60 * self.wp_distance / self.own_ship.speed

    @property
    def wp_relative_bearing(self) -> float:
        return self.own_ship.calculate_relative_bearing(self.waypoint)

    @property
    def wp_target_eta(self) -> float:
        # will ALWAYS be calculated when handling the state
        return 0.0

    def step(self, action) -> tuple[ObsType, float, bool, bool, dict[str, int]]:

        # increment the step counter
        self.step_counter += 1

        # extract own ship params from the current state
        course, speed, last_wp_distance, last_wp_eta, last_wp_relative_bearing, last_tgt_eta = self.observation[:6]

        # update the params based on action
        if isinstance(self.action_space, spaces.Discrete):
            raise NotImplementedError('This environment does not support discrete action spaces YET.')
        else:
            course_change = np.clip(action[0] * self.MAX_TURN_ANGLE, -self.MAX_TURN_ANGLE, self.MAX_TURN_ANGLE)
            speed_change = action[1] * self.MAX_SPEED_CHANGE

        # update and apply own ship parameters
        self.own_ship.course = (course + course_change) % 360
        self.own_ship.speed += speed_change
        if self.own_ship.speed > 0:
            self.own_ship.speed = min(self.own_ship.max_speed, self.own_ship.speed)
        else:
            self.own_ship.speed = max(self.own_ship.min_speed, self.own_ship.speed)

        # update next position
        self.own_ship.update_position(time_interval=self.timescale, clip_lat=self.lat_bounds, clip_lon=self.lon_bounds)

        # generate own state data
        own_ship_data = self._generate_own_ship_data()
        own_ship_data['wp_target_eta'] = last_tgt_eta - self.timescale

        # move/update all detected targets
        for target in self.own_ship.detected_targets:
            target.update_position(time_interval=self.timescale)
            self.own_ship.update_target(target)

            # check target coordinates and remove from list if out of bounds
            if target.lat <= self.lat_bounds[0] or target.lat >= self.lon_bounds[1] or \
                    target.lon <= self.lon_bounds[0] or target.lon >= self.lon_bounds[1]:
                self.own_ship.detected_targets.remove(target)

        dangerous_targets_data = self._generate_dangerous_targets_state()
        # construct the final observation
        raw_observation = {
            "own_ship": own_ship_data,
            "targets": dangerous_targets_data
        }

        previous_obs = self.observation
        current_obs = self._flatten_observation(raw_observation)

        reward, terminated, truncated, info = self.calculate_reward(previous_obs, current_obs)

        # assign the current observation
        self.observation = current_obs

        return self.observation, reward, terminated, truncated, info

    def calculate_reward(self, previous_obs: ObsType, current_obs: ObsType) -> tuple[float, bool, bool, dict[str, int]]:
        """ method to calculate the reward """

        def wp_following_reward(rwrd: float):

            # distance-based reward or penalty
            previous_wp_distance = previous_data['own_ship']['wp_distance']
            current_wp_distance = current_data['own_ship']['wp_distance']

            distance_change = previous_wp_distance - current_wp_distance
            rwrd += max(-1.0,
                        distance_change * 10)  # reward proportional to distance improvement

            # bearing alignment reward
            current_wp_relative_bearing = current_data['own_ship']['wp_relative_bearing']
            alignment_error = abs(current_wp_relative_bearing)

            # alignment reward
            rwrd += bearing_alignment_reward(alignment_error)

            return rwrd

        # bearing-alignment reward
        def bearing_alignment_reward(alignment_error: float) -> float:
            max_reward = 10
            min_positive_reward = 1
            penalty_scale = -0.1

            if alignment_error <= 30:
                reward = max_reward - ((max_reward - min_positive_reward) / 30.0) * alignment_error
            else:
                reward = penalty_scale * (alignment_error - 30)

            return reward

        def calculate_eta_reward(eta_diff: float) -> float:
            # parameters
            max_reward = 10 * self.timescale  # max reward when ETA difference = 0
            min_positive_reward = 1 * self.timescale  # reward when ETA difference = 6
            penalty_scale = -2 * self.timescale  # penalty multiplier when ETA difference > 6

            if eta_diff <= 6:
                # linear decay reward: max_reward decreases linearly to min_positive_reward
                reward = max_reward - ((max_reward - min_positive_reward) / 6) * eta_diff
            else:
                # negative penalty: linear penalty increasing with the ETA difference
                reward = penalty_scale * (eta_diff - 6)  # Penalty starts after 6 min deviation

            return reward

        previous_data = self._generate_observation_dict(previous_obs)
        current_data = self._generate_observation_dict(current_obs)

        # terminated - the episode ends because the agent has reached a goal or violated an environment rule
        # (e.g., collision, reaching the waypoint, ect).
        # truncated - the episode ends due to external constraints like a maximum time step limit,
        # regardless of whether the agent reached a goal or not.
        terminated, truncated = False, False
        info = {
            'total_steps': self.step_counter,
            'terminated': False,
            'truncated': False,
        }
        reward = 0.0

        previous_course = previous_data['own_ship']['course']
        current_course = current_data['own_ship']['course']
        # positive course change is turning to starboard
        course_change = self._normalize_course(current_course - previous_course)

        previous_speed = previous_data['own_ship']['speed']
        current_speed = current_data['own_ship']['speed']
        speed_change = current_speed - previous_speed  # negative means slowing down

        # reaching the wp -> large reward and episode termination
        if self.wp_distance < self.WP_REACH_THRESHOLD:
            info['terminated'] = 'WP Reached!'
            return self.WP_REACH_REWARD, True, False, info

        # reward for wp tracking, training stage 1
        if self.training_stage == 1:
            reward = wp_following_reward(reward)

        # TODO reward for training stage 2
        if self.training_stage == 2:

            # should check if dangerous targets in the list
            if not any(target.is_dangerous for target in self.own_ship.dangerous_targets):
                reward = wp_following_reward(reward)

            else:
                for _, target in enumerate(self.own_ship.dangerous_targets):
                    if not target.is_dangerous:
                        continue
                    # huge penalty for collision (immediate termination)
                    if target.distance < self.CPA_LIMIT:
                        reward += self.COLLISION_PENALTY
                        info['terminated'] = 'Collision!'
                        terminated = True
                        break

                    # CPA penalty (capped to prevent over-penalization)
                    if target.distance < self.CPA_THRESHOLD:
                        reward -= min(20, (2 / (target.distance + 0.1) ** 2)) * self.timescale

                    # penalize excessive avoidance (CPA > 1.2 NM)
                    if target.cpa > self.CPA_AVOIDANCE_THRESHOLD:
                        excess_cpa = target.cpa - self.CPA_AVOIDANCE_THRESHOLD
                        penalty = excess_cpa * self.AVOIDANCE_PENALTY_FACTOR
                        reward -= penalty * self.timescale

                    # TCPA penalty/reward (COLREG compliance)
                    if target.tcpa < self.TCPA_THRESHOLD - 3:
                        if target.cpa >= self.CPA_THRESHOLD:
                            if course_change >= 0:  # reward only if turning starboard or going straight
                                reward += target.tcpa * 2 * self.timescale
                            else:  # penalize if cleared via port turn
                                reward -= abs(course_change) * 5
                        else:
                            reward -= (1 / (target.cpa + 0.1) ** 2) * self.timescale

                    # head-on and crossing situations
                    if target.aspect in ['head-on', 'crossing']:
                        # strong penalty for turning to port. COLREG violation
                        if course_change < 0:
                            reward -= abs(course_change) * 5  # Increased penalty

                        # target to port bow
                        if -10 <= target.relative_bearing <= 0 and target.cpa < self.CPA_THRESHOLD:
                            if target.bcr < 0 and target.cpa > 0.5:  # alter to port if crossing stern
                                if course_change < 0:  # correct (port turn)
                                    reward += 2 * self.timescale
                                else:  # incorrect (starboard turn or no turn)
                                    reward -= abs(course_change) * 5
                            else:  # alter to starboard otherwise
                                if course_change > 0:  # correct (starboard turn)
                                    reward += 2 * self.timescale
                                else:  # incorrect (port turn or no turn)
                                    reward -= abs(course_change) * 5

                            if speed_change > 0:
                                reward += 1 * self.timescale

                        # target to starboard bow
                        if 0 < target.relative_bearing <= 10 and target.cpa < self.CPA_THRESHOLD:
                            if course_change > 0:  # correct (starboard turn)
                                reward += course_change
                            else:  # incorrect (port turn or no turn)
                                reward -= abs(course_change) * 5

                            if speed_change < 0:  # slow down is also acceptable
                                reward += 1 * self.timescale

                        # target to starboard side (10 <= relative_bearing <= 75)
                        if 10 <= target.relative_bearing <= 75 and target.cpa < self.CPA_THRESHOLD:
                            if course_change > 0:  # correct (starboard turn)
                                reward += course_change
                            elif speed_change < 0:  # slow down is also acceptable
                                reward += 1 * self.timescale
                            else:  # incorrect (port turn or no action)
                                reward -= abs(course_change) * 5

                        # target abaft the beam (75 < relative_bearing <= 112.5)
                        if 75 < target.relative_bearing <= 112.5 and target.cpa < self.CPA_THRESHOLD:
                            if speed_change < 0:  # slow down is the only option
                                # reward += 2 * self.timescale
                                reward += 2 * self.timescale * speed_change ** 4
                            else:  # incorrect (any course change or no action)
                                reward -= 2 * self.timescale * abs(course_change) ** 2

        # reward for keeping ETA steady
        current_eta = current_data['own_ship']['wp_eta']
        target_eta = current_data['own_ship']['wp_target_eta']
        eta_difference = abs(current_eta - target_eta)

        reward += calculate_eta_reward(eta_difference)

        if target_eta < -12:
            reward += self.ETA_VIOLATION_PENALTY
            info['truncated'] = 'ETA Violated!'
            truncated = True
            return reward, terminated, truncated, info

        # check if going out of bounds
        out_of_screen = self.own_ship.lat <= self.lat_bounds[0] or self.own_ship.lat >= self.lat_bounds[1] or \
                        self.own_ship.lon <= self.lon_bounds[0] or self.own_ship.lon >= self.lon_bounds[1]

        if out_of_screen:
            reward -= 100.0  # Large penalty for leaving bounds
            info['terminated'] = 'Out-of-screen!'
            terminated = True
            return reward, terminated, truncated, info

        if np.isnan(reward) or np.isinf(reward):
            raise ValueError("Invalid reward value encountered.")

        return reward * self.timescale, terminated, truncated, info

    def reset(self, seed=None, options=None) -> tuple[ObsType, dict[str, Any]]:
        # set random seed if provided
        if seed is not None:
            self._set_global_seed(seed)

        # Proceed with the rest of your reset logic...
        super().reset(seed=self.seed)

        def place_waypoint(min_range: int, max_range: int) -> tuple[float, float]:
            while True:
                waypoint_lat = np.random.uniform(self.lat_bounds[0] + 0.025, self.lat_bounds[1] - 0.025)
                waypoint_lon = np.random.uniform(self.lon_bounds[0] + 0.025, self.lon_bounds[1] - 0.025)
                distance_to_waypoint = self.own_ship.calculate_distance((waypoint_lat, waypoint_lon))
                # Ensure waypoint is at correct distance from the vessel
                if min_range < distance_to_waypoint < max_range:
                    return waypoint_lat, waypoint_lon

        # reset the step counter
        self.step_counter = 0
        self.own_ship_trace.clear()
        self.target_traces.clear()

        # initialize the data dicts for the observation space representation
        own_ship_data = {}
        dangerous_targets_data = {}

        # reset the lists so the targets does not overlap
        self.own_ship.reset()

        # TODO remove training stage
        if self.training:
            training_stage = random.randint(1, 2)
        else:
            training_stage = 2

        # random initial speed, minimum 7 kn
        self.own_ship.speed = np.random.uniform(low=7, high=self.own_ship.max_speed)

        # state for wp tracking training
        if training_stage == 1:  # training for wp tracking, no targets

            # placing the vessel in the center of the env
            self.own_ship.lat, self.own_ship.lon = np.mean(self.lat_bounds), np.mean(self.lon_bounds)

            # random initial course
            self.own_ship.course = np.random.uniform(low=0, high=360)

            # place the waypoint
            self.waypoint.lat, self.waypoint.lon = place_waypoint(3, self.ENV_RANGE - 1)

            # calculate target eta, calculated using initial speed
            target_eta = self.wp_eta

            own_ship_data = self._generate_own_ship_data()
            own_ship_data['wp_eta'] = target_eta
            own_ship_data['wp_target_eta'] = target_eta

            dangerous_targets_data = self._generate_zero_target_data(self.total_targets)

        # set the state for training with targets
        elif training_stage == 2:

            # set own position at random corner
            delta_lat = (self.lon_bounds[1] - self.lon_bounds[0]) / 6
            delta_lon = (self.lat_bounds[1] - self.lat_bounds[0]) / 6

            # Define corners
            corners = [
                (self.lat_bounds[0] + delta_lat, self.lon_bounds[0] + delta_lon),  # Lower Left
                (self.lat_bounds[0] + delta_lat, self.lon_bounds[1] - delta_lon),  # Lower Right
                (self.lat_bounds[1] - delta_lat, self.lon_bounds[0] + delta_lon),  # Top Left
                (self.lat_bounds[1] - delta_lat, self.lon_bounds[1] - delta_lon)  # Top Right
            ]

            # Select a random corner
            self.own_ship.lat, self.own_ship.lon = random.choice(corners)

            # place the waypoint
            self.waypoint.lat, self.waypoint.lon = place_waypoint(12, 17)

            target_eta = self.wp_eta
            own_ship_data = self._generate_own_ship_data()
            # course to match wp + minor deviation
            self.own_ship.course = self.own_ship.calculate_true_bearing(self.waypoint) + random.uniform(-5, 5)
            own_ship_data['course'] = self.own_ship.course
            # update the relative bearing
            own_ship_data['wp_relative_bearing'] = self.own_ship.calculate_relative_bearing(self.waypoint)
            own_ship_data['wp_eta'] = target_eta
            own_ship_data['wp_target_eta'] = target_eta

            for _ in range(self.total_targets):
                target = self._place_dangerous_target_ship()

                # add target to detected targets
                self.own_ship.detected_targets.append(target)

            # generate the targets data
            dangerous_targets_data = self._generate_dangerous_targets_state()

        raw_observation = {
            "own_ship": own_ship_data,
            "targets": dangerous_targets_data
        }

        self.training_stage = training_stage  # pass the stage to the step method
        self.observation = self._flatten_observation(raw_observation)

        return self.observation, {}

    def render(self):
        if self.window is None:
            # Initialize pygame
            pygame.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Marine Environment")
            self.clock = pygame.time.Clock()

        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

        # Clear the screen
        self.window.fill((0, 0, 50))  # Dark blue background

        # Draw the own ship
        lat, lon = self.own_ship.lat, self.own_ship.lon
        course, speed = self.observation[:2]
        px, py = self._latlon_to_pixels(lat, lon)
        pygame.draw.circle(self.window, (255, 255, 255), (px, py), self.vessel_size)  # Own ship (white)

        # Append the current position to the own ship's trace.
        self.own_ship_trace.append((px, py))
        # Optionally, limit the trace length
        if len(self.own_ship_trace) > 1000:
            self.own_ship_trace.pop(0)

        # Draw the trace line for the own ship
        if len(self.own_ship_trace) > 1:
            pygame.draw.lines(self.window, (200, 200, 200), False, self.own_ship_trace, 2)

        # Draw 1nm distance circle
        pygame.draw.circle(self.window, (255, 0, 255), (px, py), self.scale, width=1)  # Own ship (white)

        # Draw own ship's heading line
        heading_rad = np.deg2rad(course - 90)  # Adjust course for pygame's coordinate system
        line_length = int(speed * self.scale) / 6  # Line length proportional to speed
        end_x = px + int(line_length * np.cos(heading_rad))
        end_y = py + int(line_length * np.sin(heading_rad))
        pygame.draw.line(self.window, (255, 0, 0), (px, py), (end_x, end_y), 2)  # Red heading line

        # Draw the waypoint
        waypoint_px, waypoint_py = self._latlon_to_pixels(self.waypoint.lat, self.waypoint.lon)
        pygame.draw.circle(self.window, (0, 255, 0), (waypoint_px, waypoint_py), self.vessel_size)  # Waypoint (green)

        # Draw remaining waypoints
        for wp in self.waypoints:
            waypoint_px, waypoint_py = self._latlon_to_pixels(*wp)
            pygame.draw.circle(self.window, (100, 255, 100), (waypoint_px, waypoint_py),
                               self.vessel_size)  # Remaining WPs (light green)

        # Draw the target ships
        for target in self.own_ship.detected_targets:
            # --- Draw Target Ships and Their Traces ---

            # Convert target position to pixel coordinates.
            target_px, target_py = self._latlon_to_pixels(target.lat, target.lon)

            # Update trace for the target.
            # Use a unique identifier for the target. For simplicity, we use id(target) if no id is provided.
            target_id = getattr(target, 'id', id(target))
            if target_id not in self.target_traces:
                self.target_traces[target_id] = []
            self.target_traces[target_id].append((target_px, target_py))
            # Optionally limit the trace length:
            if len(self.target_traces[target_id]) > 1000:
                self.target_traces[target_id].pop(0)

            # Draw the target's trace.
            if len(self.target_traces[target_id]) > 1:
                pygame.draw.lines(self.window, (0, 255, 255), False, self.target_traces[target_id], 1)

            # Convert target ship position to pixel coordinates
            target_px, target_py = self._latlon_to_pixels(target.lat, target.lon)
            pygame.draw.circle(self.window, (0, 0, 255), (target_px, target_py), self.vessel_size)  # Target ship (blue)

            # Draw target ship's heading line
            target_heading_rad = np.deg2rad(target.course - 90)  # Adjust course for pygame
            target_line_length = int(target.speed * self.scale) / 6  # Line proportional to target speed
            target_end_x = target_px + int(target_line_length * np.cos(target_heading_rad))
            target_end_y = target_py + int(target_line_length * np.sin(target_heading_rad))
            pygame.draw.line(self.window, (0, 255, 255), (target_px, target_py), (target_end_x, target_end_y),
                             2)  # Cyan heading line

        # Update the display
        pygame.display.flip()
        self.clock.tick(30)  # Limit to 30 FPS

        # If the mode is 'rgb_array', capture and return the frame as a numpy array.
        if self.render_mode == 'rgb_array':
            # Capture the current display content as an array.
            # Note: pygame.surfarray.array3d returns an array with shape (width, height, channels)
            # so we transpose it to (height, width, channels)
            frame = pygame.surfarray.array3d(self.window)
            frame = np.transpose(frame, (1, 0, 2))
            return frame

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None

    @staticmethod
    def _normalize_course(course: float) -> float:
        """
        the method normalizes the course in range 0 - 360 degrees
        """
        # Normalize the course change to the range [-180°, 180°]
        if course > 180:
            course -= 360
        elif course < -180:
            course += 360

        return course

    def _generate_dangerous_targets_state(self):
        # initialize empty targets list with zero-filled entries
        targets_data = self._generate_zero_target_data(self.total_targets)

        if self.own_ship.detected_targets:
            # Sort detected targets by dangerous coefficient (CPA ** 2 * TCPA) and take the top n
            sorted_targets = sorted(
                self.own_ship.detected_targets, key=lambda x: x.cpa ** 2 * x.tcpa
            )[:self.total_targets]

            # Filter out resolved targets (those with tcpa <= 0)
            self.own_ship.dangerous_targets = [
                target for target in sorted_targets if target.tcpa > 0
            ]

            # Fill in detected targets and set the targets to be dangerous
            for i, target in enumerate(self.own_ship.dangerous_targets):
                if target.cpa <= self.CPA_THRESHOLD and target.tcpa <= self.TCPA_THRESHOLD:
                    target.is_dangerous = True
                    targets_data[i] = self._generate_actual_target_data(target)
                else:
                    target.is_dangerous = False  # Optionally mark non-dangerous explicitly

        return targets_data

    def _place_dangerous_target_ship(self, aspect: Optional[str] = None) -> 'Target':
        """
        Place a TargetShip on a potentially dangerous track with specified CPA and TCPA.

        :param aspect: Defines the situation, i.e. static, head-on, overtaking, crossing. The target aspect will
        define the status of the target ship as stand-on or give-way in further training
        :return: TargetShip object added to the environment.
        """

        def calculate_target_course_and_speed(rel_course: float, rel_speed: float) -> tuple[float, float]:
            course_rad, speed = np.radians(self.own_ship.course), self.own_ship.speed
            rel_course_rad, rel_speed = np.radians(rel_course), rel_speed

            # Own ship velocity components (in Cartesian coordinates)
            own_vx = speed * np.sin(course_rad)
            own_vy = speed * np.cos(course_rad)

            # Relative velocity components (in Cartesian coordinates)
            rel_vx = rel_speed * np.sin(rel_course_rad)
            rel_vy = rel_speed * np.cos(rel_course_rad)

            # Target ship absolute velocity components (vector addition)
            target_vx = own_vx + rel_vx
            target_vy = own_vy + rel_vy

            # Calculate target ship speed (magnitude of the velocity vector)
            tgt_speed = np.sqrt(target_vx ** 2 + target_vy ** 2)

            # Calculate target ship course (angle of the velocity vector)
            tgt_course = (np.degrees(np.arctan2(target_vx, target_vy)) + 360) % 360

            return tgt_course, tgt_speed

        if aspect is None:
            aspect = random.choice(self.ASPECTS)

        # Own ship's parameters
        own_lat, own_lon = self.own_ship.lat, self.own_ship.lon
        own_course = self.own_ship.course
        own_speed = self.own_ship.speed

        # Target ship
        # if the relative course == 180 + relative bearing, the target is on a collision course.
        # to generate relative course != to 180 + relative bearing bss CPA, we need distance or TCPA
        # to calculate TCPA we need relative speed
        if aspect in ['static', 'crossing']:
            initial_distance = np.random.uniform(7, 9)
        else:
            initial_distance = np.random.uniform(12, 19)

        cpa = np.random.random_sample() * 2 - 1  # random CPA between -1 and 1 NM

        # define the position of the target vessel to comply with ColReg
        relative_bearing = None
        relative_speed = None
        if aspect == 'head-on':
            relative_bearing = np.random.uniform(-5, 5)  # Relative bearing in degrees
            relative_speed = np.random.uniform(5, 15) + own_speed
        elif aspect == 'static':
            relative_bearing = np.random.uniform(-5, 5)
            relative_speed = own_speed + np.random.sample()
        # own ship is give way vessel
        elif aspect == 'crossing':
            relative_bearing = np.random.uniform(5, 117.5)
            relative_speed = np.random.uniform(2, 15 + own_speed)
        # elif scene == 'crossing':  # own ship stands on
        #     relative_bearing = np.random.uniform(-5, -117.5)
        #     relative_speed = np.random.uniform(2, 15 + own_speed)
        elif aspect == 'overtaking':
            relative_bearing = np.random.uniform(-45, 45)
            relative_speed = np.random.uniform(2, own_speed * 0.9)
            initial_distance = np.random.uniform(2, 4)

        # deviation angle for randomness
        deviation_angle = np.degrees(np.arctan2(cpa, initial_distance))
        # print('Scene: ', aspect)

        # calculating relative course in degrees and add deviation
        true_target_bearing = (own_course + relative_bearing) % 360
        reversed_true_target_bearing = (true_target_bearing + 180) % 360
        relative_course = reversed_true_target_bearing + deviation_angle

        target_course, target_speed = calculate_target_course_and_speed(
            relative_course, relative_speed
        )
        # Calculate the initial position of the TargetShip
        target_lat, target_lon = plane_sailing_position(
            [own_lat, own_lon], true_target_bearing, initial_distance
        )

        #  create target and add it to the environment
        target = Target(
            position=(target_lat, target_lon),
            course=target_course,
            speed=target_speed,
        )
        target.aspect = aspect
        self.own_ship.update_target(target)

        return target

    @staticmethod
    def _flatten_observation(raw_observation):
        """
        Flattens a nested dictionary with tuples of dictionaries into a 1D NumPy array.

        :param raw_observation: Dict with 'own_ship' and 'targets'
        :return: Flattened NumPy array (axis=1)
        """
        # Extract and flatten own_ship data
        own_ship_features = list(raw_observation['own_ship'].values())

        # Extract and flatten targets data
        target_features = []
        for target in raw_observation['targets']:  # Assuming `targets` is a tuple of dicts
            target_features.extend(target.values())  # Flatten each target dictionary

        # Combine all into a single NumPy array (1D)
        flat_obs = np.array(own_ship_features + target_features, dtype=np.float32)

        return flat_obs

    def _latlon_to_pixels(self, lat, lon):
        """Convert latitude and longitude to pixel coordinates."""
        # Get the map's latitude and longitude ranges
        lat_range = self.lat_bounds[1] - self.lat_bounds[0]
        lon_range = self.lon_bounds[1] - self.lon_bounds[0]
        px = int((lon - self.lon_bounds[0]) / lon_range * self.window_size)
        py = int((self.lat_bounds[1] - lat) / lat_range * self.window_size)
        return px, py

    def _generate_own_ship_data(self) -> dict[str, float]:
        result = dict()
        for key in self.OWN_SHIP_PARAMS:
            result[key] = getattr(self.own_ship, key)
        for key in self.WP_PARAMS:
            result[key] = getattr(self, key)

        return result

    def _generate_zero_target_data(self, targets_count: int) -> list[dict[str, float]]:
        result = []
        for _ in range(targets_count):
            spaces_dict = {}
            for key in self.TARGET_PARAMS:
                spaces_dict[key] = 0.0

            result.append(spaces_dict)

        return result

    def _generate_actual_target_data(self, target: 'Target') -> Dict[str, float]:
        result = dict()
        for key in self.TARGET_PARAMS:
            attr = key.removeprefix('target_')
            result[key] = getattr(target, attr)

        return result

    def _generate_observation_dict(self, observation: ObsType) -> dict[str, dict[str, Any]]:
        idx = 0
        own_ship_data = dict()
        targets_data = dict()
        for key in self.OWN_SHIP_PARAMS:
            own_ship_data[key] = observation[idx]
            idx += 1
        for key in self.WP_PARAMS:
            own_ship_data[key] = observation[idx]
            idx += 1
        for key in self.TARGET_PARAMS:
            targets_data[key] = observation[idx]
            idx += 1
        return {'own_ship': own_ship_data, 'targets': targets_data}


if __name__ == '__main__':
    from stable_baselines3 import PPO

    env_kwargs = dict(
        render_mode='human',
        continuous=True,
        training_stage=2,
        timescale=1 / 6,
        training=False,
        seed=42,
        total_targets=3,
    )
    env = MarineEnv(**env_kwargs)
    agent = PPO('MlpPolicy', env=env).load("ppo.zip", device='cpu')
    for i in range(5):
        state, _ = env.reset()
        print(env.training_stage)
        print(env.own_ship.detected_targets)
        # env.cpa_limit = 2
        total_reward = 0
        for _ in range(int(400 / env.timescale)):
            action = agent.predict(state, deterministic=True)
            # action = [[1, 1], 0]
            next_state, reward, terminated, truncated, info = env.step(action[0])
            total_reward += reward

            print(next_state)
            print(env.own_ship.dangerous_targets)
            print(reward)
            print(total_reward)

            if terminated or truncated:
                print(info)
                break

            state = next_state
        print('Total reward: {}'.format(total_reward))
