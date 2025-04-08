from typing import Tuple, List, Optional, Union
import numpy as np

from utils import plane_sailing_next_position, rumbline_distance, calculate_bearing, \
    calculate_relative_bearing


class StaticObject:
    def __init__(self, lat: float = 0.0, lon: float = 0.0) -> None:
        self.lat = lat
        self.lon = lon

    def __repr__(self):
        return f'{self.lat}, {self.lon}'

    def wp_distance(self, agent: 'OwnShip') -> float:
        return agent.calculate_distance(self)

    def wp_eta(self, agent: 'OwnShip') -> float:
        # will be recalculated when calling reset()
        return 60 * self.wp_distance(agent) / agent.speed

    def wp_relative_bearing(self, agent: 'OwnShip') -> float:
        return agent.calculate_relative_bearing(self)

    @property
    def wp_target_eta(self) -> float:
        # will ALWAYS be calculated when handling the state
        return 0.0


class BaseShip:
    COUNT = 1

    ASPECT_CATEGORY = {
        'head_on': 0,  # Rule 14
        'crossing': 1,  # Rule 15
        'overtaking': 2,  # Rule 13
        'adrift': 3,  # underway but stopped and making no way
    }

    VESSEL_CPA_MULTIPLIER = {
        'pwd': 1,  # power-driven vessel, lowest priority
        'sv': 0.6,  # sailing vessel not propelled by machinery
        'fv': 0.6,  # fishing vessel engaged in fishing
        'ram': 1.2,  # vessel restricted in ability to manoeuvre
        'nuc': 1.2,
    }

    VESSEL_CATEGORY = {
        'pwd': 0,  # power-driven vessel, lowest priority
        'sv': 1,  # sailing vessel not propelled by machinery
        'fv': 2,  # fishing vessel engaged in fishing
        'ram': 3,  # vessel restricted in ability to manoeuvre
        'nuc': 4,  # not under command, highes priority
    }

    def __init__(
            self,
            position: Tuple[float, float] = (0.0, 0.0),
            course: float = 0.0,
            speed: float = 0.0,
            name: str = None,  # name of the vessel for easy reference
            kind: str = 'pwd',  # type of vessel for setting responsibilities, e.g. power-driven vessel, NUC, etc.
            visible: bool = True,  # set if the target is in reduced visibility
            min_speed: float = -7.0,
            max_speed: float = 19.0,
    ) -> None:
        self.lat, self.lon = position
        self.course = course
        self.speed = speed
        self.__name = name
        self.kind = kind
        self.visible = visible
        self.min_speed = min_speed
        self.max_speed = max_speed

        # Assign a name if not provided
        if name is None:
            self.__name = f'{self.class_name}_{self.__class__.COUNT}_{self.kind}'
            self.__class__.COUNT += 1  # Increment counter correctly
        else:
            self.__name = name

    @property
    def name(self):
        return self.__name

    @property
    def class_name(self):
        return self.__class__.__name__

    @property
    def vessel_category(self):
        return self.VESSEL_CATEGORY[self.kind]

    def update_position(
            self,
            time_interval: int,
            clip_lat: Optional[tuple[float, float]] = None,
            clip_lon: Optional[tuple[float, float]] = None,
    ) -> Tuple[float, float]:
        lat, lon = plane_sailing_next_position(
            (self.lat, self.lon),
            self.course,
            self.speed,
            time_interval,
        )
        if clip_lat is not None or clip_lon is not None:
            lat = np.clip(lat, clip_lat[0], clip_lat[1])
            lon = np.clip(lon, clip_lon[0], clip_lon[1])

        self.lat = lat
        self.lon = lon

        return self.lat, self.lon

    def calculate_true_bearing(self, target: ['Target', 'StaticObject']) -> float:
        return calculate_bearing(self.lat, self.lon, target.lat, target.lon)

    def calculate_relative_bearing(self, target: ['Target', 'StaticObject', 'OwnShip']) -> float:
        true_bearing = self.calculate_true_bearing(target)
        return calculate_relative_bearing(self.course, true_bearing)

    def __repr__(self):
        return f'{self.__class__.__name__}:\nPosition: {self.lat, self.lon}\nCourse: {self.course}\nSpeed: {self.speed}'


class Target(BaseShip):
    """
    All params of the target will be updated by the own ship except for the next position (lat, lon)
    Next position of the target will be updated in the step() method of the environment for the selected time interval.
    """

    def __init__(
            self, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.relative_bearing: float = 0.0  # the relative bearing to the target as viewed from the own ship
        self.distance: float = 0.0  # the distance between own ship and target
        self.relative_course: float = 0.0  # the course of the target related to the own ship
        self.relative_speed: float = 0.0  # the speed of the target related to the own ship
        self.cpa: float = 0.0  # CPA (closest point of approach) between own ship and target
        self.tcpa: float = 0.0  # time to CPA
        self.bcr: float = 0.0  # own ship bow crossing range of the target. If positive - crossing bow of target,
        self.tbc: float = 0.0  # time to BCR
        self.is_dangerous: bool = False  # defines the status of the target
        self.__stand_on: [bool, None] = True  # defines if the target gives way or is a stand-on vessel
        self.__aspect: Optional[str] = None  # the aspect of the target as viewed from own ship
        self.__cpa_threshold: Optional[float] = None

    @property
    def cpa_threshold(self):
        return self.__cpa_threshold

    def set_cpa_threshold(self, base_cpa: float, traffic_multiplier: float, confined_multiplier: float) -> None:
        vessel_penalty = self.VESSEL_CPA_MULTIPLIER[self.kind]  # Scaling vessel category effect
        self.__cpa_threshold = base_cpa * traffic_multiplier * confined_multiplier * vessel_penalty

    @property
    def aspect(self) -> int | None:
        try:
            return self.ASPECT_CATEGORY.get(self.__aspect)
        except KeyError:
            return None

    @aspect.setter
    def aspect(self, agent: 'OwnShip') -> None:
        if self.__aspect is None:
            self.set_aspect(agent)

    def set_aspect(self, agent: 'OwnShip') -> None:

        if self.__aspect == 'overtaking' and abs(self.relative_bearing) < 112.5:
            return

        course_diff = self.course - agent.course  # Preserve sign

        # Normalize the difference within [-180, 180] to indicate relative movement to starboard or port
        if course_diff > 180:
            course_diff -= 360
        elif course_diff < -180:
            course_diff += 360

        reversed_relative_bearing = self.calculate_relative_bearing(agent)

        # Head-on, Rule 14, each vessel sees the other ahead
        if abs(self.relative_bearing) <= 10 and abs(reversed_relative_bearing) <= 10:
            self.__aspect = 'head_on'

        # this sets for crossing or overtaking
        elif abs(self.relative_bearing) <= 112.5:
            # Crossing, Rule 15, each vessel sees the other on the opposite side
            if abs(reversed_relative_bearing) <= 112.5 and \
                    np.sign(self.relative_bearing) != np.sign(reversed_relative_bearing):
                self.__aspect = 'crossing'
            # Overtaking, Rule 13, target sees the vessel abaft the beam
            elif 112.5 < abs(reversed_relative_bearing) <= 180 and agent.speed > self.speed:
                if np.sign(self.relative_bearing) != np.sign(course_diff):
                    self.__aspect = 'overtaking'
                else:
                    self.__aspect = None

    @property
    def stand_on(self) -> bool:
        return np.float32(self.__stand_on)

    @stand_on.setter
    def stand_on(self, agent: 'OwnShip') -> None:
        if self.aspect == 'overtaking':
            self.__stand_on = True
        else:
            self.set_stand_on(agent)

    def set_stand_on(self, agent: 'OwnShip') -> None:
        if self.vessel_category < agent.vessel_category:
            self.__stand_on = False

        elif self.vessel_category > agent.vessel_category:
            self.__stand_on = True

        # both vessels are same kind so the responsibility will be resolved using the aspect
        else:
            # check if visible
            if not self.visible:
                self.__stand_on = False # no stand-on vessel when visibility is restricted
                return

            if self.aspect == 0:  # head on
                self.__stand_on = False
            elif self.aspect == 1:  # crossing
                if self.relative_bearing > 0:
                    self.__stand_on = True
                else:
                    self.__stand_on = False
            elif self.aspect == 2:  # overtaking
                self.__stand_on = True

    def __repr__(self):
        return (f'{self.__class__.__name__}:\n'
                f'Name: {self.name}\n'
                f'Position: {self.lat, self.lon}\n'
                f'Course: {self.course:.2f}\n'
                f'Speed: {self.speed:.2f}\n'
                f'Relative Bearing: {self.relative_bearing:.2f}\n'
                f'Distance: {self.distance:.2f}\n'
                f'Relative Course: {self.relative_course:.2f}\n'
                f'Relative Speed: {self.relative_speed:.2f}\n'
                f'CPA Threshold: {self.cpa_threshold:.2f}\n'
                f'CPA: {self.cpa:.2f}\n'
                f'TCPA: {self.tcpa:.2f}\n'
                f'BCR: {self.bcr:.2f}\n'
                f'TBC: {self.tbc:.2f}\n'
                f'IsDangerous: {self.is_dangerous}\n'
                f'Aspect: {self.aspect}\n'
                f'Category: {self.kind}\n'
                f'Stand On: {self.stand_on}\n'
                f'Visible: {self.visible}\n')


class OwnShip(BaseShip):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.detected_targets: List['Target'] = []  # contains a list of all targets in range
        self.dangerous_targets: List['Target'] = []  # contains the top 3 dangerous targets, ordered by CPA and TCPA

    def reset(self) -> None:
        self.detected_targets.clear()
        self.dangerous_targets.clear()

    def calculate_distance(self, target: Union['Target', 'StaticObject', 'Tuple']) -> float:
        return rumbline_distance(
            (self.lat, self.lon),
            (target[0], target[1]) if isinstance(target, tuple) else (target.lat, target.lon))

    def calculate_relative_course(self, target: 'Target') -> float:
        """
        Calculate the relative course of the target ship with respect to the own ship.

        :param target: The target ship.
        :return: Relative course in degrees [0, 360).
        """

        # Compute relative velocity components
        rel_vx, rel_vy = self._relative_velocity_components(target)

        # Compute relative course using atan2
        relative_course_rad = np.arctan2(rel_vx, rel_vy)
        relative_course_deg = (np.degrees(relative_course_rad) + 360) % 360  # Normalize to [0, 360)

        return relative_course_deg

    def calculate_relative_speed(self, target: 'Target') -> float:
        """
        Calculate the relative speed between own ship and the target ship.

        :param target: The target ship.
        :return: Relative speed in knots.
        """
        # Compute relative velocity components
        rel_vx, rel_vy = self._relative_velocity_components(target)

        # Compute relative speed magnitude
        relative_speed = np.sqrt(rel_vx ** 2 + rel_vy ** 2)

        return relative_speed

    def calculate_cpa_tcpa(self, target: 'Target') -> Tuple[float, float]:
        """
        Calculate CPA and TCPA with edge case handling.
        """
        relative_speed = target.relative_speed

        if relative_speed == 0:  # Avoid division by zero
            return target.distance, float('inf')  # CPA is current distance, TCPA is infinite

        beta_rad = self._calculate_beta(target)

        cpa = abs(self._calculate_cpa(target, beta_rad))  # Always positive CPA

        tcpa = (target.distance * np.cos(beta_rad)) / relative_speed

        return cpa, max(tcpa * 60, 0)  # Ensure TCPA is non-negative

    def calculate_bcr_tbc(self, target) -> Tuple[float, float]:
        """
        Calculate Bow Crossing Range (BCR) and Time to Bow Crossing (TBC) with edge case handling.
        """
        relative_speed = target.relative_speed
        theta = target.course - target.relative_course

        # Normalize theta within [-180, 180]
        theta = (theta + 180) % 360 - 180
        theta_rad = np.radians(theta)

        cpa = self._calculate_cpa(target, self._calculate_beta(target))
        tcpa = target.tcpa / 60  # Convert to hours

        # Handle cases where theta is too small (parallel motion)
        if np.isclose(np.sin(theta_rad), 0):
            return float('inf'), float('inf')  # No bow crossing

        # Calculate Bow Crossing Range (BCR)
        bcr = abs(cpa / np.sin(theta_rad))  # Ensure BCR is positive

        # Calculate Time to Bow Crossing (TBC)
        if np.isclose(np.tan(theta_rad), 0):  # Avoid division by zero
            delta_t = float('inf')
        else:
            delta_t = (cpa / np.tan(theta_rad)) / relative_speed

        tbc = tcpa + delta_t  # Initial calculation

        # crossing astern, negative bcr
        if tbc > tcpa:
            bcr *= -1

        return bcr, max(tbc * 60, 0)  # Convert back to minutes, ensure non-negative

    def _calculate_beta(self, target: 'Target') -> float:
        """Beta is the angle between the relative course and the reversed true bearing"""

        reversed_true_bearing = (180 + self.calculate_true_bearing(target)) % 360
        beta = target.relative_course - reversed_true_bearing

        # Normalize beta within [-180, 180] to avoid errors
        beta = (beta + 180) % 360 - 180
        beta_rad = np.radians(beta)
        return beta_rad

    @staticmethod
    def _calculate_cpa(target: 'Target', beta: float) -> float:
        return target.distance * np.sin(beta)

    @staticmethod
    def set_responsibilities(target: 'Target') -> None:
        # head-on and crossing situations
        if -5 <= target.relative_bearing <= 112.5:
            target.stand_on = True

        # overtaking
        elif abs(target.calculate_relative_bearing(target)) >= 112.5:
            target.stand_on = True

        # own ship is stand-on vessel
        else:
            target.stand_on = False

    def update_target(self, target: 'Target', *args) -> None:
        target.relative_course = self.calculate_relative_course(target)
        target.relative_speed = self.calculate_relative_speed(target)
        target.relative_bearing = self.calculate_relative_bearing(target)
        target.distance = self.calculate_distance((target.lat, target.lon))
        target.cpa, target.tcpa = self.calculate_cpa_tcpa(target)
        target.bcr, target.tbc = self.calculate_bcr_tbc(target)
        target.set_cpa_threshold(*args)
        target.set_aspect(self)
        target.set_stand_on(self)

    def _relative_position_components(self, target: 'Target') -> Tuple[float, float]:
        own_lat_rad = np.radians(self.lat)
        rx = (target.lon - self.lon) * 60 * np.cos(own_lat_rad)  # Longitudinal distance (NM)
        ry = (target.lat - self.lat) * 60  # Latitudinal distance (NM)

        return rx, ry

    def _relative_velocity_components(self, target: 'Target') -> Tuple[float, float]:
        own_course_rad = np.radians(self.course)
        target_course_rad = np.radians(target.course)

        own_vx = self.speed * np.sin(own_course_rad)
        own_vy = self.speed * np.cos(own_course_rad)

        target_vx = target.speed * np.sin(target_course_rad)
        target_vy = target.speed * np.cos(target_course_rad)

        # Compute relative velocity components
        rel_vx = target_vx - own_vx
        rel_vy = target_vy - own_vy

        return rel_vx, rel_vy


if __name__ == '__main__':
    target_ship = Target(position=(0.05, 0.05), course=330, speed=5, kind='pwd')
    own_ship = OwnShip(position=(0.0, 0.0), course=0, speed=10.0)
    own_ship.update_target(target_ship, 1, 0.4, 0.5)
    print(target_ship)
