from typing import Tuple, List, Optional, Union
import numpy as np

from utils import plane_sailing_next_position, rumbline_distance, calculate_bearing, \
    calculate_relative_bearing


class StaticObject:
    def __init__(self, lat: float = 0.0, lon: float = 0.0) -> None:
        self.lat = lat
        self.lon = lon


class BaseShip:
    def __init__(
            self,
            position: Tuple[float, float] = (0.0, 0.0),
            course: float = 0.0,
            speed: float = 0.0,
            min_speed: float = -7.0,
            max_speed: float = 19.0,
    ) -> None:
        self.lat, self.lon = position
        self.course = course
        self.speed = speed
        self.min_speed = min_speed
        self.max_speed = max_speed

    def update_position(
            self,
            time_interval: int,
            clip_lat: Optional[tuple[float, float]] = None,
            clip_lon: Optional[tuple[float, float]] = None,
    ) -> Tuple[float, float]:
        lat, lon = plane_sailing_next_position(
            [self.lat, self.lon],
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
        self.stand_on: bool = True  # defines if the target gives way or is a stand-on vessel
        self.is_dangerous: bool = False  # defines the status of the target
        self.aspect: Optional[str] = None # the aspect of the target as viewed from own ship

    def __repr__(self):
        return (f'{self.__class__.__name__}:\n'
                f'Position: {self.lat, self.lon}\n'
                f'Course: {self.course:.2f}\n'
                f'Speed: {self.speed:.2f}\n'
                f'Relative Bearing: {self.relative_bearing:.2f}\n'
                f'Distance: {self.distance:.2f}\n'
                f'Relative Course: {self.relative_course:.2f}\n'
                f'Relative Speed: {self.relative_speed:.2f}\n'
                f'CPA: {self.cpa:.2f}\n'
                f'TCPA: {self.tcpa:.2f}\n'
                f'BCR: {self.bcr:.2f}\n'
                f'TBC: {self.tbc:.2f}\n'
                f'IsDangerous: {self.is_dangerous}\n'
                f'Aspect: {self.aspect}\n')



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
            [self.lat, self.lon],
            [target[0], target[1]] if isinstance(target, tuple) else [target.lat, target.lon])

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

    def calculate_cpa_tcpa(self, target: 'Target'):
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

    def calculate_bcr_tbc(self, target):
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

    def update_target(self, target: 'Target') -> None:
        target.relative_course = self.calculate_relative_course(target)
        target.relative_speed = self.calculate_relative_speed(target)
        target.relative_bearing = self.calculate_relative_bearing(target)
        target.distance = self.calculate_distance((target.lat, target.lon))
        target.cpa, target.tcpa = self.calculate_cpa_tcpa(target)
        target.bcr, target.tbc = self.calculate_bcr_tbc(target)

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
    target_ship = Target(position=(0.05, 0.05), course=270, speed=5, min_speed=5, max_speed=20)
    own_ship = OwnShip(position=(0.0, 0.0), course=0, speed=10.0, min_speed=5, max_speed=20)
    print(f'Distance: {own_ship.calculate_distance(target_ship)}')
    print(f'Relative Speed: {own_ship.calculate_relative_speed(target_ship)}')
    print(f'True bearing: {own_ship.calculate_true_bearing(target_ship)}')
    print(
        f'Relative target bearing to own ship: {target_ship.calculate_relative_bearing(own_ship)}')
    print(f'Relative bearing: {own_ship.calculate_relative_bearing(target_ship)}')
    print(f'Relative course: {own_ship.calculate_relative_course(target_ship)}')
    print(f'CPA / TCPA :{own_ship.calculate_cpa_tcpa(target_ship)}')
    print(f'BCR / TBC: {own_ship.calculate_bcr_tbc(target_ship)}')
