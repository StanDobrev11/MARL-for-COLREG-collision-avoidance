from typing import Tuple

import numpy as np


def plane_sailing_position(
    start_point: Tuple[float, float],
    bearing: float,
    distance: float
) -> np.ndarray:
    """
    Computes the new geographic position after traveling a given distance along a constant bearing
    using the plane sailing method.

    Plane sailing assumes a spherical Earth and is suitable for short to medium distances where
    curvature effects are minimal.

    Parameters:
    - start_point (Tuple[float, float]): The starting position as (latitude, longitude) in decimal degrees.
    - bearing (float): The direction of travel in degrees (0° = North, 90° = East, etc.).
    - distance (float): The traveled distance in nautical miles.

    Returns:
    - np.ndarray: The new position as a NumPy array [latitude, longitude] in decimal degrees.

    Calculation Steps:
    - Converts input coordinates and bearing to radians.
    - Converts the traveled distance from nautical miles to degrees of latitude/longitude.
    - Computes the latitude change (\(\Delta \text{lat}\)) using the bearing and starting latitude.
    - Adjusts for the mean latitude to calculate longitude change (\(\Delta \text{lon}\)).
    - Converts the final latitude and longitude back to degrees.

    Notes:
    - The method assumes **constant bearing**, meaning it follows a rhumb line rather than a great-circle route.
    - For long-distance navigation, **great-circle sailing** methods should be preferred.
    - The function handles latitude-dependent adjustments for longitude calculations.

    Example:
    >>> plane_sailing_position((34.0, -118.0), 90, 60)
    array([34.0, -117.0])  # Approximate new position after traveling 60 nautical miles east
    """

    # Convert lat/lon and course to radians
    lat1, lon1 = np.radians(start_point)
    bearing = np.radians(bearing)

    # Convert distance in nautical miles to degrees of latitude/longitude
    distance_rad = np.radians(distance / 60)  # Distance in radians (1 degree = 60 NM)

    # Calculate the change in latitude (delta_lat)
    delta_lat = distance_rad * np.cos(bearing) * np.cos(lat1)

    # Calculate the new latitude
    new_lat = lat1 + delta_lat

    # Calculate the mean latitude (average of the original and new latitude)
    mean_lat = (lat1 + new_lat) / 2

    # Calculate the change in longitude (delta_lon)
    if np.cos(mean_lat) != 0:
        delta_lon = distance_rad * np.sin(bearing) / np.cos(mean_lat)
    else:
        delta_lon = 0

    # Calculate the new longitude
    new_lon = lon1 + delta_lon

    # Convert the new latitude and longitude back to degrees
    new_lat = np.degrees(new_lat)
    new_lon = np.degrees(new_lon)

    return np.array([new_lat, new_lon])


def plane_sailing_next_position(
    start_point: Tuple[float, float],
    course: float,
    speed: float,
    time_interval: float = 6
) -> np.ndarray:
    """
    Computes the next position of a vessel using the plane sailing method.

    Plane sailing is a simplified navigation technique that assumes a spherical Earth
    and calculates new positions based on course, speed, and elapsed time.

    Parameters:
    - start_point (Tuple[float, float]): The starting position as (latitude, longitude) in decimal degrees.
    - course (float): The vessel's course (heading) in degrees (0° = North, 90° = East, etc.).
    - speed (float): The vessel's speed in knots (nautical miles per hour).
    - time_interval (float, optional): The time interval for movement in minutes. Default is 6 minutes.

    Returns:
    - np.ndarray: The new position as a NumPy array [latitude, longitude] in decimal degrees.

    Calculation Steps:
    - Converts input coordinates and course to radians.
    - Computes the distance traveled in nautical miles based on speed and time.
    - Converts distance to radians (1 degree = 60 NM).
    - Calculates the change in latitude (\(\Delta \text{lat}\)) using cosine of the course.
    - Computes the mean latitude to adjust longitude calculations.
    - Computes the change in longitude (\(\Delta \text{lon}\)), avoiding division by zero at the poles.
    - Converts the new coordinates back to degrees.

    Notes:
    - This method assumes **short distances** where Earth's curvature effects are minimal.
    - For long distances, **great-circle sailing** methods (e.g., Haversine) should be used.

    Example:
    >>> plane_sailing_next_position((34.0, -118.0), 90, 10, 6)
    array([34.0, -117.99])  # Approximate new position after 6 minutes
    """


    # Convert lat/lon and course to radians
    lat1, lon1 = np.radians(start_point)
    course = np.radians(course)

    # Calculate distance
    distance = speed * time_interval / 60

    # Convert distance in nautical miles to degrees of latitude/longitude
    distance_rad = np.radians(distance / 60)  # Distance in radians (1 degree = 60 NM)

    # Calculate the change in latitude (delta_lat)
    delta_lat = distance_rad * np.cos(course)

    # Calculate the new latitude
    new_lat = lat1 + delta_lat

    # Calculate the mean latitude (average of the original and new latitude)
    mean_lat = (lat1 + new_lat) / 2

    # Calculate the change in longitude (delta_lon)
    if np.cos(mean_lat) != 0:
        delta_lon = distance_rad * np.sin(course) / np.cos(mean_lat)
    else:
        delta_lon = 0

    # Calculate the new longitude
    new_lon = lon1 + delta_lon

    # Convert the new latitude and longitude back to degrees
    new_lat = np.degrees(new_lat)
    new_lon = np.degrees(new_lon)

    return np.array([new_lat, new_lon])


def mercator_latitude(lat: float) -> float:
    """
    Computes the Mercator-projected latitude for a given geographic latitude.

    The Mercator projection is a cylindrical map projection that preserves angles
    and directions, making it useful for navigation. This function transforms a
    latitude value into its corresponding Mercator representation.

    Parameters:
    - lat (float): Latitude in radians.

    Returns:
    - float: The Mercator-projected latitude.

    Notes:
    - The Mercator projection **distorts distances** but **preserves angles**, making
      it useful for navigation.
    - The function does **not** handle singularities at the poles (\(\pm 90^\circ\)).
    - Input values should be within **valid latitude ranges** to prevent domain errors.

    Example:
    >>> mercator_latitude(np.radians(45))
    0.881373587019543  # Approximate Mercator latitude for 45°N
    """

    return np.log(np.tan(np.pi / 4 + lat / 2))


def mercator_conversion(lat1: float, lon1: float, lat2: float, lon2: float) -> Tuple[float, float]:
    """
    Converts geographic coordinates to Mercator projection differences.

    This function computes the difference in latitude and longitude in the Mercator projection,
    which is commonly used for navigation as it preserves angles and directions.

    Parameters:
    - lat1 (float): Latitude of the starting point in decimal degrees.
    - lon1 (float): Longitude of the starting point in decimal degrees.
    - lat2 (float): Latitude of the destination point in decimal degrees.
    - lon2 (float): Longitude of the destination point in decimal degrees.

    Returns:
    - Tuple[float, float]:
      - delta_phi (float): Difference in Mercator-projected latitudes.
      - delta_lambda (float): Difference in longitudes (radians).

    Calculation:
    - Converts latitudes and longitudes to radians.
    - Applies the **Mercator latitude transformation** to compute **delta_phi**.
    - Computes **delta_lambda**, the difference in longitudes.

    Notes:
    - This function assumes a **spherical Earth** model.
    - It is primarily used in rhumb line distance calculations where a **constant bearing** is maintained.

    Example:
    >>> mercator_conversion(34.0, -118.0, 40.0, -74.0)
    (0.103, 0.768)  # Approximate Mercator coordinate differences
    """

    # Convert degrees to radians
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    lon1 = np.radians(lon1)
    lon2 = np.radians(lon2)

    delta_phi = mercator_latitude(lat2) - mercator_latitude(lat1)

    # Difference in longitudes
    delta_lambda = lon2 - lon1

    return delta_phi, delta_lambda


def rumbline_distance(start_point: Tuple[float, float], end_point: Tuple[float, float]) -> float:
    """
    Calculates the rhumb line (loxodromic) distance between two geographic coordinates.

    A rhumb line is a path of constant bearing, making it useful for maritime and
    aerial navigation where maintaining a steady course is important.

    Parameters:
    - start_point (Tuple[float, float]): A tuple (lat1, lon1) representing the starting latitude and longitude in decimal degrees.
    - end_point (Tuple[float, float]): A tuple (lat2, lon2) representing the destination latitude and longitude in decimal degrees.

    Returns:
    - float: The rhumb line distance in nautical miles.

    Calculation:
    - Uses the **Mercator projection approximation** to compute differences in latitude (Δφ) and longitude (Δλ).
    - Applies the midpoint latitude to adjust for Earth's curvature.
    - Converts the computed distance to **nautical miles** (multiplied by 3440.065, the approximate conversion factor).

    Notes:
    - This method assumes a **spherical Earth model** and is accurate for short-to-medium distances.
    - More precise geodesic distance calculations should use Vincenty's formula or Haversine formula.

    Example:
    >>> rumbline_distance((34.0, -118.0), (40.0, -74.0))
    2145.3  # Approximate nautical miles from Los Angeles to New York
    """

    lat1, lon1 = start_point
    lat2, lon2 = end_point

    # Use the midpoint latitude to minimize cosine variation
    mid_lat = (lat1 + lat2) / 2
    delta_phi, delta_lambda = mercator_conversion(lat1, lon1, lat2, lon2)

    return np.sqrt((delta_lambda * np.cos(np.radians(mid_lat))) ** 2 + delta_phi ** 2) * 3440.065


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculates the initial bearing (course angle) from the current position (lat1, lon1)
    to a given waypoint (lat2, lon2).

    The bearing is computed using the great-circle navigation formula, considering the
    Earth's curvature.

    Parameters:
    - lat1 (float): Latitude of the current position in decimal degrees.
    - lon1 (float): Longitude of the current position in decimal degrees.
    - lat2 (float): Latitude of the target waypoint in decimal degrees.
    - lon2 (float): Longitude of the target waypoint in decimal degrees.

    Returns:
    - float: The initial bearing in degrees, normalized to the range [0, 360].

    Formula:
    - Converts latitudes and longitudes to radians.
    - Uses the arctangent function to compute the directional angle between the two points.
    - Converts the result back to degrees and ensures it remains within [0, 360].

    Example:
    >>> calculate_bearing(34.0, -118.0, 40.0, -74.0)
    66.94  # Approximate bearing from Los Angeles to New York
    """

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    delta_lon = lon2 - lon1
    x = np.sin(delta_lon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon)
    bearing = np.degrees(np.arctan2(x, y))
    return (bearing + 360) % 360  # Normalize to [0, 360]


def calculate_relative_bearing(heading: float, bearing: float) -> float:
    """
    Calculates the relative bearing between the ship's heading and a target bearing.

    The relative bearing is the angular difference between the ship's **current heading**
    and the **bearing to a target**, mapped to a **-180° to +180°** range:
    - Positive values (**0° to 180°**) indicate the target is to the **starboard (right)**.
    - Negative values (**-180° to 0°**) indicate the target is to the **port (left)**.

    Parameters:
    - heading (float): The current heading of the vessel in degrees (0° = North, 90° = East, etc.).
    - bearing (float): The absolute bearing of the target in degrees (relative to North).

    Returns:
    - float: The relative bearing in degrees, normalized to the range [-180, 180].

    Calculation:
    - Normalizes both **heading** and **bearing** to the range **[0, 360]**.
    - Computes the clockwise difference between **bearing** and **heading**.
    - Adjusts values greater than **180°** by subtracting **360°** to map them to the correct
      negative range (port side).

    Example:
    >>> calculate_relative_bearing(30, 100)
    70  # Target is 70° to starboard

    >>> calculate_relative_bearing(350, 10)
    20  # Target is 20° to starboard

    >>> calculate_relative_bearing(100, 30)
    -70  # Target is 70° to port
    """


    # Normalize heading and bearing to 0-360 degrees
    heading = heading % 360
    bearing = bearing % 360

    # Calculate the difference between the bearing and the heading
    diff = (bearing - heading + 360) % 360

    # Map the difference to the semi-circle system
    if diff > 180:
        relative_bearing = diff - 360  # Convert to negative (port side)
    else:
        relative_bearing = diff  # Positive (starboard side)

    return relative_bearing


if __name__ == '__main__':
    true_bearing = calculate_bearing(30.1, 100.1, 30.4, 100.25)
    print(true_bearing)
    print(calculate_relative_bearing(45, true_bearing))
