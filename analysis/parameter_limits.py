"""
Temporary 'storage' for Parameter Limits.
"""

parameter_limits = {
    'Pressure Altitude [ft]': {
        'arinc': None,
        'rate_of_change': None,
        'max_limit': 41000,
        'min_limit': -1000
    },
    'Calibrated Airspeed [KCAS]': {
        'arinc': None,
        'rate_of_change': None,
        'max_limit': 340,
        'min_limit': None
    },
    'Weight [lbs]': {
        'arinc': None,
        'rate_of_change': None,
        'max_limit': 174700,
        'min_limit': None
    },
    'Engine Thrust/Power (lbf)': {
        'arinc': None,
        'rate_of_change': None,
        'max_limit': 22100,
        'min_limit': 0
    },
    'N1 []': {
        'arinc': None,
        'rate_of_change': None,
        'max_limit': 106,
        'min_limit': 0
    },
    'N2 []': {
        'arinc': None,
        'rate_of_change': None,
        'max_limit': 105,
        'min_limit': 0
    },
    'EGT [C]': {
        'arinc': None,
        'rate_of_change': None,
        'max_limit': 940,
        'min_limit': None
    },
    'Relative Time Count': {
        'arinc': None,
        'rate_of_change': None,
        'max_limit': 4095,
        'min_limit': 0
    },
    'Heading': {
        'arinc': None,
        'rate_of_change': None,
        'max_limit': 360,
        'min_limit': -180
    },
    'Normal Acceleration': {
        'arinc': None,
        'rate_of_change': None,
        'max_limit': 6,
        'min_limit': -3
    },
    'Pitch Attitude': {
        'arinc': None,
        'rate_of_change': None,
        'max_limit': 90,
        'min_limit': -90
    },
    'Roll Attitude': {
        'arinc': None,
        'rate_of_change': None,
        'max_limit': 180,
        'min_limit': -180
    },
    'T.E. Flap Position': {
        'arinc': None,
        'rate_of_change': None,
        'max_limit': 60,
        'min_limit': -60
    },
    'Cockpit Control Selection (Flaps)': {
        'arinc': None,
        'rate_of_change': None,
        'max_limit': 60,
        'min_limit': -60
    },
    'L.E. Flap (Slat) Position': {
        'arinc': None,
        'rate_of_change': None,
        'max_limit': 60,
        'min_limit': -60
    },
    'Cockpit Control Selection (Slats)': {
        'arinc': None,
        'rate_of_change': None,
        'max_limit': 60,
        'min_limit': -60
    },
    'Ground Spoiler Position': {
        'arinc': None,
        'rate_of_change': None,
        'max_limit': 60,
        'min_limit': -60
    },
    'Ground Spoiler Selection': {
        'arinc': None,
        'rate_of_change': None,
        'max_limit': 60,
        'min_limit': -60
    },
    'Speed Brake Position': {
        'arinc': None,
        'rate_of_change': None,
        'max_limit': 60,
        'min_limit': -60
    },
    'Speed Brake Selection': {
        'arinc': None,
        'rate_of_change': None,
        'max_limit': 60,
        'min_limit': -60
    },
    'Total Air Temperature': {
        'arinc': None,
        'rate_of_change': None,
        'max_limit': 90,
        'min_limit': -50
    },
    'Longitudinal Acceleration': {
        'arinc': None,
        'rate_of_change': None,
        'max_limit': 1,
        'min_limit': -1
    },
    'Lateral Acceleration': {
        'arinc': None,
        'rate_of_change': None,
        'max_limit': 1,
        'min_limit': -1
    },
    'Elevator Surface Position': {
        'arinc': None,
        'rate_of_change': None,
        'max_limit': 90,
        'min_limit': -90
    },
    'Elevator Surface Position Selected': {
        'arinc': None,
        'rate_of_change': None,
        'max_limit': 90,
        'min_limit': -90
    },
    'Aileron Surface Position': {
        'arinc': None,
        'rate_of_change': None,
        'max_limit': 90,
        'min_limit': -90
    },
    'Aileron Surface Position Selected': {
        'arinc': None,
        'rate_of_change': None,
        'max_limit': 90,
        'min_limit': -90
    },
    'Rudder Surface Position': {
        'arinc': None,
        'rate_of_change': None,
        'max_limit': 90,
        'min_limit': -90
    },
    'Rudder Surface Position Selected': {
        'arinc': None,
        'rate_of_change': None,
        'max_limit': 90,
        'min_limit': -90
    },
    'Elevator Trim Surface Position': {
        'arinc': None,
        'rate_of_change': None,
        'max_limit': 90,
        'min_limit': -90
    },
    'Elevator Trim Surface Position Selected': {
        'arinc': None,
        'rate_of_change': None,
        'max_limit': 90,
        'min_limit': -90
    },
    'Aileron Trim Surface Position': {
        'arinc': None,
        'rate_of_change': None,
        'max_limit': 90,
        'min_limit': -90
    },
    'Aileron Trim Surface Position Selected': {
        'arinc': None,
        'rate_of_change': None,
        'max_limit': 90,
        'min_limit': -90
    },
    'Rudder Trim Surface Position': {
        'arinc': None,
        'rate_of_change': None,
        'max_limit': 90,
        'min_limit': -90
    },
    'Rudder Trim Surface Position Selected': {
        'arinc': None,
        'rate_of_change': None,
        'max_limit': 90,
        'min_limit': -90
    },
    'Radio Altitude': {
        'arinc': None,
        'rate_of_change': None,
        'max_limit': 8000,
        'min_limit': -20
    },
    'ILS/GPS/GLS Glide Path': {
        'arinc': None,
        'rate_of_change': None,
        'max_limit': 4,
        'min_limit': -4
    },
    'MLS Elevation': {
        'arinc': None,
        'rate_of_change': None,
        'max_limit': 4,
        'min_limit': -4
    },
    'ILS/GPS/GLS Localizer': {
        'arinc': None,
        'rate_of_change': None,
        'max_limit': 4,
        'min_limit': -4
    },
    'MLS Azimuth': {
        'arinc': None,
        'rate_of_change': None,
        'max_limit': 4,
        'min_limit': -4
    },
    'Distance to Runway Threshold (GLS)': {
        'arinc': None,
        'rate_of_change': None,
        'max_limit': 200,
        'min_limit': 0
    },
    'Latitude': {
        'arinc': None,
        'rate_of_change': None,
        'max_limit': 90,
        'min_limit': -90
    },
    'Longitude': {
        'arinc': None,
        'rate_of_change': None,
        'max_limit': 180,
        'min_limit': -180
    },
    'EPR': {
        'arinc': None,
        'rate_of_change': None,
        'max_limit': 2,
        'min_limit': 0
    },
    'N1': {
        'arinc': None,
        'rate_of_change': None,
        'max_limit': 125.00,
        'min_limit': 0.00
    },
    'N2': {
        'arinc': None,
        'rate_of_change': None,
        'max_limit': 125.00,
        'min_limit': 0.00
    },
    'N3': {
        'arinc': None,
        'rate_of_change': None,
        'max_limit': 125.00,
        'min_limit': 0.00
    },
    'Yaw or Slipside Angle': {
        'arinc': None,
        'rate_of_change': None,
        'max_limit': 90,
        'min_limit': -90
    },
    'Control Wheel - Cockpit Input Forces': {
        'arinc': None,
        'rate_of_change': None,
        'max_limit': 311,
        'min_limit': -311
    },
    'Control Column - Cockpit Input Forces': {
        'arinc': None,
        'rate_of_change': None,
        'max_limit': 378,
        'min_limit': -378
    },
    'Rudder Pedal - Cockpit Input Forces': {
        'arinc': None,
        'rate_of_change': None,
        'max_limit': 734,
        'min_limit': -734
    },
}
