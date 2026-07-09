import random
import time
import threading
from datetime import datetime

# Zone boundaries (lat/lng ranges for Sangli area)
ZONE_BOUNDARIES = {
    "Zone-A": {"lat": (16.850, 16.860), "lng": (74.550, 74.560)},
    "Zone-B": {"lat": (16.860, 16.870), "lng": (74.560, 74.570)},
    "Zone-C": {"lat": (16.870, 16.880), "lng": (74.550, 74.560)},
    "Zone-D": {"lat": (16.880, 16.890), "lng": (74.560, 74.570)},
}

# In-memory store for live GPS data
live_gps_data = {}
_lock = threading.Lock()


def get_zone_from_coords(lat: float, lng: float) -> str:
    """Determine which zone a GPS coordinate falls in."""
    for zone, bounds in ZONE_BOUNDARIES.items():
        if (bounds["lat"][0] <= lat <= bounds["lat"][1] and
                bounds["lng"][0] <= lng <= bounds["lng"][1]):
            return zone
    return "Unknown"


def simulate_gps_update(vehicle_id: str, zone: str):
    """Generate a simulated GPS reading for a vehicle in a zone."""
    bounds = ZONE_BOUNDARIES.get(zone, ZONE_BOUNDARIES["Zone-A"])
    lat = round(random.uniform(*bounds["lat"]), 6)
    lng = round(random.uniform(*bounds["lng"]), 6)

    with _lock:
        live_gps_data[vehicle_id] = {
            "vehicle_id": vehicle_id,
            "latitude": lat,
            "longitude": lng,
            "zone": zone,
            "speed_kmh": round(random.uniform(0, 15), 1),
            "heading": round(random.uniform(0, 360), 1),
            "timestamp": datetime.utcnow().isoformat()
        }

    return live_gps_data[vehicle_id]


def get_all_live_positions():
    """Get all current vehicle positions."""
    with _lock:
        return dict(live_gps_data)


def get_vehicle_position(vehicle_id: str):
    """Get position of a specific vehicle."""
    with _lock:
        return live_gps_data.get(vehicle_id)


def start_gps_simulation():
    """Start background GPS simulation for demo vehicles."""
    vehicles = [
        ("TRUCK-001", "Zone-A"),
        ("TRUCK-002", "Zone-B"),
        ("TRUCK-003", "Zone-C"),
        ("TRUCK-004", "Zone-D"),
    ]

    def run():
        while True:
            for vehicle_id, zone in vehicles:
                simulate_gps_update(vehicle_id, zone)
            time.sleep(3)  # Update every 3 seconds

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    print("GPS simulation started")