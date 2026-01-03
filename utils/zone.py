from config.config import config

# expose module-level name for backward compatibility
YELLOW_BOX_ZONE = config.YELLOW_BOX_ZONE

def is_point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def is_vehicle_in_zone(bbox, zone=YELLOW_BOX_ZONE):
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2
    return is_point_in_polygon((center_x, center_y), zone)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        config.load_config(config_file)
