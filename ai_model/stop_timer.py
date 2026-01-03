import time

class StopTimer:
    def __init__(self):
        self.stop_times = {}  # vehicle_id: {'start': timestamp, 'total': seconds}

    def update(self, vehicle_id, in_zone):
        if vehicle_id not in self.stop_times:
            self.stop_times[vehicle_id] = {'start': None, 'total': 0}

        if in_zone:
            if self.stop_times[vehicle_id]['start'] is None:
                self.stop_times[vehicle_id]['start'] = time.time()
        else:
            if self.stop_times[vehicle_id]['start'] is not None:
                self.stop_times[vehicle_id]['total'] += time.time() - self.stop_times[vehicle_id]['start']
                self.stop_times[vehicle_id]['start'] = None

    def get_stop_time(self, vehicle_id):
        return self.stop_times.get(vehicle_id, {'total': 0})['total']

    def check_violation(self, vehicle_id, limit):
        return self.get_stop_time(vehicle_id) > limit
