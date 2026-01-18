import time

class StopTimer:
    def __init__(self):
        self.start_times = {}
        self.stop_times = {}
        self.is_stopped = {}
        self.violation_logged = {}  # New: to track logged violations

    def update(self, object_id, is_in_zone):
        if is_in_zone:
            if object_id not in self.start_times:
                self.start_times[object_id] = time.time()
                self.is_stopped[object_id] = True
            self.stop_times[object_id] = time.time()
        else:
            # Vehicle is outside the zone, reset its state
            if object_id in self.start_times:
                del self.start_times[object_id]
            if object_id in self.stop_times:
                del self.stop_times[object_id]
            if object_id in self.is_stopped:
                del self.is_stopped[object_id]
            if object_id in self.violation_logged:
                del self.violation_logged[object_id]

    def check_violation(self, object_id, time_limit):
        if self.violation_logged.get(object_id):
            return False
        if self.is_stopped.get(object_id):
            if self.get_stop_time(object_id) > time_limit:
                return True
        return False

    def get_stop_time(self, object_id):
        if object_id in self.start_times and object_id in self.stop_times:
            return self.stop_times[object_id] - self.start_times[object_id]
        return 0

    def log_violation(self, object_id):
        """Mark a violation as logged to prevent duplicate entries."""
        self.violation_logged[object_id] = True