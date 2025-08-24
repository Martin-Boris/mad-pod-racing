import math
import random


_MAX_ROTATION_PER_TURN = math.pi / 10
_FRICTION = 0.15
CHECKPOINT_GENERATION_MAX_GAP = 100
POD_RADIUS = 400
SPACE_BETWEEN_POD = 100
CHECKPOINT_RADIUS = 600

def from_vector(a,b):
    return Vector(b.x - a.x,b.y - a.y)

def from_tuple(a, b):
    return Vector(b[0] - a[0], b[1] - a[1])

def short_angle_dist(a0, a1):
    max_angle = math.pi * 2
    da = (a1 - a0) % max_angle
    return (2 * da) % max_angle - da

def point_to_segment_distance(px, py, x1, y1, x2, y2):
    """Returns the distance from point (px, py) to the segment from (x1, y1) to (x2, y2)."""
    # Handle degenerate segment case
    dx = x2 - x1
    dy = y2 - y1
    if dx == dy == 0:
        return math.hypot(px - x1, py - y1)

    # Project point onto the segment, computing parameterized t
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0, min(1, t))  # Clamp t to [0, 1]
    nearest_x = x1 + t * dx
    nearest_y = y1 + t * dy
    return math.hypot(px - nearest_x, py - nearest_y)


class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def angle(self):
        return math.atan2(self.y, self.x)

    def length(self):
        return math.hypot(self.x, self.y)

    def normalize(self):
        length = self.length()
        if length == 0:
            return Vector(0, 0)
        return Vector(self.x / length, self.y / length)

    def mult(self, factor):
        return Vector(self.x * factor, self.y * factor)

    def add(self, v):
        return Vector(self.x + v.x, self.y + v.y)

    def round(self):
        return Vector(round(self.x), round(self.y))

    def truncate(self):
        return Vector(math.trunc(self.x), math.trunc(self.y))

    def cross(self, s):
        return Vector(-s * self.y, s * self.x)

    def translate(self, p):
        return Vector(self.x + p[0], self.y + p[1])

    def get_tuple(self):
        return (self.x, self.y)

class Pod:
    def __init__(self, x, y):
        self.position = Vector(x, y)
        self.last_position = Vector(x, y)
        self.speed = Vector(0, 0)
        self.last_angle = None
        self.acceleration = Vector(0, 0)
        self.speed = Vector(0, 0)

    def turn_update(self, x, y):
        self.last_position.x = self.position.x
        self.last_position.y = self.position.y
        self.position.x = x
        self.position.y = y

    def apply_force(self, force):
        self.speed = self.speed.add(force).mult(1)

    def update_acceleration_from_angle(self,angle,power):
        if self.last_angle is not None:
            relative_angle = short_angle_dist(self.last_angle, angle)
            if abs(relative_angle) >= _MAX_ROTATION_PER_TURN:
                angle = self.last_angle + _MAX_ROTATION_PER_TURN * math.copysign(1, relative_angle)
        print("angle :",angle)
        self.last_angle = angle

        direction = Vector(math.cos(angle), math.sin(angle))
        self.acceleration = direction.normalize().mult(power)


    def update_acceleration(self, x, y, power):
        if self.position.x != x or self.position.y != y:
            angle = from_vector(self.position, Vector(x, y)).angle()
            self.update_acceleration_from_angle(angle, power)
        else:
            self.acceleration = Vector(0, 0)

    def step(self):
        self.last_position.x = self.position.x
        self.last_position.y = self.position.y
        self.position = self.position.add(self.speed.mult(1))

    def apply_friction(self):
        self.speed = self.speed.mult(1 - _FRICTION)

    def end_round(self):
        self.position = self.position.round()
        self.speed = self.speed.truncate()

maps = [
    [(12460, 1350), (10540, 5980), (3580, 5180), (13580, 7600)],
    [(3600, 5280), (13840, 5080), (10680, 2280), (8700, 7460), (7200, 2160)],
    [(4560, 2180), (7350, 4940), (3320, 7230), (14580, 7700), (10560, 5060), (13100, 2320)],
    [(5010, 5260), (11480, 6080), (9100, 1840)],
    [(14660, 1410), (3450, 7220), (9420, 7240), (5970, 4240)],
    [(3640, 4420), (8000, 7900), (13300, 5540), (9560, 1400)],
    [(4100, 7420), (13500, 2340), (12940, 7220), (5640, 2580)],
    [(14520, 7780), (6320, 4290), (7800, 860), (7660, 5970), (3140, 7540), (9520, 4380)],
    [(10040, 5970), (13920, 1940), (8020, 3260), (2670, 7020)],
    [(7500, 6940), (6000, 5360), (11300, 2820)],
    [(4060, 4660), (13040, 1900), (6560, 7840), (7480, 1360), (12700, 7100)],
    [(3020, 5190), (6280, 7760), (14100, 7760), (13880, 1220), (10240, 4920), (6100, 2200)],
    [(10323, 3366), (11203, 5425), (7259, 6656), (5425, 2838)],
]

class Map:
    def __init__(self, seed):
        random.seed(seed)
        points = random.choice(maps)
        # Rotate list by a random amount
        rotation = random.randint(0, len(points) - 1)
        points = points[rotation:] + points[:rotation]

        # Generate checkpoint list with random offset
        self.check_points = []
        for x, y in points:
            offset_x = random.randint(-CHECKPOINT_GENERATION_MAX_GAP, CHECKPOINT_GENERATION_MAX_GAP)
            offset_y = random.randint(-CHECKPOINT_GENERATION_MAX_GAP, CHECKPOINT_GENERATION_MAX_GAP)
            self.check_points.append((x + offset_x, y + offset_y))

        start_point = self.check_points[0]
        direction = from_tuple(start_point,self.check_points[1]).normalize().cross(1)
        podCount = 2

        pods = []
        for i in range(podCount):
            offset = ((-1 if i % 2 == 0 else 1) * (i // 2 * 2 + 1) + podCount % 2)
            position = direction.mult(offset * (POD_RADIUS + SPACE_BETWEEN_POD)).translate(start_point).round()
            pod = Pod(position.x,position.y)
            pods.append(pod)

        self.pods = pods