import math

for l in [(0, 8), (8, 0), (0, -8), (-8, 0)]:
    print(math.sqrt((4 - l[0]) ** 2 + (-2 - l[1]) ** 2) + math.sqrt((l[0]) ** 2 + (4 - l[1]) ** 2))