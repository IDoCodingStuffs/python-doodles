roles = {"BT", "CK", "CH", "GD"}

def all_diff(assignments):
    return all([e in assignments for e in roles])

def bob_not_ch(assignments):
    return assignments[1] != "CH"

def carol_not_bt(assignments):
    return assignments[2] != "BT"

def binary_1(assignments):
    return assignments[1] != "BT" or (assignments[0] != "CK" and assignments[3] != "CK")

def binary_2(assignments):
    return assignments[2] != "BT"