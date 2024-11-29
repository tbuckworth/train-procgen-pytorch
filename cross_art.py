from typing import List

# Write any import statements here


# class Point:
#  def __init__(self, x, y, ):
#    self.x = x
#    self.y = y
#    self.


init_dict = {
    'U': False,
    'D': False,
    'L': False,
    'R': False
}


def accumulate(point_info, x, y, ch, total_crosses):
    if x not in point_info.keys():
        point_info[x] = {y: init_dict.copy()}
    elif y not in point_info[x].keys():
        point_info[x][y] = init_dict.copy()

    existing_cross = all(point_info[x][y].values())

    point_info[x][y][ch] = True

    new_cross = all(point_info[x][y].values())

    if new_cross and not existing_cross:
        total_crosses[0] += 1

    if ch == 'U':
        y += 1
        from_ch = 'D'
    elif ch == 'D':
        y -= 1
        from_ch = 'U'
    elif ch == 'R':
        x += 1
        from_ch = 'L'
    elif ch == 'L':
        x -= 1
        from_ch = 'R'
    else:
        raise NotImplementedError

    if x not in point_info.keys():
        point_info[x] = {y: init_dict.copy()}
    elif y not in point_info[x].keys():
        point_info[x][y] = init_dict.copy()

    new_cross = all(point_info[x][y].values())

    point_info[x][y][from_ch] = True

    update_cross = all(point_info[x][y].values())

    if update_cross and not new_cross:
        total_crosses[0] += 1

    return x, y


def getPlusSignCount(N: int, L: List[int], D: str) -> int:
    # Write your code here
    total_crosses = [0]
    point_info = {}
    x = 0
    y = 0
    pos = 0
    ch = D[pos]

    for ch, n in zip(D, L):
        for _ in range(n):
            x, y = accumulate(point_info, x, y, ch, total_crosses)

    return total_crosses[0]


if __name__ == '__main__':
    x = getPlusSignCount(N=8,
                         L=[1, 2, 2, 1, 1, 2, 2, 1],
                         D="UDUDLRLR")
    print(x)
