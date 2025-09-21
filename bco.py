# Bee Colony Optimization (BCO) cho Traveling Salesman Problem (TSP)
# Mô tả ngắn: Cài đặt BCO theo mô hình forward–backward với tuyển mộ (recruit),
# trung thành (loyalty), nhảy múa (dance/advertise) và khai thác lân cận 2-opt.
# Chạy thử: python bco.py --n_cities 40 --iters 600 --bees 30
from __future__ import annotations
import math
import random
import argparse
from typing import List, Tuple, Optional

Point = Tuple[float, float]
Route = List[int]

# Tiện ích TSP
def euclid(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def build_distance_matrix(coords: List[Point]) -> List[List[float]]:
    n = len(coords)
    d = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dij = euclid(coords[i], coords[j])
            d[i][j] = d[j][i] = dij
    return d

def route_length(route: Route, dist: List[List[float]]) -> float:
    n = len(route)
    s = 0.0
    for i in range(n):
        a = route[i]
        b = route[(i + 1) % n]
        s += dist[a][b]
    return s

def two_opt_delta(route: Route, i: int, k: int, dist: List[List[float]]) -> float:
    # Delta chi phí khi đảo đoạn [i, k]
    n = len(route)
    a, b = route[(i - 1) % n], route[i]
    c, d = route[k], route[(k + 1) % n]
    before = dist[a][b] + dist[c][d]
    after = dist[a][c] + dist[b][d]
    return after - before

def apply_two_opt(route: Route, i: int, k: int) -> Route:
    return route[:i] + list(reversed(route[i:k + 1])) + route[k + 1:]

def local_search_2opt(route: Route, dist: List[List[float]], tries: int = 60) -> Route:
    """Thực hiện tối đa `tries` lần kiểm tra 2-opt ngẫu nhiên; áp dụng bước tốt nhất thấy được."""
    n = len(route)
    best_route = route
    best_delta = 0.0
    for _ in range(tries):
        i = random.randint(1, n - 2)
        k = random.randint(i + 1, n - 1)
        delta = two_opt_delta(route, i, k, dist)
        if delta < best_delta:
            best_delta = delta
            best_route = apply_two_opt(route, i, k)
    if best_delta < 0:
        return best_route
    # nếu không cải thiện, thử swap nhẹ để khám phá
    a, b = random.sample(range(n), 2)
    new_route = route[:]
    new_route[a], new_route[b] = new_route[b], new_route[a]
    return new_route

# Thuật toán BCO
class Bee:
    def __init__(self, route: Route, cost: float):
        self.route = route
        self.cost = cost
        self.prev_cost = cost

    def set(self, route: Route, cost: float):
        self.prev_cost = self.cost
        self.route = route
        self.cost = cost

    @property
    def improved(self) -> bool:
        return self.cost < self.prev_cost - 1e-9

class BCO_TSP:
    def __init__(
        self,
        coords: List[Point],
        n_bees: int = 30,
        recruiter_frac: float = 0.25,
        forward_steps: int = 3,
        local_tries: int = 60,
        alpha: float = 1.2,   # điều chỉnh xác suất trung thành
        beta: float = 2.0,    # độ thiên vị khi chọn recruiter theo 1/cost**beta
        max_iters: int = 500,
        stagnation_limit: int = 80,
        seed: Optional[int] = 42,
    ):
        assert 0 < recruiter_frac <= 1.0
        self.coords = coords
        self.n = len(coords)
        self.dist = build_distance_matrix(coords)
        self.n_bees = n_bees
        self.nr = max(1, int(n_bees * recruiter_frac))
        self.forward_steps = forward_steps
        self.local_tries = local_tries
        self.alpha = alpha
        self.beta = beta
        self.max_iters = max_iters
        self.stagnation_limit = stagnation_limit
        if seed is not None:
            random.seed(seed)

        self.bees: List[Bee] = []
        self.best_route: Route | None = None
        self.best_cost: float = float('inf')

    #Khởi tạo quần thể
    def _random_route(self) -> Route:
        r = list(range(self.n))
        random.shuffle(r)
        return r

    def initialize(self):
        self.bees = []
        for _ in range(self.n_bees):
            r = self._random_route()
            c = route_length(r, self.dist)
            self.bees.append(Bee(r, c))
            if c < self.best_cost:
                self.best_cost, self.best_route = c, r[:]

    #Cơ chế trung thành
    def loyalty_probability(self, bee: Bee, best_in_colony: float) -> float:
        # Xác suất trung thành cao nếu ong của bạn tốt hoặc vừa cải thiện.
        if bee.improved:
            return 0.85
        # dựa trên độ kém so với tốt nhất
        gap = (bee.cost - best_in_colony) / max(best_in_colony, 1e-9)
        p = math.exp(-self.alpha * gap)
        # kẹp trong [0.05, 0.9]
        return max(0.05, min(0.9, p))

    #Tuyển mộ & nhảy múa
    def select_recruiters(self) -> List[Bee]:
        return sorted(self.bees, key=lambda b: b.cost)[: self.nr]

    def roulette_choose_recruiter(self, recruiters: List[Bee]) -> Bee:
        # Xác suất tỉ lệ với (1/cost)^beta
        scores = [(1.0 / b.cost) ** self.beta for b in recruiters]
        s = sum(scores)
        pick = random.random() * s
        cum = 0.0
        for b, w in zip(recruiters, scores):
            cum += w
            if cum >= pick:
                return b
        return recruiters[-1]

    #Vòng lặp BCO
    def run(self, verbose: bool = True) -> Tuple[Route, float]:
        self.initialize()
        best_iter = 0
        it = 0
        while it < self.max_iters and (it - best_iter) < self.stagnation_limit:
            it += 1
            # Backward: xác định recruiter (ong nhảy múa)
            recruiters = self.select_recruiters()
            colony_best = recruiters[0].cost

            # Forward: lặp nhiều bước khám phá lân cận
            for _step in range(self.forward_steps):
                for bee in self.bees:
                    # recruiter luôn khai thác mạnh
                    if bee in recruiters:
                        new_route = local_search_2opt(bee.route, self.dist, self.local_tries)
                        new_cost = route_length(new_route, self.dist)
                        bee.set(new_route, new_cost)
                        continue

                    # quyết định trung thành hay theo recruiter
                    if random.random() < self.loyalty_probability(bee, colony_best):
                        # trung thành: khai thác quanh lời giải bản thân
                        new_route = local_search_2opt(bee.route, self.dist, self.local_tries)
                    else:
                        # không trung thành: theo một recruiter gần nhất rồi đột biến nhẹ
                        rec = self.roulette_choose_recruiter(recruiters)
                        cand = rec.route[:]
                        # đột biến: 2-opt hoặc swap nhỏ
                        new_route = local_search_2opt(cand, self.dist, self.local_tries // 2)
                    new_cost = route_length(new_route, self.dist)
                    bee.set(new_route, new_cost)

            # cập nhật best toàn cục
            for bee in self.bees:
                if bee.cost < self.best_cost:
                    self.best_cost, self.best_route = bee.cost, bee.route[:]
                    best_iter = it

            if verbose and it % 10 == 0:
                print(f"Iter {it:4d} | best = {self.best_cost:.3f}")

        if verbose:
            print(f"Kết thúc ở iter {it}, best = {self.best_cost:.3f}")
        return self.best_route[:], self.best_cost

# I/O tiện ích
def load_coords(path: str) -> List[Point]:
    coords: List[Point] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ',' in line:
                x, y = line.split(',')
            else:
                parts = line.split()
                if len(parts) < 2:
                    continue
                x, y = parts[0], parts[1]
            coords.append((float(x), float(y)))
    if len(coords) < 3:
        raise ValueError("File toạ độ phải có >= 3 dòng")
    return coords

def random_coords(n: int, width: int = 1000, height: int = 1000) -> List[Point]:
    return [(random.random() * width, random.random() * height) for _ in range(n)]

# Main
def main():
    ap = argparse.ArgumentParser(description="Bee Colony Optimization cho TSP")
    ap.add_argument('--coords', type=str, default=None, help='Đường dẫn file toạ độ (mỗi dòng: x,y hoặc x y)')
    ap.add_argument('--n_cities', type=int, default=30, help='Số thành phố (nếu không dùng file)')
    ap.add_argument('--bees', type=int, default=30, help='Số lượng ong (agents)')
    ap.add_argument('--iters', type=int, default=500, help='Số vòng lặp tối đa')
    ap.add_argument('--forward_steps', type=int, default=3, help='Số bước forward trong mỗi vòng lặp')
    ap.add_argument('--local_tries', type=int, default=60, help='Số lần thử 2-opt mỗi bước')
    ap.add_argument('--recruiter_frac', type=float, default=0.25, help='Tỷ lệ ong tuyển mộ (0-1)')
    ap.add_argument('--alpha', type=float, default=1.2, help='Hệ số xác suất trung thành')
    ap.add_argument('--beta', type=float, default=2.0, help='Độ thiên vị chọn recruiter theo 1/cost^beta')
    ap.add_argument('--seed', type=int, default=42, help='Seed ngẫu nhiên')
    ap.add_argument('--no_verbose', action='store_true', help='Tắt in log mỗi 10 vòng')
    args = ap.parse_args()

    if args.coords:
        coords = load_coords(args.coords)
    else:
        random.seed(args.seed)
        coords = random_coords(args.n_cities)

    bco = BCO_TSP(
        coords,
        n_bees=args.bees,
        recruiter_frac=args.recruiter_frac,
        forward_steps=args.forward_steps,
        local_tries=args.local_tries,
        alpha=args.alpha,
        beta=args.beta,
        max_iters=args.iters,
        seed=args.seed,
    )

    best_route, best_cost = bco.run(verbose=(not args.no_verbose))

    print("\nBest tour length:", round(best_cost, 3))
    print("Best route (index):", best_route)

    # --- Xuất tour ra CSV để tiện vẽ/kiểm tra ---
    with open("tour.csv", "w", encoding="utf-8") as f:
        f.write("order,city_id,x,y\n")
        for order, cid in enumerate(best_route):
            x, y = coords[cid]          # dùng coords đã tạo trong main()
            f.write(f"{order},{cid},{x},{y}\n")
        # thêm điểm quay về xuất phát để khép vòng
        first = best_route[0]
        fx, fy = coords[first]
        f.write(f"{len(best_route)},{first},{fx},{fy}\n")
    print("Đã ghi tour vào file tour.csv")

if __name__ == '__main__':
    main()
