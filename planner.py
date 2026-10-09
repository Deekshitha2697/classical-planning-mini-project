"""Classical planning with STRIPS: Blocks World, Air Cargo and Spare Tire.

Run in VS Code (Run > Run Without Debugging, or the terminal):
    python planner.py                         run every problem with every algorithm
    python planner.py blocks sussman          one problem, every algorithm
    python planner.py blocks sussman astar    one problem, one algorithm, prints the plan

Domains/problems: blocks (sussman, reverse, merge), cargo (cargo1, cargo2), tire (tire1)
Algorithms: bfs, dfs, astar, greedy, backward
"""
import heapq
import sys
import time
from collections import deque

LIMIT = 50_000  # stop a search after this many expansions
INF = float("inf")


# ---------------------------------------------------------------- STRIPS core
class Action:
    def __init__(self, name, pre, add, delete):
        self.name = name
        self.pre, self.add, self.delete = map(frozenset, (pre, add, delete))

    def applicable(self, state):
        return self.pre <= state

    def apply(self, state):
        return (state - self.delete) | self.add


class Problem:
    def __init__(self, name, actions, init, goal):
        self.name = name
        self.actions = actions
        self.init = frozenset(init)
        self.goal = frozenset(goal)


# ---------------------------------------------------------------- domains
def blocks_problem(name, blocks, init_towers, goal_towers):
    acts = []
    for x in blocks:
        acts.append(Action(f"pickup({x})",
                           [f"clear({x})", f"ontable({x})", "handempty"],
                           [f"holding({x})"],
                           [f"clear({x})", f"ontable({x})", "handempty"]))
        acts.append(Action(f"putdown({x})",
                           [f"holding({x})"],
                           [f"ontable({x})", f"clear({x})", "handempty"],
                           [f"holding({x})"]))
        for y in blocks:
            if x != y:
                acts.append(Action(f"stack({x},{y})",
                                   [f"holding({x})", f"clear({y})"],
                                   [f"on({x},{y})", f"clear({x})", "handempty"],
                                   [f"holding({x})", f"clear({y})"]))
                acts.append(Action(f"unstack({x},{y})",
                                   [f"on({x},{y})", f"clear({x})", "handempty"],
                                   [f"holding({x})", f"clear({y})"],
                                   [f"on({x},{y})", f"clear({x})", "handempty"]))
    init = ["handempty"]
    for t in init_towers:  # towers are listed bottom to top
        init.append(f"ontable({t[0]})")
        init += [f"on({t[i]},{t[i-1]})" for i in range(1, len(t))]
        init.append(f"clear({t[-1]})")
    goal = [f"on({t[i]},{t[i-1]})" for t in goal_towers for i in range(1, len(t))]
    return Problem(name, acts, init, goal)


def cargo_problem(name, cargos, planes, airports, init, goal):
    acts = []
    for c in cargos:
        for p in planes:
            for a in airports:
                acts.append(Action(f"Load({c},{p},{a})",
                                   [f"At({c},{a})", f"At({p},{a})"],
                                   [f"In({c},{p})"], [f"At({c},{a})"]))
                acts.append(Action(f"Unload({c},{p},{a})",
                                   [f"In({c},{p})", f"At({p},{a})"],
                                   [f"At({c},{a})"], [f"In({c},{p})"]))
    for p in planes:
        for a in airports:
            for b in airports:
                if a != b:
                    acts.append(Action(f"Fly({p},{a},{b})",
                                       [f"At({p},{a})"], [f"At({p},{b})"], [f"At({p},{a})"]))
    return Problem(name, acts, init, goal)


def tire_problem(name):
    tires = ["At(Flat,Axle)", "At(Flat,Ground)", "At(Spare,Trunk)",
             "At(Spare,Ground)", "At(Spare,Axle)"]
    acts = [
        Action("Remove(Spare,Trunk)", ["At(Spare,Trunk)"], ["At(Spare,Ground)"], ["At(Spare,Trunk)"]),
        Action("Remove(Flat,Axle)", ["At(Flat,Axle)"], ["At(Flat,Ground)", "AxleEmpty"], ["At(Flat,Axle)"]),
        Action("PutOn(Spare,Axle)", ["At(Spare,Ground)", "AxleEmpty"], ["At(Spare,Axle)"],
               ["At(Spare,Ground)", "AxleEmpty"]),
        Action("LeaveOvernight", [], [], tires),
    ]
    return Problem(name, acts, ["At(Flat,Axle)", "At(Spare,Trunk)"],
                   ["At(Spare,Axle)", "At(Flat,Ground)"])


PROBLEMS = {
    ("blocks", "sussman"): lambda: blocks_problem(
        "Blocks World: Sussman anomaly", "ABC", [["A", "C"], ["B"]], [["C", "B", "A"]]),
    ("blocks", "reverse"): lambda: blocks_problem(
        "Blocks World: reverse a 4-block tower", "ABCD", [["D", "C", "B", "A"]], [["A", "B", "C", "D"]]),
    ("blocks", "merge"): lambda: blocks_problem(
        "Blocks World: merge two towers", "ABCD", [["A", "B"], ["C", "D"]], [["D", "C", "B", "A"]]),
    ("cargo", "cargo1"): lambda: cargo_problem(
        "Air Cargo: 2 cargo, 2 airports", ["C1", "C2"], ["P1", "P2"], ["SFO", "JFK"],
        ["At(C1,SFO)", "At(C2,JFK)", "At(P1,SFO)", "At(P2,JFK)"],
        ["At(C1,JFK)", "At(C2,SFO)"]),
    ("cargo", "cargo2"): lambda: cargo_problem(
        "Air Cargo: 3 cargo, 3 airports", ["C1", "C2", "C3"], ["P1", "P2", "P3"], ["SFO", "JFK", "ATL"],
        ["At(C1,SFO)", "At(C2,JFK)", "At(C3,ATL)", "At(P1,SFO)", "At(P2,JFK)", "At(P3,ATL)"],
        ["At(C1,JFK)", "At(C2,SFO)", "At(C3,SFO)"]),
    ("tire", "tire1"): lambda: tire_problem("Spare Tire: change the flat"),
}


# ---------------------------------------------------------------- heuristics
def relaxed_cost(state, problem, combine):
    """Delete-relaxation estimate. combine=max gives h_max, combine=sum gives h_add."""
    cost = {f: 0 for f in state}
    changed = True
    while changed:
        changed = False
        for a in problem.actions:
            if all(p in cost for p in a.pre):
                c = 1 + combine(cost[p] for p in a.pre)
                for f in a.add:
                    if c < cost.get(f, INF):
                        cost[f] = c
                        changed = True
    if any(g not in cost for g in problem.goal):
        return INF
    return combine(cost[g] for g in problem.goal)


def h_max(state, problem):
    return relaxed_cost(state, problem, lambda it: max(it, default=0))


def h_add(state, problem):
    return relaxed_cost(state, problem, sum)


# ---------------------------------------------------------------- search
def forward_search(problem, kind):
    """kind: 'bfs', 'dfs', 'astar' (h_max) or 'greedy' (h_add)."""
    h = {"astar": h_max, "greedy": h_add}.get(kind, lambda s, p: 0)
    init = problem.init
    best = {init: 0}
    expanded = generated = 0
    tie = 0
    if kind in ("bfs", "dfs"):
        frontier = deque([(init, [])])
    else:
        h0 = h(init, problem)
        frontier = [(h0, 0, tie, init, [])]  # (priority, g, tie, state, plan)

    while frontier:
        if kind == "bfs":
            state, plan = frontier.popleft()
            g = len(plan)
        elif kind == "dfs":
            state, plan = frontier.pop()
            g = len(plan)
        else:
            _, g, _, state, plan = heapq.heappop(frontier)
            if kind == "astar" and g > best.get(state, INF):
                continue
        expanded += 1
        if problem.goal <= state:
            return plan, expanded, generated, False
        if expanded >= LIMIT:
            return None, expanded, generated, True
        for a in problem.actions:
            if not a.applicable(state):
                continue
            nxt, ng = a.apply(state), g + 1
            if kind == "astar":
                if ng >= best.get(nxt, INF):
                    continue
            elif nxt in best:
                continue
            best[nxt] = ng
            generated += 1
            if kind in ("bfs", "dfs"):
                frontier.append((nxt, plan + [a.name]))
            else:
                hv = h(nxt, problem)
                if hv == INF:
                    continue
                tie += 1
                priority = ng + hv if kind == "astar" else hv
                heapq.heappush(frontier, (priority, ng, tie, nxt, plan + [a.name]))
    return None, expanded, generated, False


def backward_search(problem):
    """Regress the goal through relevant actions until it holds in the initial state."""
    start = problem.goal
    frontier = deque([(start, [])])
    seen = {start}
    expanded = generated = 0
    while frontier:
        goal_set, plan = frontier.popleft()
        expanded += 1
        if goal_set <= problem.init:
            return plan, expanded, generated, False
        if expanded >= LIMIT:
            return None, expanded, generated, True
        for a in problem.actions:
            # relevant: adds something we need and deletes nothing we need
            if (a.add & goal_set) and not (a.delete & goal_set):
                prev = (goal_set - a.add) | a.pre
                if prev not in seen:
                    seen.add(prev)
                    generated += 1
                    frontier.append((prev, [a.name] + plan))
    return None, expanded, generated, False


ALGORITHMS = {
    "bfs": ("Forward BFS", lambda p: forward_search(p, "bfs")),
    "dfs": ("Forward DFS", lambda p: forward_search(p, "dfs")),
    "astar": ("Forward A* (h_max)", lambda p: forward_search(p, "astar")),
    "greedy": ("Greedy best-first (h_add)", lambda p: forward_search(p, "greedy")),
    "backward": ("Backward regression BFS", backward_search),
}


# ---------------------------------------------------------------- checking and output
def validate(problem, plan):
    """Replay the plan from the initial state and confirm the goal holds."""
    by_name = {a.name: a for a in problem.actions}
    state = problem.init
    for name in plan:
        a = by_name[name]
        if not a.applicable(state):
            return False
        state = a.apply(state)
    return problem.goal <= state


def run(problem, key, show_plan):
    label, solver = ALGORITHMS[key]
    t0 = time.perf_counter()
    plan, expanded, generated, hit = solver(problem)
    ms = (time.perf_counter() - t0) * 1000
    if plan is None:
        result = "limit reached" if hit else "no plan"
        steps, ok = "-", ""
    else:
        result = "plan found"
        steps, ok = len(plan), "verified" if validate(problem, plan) else "INVALID"
    print(f"  {label:<26} {result:<14} steps={steps!s:<4} expanded={expanded:<7} "
          f"generated={generated:<7} {ms:8.1f} ms  {ok}")
    if show_plan and plan:
        for i, name in enumerate(plan, 1):
            print(f"      {i:>2}. {name}")


def main(argv):
    keys = list(PROBLEMS)
    algos = list(ALGORITHMS)
    show_plan = False
    if len(argv) >= 2:
        keys = [(argv[0], argv[1])]
        if keys[0] not in PROBLEMS:
            sys.exit(f"Unknown problem. Choose from: {sorted(PROBLEMS)}")
    if len(argv) >= 3:
        if argv[2] not in ALGORITHMS:
            sys.exit(f"Unknown algorithm. Choose from: {algos}")
        algos, show_plan = [argv[2]], True
    for key in keys:
        problem = PROBLEMS[key]()
        print(f"\n{problem.name}")
        print(f"  initial: {sorted(problem.init)}")
        print(f"  goal:    {sorted(problem.goal)}")
        print(f"  ground actions: {len(problem.actions)}")
        for algo in algos:
            run(problem, algo, show_plan)


if __name__ == "__main__":
    main(sys.argv[1:])