from typing import List, Tuple, Set
from dataclasses import dataclass
import glob
import time
import tracemalloc
from math import sqrt
from collections import defaultdict

@dataclass
class PuzzleData:
    rows: int
    cols: int
    puzzle: List[List[int]]
    location_data: List[Tuple[int, int, int]]
    state: List[List[int]]
    factors_cache: List[List[int]]
    count: List[int]
    last_cells: List[List[Tuple[int, int]]]


class PuzzleSolver:
    def __init__(self):
        self.data = None

    def read_puzzle(self, input_filename: str) -> None:
        """Read puzzle data from file and initialize puzzle structure."""
        with open(input_filename, "r") as input_file:
            rows = int(input_file.readline())
            cols = int(input_file.readline())
            puzzle = [[-1] * cols for _ in range(rows)]
            location_data = []
            
            for row, line in enumerate(input_file):
                for col, symbol in enumerate(line.split()):
                    value = -1 if symbol == "-" else int(symbol)
                    puzzle[row][col] = value
                    if value != -1:
                        location_data.append((row, col, value))
            
            self.data = PuzzleData(
                rows=rows,
                cols=cols,
                puzzle=puzzle,
                location_data=location_data,
                state=[[-1] * cols for _ in range(rows)],
                factors_cache=[],
                count=[0] * len(location_data),
                last_cells=[[] for _ in range(len(location_data))]
            )

    @staticmethod
    def calculate_factors(n: int) -> List[int]:
        """Calculate factors of a number more efficiently."""
        factors = set()
        for i in range(1, int(sqrt(n)) + 1):
            if n % i == 0:
                factors.add(i)
                factors.add(n // i)
        return sorted(list(factors))

    def initialize_factors(self) -> None:
        """Initialize factors for all values in location data."""
        self.data.factors_cache = [
            self.calculate_factors(value) 
            for _, _, value in self.data.location_data
        ]

    @staticmethod
    def is_valid_bounds(L: List[List[int]], r1: int, r2: int, c1: int, c2: int) -> bool:
        """Check if the given bounds are valid."""
        if r1 > r2 or c1 > c2 or r1 < 0 or c1 < 0:
            return False
        return r2 < len(L) and c2 < len(L[0])

    @staticmethod
    def is_valid_region(L: List[List[int]], r1: int, r2: int, c1: int, c2: int, 
                       value: int, value2: int) -> bool:
        """Check if the region contains only valid values."""
        return all(L[r][c] in (value, value2) 
                  for r in range(r1, r2 + 1) 
                  for c in range(c1, c2 + 1))

    @staticmethod
    def set_region_value(L: List[List[int]], r1: int, r2: int, c1: int, c2: int, 
                        value: int) -> None:
        """Set a value for all cells in the given region."""
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                L[r][c] = value

    def verify_solution(self) -> bool:
        """Verify if the current state is a valid solution."""
        for i, (row, col, val) in enumerate(self.data.location_data):
            if self.data.state[row][col] != i:
                return False

            cells = [(r, c) for r in range(self.data.rows) 
                    for c in range(self.data.cols) 
                    if self.data.state[r][c] == i]
            
            if len(cells) != val:
                return False

            r_min = min(r for r, _ in cells)
            r_max = max(r for r, _ in cells)
            c_min = min(c for _, c in cells)
            c_max = max(c for _, c in cells)
            
            if (r_max - r_min + 1) * (c_max - c_min + 1) != len(cells):
                return False

        return True

    def dfs(self, next_index: int) -> bool:
        """Depth-first search to solve the puzzle."""
        if next_index >= len(self.data.location_data):
            return True

        e_row, e_col, e_value = self.data.location_data[next_index]

        while self.data.count[next_index] < len(self.data.factors_cache[next_index]):
            fac = self.data.factors_cache[next_index][self.data.count[next_index]]
            
            for i in range(e_value // fac):
                for j in range(fac):
                    r1, r2 = e_row - j, e_row + fac - 1 - j
                    c1, c2 = e_col + i - e_value // fac + 1, e_col + i

                    if (self.is_valid_bounds(self.data.state, r1, r2, c1, c2) and
                        self.is_valid_region(self.data.state, r1, r2, c1, c2, -1, next_index)):
                        
                        self.set_region_value(self.data.state, r1, r2, c1, c2, next_index)
                        
                        if all(self.data.state[r][c] != -1 
                              for r, c in self.data.last_cells[next_index]):
                            if self.dfs(next_index + 1):
                                return True
                        
                        self.set_region_value(self.data.state, r1, r2, c1, c2, -1)
                        self.data.state[e_row][e_col] = next_index

            self.data.count[next_index] += 1

        if next_index > 0:
            self.data.count[next_index] = 0
        return False

    def initialize(self) -> None:
        """Initialize the puzzle solver state."""
        self.initialize_factors()
        
        # Set initial state
        for i, (row, col, _) in enumerate(self.data.location_data):
            self.data.state[row][col] = i

        # Calculate last cells
        temp_state = [[-1] * self.data.cols for _ in range(self.data.rows)]
        for i, (row, col, _) in enumerate(self.data.location_data):
            temp_state[row][col] = i

        # Pre-calculate possible positions
        for z, (e_row, e_col, e_value) in enumerate(self.data.location_data):
            for fac in self.data.factors_cache[z]:
                for i in range(e_value // fac):
                    for j in range(fac):
                        r1, r2 = e_row - j, e_row + fac - 1 - j
                        c1, c2 = e_col + i - e_value // fac + 1, e_col + i
                        
                        if self.is_valid_bounds(temp_state, r1, r2, c1, c2):
                            self.set_region_value(temp_state, r1, r2, c1, c2, z)

        # Store last cells
        for row in range(self.data.rows):
            for col in range(self.data.cols):
                value = temp_state[row][col]
                if value != -1:
                    self.data.last_cells[value].append((row, col))

        # Reset state
        self.data.state = [[-1] * self.data.cols for _ in range(self.data.rows)]
        for i, (row, col, _) in enumerate(self.data.location_data):
            self.data.state[row][col] = i

    def solve(self, filename: str) -> Tuple[bool, float,float]:
        """Solve the puzzle and return solution status and time taken."""
        self.read_puzzle(filename)
        
        print("\nStart Grid:")
        self.print_grid(self.data.puzzle)
        
        tracemalloc.start()
        start_time = time.time()
        self.initialize()
        
        success = self.dfs(0)
        end_time = time.time()
        mem_usage = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return success, end_time - start_time, mem_usage[1] / 1024

    @staticmethod
    def print_grid(grid: List[List[int]]) -> None:
        """Print the grid in a formatted manner."""
        for row in grid:
            print(''.join(f"{str(symbol):>4}" for symbol in row))


def main():
    solver = PuzzleSolver()
    filenames = sorted(glob.glob("input/001.txt"))
    
    if not filenames:
        print("No input files found in inputSubmit/")
        return
    
    for filename in filenames:
        print(f"\nProcessing file: {filename}")
        success, time_taken, memory_usage = solver.solve(filename)
        
        print("\nFinal Grid:")
        solver.print_grid(solver.data.state)
        print(f"\nStatus: {'Solved' if success else 'Not Solved'}")
        print(f"Time taken: {time_taken:.3f} seconds")
        print(f"Memory used: {memory_usage:.2f} KB")
        print("=" * 40)


if __name__ == "__main__":
    main()