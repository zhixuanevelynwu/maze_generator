#! /usr/bin/env python3
import random
import time


class Cell:
    ''' Represents a single cell of a maze.  Cells know their neighbors
        and know if they are linked (connected) to each.  Cells have
        four potential neighbors, in NSEW directions.
    '''

    def __init__(self, row, column):
        assert row >= 0
        assert column >= 0
        self.row = row
        self.column = column
        self.links = {}
        self.north = None
        self.south = None
        self.east = None
        self.west = None

    def link(self, cell, bidirectional=True):
        assert isinstance(cell, Cell)
        self.links[cell] = True
        if bidirectional:
            cell.link(self, bidirectional=False)

    def unlink(self, cell, bidirectional=True):
        assert isinstance(cell, Cell)
        self.links[cell] = False
        if bidirectional:
            cell.unlink(self, bidirectional=False)

    def is_linked(self, cell):
        assert isinstance(cell, Cell)
        return self.links[cell]

    def all_links(self):
        list = []
        for cell in self.links:
            if self.links[cell]:
                list.append(cell)
        return list

    def link_count(self):
        count = 0
        for cell in self.links:
            if self.links[cell]:
                count += 1
        return count

    def neighbors(self):
        list = []
        if self.north != None:
            list.append(self.north)
        if self.south != None:
            list.append(self.south)
        if self.east != None:
            list.append(self.east)
        if self.west != None:
            list.append(self.west)
        return list

    def __str__(self):
        return f'Cell at {self.row}, {self.column}'


class Grid:
    ''' A container to hold all the cells in a maze. The grid is a
        rectangular collection, with equal numbers of columns in each
        row and vis versa.
    '''

    def __init__(self, num_rows, num_columns):
        assert num_rows > 0
        assert num_columns > 0
        self.num_rows = num_rows
        self.num_columns = num_columns
        self.grid = self.create_cells()
        self.connect_cells()

    def create_cells(self):
        cells = [[Cell(x, y) for x in range(self.num_columns)]
                 for y in range(self.num_rows)]
        for row in range(self.num_rows):
            for col in range(self.num_columns):
                cell = cells[row][col]
                if row > 0:
                    cell.north = cells[row-1][col]
                if row < self.num_rows-1:
                    cell.south = cells[row+1][col]
                if col > 0:
                    cell.west = cells[row][col-1]
                if col < self.num_columns-1:
                    cell.east = cells[row][col+1]
        return cells

    def connect_cells(self):
        for row in self.grid:
            for cell in row:
                for neighbor in cell.neighbors():
                    cell.link(neighbor, bidirectional=True)

    def unconnect_cells(self):
        for row in self.grid:
            for cell in row:
                for neighbor in cell.neighbors():
                    cell.unlink(neighbor, bidirectional=True)

    def cell_at(self, row, column):
        return self.grid[row][column]

    def deadends(self):
        deadends = []
        for row in self.grid:
            for cell in row:
                if cell.link_count() == 1:
                    deadends.append(cell)
        return deadends

    def each_cell(self):
        for row in range(self.num_rows):
            for col in range(self.num_columns):
                c = self.cell_at(row, col)
                yield c

    def each_row(self):
        for row in self.grid:
            yield row

    def random_cell(self):
        rand_row = random.randrange(0, self.num_rows)
        rand_col = random.randrange(0, self.num_columns)
        return self.cell_at(rand_row, rand_col)

    def size(self):
        return self.num_columns * self.num_rows

    def set_markup(self, markup):
        self.markup = markup

    def __str__(self):
        ret_val = '+' + '---+' * self.num_columns + '\n'
        for row in self.grid:
            ret_val += '|'
            for cell in row:
                cell_value = self.markup[cell]
                ret_val += '{:^3s}'.format(str(cell_value))
                if not cell.east:
                    ret_val += '|'
                elif cell.east.is_linked(cell):
                    ret_val += ' '
                else:
                    ret_val += '|'
            ret_val += '\n+'
            for cell in row:
                if not cell.south:
                    ret_val += '---+'
                elif cell.south.is_linked(cell):
                    ret_val += '   +'
                else:
                    ret_val += '---+'
            ret_val += '\n'
        return ret_val


class Markup:

    def __init__(self, grid, default=' '):
        self.grid = grid
        self.marks = {}  # Key: cell, Value = some object
        self.default = default

    def reset(self):
        self.marks = {}

    def __setitem__(self, cell, value):
        self.marks[cell] = value

    def __getitem__(self, cell):
        return self.marks.get(cell, self.default)

    def set_item_at(self, row, column, value):
        assert row >= 0 and row < self.grid.num_rows
        assert column >= 0 and column < self.grid.num_columns
        cell = self.grid.cell_at(row, column)
        if cell:
            self.marks[cell] = value
        else:
            raise IndexError

    def get_item_at(self, row, column):
        assert row >= 0 and row < self.grid.num_rows
        assert column >= 0 and column < self.grid.num_columns
        cell = self.grid.cell_at(row, column)
        if cell:
            return self.marks.get(cell)
        else:
            raise IndexError

    def max(self):
        ''' Return the cell with the largest markup value. '''
        if len(self.marks.keys()) == 0:
            return 0
        return max(self.marks.keys(), key=self.__getitem__)

    def min(self):
        ''' Return the cell with the largest markup value. '''
        return min(self.marks.keys(), key=self.__getitem__)


class DijkstraMarkup(Markup):

    def __init__(self, grid, root_cell, default=0):
        ''' Execute the algorithm and store each cell's value in self.marks[]
        '''
        super().__init__(grid, default)
        self.root_cell = root_cell
        self.__setitem__(root_cell, 0)
        # Dijkstra
        self.marks[root_cell] = 0
        frontier = [root_cell]
        while len(frontier) > 0:
            min_cell = self.min_cell(frontier)
            frontier.remove(min_cell)
            neighbors = self.unmarkedNeighbors(min_cell)
            frontier.extend(neighbors)

    def farthest_cell(self):
        cell = self.max()
        distance = self.marks[cell]
        return (cell, distance)

    def unmarkedNeighbors(self, cell):
        unmarked = []
        for n in cell.neighbors():
            if n not in self.marks and cell.is_linked(n):
                unmarked.append(n)
                self.marks[n] = 1 + self.marks[cell]
        return unmarked

    def min_cell(self, frontier):
        min_cell = frontier[0]
        for cell in frontier:
            if self.marks[cell] < self.marks[min_cell]:
                min_cell = cell
        return min_cell


class ShortestPathMarkup(DijkstraMarkup):
    def __init__(self, grid, start_cell, goal_cell,
                 path_marker='*', non_path_marker=' '):
        super().__init__(grid, start_cell)
        self.path_marker = path_marker
        self.non_path_marker = non_path_marker
        # process
        path = [goal_cell]
        cell = goal_cell
        while self.marks[cell] != 0:
            cell = self.minMarkNeighbor(cell, path)
            path.insert(0, cell)
        for p in path:
            self.marks[p] = path_marker

    def minMarkNeighbor(self, cell, path):
        min_cell = None
        for n in cell.neighbors():
            if n in path:
                continue
            if not cell.is_linked(n):
                continue
            if min_cell == None:
                min_cell = n
            if self.marks[n] <= self.marks[min_cell] and cell.is_linked(n):
                min_cell = n
        return min_cell


class LongestPathMarkup(ShortestPathMarkup):

    def __init__(self, grid, path_marker='*', non_path_marker=' '):
        start_cell = grid.random_cell()
        dm = DijkstraMarkup(grid, start_cell)
        farthest, _ = dm.farthest_cell()
        dm = DijkstraMarkup(grid, farthest)
        next_farthest, _ = dm.farthest_cell()
        super().__init__(grid, farthest, next_farthest, path_marker, non_path_marker)


class ColorizedMarkup(Markup):

    def __init__(self, grid, channel='R'):
        assert channel in 'RGB'
        super().__init__(grid)
        self.channel = channel

    def colorize_dijkstra(self, start_row=None, start_column=None):
        if not start_row:
            start_row = self.grid.num_rows // 2
        if not start_column:
            start_column = self.grid.num_columns // 2
        start_cell = self.grid.cell_at(start_row, start_column)
        dm = DijkstraMarkup(self.grid, start_cell)
        self.intensity_colorize(dm)

    def intensity_colorize(self, markup):
        max = markup.max()
        max_value = markup[max]
        if max == 0:
            return
        for c in self.grid.each_cell():
            cell_value = markup[c]
            intensity = (max_value - cell_value) / max_value
            dark = round(255 * intensity)
            bright = round(127 * intensity) + 128
            if self.channel == 'R':
                self.marks[c] = [bright, dark, dark]
            elif self.channel == 'G':
                self.marks[c] = [dark, bright, dark]
            else:
                self.marks[c] = [dark, dark, bright]


def binary_tree(grid):
    ''' The Binary Tree Algorithm.
    '''
    grid.unconnect_cells()
    for y in range(grid.num_rows):
        for x in range(grid.num_columns):
            cell = grid.cell_at(y, x)
            neighbors = []
            if cell.north != None:
                neighbors.append(cell.north)
            if cell.east != None:
                neighbors.append(cell.east)
            if len(neighbors) == 1:
                cell.link(neighbors[0], bidirectional=True)
            elif len(neighbors) == 2:
                connect = neighbors[random.randrange(0, 2)]
                cell.link(connect, bidirectional=True)


def sidewinder(grid, odds=.5):
    ''' The Sidewinder algorithm.
    '''
    grid.unconnect_cells()
    assert odds >= 0.0
    assert odds < 1.0
    runlist = []
    for y in range(grid.num_rows):
        for x in range(grid.num_columns):
            cell = grid.cell_at(y, x)
            if x < grid.num_columns - 1:
                if y == 0:
                    cell.link(cell.east, bidirectional=True)
                elif y > 0:
                    if len(runlist) == 0:
                        runlist.append(cell)
                    contd = random.uniform(0, 1)
                    if contd > odds:
                        cell.link(cell.east, bidirectional=True)
                        runlist.append(cell.east)
                    elif contd < odds:
                        connect = runlist[random.randrange(0, len(runlist))]
                        connect.link(connect.north, bidirectional=True)
                        runlist.clear()
            else:
                if y > 0 and len(runlist) > 0:
                    connect = runlist[random.randrange(0, len(runlist))]
                    connect.link(connect.north, bidirectional=True)
                    runlist.clear()
                elif y > 0:
                    cell.link(cell.north, bidirectional=True)
                    runlist.clear()


def aldous_broder(grid):
    ''' The Aldous-Broder algorithm is a random-walk algorithm.
    '''
    grid.unconnect_cells()
    start_time = time.time()
    cell = grid.random_cell()
    visited = {cell}
    iteration_count = 0
    while True:
        iteration_count += 1
        neighbors = cell.neighbors()
        next_step = neighbors[random.randrange(0, len(neighbors))]
        if not next_step in visited:
            cell.link(next_step, bidirectional=True)
            visited.add(next_step)
        cell = next_step
        if len(visited) == grid.size():
            break
    print(
        f'Aldous-Broder executed on a grid of size {grid.size()} in {iteration_count} steps.')
    print("--- ab generated a maze in %s seconds ---" %
          (time.time() - start_time))


def wilson(grid):
    ''' Wilson's algorithm
    '''
    grid.unconnect_cells()
    start_time = time.time()
    random_choices = 0
    loops_removed = 0
    unvisited = []
    cell = grid.random_cell()
    for y in range(grid.num_rows):
        for x in range(grid.num_columns):
            unvisited.append(grid.cell_at(y, x))
    unvisited.remove(cell)
    current = unvisited[random.randrange(0, len(unvisited))]
    path = [current]
    while True:
        neighbors = current.neighbors()
        next_step = neighbors[random.randrange(0, len(neighbors))]
        random_choices += 1
        if next_step not in unvisited:
            connectPath(path, unvisited, next_step)
            if len(unvisited) == 0:
                break
            path.clear()
            current = unvisited[random.randrange(0, len(unvisited))]
            path.append(current)
        elif next_step in path:
            loops_removed += 1
            path = chop(path, next_step)
            current = next_step
        else:
            path.append(next_step)
            current = next_step
    print(
        f'Wilson executed on a grid of size {grid.size()} with {random_choices}', end='')
    print(f' random cells choosen and {loops_removed} loops removed')
    print("--- wilson generated a maze in %s seconds ---" %
          (time.time() - start_time))


def connectPath(path, unvisited, next_step):
    for i in range(len(path)-1):
        path[i].link(path[i+1], bidirectional=True)
        unvisited.remove(path[i])
    path[len(path)-1].link(next_step, bidirectional=True)
    unvisited.remove(path[len(path)-1])


def chop(path, loop_cause):
    after = path.index(loop_cause)
    chop = []
    for cell in path:
        if path.index(cell) > after:
            chop.append(cell)
    return [cell for cell in path if cell not in chop]


def recursive_backtracker(grid, start_cell=None):
    ''' Recursive Backtracker
    '''
    grid.unconnect_cells()
    start_cell = grid.random_cell()
    stack = []
    visited = set()
    n = allNeighborsNotVisited(start_cell.neighbors(), visited)
    while True:
        if start_cell == None:
            start_cell = grid.random_cell()
        visited.add(start_cell)
        if len(visited) == grid.size():
            break
        n = allNeighborsNotVisited(start_cell.neighbors(), visited)
        while len(n) == 0:
            if len(stack) == 0:
                stack.append(grid.random_cell())
            start_cell = stack.pop()
            n = allNeighborsNotVisited(start_cell.neighbors(), visited)
        next_step = n[random.randrange(0, len(n))]
        start_cell.link(next_step, bidirectional=True)
        stack.append(next_step)
        start_cell = next_step


def allNeighborsNotVisited(neighbors, visited):
    list = []
    for n in neighbors:
        if n not in visited:
            list.append(n)
    return list


def ab_wilson(grid, switch=2):
    '''
    Hybrid of ab and wilson algorithm'''
    grid.unconnect_cells()
    start_time = time.time()
    cell = grid.random_cell()
    unvisited = []
    for y in range(grid.num_rows):
        for x in range(grid.num_columns):
            unvisited.append(grid.cell_at(y, x))
    unvisited.remove(cell)
    iteration_count = 0
    # ab
    while True:
        iteration_count += 1
        neighbors = cell.neighbors()
        next_step = neighbors[random.randrange(0, len(neighbors))]
        if next_step in unvisited:
            cell.link(next_step, bidirectional=True)
            unvisited.remove(next_step)
        cell = next_step
        if len(unvisited) == grid.size() // switch:
            break
    # wilson
    random_choices = 0
    loops_removed = 0
    current = unvisited[random.randrange(0, len(unvisited))]
    path = [current]
    while True:
        iteration_count += 1
        neighbors = current.neighbors()
        next_step = neighbors[random.randrange(0, len(neighbors))]
        random_choices += 1
        if next_step not in unvisited:
            connectPath(path, unvisited, next_step)
            if len(unvisited) == 0:
                break
            path.clear()
            current = unvisited[random.randrange(0, len(unvisited))]
            path.append(current)
        elif next_step in path:
            loops_removed += 1
            path = chop(path, next_step)
            current = next_step
        else:
            path.append(next_step)
            current = next_step
    print(
        f'hybrid_ab_wilson executed on a grid of size {grid.size()} with {random_choices}', end='')
    print(
        f' random cells choosen and {loops_removed} loops removed in {iteration_count} steps.')
    print("--- hybrid_ab_wilson generated a maze in %s seconds ---" %
          (time.time() - start_time))
