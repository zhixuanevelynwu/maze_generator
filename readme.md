# Random Maze Generator with Dfferent Algorithms

## Table of Contents

- [About](#about)
- [Getting Started](#getting_started)
- [Usage](#usage)

## About <a name = "about"></a>

This is a visualization of different maze generation algorithms. The functions for solving the longest paths in each maze and marking the deadends are also implemented..

## Getting Started <a name = "getting_started"></a>

### Prerequisites

Have Python3 environment and Pygame Library installed to run the map visualization locally.

### Installing

A step by step series of examples that tell you how to get a development env running.

Clone the repo to your local folder

```
git clone https://github.com/zhixuanevelynwu/maze_generator.git
```

## Usage <a name = "usage"></a>

On your terminal, navigate to the project folder and run

```
python3 show_maze.py
```

You will open up a black pygame window. Now, to generate random mazes with different algorithms, use the control below:

```
- "A": Run Aldous-Broder random walk algorithm.
- "B": Run Binary Tree algorithm.
- "S": Run Sidewinder algorithm.
- "W": Run Wilson's algorithm.
- "R": Run Recursive Backtracker algorithm.
- "H": Run Hybrid Aldous-Broder-Wilson algorithm.
- "N": Erase the previous maze. You will need to type "N" each time before running a new algorithm visualization.
- "L": Run a Dijkstra’s algorithm to solve for the longest path in the maze.
- "D": Colorcode all the deadends in the map.
- "C": Colorize the distance from each cell on the map to the center.
```

Below are some example outputs:
![initial window](images/r.png?raw=true "Title")
![initial window](images/d.png?raw=true "Title")
![initial window](images/c.png?raw=true "Title")
![initial window](images/l.png?raw=true "Title")
