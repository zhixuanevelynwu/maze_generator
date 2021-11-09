#! /usr/bin/env python3
import pygame
import pygame.gfxdraw
from pygame.locals import *
import mazes


def main():
    pygame.init()
    screen = pygame.display.set_mode([1034, 778])
    running = True
    markup = None
    g = mazes.Grid(24, 32)
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == K_q:  # Quit
                    running = False
                elif event.key == K_n:  # New
                    g = mazes.Grid(24, 32)
                    markup = None
                elif event.key == K_f:  # Save to FILE
                    pygame.image.save(screen, 'maze.png')
                elif event.key == K_b:  # Binary Tree
                    mazes.binary_tree(g)
                    markup = None
                elif event.key == K_s:  # Sidewinder
                    mazes.sidewinder(g, .5)
                    markup = None
                elif event.key == K_a:  # Aldous-Broder random walk
                    mazes.aldous_broder(g)
                    markup = None
                elif event.key == K_w:  # Wilson
                    mazes.wilson(g)
                    markup = None
                elif event.key == K_r:  # Recursive Backtracker
                    mazes.recursive_backtracker(g)
                    markup = None
                elif event.key == K_h:  # Hybrid
                    mazes.ab_wilson(g)
                    markup = None
                elif event.key == K_l:  # longest path
                    markup = mazes.LongestPathMarkup(g)
                elif event.key == K_d:  # deadend count and color
                    m = mazes.Markup(g, default=0)
                    print(f'There are {len(g.deadends())} deadends')
                    for c in g.deadends():
                        m[c] = 1
                    markup = mazes.ColorizedMarkup(g)
                    markup.intensity_colorize(m)
                elif event.key == K_c:  # Colorize a Dijkstra from the center
                    markup = mazes.ColorizedMarkup(g, channel='G')
                    markup.colorize_dijkstra()
        display_grid(g, markup, screen)
        pygame.display.flip()


def display_grid(g, markup, screen):
    screen.fill((0, 0, 0))
    for row in range(g.num_rows):
        for col in range(g.num_columns):
            c = g.cell_at(row, col)
            cell_x = col * 32 + 5
            cell_y = row * 32 + 5
            # Draw top row
            if markup:
                value = markup.get_item_at(row, col)
                if not value:
                    continue
                if value == '*':  # Path marker
                    pygame.draw.circle(screen,
                                       (150, 80, 50),
                                       (cell_x+15, cell_y+15),
                                       7,  # radius
                                       0)  # filled
                if isinstance(value, list) and len(value) == 3:
                    pygame.draw.rect(screen,
                                     value,  # color
                                     (cell_x, cell_y, 32, 32))

            if not c.north or not c.is_linked(c.north):
                pygame.gfxdraw.hline(screen,
                                     cell_x, cell_x+31, cell_y,
                                     (100, 100, 100))
            if not c.south or not c.is_linked(c.south):
                pygame.gfxdraw.hline(screen,
                                     cell_x, cell_x+31, cell_y+31,
                                     (100, 100, 100))
            if not c.east or not c.is_linked(c.east):
                pygame.gfxdraw.vline(screen,
                                     cell_x+31, cell_y, cell_y+31,
                                     (100, 100, 100))
            if not c.west or not c.is_linked(c.west):
                pygame.gfxdraw.vline(screen,
                                     cell_x, cell_y, cell_y+31,
                                     (100, 100, 100))


if __name__ == "__main__":
    main()
