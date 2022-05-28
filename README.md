# Desicion-tree-from-scratch
Assignment in AI Foundations course at BGU

Problem 1: A Board of Trouble

In this exercise you will build a function that starts from one board and tries
to reach another using only a chain of legal moves. We will use 6 × 6 boards which you
will receive as a 2 dimensional array.

A value of 0 indicates the place is empty,
A value of 1 indicates there is a forcefield there, blocking your advance (marked in the output by @).
A value of 2 indicates you have an agent in that location (marked in the output by *).

Your agent can only move on straight lines (forward/back or left/right)
a distance of 1 each turn. A piece can disappear if it makes a move to
go beyond the final (6th) line, i.e., an agent making a forward move in line 6 will
disappear.

The cost is the number of moves that is needed to reach the goal, so we
wish to minimize the number of steps that it will take to reach the goal board



In this exercise you will only implement A* algorithm to solve this. 

You will write a function in python called find path:
def find_path(starting_board,goal_board,search_method,detail_output):

The function takes 4 variables (in this order of input):

starting board This is the beginning of your search. This is a 2-dimensional array
populated with the values 0,1 and 2 as explained above. You can assume this
is a valid board (though you can write a function to check this, which will
probably help in your debugging).

goal board This is the board you wish to reach from the starting board via the
process. This is a 2-dimensional array populated with the values 0,1 and 2 as
explained above. You can assume this is a legal board, with the forcefields in the
same location (again, a function checking it’s legality will probably be helpful
to you)

search method This will be an integer. It will have multiple possible values added
in Problem Set 2, but for now you will only implement:
1. An A*-heuristic search. You choose the heuristic. In your submitted answers, containing the answers to the questions, you will detail your
heuristic. It cannot be trivial (i.e., all 0). You need to explain in your
submitted answers if your heuristic is admissible, consistent, or neither.

detail output This is a binary variable. When it is false, your output is like the text
above – you give the full chain of locations. The first one contains the starting
board, and following that, boards with a single legal move from the board before
them, until the last line contains the goal board. If no path was found from the
starting board to the goal board, the output is No path found.

If the binary variable is true, for the first transformation (from the first set
of locations to your second set of locations) you need to print out your work
process, so for search method:
1-3
