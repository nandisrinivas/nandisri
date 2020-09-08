Input Format:

c

c comments Weighted Max-SAT

c

p wcnf 2 4

5 1 -2 0

10 1 2 0

15 -1 -2 0

20 -1 2 0

In Weighted Max-SAT, the parameters line is "p wcnf nbvar nbclauses".
The weights of each clause will be identified by the first integer in each clause line.
The weight of each clause is an integer greater than or equal to 1, and the sum of all clauses must be smaller than 2^63.
Comments will be present in the input file, and your solver must correctly parse and ignore them.
Output Format:

 

o 15

o 5

s OPTIMUM FOUND

v -1 2