You are to build a SAT solver using CDCL algorithm with unit propagation as discussed in class. (Section 4.3, Algorithm 4.1 from the Handbook of Satisfiability.
The input will be given in DIMACS format via STDIN. Look at the "File format" in the link given. Ignore all the comments (line beginning with 'c' as the first character).
The output format is the following.
If the formula is unsatisfiable then just output "UNSAT" (without the quotes) in the first and only line without any extra space or new line or any other text. For example,
UNSAT

If the formula is SAT then on the first line, output "SAT", on the second line output the satisfying assignment similar to MiniSAT-2.2.0. Look here for "MiniSAT output format".  For example,
SAT

1 -2 -3 4 5 0

Provide a complete satisfying interpretation of the formula in sorted variable order. For example, even if "1 -3" is sufficient to satisfy the formula, you have to print "1 2 -3". Note that the normal sorting would print "-3 1 2" which is NOT ACCEPTABLE. The order of printing has to be increasing in variable numbering (not literals)
All the output should be written on STDOUT. No other output should be present on STDOUT. For example, DO NOT print "Please enter number of variables: " etc., on STDOUT.
The input is NOT going to be interactive. DO NOT print anything on STDERR.