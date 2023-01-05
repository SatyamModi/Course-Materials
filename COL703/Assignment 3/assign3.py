# from msilib.schema import Component
import sys

# Do not change the name of the function. 
# Do not use global variables as we will run your code on multiple test cases.
# 
def check_horn(clauses):

    for clause in clauses:
        pos_count = 0
        for atom in clause:
            if atom > 0:
                pos_count += 1
                if (pos_count >= 2):
                    return False
                else:
                    continue
            else:
                continue
    return True


def get_G(clauses):
    G = []
    for clause in clauses:
        for i in range(len(clause)):
            for j in range(i+1, len(clause)):
                G.append([clause[i], clause[j]])
    return G


def dfs1(v, edges, used, order):
    used[v] = True
    for u in edges[v]:
        if used[u] == False:
            dfs1(u, edges, used, order)
        else:
            continue

    order.append(v)


def dfs2(v, rev_edges, component, num):
    component[v] = num
    for u in rev_edges[v]:
        if component[u] == -1:
            dfs2(u, rev_edges, component, num)
        else:
            continue


def solve_2sat(edges, rev_edges, n):

	used = {}
	order = []
	component = {}

	for i in range(1, n+1):
		used[i] = False

	for i in range(1, n+1):
		if used[i] == False:
			dfs1(i, edges, used, order)
		else:
			continue

	for i in range(1, n+1):
		component[i] = -1

	num = 1
	for i in range(n):
		v = order[n-i-1]
		if component[v] == -1:
			dfs2(v, rev_edges, component, num)
			num += 1
		else:
			continue

	assignment = {}
	for i in range(1, n//2 + 1):
		assignment[i] = 0

	i = 1
	while (i <= n):
		if (component[i] == component[i+1]):
			return (False, {})
		else:
			assignment[(i+1)//2] = component[i] > component[i+1]
			i += 2

	return (True, assignment)


def get_satisfiable(clauses, num_vars):

	edges = {}
	rev_edges = {}
	for i in range(1, 2*num_vars+1):
		edges[i] = []
		rev_edges[i] = []

	for clause in clauses:

		if clause[0] > 0 and clause[1] > 0:
			a = abs(clause[0])
			b = abs(clause[1])
			edges[2*a].append(2*b-1)
			edges[2*b].append(2*a-1)
			rev_edges[2*b-1].append(2*a)
			rev_edges[2*a-1].append(2*b)

		elif clause[0] > 0 and clause[1] < 0:
			a = abs(clause[0])
			b = abs(clause[1])
			edges[2*b-1].append(2*a-1)
			edges[2*a].append(2*b)
			rev_edges[2*a-1].append(2*b-1)
			rev_edges[2*b].append(2*a)

		elif clause[0] < 0 and clause[1] > 0:
			a = abs(clause[0])
			b = abs(clause[1])
			edges[2*a-1].append(2*b-1)
			edges[2*b].append(2*a)
			rev_edges[2*b-1].append(2*a-1)
			rev_edges[2*a].append(2*b)

		elif clause[0] < 0 and clause[1] < 0:
			a = abs(clause[0])
			b = abs(clause[1])
			edges[2*a-1].append(2*b)
			edges[2*b-1].append(2*a)
			rev_edges[2*b].append(2*a-1)
			rev_edges[2*a].append(2*b-1)

		# print(edges)

	result = solve_2sat(edges, rev_edges, 2*num_vars)
	return result	

def solve(inputString, n):
	lines = inputString.split('\n')
	clauses = []
	num_vars = 0
	num_clause = 0
	
	for line in lines:

		if line.startswith('c'):
			continue

		if line.startswith('p'):
			line = line.rstrip('\n')
			line = line.split(' ')
			num_vars = int(line[2])
			num_clause = int(line[3])

		else:
			line = line.rstrip('\n')
			line = line.split(' ')
			print("line: ", line)
			clause = []
			for i in range(len(line)):
				# to check the end of the clause
				if (line[i] != '0'):
					clause.append(int(line[i]))

			clauses.append(clause)
	
	horn_check = check_horn(clauses)

	if (n == 1):
		if (horn_check):
			return "already horn"
		else:
			return "not horn"
	
	elif (n == 2):
		G = get_G(clauses)
		s = set()
		for clause in G:
			for l in clause:
				s.add(abs(l))

		res = "c 2-CNF formula which is sat iff input is renamable Horn\n"
		res += "p cnf {} {}".format(len(s), len(G)) + '\n'
		for clause in G:
			res += "{} {} {}".format(clause[0], clause[1], 0) + '\n'
		
		return res 

	elif (n == 3):
		if (horn_check):
			return "already horn"
		else:
			G = get_G(clauses)
			s = set()
			for clause in G:
				for l in clause:
					s.add(abs(l))

			result = get_satisfiable(G, len(s))
			if result == False:
				return 'not renamable'
			else:
				return 'renamable'
	
	else:
		if (horn_check):
			return "already horn"
		else:
			
			G = get_G(clauses)
			s = set()
			for clause in G:
				for l in clause:
					s.add(abs(l))

			result = get_satisfiable(G, len(s))
			
			if result[0] == False:
				return 'not renamable'
			else:
				assignment = result[1]
				res = ""
				for p in assignment:
					if assignment[p] == True:
						res += "{} ".format(p) 
					else:
						continue
				return res



# Main function: do NOT change this.
if __name__=="__main__":
	inputFile = sys.argv[1]
	n = int(sys.argv[2])
	with open(inputFile, 'r') as f:
		inputString = f.read()
		print(solve(inputString, n))
