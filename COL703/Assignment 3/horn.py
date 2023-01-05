edges = {}
rev_edges = {}
used = {}
order = []
component = {}


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


def dfs1(v):
    used[v] = True
    for u in edges[v]:
        if used[u] == False:
            dfs1(u)
        else:
            continue

    order.append(v)


def dfs2(v, num):
    component[v] = num
    for u in rev_edges[v]:
        if component[u] == -1:
            dfs2(u, num)
        else:
            continue


def solve_2sat(n):

    for i in range(1, n+1):
        used[i] = False

    for i in range(1, n+1):
        if used[i] == False:
            dfs1(i)
        else:
            continue

    for i in range(1, n+1):
        component[i] = -1

    num = 1
    for i in range(n):
        v = order[n-i-1]
        if component[v] == -1:
            dfs2(v, num)
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

    result = solve_2sat(2*num_vars)
    return result

file = "formula.txt"
clauses = []
num_vars = 0
num_clause = 0
with open(file) as f:
    f_lines = f.readlines()

for line in f_lines:

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
        clause = []
        for i in range(len(line)):
            # to check the end of the clause
            if (line[i] != '0'):
                clause.append(int(line[i]))

        clauses.append(clause)

horn_check = check_horn(clauses)
if (horn_check):
    print("already horn")
else:
    print("not horn")

G = get_G(clauses)
s = set()
for clause in G:
    for l in clause:
        s.add(abs(l))

print("c 2-CNF formula which is sat iff input is renamable Horn")
print("p cnf {} {}".format(len(s), len(G)))
for clause in G:
    print(clause[0], clause[1], 0)


result = get_satisfiable(G, len(s))
if result == False:
    print('not renamable')
else:
    print('renamable')
    assignment = result[1]
    for p in assignment:
        if assignment[p] == True:
            print(p, end = " ")
        else:
            continue
    print()
