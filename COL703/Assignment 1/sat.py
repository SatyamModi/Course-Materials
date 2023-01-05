import sys

proof_filename = sys.argv[2]    #proofN.txt
formula_filename = sys.argv[1]  #formula.txt

p_clauses = []  #list to store proof clauses
p_lines = []    #list to store proof lines 
f_clauses = []  #list to store formula clauses
f_lines = []    #list to store formula lines

with open(proof_filename) as p:
    p_lines = p.readlines()

print(p_lines)
for line in p_lines:
    line = line.rstrip('\n')
    line = line.split(' ')

    clause = []

    # remove starting 1p 7f
    for i in range(2, len(line)):
        if line[i] != '0':
            clause.append(int(line[i]))
    
    p_clauses.append(set(clause))


with open(formula_filename) as f:
    f_lines = f.readlines()

# reading clauses from formula file
for line in f_lines:

    if line.startswith('c'):
        continue
    if line.startswith('p'):
        continue
    else:
        line = line.rstrip('\n')
        line = line.split(' ')
        clause = []
        for i in range(len(line)): 
            # to check the end of the clause
            if (line[i] != '0'):
                clause.append(int(line[i]))

        f_clauses.append(set(clause))

flag = 1

for i in range(len(p_lines)):

    line = p_lines[i].split(' ')

    clause1 = set()
    clause2 = set()

    if (line[0].endswith('f')):
        idx = int(line[0].rstrip('f')) - 3
        clause1 = f_clauses[idx]
    
    if (line[0].endswith('p')):
        idx = int(line[0].rstrip('p'))-1

        # If the clause being used has not been derived yet
        if (idx >= i):
            flag = 0
            break

        clause1 = p_clauses[idx]
    
    if (line[1].endswith('f')):
        idx = int(line[1].rstrip('f')) - 3
        clause2 = f_clauses[idx]
    
    if (line[1].endswith('p')):
        idx = int(line[1].rstrip('p'))-1

        # If the clause being used has not been derived yet
        if (idx >= i):
            flag = 0
            break

        clause2 = p_clauses[idx]

    possibles = []
    for var in clause1:
        if -var in clause2:
            case = clause1 | clause2
            case.remove(var)
            case.remove(-var)
            possibles.append(case)
    
    proof = p_clauses[i]
    if proof not in possibles:
        flag = 0
        break

if flag == 0:
    print("Incorrect")
else:
    print("Correct")