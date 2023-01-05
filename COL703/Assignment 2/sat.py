import sys

blanks = []     #list to store blank results
p_clauses = []  #list to store proof clauses
f_clauses = []  #list to store formula clauses
p_lines = []    #list to store proof lines 
f_lines = []    #list to store formula lines
num_atoms = 0
num_formulas = 0

NOT_RESOLVED = set({999})

def resolve(c1, c2, num_atoms):
    # enumerates in increasing order
    for i in range(1, num_atoms+1):
        if i in c1:
            if -i in c2:
                r_clause = c1 | c2
                r_clause.remove(i)
                r_clause.remove(-i)
                return r_clause
            else:
                continue
        
        elif -i in c1:
            if i in c2:
                r_clause = c1 | c2
                r_clause.remove(i)
                r_clause.remove(-i)
                return r_clause
            else:
                continue
        
        else:
            continue

    return NOT_RESOLVED


def proof_check(i):
    global p_clauses
    global blanks

    if i == len(p_lines):
        # check whether the last clause in the p_clause is empty clause
        last_p_clause = p_clauses[-1]
        if last_p_clause == set():
            return 1
        else:
            return 0

    line = p_lines[i]
    if '??' in line:
        #here atoms mean 3f, 4p
        for atoms in line:
            if (atoms != '??'):
                # if there is a clause from formula file in the proof step
                c1 = set()
                if atoms[-1] == 'f':
                    # line number of the clause
                    c_num = int(atoms[:len(atoms)-1])
                    c1 = f_clauses[c_num-3]
                else:
                    c_num = int(atoms[:len(atoms)-1])
                    c1 = p_clauses[c_num-1]

                # first search in the formula_clauses
                for j in range(num_formulas):
                    c2 = f_clauses[j]
                    r_clause = resolve(c1, c2, num_atoms)

                    # check if resolvable
                    if r_clause != NOT_RESOLVED:
                        p_clauses.append(r_clause)
                        blanks.append(str(j+3) + 'f')
                        result = proof_check(i+1)

                        if result == 1:
                            return 1
                        else:
                            p_clauses.pop()
                            blanks.pop()

                    #if not resolvable
                    else:
                        continue
                
                #If not found, now search in proof_clauses
                for j in range(len(p_clauses)):
                    c2 = p_clauses[j]
                    r_clause = resolve(c1, c2, num_atoms)

                    #check if resolvable
                    if r_clause != NOT_RESOLVED:
                        p_clauses.append(r_clause)
                        blanks.append(str(j+1)+'p')
                        result = proof_check(i+1)

                        if (result == 1):
                            return 1
                        else:
                            p_clauses.pop()
                            blanks.pop()
                    else:
                        continue
                
                # if no possible clause for resolving found
                return 0

            else:
                continue

    else:
        c1_info = line[0]
        c2_info = line[1]
        c1 = set()
        c2 = set()
        if (c1_info[-1] == 'f'):
            c_num = int(c1_info.rstrip('f'))
            c1 = f_clauses[c_num-3]
        else:
            c_num = int(c1_info.rstrip('p'))
            c1 = p_clauses[c_num-1]
        
        if (c2_info[-1] == 'f'):
            c_num = int(c2_info.rstrip('f'))
            c2 = f_clauses[c_num-3]
        else:
            c_num = int(c2_info.rstrip('p'))
            c2 = p_clauses[c_num-1]

        r_clause = resolve(c1, c2, num_atoms)
        # if resolvable
        if r_clause != NOT_RESOLVED:
            p_clauses.append(r_clause)
            result = proof_check(i+1)

            if result == 1:
                return 1
            else:
                p_clauses.pop()
                return 0
        # if not resolvable
        else:
            return 0

def solve(formula_filename, proof_filename, output_filename):
    global p_lines
    global num_atoms
    global num_formulas
    global f_clauses
    
    with open(formula_filename) as f:
        f_lines = f.readlines()

    # reading clauses from formula file
    for line in f_lines:
        
        if line.startswith('c'):
            continue

        if line.startswith('p'):
            line = line.rstrip('\n')
            line = line.split(' ')
            num_atoms = int(line[2])
            num_formulas = int(line[3])

        else:
            line = line.rstrip('\n')
            line = line.split(' ')
            clause = []
            for i in range(len(line)): 
                # to check the end of the clause
                if (line[i] != '0'):
                    clause.append(int(line[i]))

            f_clauses.append(set(clause))

    with open(proof_filename) as p:
        p_lines = p.readlines()

    for i in range(len(p_lines)):
        line = p_lines[i]
        line = line.rstrip('\n')
        line = line.split(' ')
        p_lines[i] = line

    result = proof_check(0)
    if (result == 1):
        idx = 0
        for i in range(len(p_lines)):
            for j in range(len(p_lines[i])):
                if (p_lines[i][j] == '??'):
                    p_lines[i][j] = blanks[idx]
                    idx += 1
                else:
                    continue
            
            for atom in p_clauses[i]:
                p_lines[i].append(str(atom))
            p_lines[i].append(str(0))
        
    else:
        for i in range(len(p_lines)):
            for j in range(len(p_lines[i])):
                if (p_lines[i][j] == '??'):
                    p_lines[i][j] = 'np'
                else:
                    continue
        
    with open("new_proof.txt", "w") as f:
        for line in p_lines:
            result = ' '.join(line)
            result += '\n'
            f.write(result)


proof_filename = sys.argv[2]
formula_filename = sys.argv[1]
output_filename = sys.argv[3]
solve(formula_filename, proof_filename, output_filename)

