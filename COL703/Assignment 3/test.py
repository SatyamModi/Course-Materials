s = ""
with open('formula.txt', 'r') as f:
    s = f.read()

lines = s.split('\n')
print(lines)