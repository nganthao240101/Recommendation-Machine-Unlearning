with open('visualize_final.py', 'r') as f:
 content = f.read()

lines = content.split('\n')
new_lines = []
for line in lines:
 stripped = line.lstrip(' ')
 spaces = len(line) - len(stripped)
 new_lines.append(' ' * (spaces * 4) + stripped)

with open('visualize_final.py', 'w') as f:
 f.write('\n'.join(new_lines))

import ast
ast.parse(open('visualize_final.py').read())
print('OK')