import re
with open('visualize_final.py', 'r') as f:
 content = f.read()
lines = content.split('\n')
result = []
for line in lines:
 prefix = ' ' if line.startswith(' ') else ''
 result.append(prefix + line)
with open('visualize_final.py', 'w') as f:
 f.write('\n'.join(result))
import ast
ast.parse(open('visualize_final.py').read())
print('OK')