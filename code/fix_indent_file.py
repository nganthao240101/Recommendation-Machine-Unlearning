with open('compare_with_retrain.py', 'r') as f:
 content = f.read()

lines = content.split('\n')
result = []
for line in lines:
 if line.startswith(' '):
 n = len(line) - len(line.lstrip(' '))
 result.append(' ' * n + line.lstrip(' '))
 else:
 result.append(line)

with open('compare_with_retrain.py', 'w') as f:
 f.write('\n'.join(result))

import ast
try:
 ast.parse(open('compare_with_retrain.py').read())
 print('Valid!')
except Exception as e:
 print('Error:', e)