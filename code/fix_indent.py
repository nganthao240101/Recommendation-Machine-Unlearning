with open('visualize_final.py', 'r') as f:
 lines = f.readlines()

new_lines = []
for line in lines:
 stripped = line.lstrip(' ')
 count = len(line) - len(stripped)
 if count > 0:
 new_lines.append(' ' * (count // 4) + stripped)
 else:
 new_lines.append(line)

with open('visualize_final.py', 'w') as f:
 f.writelines(new_lines)
print('Fixed')