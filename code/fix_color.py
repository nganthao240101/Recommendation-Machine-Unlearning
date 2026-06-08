files = ['make_table.py', 'viz_4partitions_table.py', 'viz_before_after.py', 'viz_unlearn_3types.py']
for f in files:
 with open(f, 'rb') as fh:
 content = fh.read().decode('utf-8')
 count = content.count("'#444'")
 content = content.replace("'#444'", "'#444444'")
 with open(f, 'wb') as fh:
 fh.write(content.encode('utf-8'))
 print('Fixed', f, count, 'occurrences')