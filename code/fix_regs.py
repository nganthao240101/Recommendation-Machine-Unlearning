content = open('RecEraser_BPR.py').read()
old = 'self.regs = eval(args.regs)\n        self.decay = self.regs[0]'
new = "self.regs = eval(args.regs) if args.regs.startswith('[') else [float(args.regs)]\n        self.decay = self.regs[0]"
if old in content:
 content = content.replace(old, new)
 open('RecEraser_BPR.py', 'w').write(content)
 print('Fixed')
else:
 print('NOT FOUND')