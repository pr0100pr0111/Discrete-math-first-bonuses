import re
from itertools import product

import numpy as np

BATCH_SIZE = 1 << 19

def union(a,b): return a | b
def intersection(a,b): return a & b
def difference(a,b): return a & ~b
def symmetric_difference(a,b): return a ^ b
def complement(a,n): return a ^ ((1<<n)-1)
def subset(a,b): return a & ~b == 0
def cardinality(a): return a.bit_count()
def powerset(a):
    out=[]; s=a
    while True:
        out.append(s)
        if s==0: return out
        s=(s-1)&a

def parse(s):
    s=s.translate(str.maketrans({'∪':'|','∩':'&','∖':'-','△':'^','¬':'~'}))
    tokens=re.findall(r'[A-Za-z][A-Za-z0-9_]*|[^\s]',s); i=0
    prec={'|':1,'^':2,'&':3,'-':3}
    def expr(level=0):
        nonlocal i
        if i==len(tokens): raise ValueError('Ожидалось выражение')
        t=tokens[i]; i+=1
        if t=='~': node=('~',expr(4))
        elif t=='(':
            node=expr()
            if i==len(tokens) or tokens[i]!=')': raise ValueError('Ожидалась закрывающая скобка')
            i+=1
        elif re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*',t): node=t
        else: raise ValueError(f'Неожиданный токен: {t}')
        while i<len(tokens) and tokens[i] in prec and prec[tokens[i]]>=level:
            op=tokens[i]; i+=1; node=(op,node,expr(prec[op]+1))
        return node
    node=expr()
    if i!=len(tokens): raise ValueError(f'Лишний токен: {tokens[i]}')
    return node

def variables(t):
    if isinstance(t,str): return {t}
    return set().union(*(variables(x) for x in t[1:]))

def compile_tree(t):
    if isinstance(t,str): return lambda env,mask: env[t]
    op=t[0]; f=compile_tree(t[1])
    if op=='~': return lambda env,mask: mask ^ f(env,mask)
    g=compile_tree(t[2]); fn={'|':union,'&':intersection,'-':difference,'^':symmetric_difference}[op]
    return lambda env,mask: fn(f(env,mask),g(env,mask))

def check_batches(f, g, names, n, mask):
    total = 1 << (n * len(names))
    shifts = range(n * (len(names) - 1), -1, -n)
    dtype = np.min_scalar_type(mask)
    for start in range(0, total, BATCH_SIZE):
        indices = np.arange(start, min(start + BATCH_SIZE, total), dtype=np.uint64)
        env = {
            name: ((indices >> shift) & mask).astype(dtype)
            for name, shift in zip(names, shifts)
        }
        mismatches = np.flatnonzero(f(env, mask) != g(env, mask))
        if mismatches.size:
            first = int(mismatches[0])
            values = {name: int(column[first]) for name, column in env.items()}
            return False, start + first + 1, values
    return True, total, None

def check(left,right,n=1,fast=True):
    if n<0: raise ValueError('n must be nonnegative')
    a,b=parse(left),parse(right); names=sorted(variables(a)|variables(b)); f,g=map(compile_tree,(a,b))
    m=1 if fast and n else n; mask=(1<<m)-1
    index_bits = n * len(names)
    if not fast and 0 < index_bits <= 63:
        ok, count, values = check_batches(f, g, names, n, mask)
        return ok, count, None if values is None else decode_witness(values)
    for count,values in enumerate(product(range(1<<m),repeat=len(names)),1):
        env=dict(zip(names,values))
        if f(env,mask)!=g(env,mask):
            return False,count,decode_witness(env)
    return True,count,None

def decode_witness(env):
    return {
        name: {i + 1 for i in range(value.bit_length()) if (value >> i) & 1}
        for name, value in env.items()
    }

CASES=[('~(A|B)','~A&~B',True),('~(A&B)','~A|~B',True),('A&(B|C)','(A&B)|(A&C)',True),('~~A','A',True),('A-B','A',False),('A^B','B^A',True),('(A-B)-C','A-(B-C)',False)]
if __name__=='__main__':
    for a,b,expected in CASES:
        result=check(a,b,3)
        print(a,'=',b,'prediction:',expected,'result:',result)
        assert result[0]==expected