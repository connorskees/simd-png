def wrap(a):
    return [x%256 for x in a]

def add(a, b):
    return wrap([x+y for x, y in zip(a, b)])

def bit_and(a, b):
    return [x&y for x, y in zip(a, b)]

def div(a, n):
    return [x//n for x in a]

half = lambda a: div(a, 2)

def shift(a, n):
    c = [0]*n+a[:len(a)-n]
    return c

def pf_sum(a):
    c = [*a]
    for i in range(1, len(c)):
            c[i] += c[i-1]
            c[i] %= 256
    return c

def half_pf_sum(a):
    c = [*a]
    for i in range(1, len(c)):
            c[i] += c[i-1]//2
            c[i] %= 256
    return c

ones = [1 for _ in range(5)]

def test1(up, row):
    c = add(row, add(div(up, 2), shift(div(row, 2), 1)))
    d = add(c, shift(div(c, 2), 2))
    return d

supplement = lambda x: add(x, bit_and(half(up), bit_and(shift(row, 1), ones)))


import random

def rand_u8():
    return random.randint(0, 255)

def rand_arr(n = 5):
    return [rand_u8() for _ in range(n)]




# def half_prefix_sum(a):
#     s = 0
#     result = []
#     for x in a:
#         s //= 2
#         s += x
#         result.append(s)
#     return result

