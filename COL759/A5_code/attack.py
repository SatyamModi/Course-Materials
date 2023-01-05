from rsa import *
from math import ceil, floor
import sys

MAX_INT = sys.maxsize
"""
Takes a ciphertext, public modulus and public exponent as input as input
PARAMS:
ciphertext: a list of integers of size 128 bytes
N: the public modulus of size 128 bytes
e: the public exponent
"""
def ceil(a,b):
    return (a//b + (a%b>0))

def floor(a, b):
    return a//b

def merge_intervals(M, a, b):

    for i in range(len(M)):
        lo = M[i][0]
        hi = M[i][1]

        if lo <= b and a <= hi:
            a = min(a, lo)
            b = max(b, hi)
            M[i] = (a, b)
            return

    M.append((a, b))
    return

def blinding(N, e, ct):

    s = 1
    c = ct 
    flag = 0
    while not check_padding(c):
        s = randrange(2, N)
        c1 = int.from_bytes(c, "big")
        val = (c1 * pow(s, e, N)) % N 
        c = list(val.to_bytes(96, "big"))

    return (s, c)

def search_a(N, e, c, B):

    s = ceil(N , 3*B)

    x = int.from_bytes(c, "big")
    val = (x * pow(s, e, N)) % N

    y = list(val.to_bytes(96, "big"))
    while not check_padding(y):
        s += 1
        val = (x * pow(s, e, N)) % N
        y = list(val.to_bytes(96, "big"))
        # if s % 1e3 == 0:
        #     print("s: ", s)
    return s 

def search_b(N, e, c, s):

    s += 1
    x = int.from_bytes(c, "big")
    val = (x * pow(s, e, N)) % N 
    y = list(val.to_bytes(96, "big"))

    while not check_padding(y):
        s += 1
        val = (x * pow(s, e, N)) % N
        y = list(val.to_bytes(96, "big"))

    return s 

def search_c(N, e, c, s, a, b, B):

    r = ceil(2 * (b*s - 2*B), N)
    x = int.from_bytes(c, "big")

    while True:
        lo = ceil(2*B + r*N, b)
        hi = floor(3*B + r*N, a)
        for s in range(lo, hi + 1):
            val = (x * pow(s, e, N)) % N  
            y = list(val.to_bytes(96, "big"))      
            if check_padding(y):
                return s 

        r += 1

def narrowing(N, s, B, M):
    intervals = []
    for i in range(len(M)):
        a, b = M[i]
        lo = ceil(a*s - 3*B+1, N)
        hi = floor(b*s-2*B, N)
        for r in range(lo, hi+1):
            a1 = max(a, ceil(2*B + r*N, s))
            b1 = min(b, floor(3*B-1+r*N, s))
            merge_intervals(intervals, a1, b1)

    return intervals 

def attack(cipher_text, N, e):
    """
    TODO: Implement your code here
    """
    k = 96
    B = pow(2, 8*(k-2))
    M = [(2*B, 3*B-1)]
    # s_, c_ = blinding(N, e, cipher_text)
    s_ = ceil(N , 3*B)
    c_ = cipher_text

    s = search_a(N, e, c_, B)

    M = narrowing(N, s, B, M)

    while True:

        if len(M) == 1:
            a, b = M[0]
            if a == b:
                msg = a % N
                break
            else:
                s = search_c(N, e, c_, s, a, b, B)
        else:
            s = search_b(N, e, c_, s)
        M = narrowing(N, s, B, M)

    """
    Return a list of integers representing the original message
    """
    res = list(msg.to_bytes(96, "big"))
    idx = 0
    for i in range(len(res)):
        if res[i] == 0:
            idx = i+1
        else:
            continue

    return res[idx:]

msg = 'Hi, I am Satyam'
msg = list(bytes(msg, 'raw_unicode_escape'))
msg = pad(msg)

ct = encryption(msg)

e = pub_key[0]
N = pub_key[1]

list_of_bytes = attack(ct, N, e)
dec_msg = ''.join(map(chr, list_of_bytes))
print(len(list_of_bytes))
print(dec_msg)