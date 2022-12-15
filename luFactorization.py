import time
import pprint
import scipy
import scipy.linalg   # SciPy Linear Algebra Library
import numpy
import argparse, os
import sys
import subprocess
parser = argparse.ArgumentParser()
parser.add_argument(
       '-d',
        action="store_true",
        default=False,
        help='Use custom array'
)
parser.add_argument(
    "-n",
    type=int,
    default=64,
    help="Matrix Size"
)

args = parser.parse_args()

n = args.n

if args.d:
    A = args.n*numpy.random.rand(args.n, args.n)
else:
    A = scipy.array([ [8, 4, -2, 3], [4, 9, 2, -5], [-2, 2, 5, -2], [3, -5, -2, 7] ])
start = time.time()
P, L, U = scipy.linalg.lu(A)
end = time.time()
fpA = open("inputMatrix_n" + str(args.n) + ".bin", "wb")
fpL = open("LMatrix_n" + str(args.n) + ".bin", "wb")
fpU = open("UMatrix_n" + str(args.n) + ".bin", "wb")

#print ("A:")
#pprint.pprint(A)
#
#print ("P:")
#pprint.pprint(P)
#
#print ("L:")
#pprint.pprint(L)
#
#print ("U:")
#pprint.pprint(U)

print ("Python Time:", (end - start)*1000)

numpy.savetxt(fpA, A)
numpy.savetxt(fpL, L)
numpy.savetxt(fpU, U)
