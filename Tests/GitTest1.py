from ad import adnumber
#'/home/othe/Desktop/HIIT/Moduleita/pyautodiff-python2-ast'
import sys
sys.path.append('/home/othe/Desktop/HIIT/Moduleita/pyautodiff-python2-ast')
from autodiff import function, gradient

def f(x):
    return x*x + 1

def g(x):
    return x*x*x + 3

x = adnumber(3.0)
y = g(x)

z = y.d(x)
print(z)

def h1(x):
    return x*x + 1

@gradient
def h2(x):
    return x*x + 1

a = 3.0

print(h1(a))
print(h2(a))