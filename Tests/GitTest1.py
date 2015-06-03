from ad import adnumber

def f(x):
    return x*x + 1

def g(x):
    return x*x*x + 3

x = adnumber(3.0)
y = g(x)

z = y.d(x)
print(z)
