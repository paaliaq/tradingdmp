def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

def square(a):
    return multiply(a, a)

if __name__ == '__main__':

    a = 2
    b = 5
    c = add(a, b)
    print(c)
    d = multiply(b, c)
    print(d)
    e = square(d)
    print(e)