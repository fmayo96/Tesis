def K1(f, x, y, dt, V):
    return dt * f(x, y, V)
def K2(f, x, y, dt, V):
    return dt * (f(x + 0.5 * K1(f, x, y, dt, V), y, V))
def K3(f, x, y, dt, V):
    return dt * (f(x + 0.5 * K2(f, x, y, dt, V), y, V))
def K4(f, x, y, dt, V):
    return dt * (f(x + K3(f, x, y, dt, V), y, V))

def RK4(f, x, y, dt, V):
    return x + (1.0/6) * (K1(f, x, y, dt, V)+2*K2(f, x, y, dt, V)+2*K2(f, x, y, dt, V)+K4(f, x, y, dt, V))

