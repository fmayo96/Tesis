def K1(f, H, x, y, dt, V):
    return dt * f(H, x, y, V)
def K2(f, H, x, y, dt, V):
    return dt * (f(H, x + 0.5 * K1(f, H, x, y, dt, V), y, V))
def K3(f, H, x, y, dt, V):
    return dt * (f(H, x + 0.5 * K2(f, H, x, y, dt, V), y, V))
def K4(f, H, x, y, dt, V):
    return dt * (f(H, x + K3(f, H, x, y, dt, V), y, V))

def RK4(f, H, x, y, dt, V):
    return x + (1.0/6) * (K1(f, H, x, y, dt, V)+2*K2(f, H, x, y, dt, V)+2*K2(f, H, x, y, dt, V)+K4(f, H, x, y, dt, V))

