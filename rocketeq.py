from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# rocket equation for velocity
def rhs(t, v, m0, l, R, g, k):
    return [((R - (m0 - l*t)*g - k*v[0]**2 + l*v[0])*(1/(m0 - l*t)))]

solve = solve_ivp(rhs, (0,10), [0], args=(2480, 131.25, 192000, 9.8, 0.3), dense_output=True)

print(solve.sol(0))

f1 = plt.figure(1)
plt.plot(solve.t, solve.y[0], label=f'$v(t)$')
plt.xlabel('$t$')
plt.legend()


distance = quad(lambda t : solve.sol(t), 0, 9)

print(distance)

# twr - thrust to weight ratio ; g - gravitational acceleration ; v[0] - speed ; v[1] - angle between vertical axis and velocity vector
def gravity_turn(t, v, twr, g):
    return [g*twr - (g * np.cos(v[1])), (g/v[0]) * np.sin(v[1])]


solve_gravity_turn = solve_ivp(gravity_turn, (0,500), [60,(17 * np.pi / 180)], args=(2.97, 9.8), dense_output=True)

f2 = plt.figure(2)
plt.plot(solve_gravity_turn.t, solve_gravity_turn.y[0], label=f'$v(t)$')
plt.xlabel('$t$')
plt.legend()

f3 = plt.figure(3)
plt.plot(solve_gravity_turn.t, solve_gravity_turn.y[1], label=f'$\\beta(t)$')
plt.xlabel('$t$')
plt.legend()

f4 = plt.figure(4)
x = np.linspace(0,500,500)
y = np.array(list(map(lambda t : quad(lambda v : (solve_gravity_turn.sol(v)[0] * np.cos(solve_gravity_turn.sol(v)[1])), 0, t)[0], x)))

print(x.shape)
print(y.shape)
print(y[0])

plt.scatter(x,y)

plt.show()

## TO TURN TRAJECTORY INTO ORBIT
## Consider trajectory in polar form, then find t s.t dr/dt = 0. This will be closest or furthest point in trajectory from planet. 
## Then use calculated t  