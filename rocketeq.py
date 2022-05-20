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

# System of differential equations for gravity turn:
# parameters: I_sp = specific impulse, m0 = initial mass, m_dry = final mass, burn_time = burn time, G = gravitional constant, rho_0 = initial atmospheric density, d = drag coefficient, A = cross-sectional area, H = scale height, R = planet radius, M = planet mass
# variables: v = velocity, beta = angle between vertical and velocity vector, m = mass, h = height, rho = atmospheric density, g = gravitational acceleration
# --------------------------------
# dv/dt = ([(g * I_sp) * (dm/dt)) - ((1/2) * rho * v^2 * d * A))]/ m) - g*cos(beta)
# dbeta/dt = (g/v)*sin(beta)
# dm/dt = (m_dry - m0) / burn_time
# dh/dt = v * cos(beta)
# drho/dt = rho_0 * exp(-h/H)
# dg/dt = GM/((R + h)^3)dh/dt

# v[0] - speed ; v[1] - angle between vertical axis and velocity vector v[2] = mass ; v[3] = height ; v[4] = atmospheric
#  (rho) ; v[5] = gravitational acceleration
def gravity_turn(t, v, I_sp, m0, m_dry, burn_time, G, rho_0, d, A, H, R, M):
    return [(((v[5] * I_sp * -((m_dry - m0) / burn_time)) - ((1/2) * v[4] * v[0]**2 * d * A))/ v[2]) - (v[5] * np.cos(v[1])), (v[5]/v[0]) * np.sin(v[1]), (m_dry - m0) / burn_time, v[0] * np.cos(v[1]), rho_0 * np.exp(-(v[3]/H)), -(G*M / ((R + v[3])**3)) * (v[0] * np.cos(v[1]))]


start_time = 15
start_height = 787
end_time = 70

G = 6.67e-11

planet_mass = 5.29e22
planet_radius = 600000

I_sp = 218 # seconds
m0 = 67600
m_dry = 36800
burn_time = 70 - start_time
gravity = 9.8
rho_0 = 1
d = 0.3
A = 1
H = 70000

solve_gravity_turn = solve_ivp(gravity_turn, (start_time, end_time), [96.8,(17 * np.pi / 180), m0, start_height, rho_0, gravity], args=(I_sp, m0, m_dry, burn_time, G, rho_0, d, A, H, planet_radius, planet_mass), dense_output=True)

f2 = plt.figure(2)
plt.plot(solve_gravity_turn.t, solve_gravity_turn.y[0], label=f'$v(t)$')
plt.xlabel('$t$')
plt.legend()

f3 = plt.figure(3)
plt.plot(solve_gravity_turn.t, solve_gravity_turn.y[1], label=f'$\\beta(t)$')
plt.xlabel('$t$')
plt.legend()



#f4 = plt.figure(4)
#x = np.linspace(start_time,end_time,end_time)
#y = np.array(list(map(lambda t : start_height + quad(lambda v : (solve_gravity_turn.sol(v)[0] * np.cos(solve_gravity_turn.sol(v)[1])), start_time, t)[0], x)))

f4 = plt.figure(4)
plt.plot(solve_gravity_turn.t, solve_gravity_turn.y[3], label=f'$h(t)$')
plt.xlabel('$t$')
plt.legend()


#print(x.shape)
#print(y.shape)
#print(y[0])

#plt.scatter(x,y, label=f'$h(t)$')
#plt.xlabel("$t")
#plt.legend()


f5 = plt.figure(5)
plt.plot(solve_gravity_turn.t, solve_gravity_turn.y[2], label=f'$m(t)$')
plt.xlabel('$t$')
plt.legend()

f6 = plt.figure(6)
plt.plot(solve_gravity_turn.t, solve_gravity_turn.y[5], label=f'$g(t)$')
plt.xlabel('$t$')
plt.legend()


plt.show()

## TO TURN TRAJECTORY INTO ORBIT
## Consider trajectory in polar form, then find t s.t dr/dt = 0. This will be closest or furthest point in trajectory from planet. 
## Then use calculated t  