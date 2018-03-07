import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

fig1 = plt.figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')

point = [random.random(), random.random()]
speed = [random.random()/5, random.random()/5]

speed = [1/random.randint(1, 10), 1/random.randint(1, 10)]
point = [1/random.randint(1, 10), 1/random.randint(1, 10)]

line_x = [point[0]]
line_y = [point[1]]

b = 0.005
v = 0.01

plt.xlim(0, 1)
plt.ylim(0, 1)

l, = plt.plot(point[0], point[1], '-')


def func(num):

    f = np.cross(np.array([speed[0], speed[1], 0]), np.array([0, 0, b]))

    speed[0] += f[0]
    speed[1] += f[1]

    v_t = np.sqrt(speed[1]*speed[1]+speed[0]*speed[0])

    speed[0] = speed[0]/v_t*v
    speed[1] = speed[1]/v_t*v
    
    point[0] += speed[0]
    point[1] += speed[1]
    
    for i in range(2):
        if (point[i] > 1):
            point[i] = 1
            speed[i] = -speed[i]
        elif (point[i] < 0):
            point[i] = 0
            speed[i] = -speed[i]

    
    line_x.append(point[0])
    line_y.append(point[1])
    
    #l.set_color((random.random(), random.random(), random.random(), 1))

    l.set_data(line_x, line_y)
    return point

def main():
    red = 0
    gre = 0
    blu = 0
    ani = animation.FuncAnimation(fig1, func, 1000, interval=2)
    plt.show()

if __name__ == "__main__":
    main()
