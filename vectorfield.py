import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig1 = plt.figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')

#vector_field = (np.ones((100, 100, 2))
vector_field = None

point = np.random.rand(2)
speed = np.random.rand(2)/2
mass = 1

point = np.array([0.8, 0.4])
speed = np.array([0, 0.5])

line_x = [point[0]]
line_y = [point[1]]
line_len = 200


dt = 0.005

plt.xlim(0, 1)
plt.ylim(0, 1)

l, = plt.plot(point[0], point[1], '-')
plt.plot(0.5, 0.5, 'o', color='r', markersize=20)

particles = []

class Particle:

    def __init__(self, pos, vel, mass):
        self.r = pos
        self.v = vel
        self.m = mass
        self.line_x = [self.r[0]]
        self.line_y = [self.r[1]]
        self.l, = plt.plot(self.r[0], self.r[1], '-')
        self.p, = plt.plot(self.r[0], self.r[1], 'o', markersize=3)

    def update_r(self, vec_field):
            x = clamp(int(self.r[0]*100), 0, 99)
            y = clamp(int(self.r[1]*100), 0, 99)

            f_x = vector_field[x][y][0]
            f_y = vector_field[x][y][1]

            dv_x = f_x/self.m
            dv_y = f_y/self.m
    
            self.v[0] += dv_x*dt
            self.v[1] += dv_y*dt
    
            self.r[0] += self.v[0]*dt
            self.r[1] += self.v[1]*dt

    def update_line(self):
        self.line_x.append(self.r[0])
        self.line_y.append(self.r[1])
        if len(self.line_x) > line_len:
            self.line_x = self.line_x[1:]
        if len(self.line_y) > line_len:
            self.line_y = self.line_y[1:]
        self.l.set_data(self.line_x, self.line_y)
        self.p.set_data(self.r[0], self.r[1])

    def collision_check(self, x, y):
        d = np.sqrt((x-self.r[0])**2 + (y-self.r[1])**2)
        if d < 0.03:
            self.delete()
        elif self.wall_check():
            self.delete()

    def wall_check(self):
        for i in range(2):
            if (self.r[i] > 1):
                self.r[i] = 1
                self.v[i] = -self.v[i]
                return True
            elif (self.r[i] < 0):
                self.r[i] = 0
                self.v[i] = -self.v[i]
                return True
        return False

    def delete(self):
        particles.remove(self)
        self.l.set_data(0, 0)
        self.p.set_data(-1, -1)

class Ani:
    
    def __init__(self, interval, i):
        self.time = 0
        self.ani = animation.FuncAnimation(fig1, self.class_func, i, interval=interval)

        
    def class_func(self, num):
        for p in particles:
            p.wall_check()
            p.update_r(vector_field)
            p.update_line()
            p.collision_check(0.5, 0.5)
        self.time += dt
        plt.title("Particles: "+ str(len(particles)) + "  Time: " + str("{0:.1f}".format(self.time)))
        return

def grav(x, y):
    a = np.ones((101, 101, 2))
    for i in range(101):
        for j in range(101):
            if i == 50 and j == 50:
                continue
            r = np.array([x*100-i, y*100-j])
            a[i][j] = np.sqrt(r[0]**2+r[1]**2)**(-3) * r - 40 * np.sqrt(r[0]**2+r[1]**2)**(-5) * r
    print(np.sum(a))
    print()
    return a*1000

vector_field = grav(0.5, 0.5)

def main():

    time = 0
    
    for i in range(100):
        th = np.random.random()*2*np.pi
        r = 0.1+np.random.random()*0.5
        x = 0.5+np.cos(th)*r
        y = 0.5+np.sin(th)*r
        v = np.random.random()
        vx = np.cos(th+np.pi/2+np.random.random()*0.2)*v
        vy = np.sin(th+np.pi/2+np.random.random()*0.2)*v
        p = Particle([x, y], [vx, vy], 1)
        particles.append(p)

    ani = Ani(1, 100)
    plt.show()

def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)
    
if __name__ == "__main__":
    main()
