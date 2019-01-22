import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

fig1 = plt.figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')

point = np.random.rand(2)
speed = np.random.rand(2)/2
mass = 1

point = np.array([0.8, 0.4])
speed = np.array([0, 0.5])

line_x = [point[0]]
line_y = [point[1]]
line_len = 25

dt = 0.002


plt.xlim(0, 1)
plt.ylim(0, 1)

l, = plt.plot(point[0], point[1], '-')
#plt.plot(p1_x, p1_y, 'o', color='black', markersize=20)

particles = []

class Particle:

    def __init__(self, pos, vel, mass):
        self.r = pos
        self.v = vel
        self.m = mass
        self.line_x = [self.r[0]]
        self.line_y = [self.r[1]]
        self.l, = plt.plot(self.r[0], self.r[1], '-', color="w")
        self.p, = plt.plot(self.r[0], self.r[1], 'o', markersize=3, color="w")

    def update_r(self, vec_field):
        x = clamp(int(self.r[0]*100), 0, 99)
        y = clamp(int(self.r[1]*100), 0, 99)

        f_x = vec_field[x][y][0]
        f_y = vec_field[x][y][1]

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

    def wall_check1(self):
        x = self.r[0]
        y = self.r[1]
        
        if x < 0 or x > 1:
            self.r[0] = np.clip(self.r[0],0,1)
            self.v[0] *= -1
        
        if y < 0 or y > 1:
            self.r[1] = np.clip(self.r[1],0,1)
            self.v[1] *= -1
            

    def wall_check2(self):
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

class Field:
    """
    def field1(self, x, y, t, k):
        a = np.ones((101, 101, 2))
        for i in range(101):
            for j in range(101):
                if i == p1_x*100 and j == p1_y*100:
                    continue
                r = np.array([x*100-i, y*100-j])
                a[i][j] = np.sqrt(r[0]**2+r[1]**2)**(-3) * r - 40 * np.sqrt(r[0]**2+r[1]**2)**(-5) * r
        print(np.sum(a))
        print()
        return a*k

    def field2(self, x, y, t, k):
        a = np.ones((101, 101, 2))
        for i in range(101):
            for j in range(101):
                if i == p1_x*100 and j == p1_y*100:
                    continue
                r = np.array([x*100-i, y*100-j])
                a[i][j] = (1 + 2 * np.sin(10*np.pi*t))*1*(np.sqrt(r[0]**2+r[1]**2)**(-3) * r - 40 * np.sqrt(r[0]**2+r[1]**2)**(-5) * r)
        return a*k
        
    def field3(self, x, y, t, k):
        a = np.ones((101, 101, 2))
        for i in range(101):
            for j in range(101):
                if i == p1_x*100 and j == p1_y*100:
                    continue
                r = np.array([x*100-i, y*100-j])
                adbsR = np.sqrt(r[0]**2+r[1]**2)
                a[i][j] =  adbsR**(-3) * r - 40 * adbsR**(-5) * r
                a[i][j] += np.sin(np.sqrt(r[0] ** 2 + r[1] ** 2) / 10 + 20 * t)**4 / 200 * (r / adbsR)
        return a*k
    
    def field4(self, x, y, t, k):
        a = np.ones((101, 101, 2))
        for i in range(101):
            for j in range(101):
                if i == p1_x*100 and j == p1_y*100:
                    continue
                r = np.array([x*100-i, y*100-j])
                adbsR = np.sqrt(r[0]**2+r[1]**2)
                a[i][j] =  adbsR**(-3) * r
                a[i][j] += np.sin(np.sqrt(r[0] ** 2 + r[1] ** 2) / 10 + 20 * t)**4 / 200 * (r / adbsR)
        return a*k
    """
    def particle_field(self, t, k):
        a = np.ones((101, 101, 2))
        form = lambda r, k: k * r**(-2)
        for i in range(101):
            for j in range(101):
                for p in particles:
                    r = np.array([j/101-p.r[0], (1-p.r[1])-i/101])
                    ab = np.sqrt(r[0]**2+r[1]**2)
                    a[i][j] += form(ab, k)*r/ab
        return a

class Ani:

    vector_field = None
    
    def __init__(self, interval, i):
        self.time = 0
        self.ani = animation.FuncAnimation(fig1, self.class_func, frames=np.arange(0, i), interval=interval)
        self.vector_field = Field()
        self.tp = self.time
        self.field = self.vector_field.particle_field(self.time, -20)
 
    def class_func(self, num):
        
        for p in particles:
            p.wall_check1()
            p.update_r(self.field)
            p.update_line()
        self.field = self.vector_field.particle_field(self.time, -20)
        
        self.time += dt
        if self.tp != "{0:.2f}".format(self.time):
            self.tp = "{0:.2f}".format(self.time)
            print(self.tp)
        
        im = plt.imshow([[np.clip(np.log(np.sqrt(x**2+y**2)/5), 0, 1000) for x, y in f] for f in self.field], interpolation="nearest", extent=(0,1,0,1), cmap="viridis")
        cb = plt.colorbar(im)
        
        plt.title("Particles: "+ str(len(particles)) + "  Time: " + str("{0:.1f}".format(self.time)))
        return


def main():

    if True:
        for i in range(2):
            th = np.random.random()*2*np.pi
            x = random.random()
            y = random.random()
            v = np.random.random()/4
            vx = np.cos(th+np.pi/2+np.random.random()*0.2)*v
            vy = np.sin(th+np.pi/2+np.random.random()*0.2)*v
            p = Particle([x, y], [vx, vy], 1)
            particles.append(p)

    if False:
        for i in range(100):
            x = 0.05 + 0.1*(i%10)
            y = 0.05 + int(i/10)/10
            p = Particle([x, y], [0, 0], 1)
            particles.append(p)

    ani = Ani(17, 400)
    if len(sys.argv) > 1 and sys.argv[1] == 'save':
        ani.ani.save('field.gif', writer="imagemagick")
    else:
        plt.show()


def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)
    
if __name__ == "__main__":
    main()
