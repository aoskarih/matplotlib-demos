import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import time

fig1 = plt.figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='black')

plt.xlim(0, 1)
plt.ylim(0, 1)


p_color = (1.0, 0.9, 0.9)
line_color = (1, 1, 1)

random.seed(934)
resolution = 100
k = -150
line_len = 15
dt = 0.000001
particles = []

# saving
steps = 4000
file_name = 'collision2.mp4'

class Particle:

    def __init__(self, pos, vel, mass):
        self.r = pos
        self.v = vel
        self.m = mass
        self.to_be_deleted = False
        self.ignore = False
        self.line_x = [self.r[0]]
        self.line_y = [self.r[1]]
        self.l, = plt.plot(self.r[0], self.r[1], '-', color=line_color)
        self.p, = plt.plot(self.r[0], self.r[1], 'o', markersize=int(np.log(self.m)), color=p_color)

    def update_r(self):
        f = 0
        for p in particles:
            f += force(self, p)
        dv = f/self.m * dt
        #print(f)
        self.v += dv * dt
        self.r += self.v

    def update_line(self):
        self.line_x.append(self.r[0])
        self.line_y.append(self.r[1])
        if len(self.line_x) > line_len:
            self.line_x = self.line_x[1:]
        if len(self.line_y) > line_len:
            self.line_y = self.line_y[1:]
        self.l.set_data(self.line_x, self.line_y)
        self.p.set_data(self.r[0], self.r[1])

    def collision_check(self):
        for p in particles:
            d = np.array(p.r-self.r)
            l = np.sqrt(np.dot(d,d))
            if l < 0.0005*self.m**(1/3) and l != 0 and not p.ignore:
                pm1 = self.m * self.v
                pm2 = p.m * p.v
                v3 = (pm1+pm2)/(self.m + p.m)
                r3 = (self.m*self.r+p.m*p.r)/((self.m + p.m))
                npar = Particle(r3, v3, self.m + p.m)
                npar.ignore = True
                particles.append(npar)
                self.to_be_deleted = True
                p.to_be_deleted = True

    def wall_check1(self):
        x = self.r[0]
        y = self.r[1]
        
        if x < 0 or x > 1:
            self.r[0] -= self.v[0]
            self.v[0] *= -1
        
        if y < 0 or y > 1:
            self.r[1] -= self.v[1]
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
        if self in particles:
            particles.remove(self)
        self.l.set_data(0, 0)
        self.p.set_data(-1, -1)

class Ani:

    vector_field = None
    
    def __init__(self, interval, i):
        self.p_time = time.clock()
        self.time = 0
        self.step = 0
        self.ani = animation.FuncAnimation(fig1, self.class_func, frames=np.arange(0, i), interval=interval)
        self.tp = self.time
        self.im = plt.imshow(np.log(potential_field()), interpolation="nearest", extent=(0,1,0,1), cmap="viridis")
        self.cb = plt.colorbar(self.im)
        
    def class_func(self, num):
        
        for p in particles:
            if p.to_be_deleted or p.ignore:
                continue
            p.collision_check()
            p.wall_check1()
            p.update_r()
            p.update_line()
        
        for p in particles:
            if p.ignore:
                p.ignore = False
            if p.to_be_deleted:
                p.delete()
        
        u = np.log(potential_field())
        self.im.set_data(u)
                
        self.time += dt
        self.step += 1
        if True:
            os.system('clear')
            perc = self.time/(steps*dt)*100
            bar = int(perc/5)*"#" + int(20-perc/5)*" "
            print("Progress: "+"["+bar+"] "+str("{0:.1f}".format(perc)+"%"))
            
        plt.title("Particles: "+ str(len(particles)) + "  Step: " + str(self.step))
        return

def potential_field():
    u = np.zeros((resolution, resolution))
    for i in range(resolution):
        for j in range(resolution):
            r = np.array([(j+0.5)/resolution, 1-(i+0.5)/resolution])
            for p in particles:
                dr = np.array(r-p.r)
                d = np.sqrt(np.dot(dr,dr))
                u[i][j] += potential(d, p.m)
    return u

def main():

    for i in range(25):
        th = np.random.random()*2*np.pi
        x = random.random()
        y = random.random()
        v = np.random.random()/200
        vx = np.cos(th+np.pi/2+np.random.random()*0.2)*v
        vy = np.sin(th+np.pi/2+np.random.random()*0.2)*v
        p = Particle(np.array([x, y]), np.array([vx, vy]), 2**random.randint(4, 13))
        particles.append(p)

    ani = Ani(17, steps)
    if len(sys.argv) > 1 and sys.argv[1] == 'save':
        ani.ani.save(file_name, writer="ffmpeg")
    else:
        plt.show()

def force(p1, p2):
    d = np.array(p2.r-p1.r)
    l = np.sqrt(np.dot(d,d))
    if l == 0:
        return np.array([0.0,0.0])
    f = k * p1.m * p2.m / l**2 * d/l
    return f

def potential(r, m):
    return -k * m / r

def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)
    
if __name__ == "__main__":
    main()