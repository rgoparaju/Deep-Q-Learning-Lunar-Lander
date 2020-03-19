from graphics import *
import numpy as np
from random import random, randint

class Lander():
    def __init__(self):
        self.fuel = 1000
        # Initial amount of fuel for every instance of the lander
        
        
        # Given by graphics.py # Point(120,520)
        self.lander = Polygon(Point(105,512.5), Point(115,512.5), Point(120,500), Point(100,500))
        self.reset_points = [Point(105,512.5), Point(115,512.5), Point(120,500), Point(100,500)]
        
        
        self.lander.setFill("white")
        
        self.rotate(30) # Initial rotation angle of the lander, given in degrees
        
        # Initial x and y velocities of the lander. Positive values mean +v_x and +v_y
        self.dy = 0
        self.dx = 0.75
        
        # The angle of the lander, not initialized here since it is only a self-referential variable
        # that is assigned to later.
        self.angle = 0
        
    def fireThruster(self):
        if (self.fuel > 0): 
            self.fuel -= 1
            angle = np.degrees(self.getAngle())
#            print(angle)
            
            # The thruster is being fired perpendicularly from the bottom of the lander,
            # thus it is necessary to decompose the change-in-velocity vector into x and y components        
            self.dy += 0.005*np.cos(angle)
            self.dx += 0.005*np.sin(angle)#*np.sign(angle)
            
#            if angle > 0:
#                self.dx -= 0.005*np.sin(angle)
#            elif angle < 0:
#                self.dx += 0.005*np.sin(angle)
        
    def rotate(self, theta):
        center = self.getCenter()
        angle = np.radians(theta)
#        print(np.degrees(self.getAngle()))
        R = [[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]
        # This is the well-known 2-D rotation matrix that rotates a vector counterclockwise by some
        # angle theta. When you take the dot product of the matrix and a vector, the result is
        # another vector that is the rotated image of the original vector by angle theta. In order
        # to rotate clockwise, a negative angle must be provided, but this is automatically
        # satisfied by the constraints of the program.
        # Since the origin of the display window's coordinate system is the top-left corner, the 
        # coordinates must be corrected for that, so that the points rotate about the center of the
        # lander rather than the true origin of the window. This is accomplished simply subtracting
        # the x- and y-coordinates of the center from those of each point that is rotating.
        
        # If the lander is completely vertical, the angle calculated is always +90 degrees, no 
        # matter if the base of the lander is facing right or left, so the lander continues to 
        # rotate in that direction if given a continuous input. This is simply a result of how the 
        # the angle is calculated. To fix this, I made the maximum rotation angle 89.999 degrees in 
        # both directions, which is close enough to 90 degrees that for all purposes it is the same.
        old_points = self.lander.getPoints()
        if np.degrees(self.getAngle()) < 89.999 and np.degrees(self.getAngle()) > -89.999:
            new_points = []
            new_vectors = []
            for p in old_points:
                v = [p.getX() - center.getX(),p.getY() - center.getY()]
                new_vectors.append(np.dot(R,v))
                
            for v in new_vectors:
                new_points.append(Point(v[0] + center.getX(), v[1] + center.getY()))
                
            self.lander.points = new_points
        
        else:
            self.lander.points = old_points
               
    def getAngle(self):
        # Method to determine the angle of the lander with respect to the x-axis. Used to determine
        # if the lander can rotate past the allowed values. Returns value in radians
        points = self.lander.getPoints()
        p3 = points[len(points)-2]
        p4 = points[len(points)-1]
        
        # If we let (p_3, p_4) be a vector that describes the bottom surface of the lander, we 
        # can easily compute the angle of the vector with respect to the horizontal axis passing
        # through p_3
        if p3.getX() - p4.getX() != 0:
            self.angle = np.arctan((p4.getY() - p3.getY()) / (p4.getX() - p3.getX()))
        elif p4.getY() > p3.getY(): 
            self.angle = -90
        elif p3.getY() > p4.getY():
            self.angle = 90
        return self.angle
        
    def getCenter(self):
#        print(self.lander.getPoints())
        
        # The center of the lander, i.e. the center of the shape the lander takes, is given by the
        # the average values of the x and y distances. This code is modified from the same function
        # in graphics.py that calculates the center of a rectangle or oval, but here instead of
        # just two points, there are 3 or 4. This is because in Zelle's code, the Polygon class has
        # no method for calculating the center, only in the Rectangle or Oval classes
        arrx = []
        for x in self.lander.getPoints():
            arrx.append(x.getX())
        avg_x = np.average(arrx)
        
        arry = []
        for y in self.lander.getPoints():
            arry.append(y.getY())
        avg_y = np.average(arry)
        
        return Point(avg_x,avg_y)
        