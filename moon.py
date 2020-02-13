#Deep Q-Learning Lunar Lander

#Import necessary Libraries
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time
from graphics import *


wn = GraphWin("Lunar Lander", 1000, 600)
wn.setCoords(0,0,1000,600)
#wn.setBackground("black")



class Lander():
    def __init__(self):
        self.fuel = 1000
        # Initial amount of fuel for every instance of the lander
        
        # Given by graphics.py # Point(120,520)
        self.lander = Polygon(Point(105,512.5), Point(115,512.5), Point(120,500), Point(100,500))
        self.reset_points = [Point(105,512.5), Point(115,512.5), Point(120,500), Point(100,500)]
        
        # Testing and debugging code
#        self.lander = Polygon(Point(100,100), Point((100+(25/2)*np.sqrt(2)),(100+(25/2)*np.sqrt(2))), 
#            Point(100+25*np.sqrt(2),100), Point((100+(25/2)*np.sqrt(2)),(100-(25/2)*np.sqrt(2))))
#        self.lander = Polygon(Point((100+(25/2)*np.sqrt(2)),(100+(25/2)*np.sqrt(2))),
#            Point(100+25*np.sqrt(2),100), Point((100+(25/2)*np.sqrt(2)),(100-(25/2)*np.sqrt(2))), Point(100,100))
#        print(self.getAngle())
#        print(np.arctan(0))
        
        self.lander.setFill("white")
        
        self.rotate(30) # Initial rotation angle of the lander, given in degrees
        
        # Initial x and y velocities of the lander. Positive values mean +v_x and +v_y
        self.dy = 0
        self.dx = 0.5
        
        # The angle of the lander, not initialized here since it is only a self-referential variable
        # that is assigned to later.
        self.angle = 0
        
    def fireThruster(self):
        if (self.fuel > 0): 
            self.fuel -= 1
        
        # I assume the thruster is being fired perpendicularly from the bottom of the lander,
        # thus it is necessary to decompose the change-in-velocity vector into x and y components        
        self.dy += 0.0025*np.cos(self.getAngle())
        
        if self.dx > 0:
            self.dx -= 0.0025*np.sin(self.getAngle())
        elif self.dx < 0:
            self.dx += 0.0025*np.sin(self.getAngle())
#        elif self.dx == 0:
#            if self.getAngle() > 0:
#                self.dx -= 0.004*np.sin(self.getAngle())
#            else:
#                self.dx += 0.004*np.sin(self.getAngle())
        
    def rotate(self,theta):
        center = self.getCenter()
        angle = np.radians(theta)
        print(np.degrees(self.getAngle()))
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
        
        # More debugging code
#        return new_points
        
#        p1 = self.lander.getP1()
#        p2 = self.lander.getP2()
       
    def getAngle(self):
        # Internal method to determine the angle of the lander with respect to the x-axis. Used to 
        # determine if the lander can rotate past the allowed values. Returns value in radians
        points = self.lander.getPoints()
        p3 = points[len(points)-2]
        p4 = points[len(points)-1]
        
        # If we let (p_3, p_4) be a vector that describes the bottom surface of the lander, we 
        # can easily compute the angle of the vector with respect to the horizontal axis passing
        # through p_3
        if p3.getX() - p4.getX() != 0:
            self.angle = np.arctan((p3.getY() - p4.getY()) / (p3.getX() - p4.getX()))
        elif p4.getY() > p3.getY(): 
            self.angle = -90
        elif p3.getY() > p4.getY():
            self.angle = 90
        return self.angle
        
#        center = self.lander.getCenter()
#        xCoord = center.x
#        yCoord = center.y
        
    def getCenter(self):
#        print(self.lander.getPoints())
        
        # The center of the lander, i.e. the center of the shape the lander takes, is given by the
        # the average values of the x and y distances. This code is modified from the same function
        # in graphics.py that calculates the center of a rectangle or oval, but here instead of
        # just two points, there are 3 or 4. This is because in Zelle's code, the Polygon class has
        # no method for retrieving or calculating the center, only in the Rectangle or Oval classes
        arrx = []
        for x in self.lander.getPoints():
            arrx.append(x.getX())
        avg_x = np.average(arrx)
        
        arry = []
        for y in self.lander.getPoints():
            arry.append(y.getY())
        avg_y = np.average(arry)
        
        return Point(avg_x,avg_y)


# Initialize the moon lander outside of the classes as a global variable
apollo1 = Lander()
#from lander import Dqn
#rotationAction = [0,1,-1] #No rotation = index 0, counterclockwise rotation = index 1, clockwise = index 2.
# Has not implemented yet
gravity = 0.005 # This is the amount by which the velocity changes in each loop of the update function

class LanderGame():   
    def __init__(self):
        global apollo1
        apollo1.lander.draw(wn)
        # Need to draw moon surface
        #-------------------------------------------------------------------------------------
        self.terrain_values = []
        # List containing the heights of the endpoints of the segments that create the terrain
        self.terrain = []
        # List containing the physical Line objects that are drawn to the canvas
        self.difficulty = 35
        # This parameter will determine how long each segment of land is, and and the higher the 
        # number, the easier it will be to land
        self.drawTerrain()
        # Internal method to carry out the actual drawing
        #-------------------------------------------------------------------------------------
        # Draw amount of fuel the lander has to the upper left of the screen
        self.amount_of_fuel = Text(Point(45,575), "Fuel: " + str(apollo1.fuel))
        self.amount_of_fuel.draw(wn)
        #-------------------------------------------------------------------------------------
        # Update checker to monitor the win/lose conditions
        while(self.isCrashed(self.computeElevation()[1]) == False
              and self.isLanded() == False and wn.isOpen()):
            self.update()
#        print(self.isCrashed(self.computeElevation()[1]))
        final_lander_speed = np.sqrt(apollo1.dx**2 + apollo1.dy**2)
        print(final_lander_speed)
        
#    def playGame(self):    
#        while(apollo1.isCrashed() == False and apollo1.isLanded() == False):
#            self.update()
    
    def drawTerrain(self):
        # Randomly generated heights of the endpoints of the segments
        for i in range(51):
            self.terrain_values.append(25*np.random.rand() + 35)
            
        # On the surface, there needs to be a completely flat region on which the lander needs to
        # be trained to land. It position is randomly generated, and is within certain bounds of
        # the edges of the window, determined by the difficulty of the game
        f = np.random.randint(low = 5, high = 24)
        self.terrain_values[f] = 35
        self.terrain_values[f+1] = 35
        
        # For each index of 'terrain_values' there exists a Line object in 'terrain'
        for i in range(1000):
            if i % self.difficulty == 0:
                x = Line(Point(i,self.terrain_values[int(i/self.difficulty)]),
                         Point(i+self.difficulty,self.terrain_values[int((i/self.difficulty)+1)]))
                self.terrain.append(x)                
#                x.setFill("white")
                x.setWidth(5)
                x.draw(wn)
#                Line(Point(i,terrain[int(i/20)]),Point(i+20,terrain[int((i/20)+1)])).draw(wn)
         
        # Debugging code to test terrain drawing
#        test1 = Line(Point(0,100),Point(200,0),[None,"white",5])
#        test1.setWidth(5)
#        test1.setFill("white")
#        test1 = Line(Point(0,100),Point(200,0)).draw(wn)
#        print(test1.getP2())
    
    def isCrashed(self, elevation):
        # If the lander makes contact with the ground when it is going too fast, this checker returns True,
        # otherwise it returns False
        lander_speed = np.sqrt(apollo1.dx**2 + apollo1.dy**2)
        tilt_angle = np.degrees(apollo1.getAngle())
        
#        print(lander_speed)
        if((lander_speed >= 1 or np.abs(tilt_angle) >= 5) and elevation <= 0.5):
            print("Crashed")
            return True
        
        return False
    
    def isLanded(self):
        # This method has yet to be implemented
        return False
    
    def reset(self):
        if(apollo1.fuel != 0):
            apollo1.undraw()
            apollo1.points = apollo1.reset_points
            apollo1.dx = 0.75
            apollo1.dy = 0
            apollo1.draw()
        else: print("Out of Fuel!")
    
    def computeSlope(self):
        # Calculate the slope of the terrain immediately underneath the body of the lander        
        xCoord = apollo1.getCenter().getX()
        val = int(xCoord/self.difficulty)
        terrain_segment_x1 = self.terrain[val].getP1().getX()
        terrain_segment_x2 = self.terrain[val + 1].getP1().getX()
        
        terrain_segment_y1 = self.terrain[val].getP1().getY()
        terrain_segment_y2 = self.terrain[val + 1].getP1().getY()
        
        slope = (terrain_segment_y2 - terrain_segment_y1)/(terrain_segment_x2 - terrain_segment_x1)
        return np.abs(slope)
    
    def computeElevation(self):
        # Calculate the height of the lander from the ground, as well as the height of the surface
        #-------------------------------------------------------------------------------------
        points = apollo1.lander.getPoints()
        p4 = points[len(points)-1]
        p3 = points[len(points)-2]
        low_point = None
        
        # Find which of the lander's legs is closer to the ground. if both are equal, then take avg
        if p3.getY() < p4.getY():
            xCoord = p3.getX()
            low_point = p3
        elif p4.getY() < p3.getY():
            xCoord = p4.getX()
            low_point = p4
        elif p3.getY() == p4.getY():
            xCoord = (p3.getX() + p4.getX())/2
            low_point = Point(xCoord,p3.getY())
        
        val = int(xCoord/self.difficulty)
        terrain_segment_x1 = self.terrain[val].getP1().getX()
        terrain_segment_x2 = self.terrain[val + 1].getP1().getX()
        
        terrain_segment_y1 = self.terrain[val].getP1().getY()
        terrain_segment_y2 = self.terrain[val + 1].getP1().getY()
        
        # Need to calculate the equation of the line of the given terrain segment
        slope = (terrain_segment_y2 - terrain_segment_y1)/(terrain_segment_x2 - terrain_segment_x1)
        y_intercept = terrain_segment_y1 - slope*terrain_segment_x1
        
        # y-value (ground_height) of the point directly underneath the lowest leg of the lander,
        # given by plugging in the x-coordinate of the lowest point of the lander into the equation
        # calculated above
        ground_height = xCoord * slope + y_intercept
        elevation = low_point.getY() - ground_height
        return [ground_height,elevation]
    
    def update(self):
        # Optimal amount of delay for smooth animation speed
        time.sleep(0.035)
        
        # Testing code to see if the lander can take every action available to it
        
#        apollo1.fireThruster()
#        apollo1.lander.undraw()
#        apollo1.rotate(-1)
#        apollo1.lander.draw(wn)
        
        # Update the amount of fuel the lander has
        self.amount_of_fuel.setText("Fuel: " + str(apollo1.fuel))
#        print(self.computeElevation())
#        if apollo1.getCenter().y < 300:
#            apollo1.fireThruster()
#            print(apollo1.dy)
        
        if apollo1.getCenter().y > 10:
            apollo1.dy -= gravity
#            apollo1.lander.move(0,apollo1.dy)        
        else: 
            apollo1.dy = 0
            apollo1.dx = 0

#        print(apollo1.dx,apollo1.dy)
        apollo1.lander.move(apollo1.dx,apollo1.dy)
#        print(apollo1.fuel)
    

#def update():
##    while(apollo1.isCrashed() == False and apollo1.isLanded() == False and wn.isOpen()):
#    for i in range(20):
#        apollo1.lander.move(0,-10)
#        time.sleep(0.5)
#        if i == 10: 
#            apollo1.rotate(45)
#            apollo1.lander.undraw()
#            apollo1.lander.draw(wn)

    
#    time.sleep(0.01)
#    if apollo1.getCenter().y > 10:
#        apollo1.dy -= gravity
##            apollo1.lander.move(0,apollo1.dy)        
#    else: 
#        apollo1.dy = 0
##            apollo1.lander.points = apollo1.rotate(45)
##            apollo1.lander.move(0,apollo1.dy)
##            apollo1.rotate(45)
##        if apollo1.getCenter().getY() <= 250: apollo1.rotate(45)
#
#    apollo1.lander.move(0,apollo1.dy)

game = LanderGame()
#update()
# For some reason these two lines are necessary for the program to work
wn.getMouse()
wn.close()
