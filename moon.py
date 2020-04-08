#Deep Q-Learning Lunar Lander

#Import necessary Libraries
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time
from graphics import *
from lander import Lander

from landing_seq import Landing_Sequence

wn = GraphWin("Lunar Lander", 1250, 600)
wn.setCoords(0,0,1250,600)

last_reward = 0
scores = []

# AI that is landing the lunar lander. 4 inputs, 3 actions, gamma (q-learning parameter) 0.9
pilot = Landing_Sequence(5, 3, 0.9)

# Initialize the moon lander outside of the classes as a global variable
apollo1 = Lander()

# Amount of gravitational acceleration applied in each loop of time dt of the update function
gravity = 0.00075

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
        # number, the easier it will be to land. Lowest is 20. 
        self.drawTerrain()
        #-------------------------------------------------------------------------------------
        # Draw amount of fuel the lander has to the upper left of the screen
        self.amount_of_fuel = Text(Point(45,575), "Fuel: " + str(apollo1.fuel))
        self.amount_of_fuel.setFill("white")
        self.amount_of_fuel.draw(wn)
        
        self.x_speed = Text(Point(100, 550), "")
        self.x_speed.setFill("white")
        self.x_speed.draw(wn)
        self.y_speed = Text(Point(100, 525), "")
        self.y_speed.setFill("white")
        self.y_speed.draw(wn)
        #-------------------------------------------------------------------------------------
        self.game_state = True
        self.playGame()
        
    def playGame(self):
        global pilot
        pilot.load()
        
        while(self.game_state): # Main game loop
            self.update()
        print("complete")        
    
    def drawTerrain(self):
        # Randomly generated heights of the endpoints of the segments
        for i in range(76):
            self.terrain_values.append(25*np.random.rand() + 35)
            
        # On the surface, there needs to be a completely flat region on which the lander needs to
        # be trained to land. It position is randomly generated, and is within certain bounds of
        # the edges of the window, determined by the difficulty of the game
        f = np.random.randint(low = 5, high = 24)
        self.terrain_values[f] = 35
        self.terrain_values[f+1] = 35
        
        # For each index of 'terrain_values' there exists a Line object in 'terrain'
        for i in range(1500):
            if i % self.difficulty == 0:
                x = Line(Point(i,self.terrain_values[int(i/self.difficulty)]),
                         Point(i+self.difficulty,self.terrain_values[int((i/self.difficulty)+1)]))
                self.terrain.append(x)                
                x.setFill("white")
                x.setWidth(5)
                x.draw(wn)
    
    def isCrashed(self, elevation, slope):
        lander_speed = np.sqrt(apollo1.dx**2 + apollo1.dy**2)
        tilt_angle = np.degrees(apollo1.getAngle())
        
        if apollo1.getCenter().getX() < 10 or apollo1.getCenter().getX() > 1240: return True
        if apollo1.getCenter().getY() >= 590: return True
        if elevation < 1 and slope > 0.05: return True
        
        if((lander_speed > 0.05 or np.abs(tilt_angle) > 2) and elevation < 1):
            print("Crashed. Lander Speed: ", str(lander_speed))
            return True
        
        return False
    
    def isLanded(self, elevation, slope):
        lander_speed = np.sqrt(apollo1.dx**2 + apollo1.dy**2)
        tilt_angle = np.degrees(apollo1.getAngle())
        if((lander_speed <= 0.05 and np.abs(tilt_angle) <= 2) and elevation < 1 and slope < 0.05):
            print("Landed")
            return True

        return False
    
    def reset(self):
        if(apollo1.fuel != 0):
            apollo1.lander.undraw()
            apollo1.lander.points = apollo1.reset_points
            apollo1.rotate(30)
            apollo1.dy = 0
            apollo1.dx = 0.75
            apollo1.lander.draw(wn)
        else:
            print("Out of Fuel!")
            self.game_state = False
    
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
        # Calculate the height of the lowest point of the lander from the ground, as well as the 
        # height of the surface
        #-------------------------------------------------------------------------------------
        points = apollo1.lander.getPoints()
        p4 = points[len(points)-1]
        p3 = points[len(points)-2]
        p0 = Point((p3.getX() + p4.getX())/2, (p3.getY() + p3.getY())/2 )
        
        xCoord_p3 = p3.getX()
        val_p3 = int(xCoord_p3/self.difficulty)
        terrain_segment_x1_p3 = self.terrain[val_p3].getP1().getX()
        terrain_segment_x2_p3 = self.terrain[val_p3 + 1].getP1().getX()
        terrain_segment_y1_p3 = self.terrain[val_p3].getP1().getY()
        terrain_segment_y2_p3 = self.terrain[val_p3 + 1].getP1().getY()
        slope_p3 = (terrain_segment_y2_p3 - terrain_segment_y1_p3)/(terrain_segment_x2_p3 - terrain_segment_x1_p3)
        y_intercept_p3 = terrain_segment_y1_p3 - slope_p3*terrain_segment_x1_p3
        ground_height_p3 = xCoord_p3 * slope_p3 + y_intercept_p3
        elevation_p3 = p3.getY() - ground_height_p3
        
        xCoord_p4 = p4.getX()
        val_p4 = int(xCoord_p4/self.difficulty)
        terrain_segment_x1_p4 = self.terrain[val_p4].getP1().getX()
        terrain_segment_x2_p4 = self.terrain[val_p4 + 1].getP1().getX()
        terrain_segment_y1_p4 = self.terrain[val_p4].getP1().getY()
        terrain_segment_y2_p4 = self.terrain[val_p4 + 1].getP1().getY()
        slope_p4 = (terrain_segment_y2_p4 - terrain_segment_y1_p4)/(terrain_segment_x2_p4 - terrain_segment_x1_p4)
        y_intercept_p4 = terrain_segment_y1_p4 - slope_p4*terrain_segment_x1_p4
        ground_height_p4 = xCoord_p4 * slope_p4 + y_intercept_p4
        elevation_p4 = p4.getY() - ground_height_p4
        
        xCoord_p0 = p0.getX()
        val_p0 = int(xCoord_p0/self.difficulty)
        terrain_segment_x1_p0 = self.terrain[val_p0].getP1().getX()
        terrain_segment_x2_p0 = self.terrain[val_p0 + 1].getP1().getX()
        terrain_segment_y1_p0 = self.terrain[val_p0].getP1().getY()
        terrain_segment_y2_p0 = self.terrain[val_p0 + 1].getP1().getY()
        slope_p0 = (terrain_segment_y2_p0 - terrain_segment_y1_p0)/(terrain_segment_x2_p0 - terrain_segment_x1_p0)
        y_intercept_p0 = terrain_segment_y1_p0 - slope_p0*terrain_segment_x1_p0
        ground_height_p0 = xCoord_p0 * slope_p0 + y_intercept_p0
        elevation_p0 = p0.getY() - ground_height_p0

        return [ground_height_p3, ground_height_p4, 
                np.minimum( np.minimum(elevation_p3, elevation_p4), 
                           np.minimum(elevation_p3, elevation_p0) )]
            
    def update(self):
        global action
        global last_reward
        global scores
        
        elevation = self.computeElevation()[2]
        lander_angle = np.abs(np.degrees(apollo1.getAngle()))
        lander_speed = np.sqrt(apollo1.dx**2 + apollo1.dy**2)
        slope = self.computeSlope()
        isCrash = self.isCrashed(elevation, slope)
        isLand = self.isLanded(elevation, slope)
        
        
        signal = [elevation, slope, lander_angle, apollo1.dx, apollo1.dy]
        action = pilot.update(last_reward, signal)
        scores.append(pilot.score())
        
        if action == 0:
            for i in range(2):
                apollo1.fireThruster()
        if action == 1:
            for i in range(5):
                apollo1.lander.undraw()
                apollo1.rotate(0.5)
                apollo1.lander.draw(wn)
        if action == 2:
            for i in range(5):
                apollo1.lander.undraw()
                apollo1.rotate(-0.5)
                apollo1.lander.draw(wn)
        
        if(isCrash == False and isLand == False):
            last_reward = -0.2
            if slope <= 0.05: last_reward = 0.1
            if lander_angle <= 2: last_reward = 0.1
            if lander_speed <= 0.15: last_reward = 0.1
            if lander_speed <= 0.15 and lander_angle <= 2: last_reward = 0.2
            
            # Optimal amount of delay for smooth animation speed
            time.sleep(0.035)
            if apollo1.getCenter().y > 10:
                apollo1.dy -= gravity
            else: 
                apollo1.dy = 0
                apollo1.dx = 0
            apollo1.lander.move(apollo1.dx,apollo1.dy)
            
            # Update the amount of fuel the lander has
            self.amount_of_fuel.setText("Fuel: " + str(apollo1.fuel))
            self.x_speed.setText("Lander X-Speed: " + str(round(apollo1.dx, 4)))
            self.y_speed.setText("Lander Y-Speed: " + str(round(apollo1.dy, 4)))
            
        if isCrash:
            print("Crashed!")
            time.sleep(1)
            last_reward = -1 # Bad reward for crashing
            self.reset()
        if isLand:
            print("Landed!")
            time.sleep(1)
            last_reward = 2 # Good reward for landing
            self.reset()
       

game = LanderGame()
wn.getMouse()
wn.close()

if wn.isClosed():
    print("Closed. Saving...")
    pilot.save()
    plt.plot(scores)
    plt.show()    
