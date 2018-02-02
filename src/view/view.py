import pygame 
import os

WHITE = (255, 255, 255)

class View():


	def __init__(self, sizeX, sizeY, model):
		self.model = model
		model.register_listener(self.model_event)
        self.screen = pygame.display.set_mode((640,480))
        pygame.display.set_caption('A.I.gar')
        
    def draw(self):
    	self.screen.fill(WHITE)
    	for blob in model.getBlobs():
    	    pygame.draw.circle(self.screen, blob.getColor(), blob.getPos(), blobl.getRadius(), width = 0 )


    def model_event(self, event_name):
        self.draw()
	 