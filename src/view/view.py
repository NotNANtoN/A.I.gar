import pygame 
import os

WHITE = (255, 255, 255)

class View():


    def __init__(self, sizeX, sizeY, model):
        self.model = model
        model.register_listener(self.model_event)
        self.screen = pygame.display.set_mode((640,480))
        pygame.display.set_caption('A.I.gar')
        
    def drawCells(self, cells):
        for cell in cells:
            pygame.draw.circle(self.screen, cell.getColor(), cell.getPos(), cell.getRadius(), width = 0 )


    def drawAllCells(self):
        self.drawCells(model.getCollectibles())
        self.drawCells(model.getViruses())
        self.drawCells(model.getPlayerCells())

    def draw(self):
        self.screen.fill(WHITE)
        self.drawAllCells()

        

    def model_event(self, event_name):
        self.draw()
     