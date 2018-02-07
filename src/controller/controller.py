from view.view import *
from model.model import *
from numpy import *


class Controller:
    """
    Initializing the 'root' main container, the model, the view,
    """

    def __init__(self, model, view):

        self.model = model
        self.view = view
        self.running = True

    def process_input(self):
        human = self.model.getHuman()
        for event in pygame.event.get():
            # Event types
            if event.type == pygame.QUIT:
                self.running = False
            if( event.type == pygame.KEYDOWN ):
                if( event.key == pygame.K_SPACE  and self.human.getCanSplit()):
                    human.setSplit(True)
                elif( event.key == pygame.K_w and self.human.getCanEject()):
                    human.setEject(True)

        if( self.model.hasHuman() ):
            self.mousePosition()

    # Find the point where the player moved, taking into account that he only sees the fov
    def mousePosition(self):
        mousePos = [self.view.width,0]#pygame.mouse.get_pos()
        fovPos = self.model.human.getFovPos()
        fovDims = self.model.human.getFovDims()
        difference = numpy.subtract(mousePos, [fovDims[0] / 2,fovDims[1] / 2])
        relativeMousePos = numpy.add(difference, [fovPos[0], fovPos[1]])
        print("mousePos", mousePos)
        print("fovPos", fovPos)
        print("fovDims", fovDims)  
        print("diff", difference)
        print("relMouPos", relativeMousePos, "\n")
        self.model.human.setMoveTowards(relativeMousePos)

