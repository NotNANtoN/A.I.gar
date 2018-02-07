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
                # "Escape" to Quit
                if( event.key == pygame.K_ESCAPE ):
                    self.running = False
                # "space" to Split
                elif( event.key == pygame.K_SPACE  and self.human.getCanSplit()):
                    human.setSplit(True)
                # "w" to Eject
                elif( event.key == pygame.K_w and self.human.getCanEject()):
                    human.setEject(True)

        if( self.model.hasHuman() ):
            self.mousePosition()

    # Find the point where the player moved, taking into account that he only sees the fov
    def mousePosition(self):
        mousePos = pygame.mouse.get_pos()
        fovPos = numpy.array(self.model.human.getFovPos())
        fovDims = numpy.array(self.model.human.getFovDims()  )
        screenDims = self.view.getScreenDims()
        relativeMousePos = self.viewToModel(mousePos, fovPos, fovDims, screenDims)
        self.model.human.setMoveTowards(relativeMousePos)

    def modelToView(self, pos, fovPos, fovDims, screenDims):
        adjustedPos = pos - fovPos + (fovDims / 2)
        scaledPos = adjustedPos * (screenDims / fovDims)
        return scaledPos

    def viewToModel(self, pos, fovPos, fovDims, screenDims):
        scaledPos = pos / (screenDims / fovDims)
        adjustedPos = scaledPos + fovPos - (fovDims / 2)
        return adjustedPos

