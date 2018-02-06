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
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if( self.model.hasHuman() ):
                if( event.type == pygame.KEYDOWN ):
                    if( event.key == pygame.K_SPACE  and self.human.getCanSplit()):
                        human.setSplit(True)
                    elif( event.key == pygame.K_w and self.human.getCanEject()):
                        human.setEject(True)

                #elif event.type == pygame.MOUSEBUTTONDOWN:
                    #if pygame.mouse.get_pressed()[0]:
                    #    pass
                # Find the point where the player clicked, taking into account that he only sees the fov
                elif event.type == pygame.MOUSEMOTION:
                    mousePos = pygame.mouse.get_pos()
                    fovPos = self.model.human.getFovPos()
                    fovDims = self.model.human.getFovDims()
                    difference = numpy.subtract(mousePos, [fovDims[0] / 2,fovDims[1] / 2])
                    relativeMousePos = numpy.add(difference, [fovPos[0], fovPos[1]])
                    self.model.human.setMoveTowards(relativeMousePos)

