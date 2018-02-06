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
            if event.type == pygame.QUIT:
                self.running = False
            if( human is not None ):
                if( event.type == pygame.KEYDOWN ):
                    if( event.key == pygame.K_SPACE  and human.getCanSplit()):
                        human.setSplit(True)
                    elif( event.key == pygame.K_w and human.getCanEject()):
                        human.setEject(True)

                #elif event.type == pygame.MOUSEBUTTONDOWN:
                    #if pygame.mouse.get_pressed()[0]:
                    #    pass
                # Find the point where the player clicked, taking into account that he only sees the fov
                elif event.type == pygame.MOUSEMOTION:
                    mousePos = pygame.mouse.get_pos()
                    fovPos = human.getFovPos()
                    fovDims = human.getFovDims()
                    difference = numpy.subtract(mousePos, [fovDims[0] / 2,fovDims[1] / 2])
                    relativeMousePos = numpy.add(difference, [fovPos[0], fovPos[1]])
                    human.setMoveTowards(relativeMousePos)

