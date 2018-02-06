from view.view import *


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
                quit()
            if( event.type == pygame.KEYDOWN ):
                if( event.key == pygame.K_SPACE  and self.human.getCanSplit()):
                    human.setSplit(True)
                elif( event.key == pygame.K_w and self.human.getCanEject()):
                    human.setEject(True)

            #elif event.type == pygame.MOUSEBUTTONDOWN:
                #if pygame.mouse.get_pressed()[0]:
                #    pass
            elif event.type == pygame.MOUSEMOTION:
                pass