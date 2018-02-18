from view.view import *
import pygame


class Controller:
    """
    Initializing the 'root' main container, the model, the view,
    """

    def __init__(self, model, viewEnabled, view):
        self.model = model
        self.view = view
        self.screenWidth, self.screenHeight = self.view.getScreenDims()
        self.running = True
        self.viewEnabled = viewEnabled

    def process_input(self):
        humanList = self.model.getHumans()
        humanCommandPoint = []
        if self.model.hasHuman():
            for human in humanList:
                humanCommandPoint.append([self.screenWidth/2, self.screenHeight/2])
                if human.getIsAlive():
                    human.setSplit(False)
                    human.setEject(False)
            #Human1 direction control
            humanCommandPoint[0] = pygame.mouse.get_pos()
            #Human2 direction control
            if len(humanList) > 1:
                keys = pygame.key.get_pressed()
                if humanList[1].getIsAlive():
                    if keys[pygame.K_UP]:
                        humanCommandPoint[1][1] -= self.screenHeight/2
                    if keys[pygame.K_DOWN]:
                        humanCommandPoint[1][1] += self.screenHeight/2
                    if keys[pygame.K_LEFT]:
                        humanCommandPoint[1][0] -= self.screenWidth/2
                    if keys[pygame.K_RIGHT]:
                        humanCommandPoint[1][0] += self.screenWidth/2
                    #Human3 direction controls
                if humanList[2].getIsAlive():
                    if keys[pygame.K_w]:
                        humanCommandPoint[2][1] -= self.screenHeight/2
                    if keys[pygame.K_s]:
                        humanCommandPoint[2][1] += self.screenHeight/2
                    if keys[pygame.K_a]:
                        humanCommandPoint[2][0] -= self.screenWidth/2
                    if keys[pygame.K_d]:
                        humanCommandPoint[2][0] += self.screenWidth/2


            for i in range(len(humanList)):
                self.mousePosition(humanList[i], humanCommandPoint[i], i)

        for event in pygame.event.get():
            # Event types
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN:
                # "Escape" to Quit
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                #elif event.key == pygame.K_b:
                #    self.model.createBot()
                #    player = self.model.players[-1]
                #    self.model.getField().initializePlayer(player)
                if humanList:
                    #Human1 controls
                    human1 = humanList[0]
                    if human1.getIsAlive():
                        # "space" to Split
                        if event.key == pygame.K_SPACE and human1.getCanSplit():
                            human1.setSplit(True)
                        # "w" to Eject
                        elif event.key == pygame.K_b and human1.getCanEject():
                            human1.setEject(True)
                        elif event.key == pygame.K_m:
                            human1.addMass(human1.getTotalMass() * 0.2)
                    if len(humanList) > 1:
                        #Human2 controls
                        human2 = humanList[1]
                        if human2.getIsAlive():
                            # "space" to Split
                            if event.key == pygame.K_KP0 and human2.getCanSplit():
                                human2.setSplit(True)
                            # "w" to Eject
                            elif event.key == pygame.K_KP1 and human2.getCanEject():
                                human2.setEject(True)
                            elif event.key == pygame.K_KP2:
                                human2.addMass(human2.getTotalMass() * 0.2)
                            humanList[1] = human2
                    if len(humanList) > 2:
                        #Human3 controls
                        human3 = humanList[2]
                        if human3.getIsAlive():
                            # "space" to Split
                            if event.key == pygame.K_e and human3.getCanSplit():
                                human3.setSplit(True)
                            # "w" to Eject
                            elif event.key == pygame.K_q and human3.getCanEject():
                                human3.setEject(True)
                            elif event.key == pygame.K_r:
                                human3.addMass(human3.getTotalMass() * 0.2)

                if self.model.hasPlayerSpectator() and not humanList:
                    spectatedPlayer = self.model.getSpectatedPlayer()
                    if event.key == pygame.K_RIGHT:
                        players = self.model.getPlayers()
                        nextPlayerIndex = (players.index(spectatedPlayer) + 1) % len(players)
                        nextPlayer = players[nextPlayerIndex]
                        self.model.setSpectatedPlayer(nextPlayer)
                    if event.key == pygame.K_LEFT:
                        players = self.model.getPlayers()
                        nextPlayerIndex = (players.index(spectatedPlayer) - 1) % len(players)
                        nextPlayer = players[nextPlayerIndex]
                        self.model.setSpectatedPlayer(nextPlayer)
            if not self.viewEnabled:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    self.model.setViewEnabled(True)
                if event.type == pygame.MOUSEBUTTONUP:
                    self.model.setViewEnabled(False)


    # Find the point where the player moved, taking into account that he only sees the fov
    def mousePosition(self, human, mousePos, humanNr):
        relativeMousePos = self.view.viewToModelScaling(mousePos, humanNr)
        human.setMoveTowards(relativeMousePos)
