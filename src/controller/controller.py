from view.view import *
import pygame


class Controller:
    """
    Initializing the 'root' main container, the model, the view,
    """

    def __init__(self, model, viewEnabled, view):

        self.model = model
        self.view = view
        self.fieldWidth = model.getField().getWidth()
        self.fieldHeight = model.getField().getHeight()
        self.running = True
        self.viewEnabled = viewEnabled

    def process_input(self):
        humanList = self.model.getHumans()
        humanCommandPoint = []
        if self.model.hasHuman():
            for human in humanList:
                humanCommandPoint.append([self.fieldWidth / 2, self.fieldHeight / 2])
                if human.getIsAlive():
                    human.setSplit(False)
                    human.setEject(False)
            humanCommandPoint[0] = pygame.mouse.get_pos()

        for event in pygame.event.get():
            # Event types
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN:
                # "Escape" to Quit
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_b:
                    self.model.createBot()
                    player = self.model.players[-1]
                    self.model.getField().initializePlayer(player)
                if bool(humanList):
                    #Human1 controls
                    human1 = humanList[0]
                    keys = pygame.key.get_pressed()
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
                            if keys[pygame.K_UP] and human2.getCanSplit():
                                humanCommandPoint[1][1] -= self.fieldHeight / 2
                            if keys[pygame.K_DOWN] and human2.getCanSplit():
                                humanCommandPoint[1][1] += self.fieldHeight / 2
                            if keys[pygame.K_LEFT] and human2.getCanSplit():
                                humanCommandPoint[1][0] -= self.fieldWidth / 2
                            if keys[pygame.K_RIGHT] and human2.getCanSplit():
                                humanCommandPoint[1][0] += self.fieldWidth / 2
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
                            if keys[pygame.K_w] and human2.getCanSplit():
                                humanCommandPoint[2][1] -= self.fieldHeight / 2
                            if keys[pygame.K_s] and human2.getCanSplit():
                                humanCommandPoint[2][1] += self.fieldHeight / 2
                            if keys[pygame.K_a] and human2.getCanSplit():
                                humanCommandPoint[2][0] -= self.fieldWidth / 2
                            if keys[pygame.K_d] and human2.getCanSplit():
                                humanCommandPoint[2][0] += self.fieldWidth / 2
                            # "space" to Split
                            if event.key == pygame.K_e and human3.getCanSplit():
                                human3.setSplit(True)
                            # "w" to Eject
                            elif event.key == pygame.K_q and human3.getCanEject():
                                human3.setEject(True)
                            elif event.key == pygame.K_r:
                                human3.addMass(human3.getTotalMass() * 0.2)

                if self.model.hasPlayerSpectator() and not bool(humanList):
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
        if self.model.hasHuman():
            for i in range(len(humanList)):
                print(humanCommandPoint[i])
                self.mousePosition(humanList[i], humanCommandPoint[i])

    # Find the point where the player moved, taking into account that he only sees the fov
    def mousePosition(self, human, mousePos):
        relativeMousePos = self.view.viewToModelScaling(mousePos)
        human.setMoveTowards(relativeMousePos)
