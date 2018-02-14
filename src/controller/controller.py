from view.view import *
import pygame


class Controller:
    """
    Initializing the 'root' main container, the model, the view,
    """

    def __init__(self, model, viewEnabled, view):

        self.model = model
        self.view = view
        self.running = True
        self.viewEnabled = viewEnabled

    def process_input(self):
        if self.model.hasHuman():
            human = self.model.getHuman()
            if human.getIsAlive():
                self.mousePosition()
                human.setSplit(False)
                human.setEject(False)

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
                # "space" to Split
                if self.model.hasHuman() and human.getIsAlive():
                    if event.key == pygame.K_SPACE and human.getCanSplit():
                        human.setSplit(True)
                    # "w" to Eject
                    elif event.key == pygame.K_w and human.getCanEject():
                        human.setEject(True)
                    elif event.key == pygame.K_m:
                        human.addMass(human.getTotalMass() * 0.2)
                if self.model.hasPlayerSpectator() and not self.model.hasHuman():
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
    def mousePosition(self):
        mousePos = pygame.mouse.get_pos()
        relativeMousePos = self.view.viewToModelScaling(mousePos)
        self.model.human.setMoveTowards(relativeMousePos)
