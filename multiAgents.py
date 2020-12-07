# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        #newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        infinity = float('inf')
        result = 0

        #food
        minFoodDistance = infinity
        listOfFood = newFood.asList()
        for food in listOfFood:
            foodDistance = manhattanDistance(food, newPos)
            minFoodDistance = min(minFoodDistance, foodDistance)
        
        result +=  + 1/minFoodDistance

        #ghost
        minGhostDistance = infinity
        for ghostState in newGhostStates:
            ghostPosition = ghostState.getPosition()
            ghostDistance = manhattanDistance(ghostPosition, newPos)
            minGhostDistance = min(minGhostDistance, ghostDistance)
            if ghostState.scaredTimer == 0 and minGhostDistance < 2: return -infinity
        
        result += successorGameState.getScore()

        return result

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

    def isTerminalState(self, gameState):
        return gameState.isWin() or gameState.isLose()

    def isPacman(self, agentIndex):
        return agentIndex == 0

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        global infinity 
        infinity = float('inf')

        self.minimaxFunction(gameState)
        return self.action

    def minimaxFunction(self, gameState, agentIndex = 0, depth = 0):
        if gameState.isWin() or gameState.isLose(): return self.evaluationFunction(gameState)

        agentIndex = agentIndex % gameState.getNumAgents()

        if agentIndex == 0:
            if depth < self.depth:
                return self.maxValue(gameState, agentIndex, depth+1)
            else:
                return self.evaluationFunction(gameState)
        else:
            return self.minValue(gameState, agentIndex, depth)

    def maxValue(self, gameState, agentIndex, depth):
        result = -infinity
        actionList = gameState.getLegalActions(agentIndex)
        for action in actionList:
            successorState = gameState.generateSuccessor(agentIndex, action)
            value = self.minimaxFunction(successorState, agentIndex + 1, depth)
            result = max(result, value)
            if depth == 1 and result == value: self.action = action
        return result

    def minValue(self, gameState, agentIndex, depth):
        result = infinity
        actionList = gameState.getLegalActions(agentIndex)
        for action in actionList:
            successorState = gameState.generateSuccessor(agentIndex, action)
            value = self.minimaxFunction(successorState, agentIndex + 1, depth)
            result = min(result, value)
        return result

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        global infinity 
        infinity = float('inf')

        self.minimaxFunction(gameState)
        return self.action

    def minimaxFunction(self, gameState, agentIndex = 0, depth = 0, alpha = -float('inf'), beta = float('inf')):
        if gameState.isWin() or gameState.isLose(): return self.evaluationFunction(gameState)

        agentIndex = agentIndex % gameState.getNumAgents()

        if agentIndex == 0:
            if depth < self.depth:
                return self.maxValue(gameState, agentIndex, depth+1, alpha, beta)
            else:
                return self.evaluationFunction(gameState)
        else:
            return self.minValue(gameState, agentIndex, depth, alpha, beta)

    def maxValue(self, gameState, agentIndex, depth, alpha, beta):
        result = -infinity
        actionList = gameState.getLegalActions(agentIndex)
        for action in actionList:
            successorState = gameState.generateSuccessor(agentIndex, action)
            value = self.minimaxFunction(successorState, agentIndex + 1, depth, alpha, beta)
            result = max(result, value)
            if depth == 1 and result == value: self.action = action
            if result > beta: return result
            alpha = max(alpha, result)
        return result

    def minValue(self, gameState, agentIndex, depth, alpha, beta):
        result = infinity
        actionList = gameState.getLegalActions(agentIndex)
        for action in actionList:
            successorState = gameState.generateSuccessor(agentIndex, action)
            value = self.minimaxFunction(successorState, agentIndex + 1, depth, alpha, beta)
            result = min(result, value)
            if result < alpha: return result
            beta = min(beta, result)
        return result

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction
          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"

        global infinity 
        infinity = float('inf')

        self.expectimaxFunction(gameState)
        return self.action

    def expectimaxFunction(self, gameState, agentIndex=0, depth=0):
        if gameState.isWin() or gameState.isLose(): return self.evaluationFunction(gameState)

        agentIndex = agentIndex % gameState.getNumAgents()

        if agentIndex == 0:
            if depth < self.depth:
                return self.maxValue(gameState, agentIndex, depth+1)
            else:
                return self.evaluationFunction(gameState)
        else:
            return self.expectimaxValue(gameState, agentIndex, depth)

    def maxValue(self, gameState, agentIndex, depth):
        result = -infinity
        actionList = gameState.getLegalActions(agentIndex)
        for action in actionList:
            successorState = gameState.generateSuccessor(agentIndex, action)
            value = self.expectimaxFunction(successorState, agentIndex + 1, depth)
            result = max(result, value)
            if depth == 1 and result == value: self.action = action
        return result

    def probability(self, actionList):
        return 1.0 / len(actionList)

    def expectimaxValue(self, gameState, agentIndex, depth):
        actionList = gameState.getLegalActions(agentIndex)
        value = 0
        for action in actionList:
            successorState = gameState.generateSuccessor(agentIndex, action)
            p = self.probability(actionList)
            value += p * self.expectimaxFunction(successorState, agentIndex+1, depth)
        return value

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    infinity = float('inf')
    result = 0

    #food
    closestFood = None
    minFoodDistance = infinity
    listOfFood = newFood.asList()
    for food in listOfFood:
        foodDistance = manhattanDistance(food, newPos)
        if foodDistance < minFoodDistance:
            closestFood = food
            minFoodDistance = foodDistance
    if closestFood: result -= minFoodDistance * 0.1

    #ghost
    minGhostDistance = infinity
    for ghostState in newGhostStates:
        ghostPosition = ghostState.getPosition()
        ghostDistance = manhattanDistance(ghostPosition, newPos)
        minGhostDistance = min(minGhostDistance, ghostDistance)

        if ghostState.scaredTimer == 0 and minGhostDistance < 2: return -infinity

        else:
            for time in newScaredTimes:
                scaredGhostPosition = newGhostStates[0].getPosition()
                scaredGhostDistance = manhattanDistance(newPos, scaredGhostPosition)
                if time > 0 and scaredGhostDistance < 8: result += scaredGhostDistance

    result += currentGameState.getScore()

    return result

# Abbreviation
better = betterEvaluationFunction
