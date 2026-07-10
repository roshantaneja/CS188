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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        foodlist = newFood.asList()
        score = successorGameState.getScore()

        if len(foodlist) > 0:
            closestFood = manhattanDistance(newPos, foodlist[0])
            for food in foodlist:
                dist = manhattanDistance(newPos, food)
                if dist < closestFood:
                    closestFood = dist
            score = score + 1.0 / closestFood


        for i in range(len(newGhostStates)):
            ghoststate = newGhostStates[i]
            scaredtime = newScaredTimes[i]
            ghostdist = manhattanDistance(newPos, ghoststate.getPosition())

            if scaredtime > 0:
                score = score + 1.0/(ghostdist + 1)
            else:
                if ghostdist <= 1:
                    return float('-inf')
                score = score - 1.0/ghostdist

        return score

def scoreEvaluationFunction(currentGameState: GameState):
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

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
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

        def value(state, agentIndex, depth):
            if depth == 0:
                return self.evaluationFunction(state)
            if state.isWin():
                return self.evaluationFunction(state)
            if state.isLose():
                return self.evaluationFunction(state)
            

            numagents = state.getNumAgents()
            agent = (agentIndex + 1) % numagents

            if agent == 0:
                nextDepth = depth - 1
            else:
                nextDepth = depth

            actions = state.getLegalActions(agentIndex)
            outcomes = []
            for a in actions:
                successor = state.generateSuccessor(agentIndex, a)
                outcomes.append(value(successor, agent, nextDepth))
            
            
            if agentIndex == 0:
                return max(outcomes)
            else:
                return min(outcomes)

        bestAction = None
        bestValue = float('-inf')
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            v = value(successor, 1, self.depth)
            if v > bestValue:
                bestValue = v
                bestAction = action
        
        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def maxvalue(state, agentIndex, depth, alpha, beta):
            if depth == 0:
                return self.evaluationFunction(state)
            if state.isWin():
                return self.evaluationFunction(state)
            if state.isLose():
                return self.evaluationFunction(state)
            

            numagents = state.getNumAgents()
            agent = (agentIndex + 1) % numagents

            if agent == 0:
                nextDepth = depth - 1
            else:
                nextDepth = depth

            v = float('-inf')
            actions = state.getLegalActions(agentIndex)
            outcomes = []
            for a in actions:
                successor = state.generateSuccessor(agentIndex, a)
                v = max(v, value(successor, agent, nextDepth, alpha, beta))
                if v > beta:
                    return v
                alpha = max(alpha, v)

            return v
            
        def minvalue(state, agentIndex, depth, alpha, beta):
            if depth == 0:
                return self.evaluationFunction(state)
            if state.isWin():
                return self.evaluationFunction(state)
            if state.isLose():
                return self.evaluationFunction(state)
            

            numagents = state.getNumAgents()
            agent = (agentIndex + 1) % numagents

            if agent == 0:
                nextDepth = depth - 1
            else:
                nextDepth = depth

            v = float('inf')
            actions = state.getLegalActions(agentIndex)
            outcomes = []
            for a in actions:
                successor = state.generateSuccessor(agentIndex, a)
                v = min(v, value(successor, agent, nextDepth, alpha, beta))
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v
        
        def value(state, agentIndex, depth, alpha, beta):
            if agentIndex == 0:
                return maxvalue(state, agentIndex, depth, alpha, beta)
            else:
                return minvalue(state, agentIndex, depth, alpha, beta)

        bestAction = None
        v = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            score = value(successor, 1, self.depth, alpha, beta)
            if score > v:
                v = score
                bestAction = action
            alpha = max(alpha, v)
        
        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def value(state, agentIndex, depth):
            if depth == 0:
                return self.evaluationFunction(state)
            if state.isWin():
                return self.evaluationFunction(state)
            if state.isLose():
                return self.evaluationFunction(state)
            

            numagents = state.getNumAgents()
            agent = (agentIndex + 1) % numagents

            if agent == 0:
                nextDepth = depth - 1
            else:
                nextDepth = depth

            actions = state.getLegalActions(agentIndex)
            outcomes = []
            for a in actions:
                successor = state.generateSuccessor(agentIndex, a)
                outcomes.append(value(successor, agent, nextDepth))
            
            
            if agentIndex == 0:
                return max(outcomes)
            else:
                total = sum(outcomes)
                return 1.0*total/len(outcomes)

        bestAction = None
        bestValue = float('-inf')
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            v = value(successor, 1, self.depth)
            if v > bestValue:
                bestValue = v
                bestAction = action
        
        return bestAction
        

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: i starrted with manhattan distance to food and inverse of the same for the ghost then i really wanted to prioritize killing the ghost so i added a section where if the time to catch the ghost is shorter than the distance then priorize killing it asap. After that i still wasnt happy with my 1100 average, so i wanted to implement bfs mazedistance like we did in project 1 and it actually worked pretty well, after that i tuned the values a bit and this is what is submitted now!
    """
    "*** YOUR CODE HERE ***"

    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghoststates = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()
    walls = currentGameState.getWalls()

    score = currentGameState.getScore()
    foodlist = food.asList()

    # if len(foodlist) > 0:
    #     closest = manhattanDistance(pos, foodlist[0])
    #     for f in foodlist:
    #         d = manhattanDistance(pos, f)
    #         if d < closest:
    #             closest = d
    #     score = score + 1.0/closest

    foodset = {}
    for f in foodlist:
        foodset[f] = True
    
    queue = [(pos, 0)]
    visited = {}
    visited[pos] = True
    nearest = None
    while len(queue) > 0:
        current, dist = queue[0]
        queue = queue[1:]
        if current in foodset:
            nearest = dist
            break
        x, y = current
        neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        for dx, dy in neighbors:
            if not walls[dx][dy]:
                if (dx, dy) not in visited:
                    visited[(dx, dy)] = True
                    queue.append(((dx, dy), dist + 1))
    if nearest is not None:
        score = score + 1.0 / (nearest + 1)



    score = score - 10 * len(foodlist)
    score = score - 20 * len(capsules)

    for ghost in ghoststates:
        ghostdist = manhattanDistance(pos, ghost.getPosition())
        if ghost.scaredTimer > 0:
            if ghost.scaredTimer > ghostdist:
                score = score + 8.0/(ghostdist + 1)
            else:
                score = score + 1.0/(ghostdist + 1)
        else:
            if ghostdist <= 1:
                return float('-inf')
            score = score - 2.0/ghostdist

    return score 


# Abbreviation
better = betterEvaluationFunction
