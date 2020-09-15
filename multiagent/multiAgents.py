from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent_num1(Agent):
    def getAction(self, gameState):
        legalMoves = gameState.getLegalActions()

        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        food = currentGameState.getFood()
        currentPos = list(successorGameState.getPacmanPosition())
        dist = float("-Info")

        foods = food.asList()

        if (action == 'Stop'):
            return float("-Info")

        for state in newGhostStates:
            if state.getPosition() == tuple(currentPos) and (state.scaredTimer == 0):
                return float("-Info")

        for x in foods:
            tempDistance = -1 * (manhattanDistance(currentPos, x))
            if (tempDistance > dist):
                dist = tempDistance

        return dist

def evaluationFunction(self, currentGameState, action):
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        if successorGameState.isWin():
            return float("Info")

        for ghostState in newGhostStates:
            if util.manhattanDistance(ghostState.getPosition(), newPos) < 2:
                return float("-Info")

        foodDist = []
        for food in list(newFood.asList()):
            foodDist.append(util.manhattanDistance(food, newPos))

        foodSuccessor = 0
        if (currentGameState.getNumFood() > successorGameState.getNumFood()):
            foodSuccessor = 300

        return successorGameState.getScore() - 5 * min(foodDist) + foodSuccessor 

def scoreEvaluationFunction(currentGameState):
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        "*** YOUR CODE HERE ***"

        Fantasmas = [i for i in range(1, gameState.getNumAgents())]

        def term(state, d):
            return state.isWin() or state.isLose() or d == self.depth

        def min_value(state, d, ghost): 

            if term(state, d):
                return self.evaluationFunction(state)

            v = 999999999999999999
            for action in state.getLegalActions(ghost):
                if ghost == Fantasmas[-1]:
                    v = min(v, max_value(state.generateSuccessor(ghost, action), d + 1))
                else:
                    v = min(v, min_value(state.generateSuccessor(ghost, action), d, ghost + 1))
            return v

        def max_value(state, d): 

            if term(state, d):
                return self.evaluationFunction(state)

            v = -999999999999
            for action in state.getLegalActions(0):
                v = max(v, min_value(state.generateSuccessor(0, action), d, 1))
            return v

        res = [(action, min_value(gameState.generateSuccessor(0, action), 0, 1)) for action in
               gameState.getLegalActions(0)]
        res.sort(key=lambda k: k[1])

        return res[-1][0]

        util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        "*** YOUR CODE HERE ***"

        Fantasmas = [i for i in range(1, gameState.getNumAgents())]
        Info = 99999999

        def term(state, d):
            return state.isWin() or state.isLose() or d == self.depth

        def min_value(state, d, ghost, A, B): 

            if term(state, d):
                return self.evaluationFunction(state)

            v = Info
            for action in state.getLegalActions(ghost):
                if ghost == Fantasmas[-1]:
                    v = min(v, max_value(state.generateSuccessor(ghost, action), d + 1, A, B))
                else:
                    v = min(v, min_value(state.generateSuccessor(ghost, action), d, ghost + 1, A, B))

                if v < A:
                    return v
                B = min(B, v)

            return v

        def max_value(state, d, A, B): 

            if term(state, d):
                return self.evaluationFunction(state)

            v = -Info
            for action in state.getLegalActions(0):
                v = max(v, min_value(state.generateSuccessor(0, action), d, 1, A, B))

                if v > B:
                    return v
                A = max(A, v)

            return v

        def alphabeta(state):

            v = -Info
            act = None
            A = -Info
            B = Info

            for action in state.getLegalActions(0):  
                tmp = min_value(gameState.generateSuccessor(0, action), 0, 1, A, B)

                if v < tmp:  
                    v = tmp
                    act = action

                if v > B:
                    return v
                A = max(A, tmp)

            return act

        return alphabeta(gameState)

        util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        "*** YOUR CODE HERE ***"

        Fantasmas = [i for i in range(1, gameState.getNumAgents())]

        def term(state, d):
            return state.isWin() or state.isLose() or d == self.depth

        def exp_value(state, d, Fantasma): #minimizadora

            if term(state, d):
                return self.evaluationFunction(state)

            v = 0
            prob = 1 / len(state.getLegalActions(Fantasma))

            for action in state.getLegalActions(Fantasma):
                if Fantasma == Fantasmas[-1]:
                    v += prob * max_value(state.generateSuccessor(Fantasma, action), d + 1)
                else:
                    v += prob * exp_value(state.generateSuccessor(Fantasma, action), d, Fantasma + 1)
            return v

        def max_value(state, d): #maximizadora

            if term(state, d):
                return self.evaluationFunction(state)

            v = -99999999999999999
            for action in state.getLegalActions(0):
                v = max(v, exp_value(state.generateSuccessor(0, action), d, 1))
            return v

        res = [(action, exp_value(gameState.generateSuccessor(0, action), 0, 1)) for action in
               gameState.getLegalActions(0)]
        res.sort(key=lambda k: k[1])

        return res[-1][0]

        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):

    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    Fantasma_mat = [ghostState.scaredTimer for ghostState in newGhostStates]
    Paredes = currentGameState.getWalls()
    newFood = newFood.asList()
    Pos_Fantasma = [(G.getPosition()[0], G.getPosition()[1]) for G in newGhostStates]
    scared = min(Fantasma_mat) > 0

    if currentGameState.isLose():
        return float('-Info')

    if newPos in ghostPos:
        return float('-Info')


    Dist_food_perto = sorted(newFood, key=lambda fDist: util.manhattanDistance(fDist, newPos))
    Dist_fanstasma_perto = sorted(ghostPos, key=lambda gDist: util.manhattanDistance(gDist, newPos))

    score = 0

    fd = lambda fDis: util.manhattanDistance(fDis, newPos)
    gd = lambda gDis: util.manhattanDistance(gDis, newPos)

    if gd(Dist_fanstasma_perto[0]) <3:
        score-=300
    if gd(Dist_fanstasma_perto[0]) <2:
        score-=1000
    if gd(Dist_fanstasma_perto[0]) <1:
        return float('-Info')

    if len(currentGameState.getCapsules()) < 2:
        score+=100

    if len(Dist_food_perto)==0 or len(Dist_fanstasma_perto)==0 :
        score += scoreEvaluationFunction(currentGameState) + 10
    else:
        score += (   scoreEvaluationFunction(currentGameState) + 10/fd(Dist_food_perto[0]) + 1/gd(closestGhostDist[0]) + 1/gd(closestGhostDist[-1])  )

    return score

    util.raiseNotDefined()

better = betterEvaluationFunction
