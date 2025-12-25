#!/usr/bin/env python3
"""
Simple Perceptron-based Cribbage player.
Uses basic game features and learned weights to make decisions.
"""

import numpy as np
from Player import Player
from Deck import Deck


class Perceptron(Player):
    """
    A simple perceptron-based player that learns weights for hand throwing and pegging.
    Features are basic game state indicators (hand composition, count, etc.).
    """

    def __init__(self, number, alpha=0.1, verboseFlag=False):
        """
        Initialize the Perceptron player.
        
        Args:
            number: Player number (1 or 2)
            alpha: Learning rate (default 0.1)
            verboseFlag: Verbose output (default False)
        """
        super().__init__(number, verboseFlag)
        self.alpha = alpha  # Learning rate
        
        # Initialize weights for throwing (12 features)
        self.throwingWeights = np.zeros(12)
        
        # Initialize weights for pegging (8 features)
        self.peggingWeights = np.zeros(8)
        
        # For tracking state during learning
        self.lastState = None
        self.lastAction = None

    def getThrowCards(self):
        """
        Decide which 4 cards to throw from hand of 6.
        Uses perceptron to score each possible 4-card combination.
        """
        if len(self.hand) != 6:
            # Fallback if hand size is wrong
            return self.hand[:2]
        
        best_score = -float('inf')
        best_combo = self.hand[:4]
        
        # Try all combinations of 4 cards from 6
        from itertools import combinations
        for combo in combinations(self.hand, 4):
            combo_list = list(combo)
            features = self._getThrowFeatures(combo_list)
            score = np.dot(self.throwingWeights, features)
            
            if score > best_score:
                best_score = score
                best_combo = combo_list
        
        # Cards to throw are the 2 not in the best combo
        throw = [c for c in self.hand if c not in best_combo]
        return throw

    def throwCribCards(self):
        """Abstract method - redirect to getThrowCards."""
        return self.getThrowCards()

    def playCard(self, legalCards, count):
        """Abstract method - redirect to getPlay."""
        return self.getPlay(legalCards, count)

    def getPlay(self, legalCards, count):
        """
        Choose which card to play during pegging.
        Uses perceptron to score each legal card.
        """
        if not legalCards:
            return None
        
        best_score = -float('inf')
        best_card = legalCards[0]
        
        for card in legalCards:
            features = self._getPeggingFeatures([card], count, len(self.hand))
            score = np.dot(self.peggingWeights, features)
            
            if score > best_score:
                best_score = score
                best_card = card
        
        self.lastState = (best_card, count, len(self.hand))
        self.lastAction = best_card
        return best_card

    def _getThrowFeatures(self, cards):
        """
        Extract features from 4 cards to keep (for throwing decision).
        Returns 12-dimensional feature vector.
        """
        features = np.zeros(12)
        
        # Feature 0: Sum of card values
        features[0] = sum(c.value() for c in cards) / 52.0
        
        # Feature 1: Number of cards in 5-15 range (good for pegging)
        features[1] = sum(1 for c in cards if 5 <= c.value() <= 15) / 4.0
        
        # Feature 2: Number of face cards (value 10)
        features[2] = sum(1 for c in cards if c.value() == 10) / 4.0
        
        # Feature 3: Number of low cards (value < 5)
        features[3] = sum(1 for c in cards if c.value() < 5) / 4.0
        
        # Feature 4: Number of 5s
        features[4] = sum(1 for c in cards if c.value() == 5) / 4.0
        
        # Feature 5-10: Pairs/runs potential (simplified)
        ranks = [c.rank for c in cards]
        features[5] = len(ranks) / 4.0  # Baseline
        
        # Feature 11: Diversity of suits
        suits = [c.suit for c in cards]
        features[11] = len(set(suits)) / 4.0
        
        return features

    def _getPeggingFeatures(self, cards, count, oppCards):
        """
        Extract features from a single card for pegging decision.
        Returns 8-dimensional feature vector.
        """
        features = np.zeros(8)
        
        card = cards[0]
        
        # Feature 0: Card value (normalized)
        features[0] = card.value() / 31.0
        
        # Feature 1: Sum to 31 (go)
        if count + card.value() == 31:
            features[1] = 1.0
        elif count + card.value() < 31:
            features[1] = 0.5
        else:
            features[1] = 0.0
        
        # Feature 2: Distance to 31
        features[2] = (31 - (count + card.value())) / 31.0
        
        # Feature 3: Makes 15
        if count + card.value() == 15:
            features[3] = 1.0
        else:
            features[3] = 0.0
        
        # Feature 4: Remaining count safety (avoid busting)
        if count + card.value() <= 31:
            features[4] = 1.0 - (count + card.value()) / 31.0
        else:
            features[4] = -1.0
        
        # Feature 5: Opponent cards remaining (more cards = more risk)
        features[5] = oppCards / 4.0
        
        # Feature 6: Low card (safe)
        if card.value() < 5:
            features[6] = 1.0
        else:
            features[6] = 0.0
        
        # Feature 7: High card (risky)
        if card.value() > 10:
            features[7] = 1.0
        else:
            features[7] = 0.0
        
        return features

    def learnFromThrowing(self, scores):
        """
        Learn from throwing decisions.
        Called after hand scoring to update throwing weights.
        """
        # Simple update: if we scored well, reinforce the features
        # This is simplified; a full implementation would track the thrown cards
        if scores[self.number - 1] > 10:
            # Good outcome, small positive update
            pass

    def learnFromHandScores(self, scores):
        """
        Learn from hand scoring (required abstract method).
        Called after hand scoring phase completes.
        """
        pass

    def learnFromPegging(self, gameState):
        """
        Learn from pegging decisions using Q-learning style update.
        Called during/after pegging to update pegging weights.
        """
        if self.lastState is None or self.lastAction is None:
            return
        
        card, prev_count, opp_cards_left = self.lastState
        
        # Simple reward: scoring points or hitting 31 is good
        reward = 0
        if prev_count + card.value() == 31:
            reward = 2.0
        elif prev_count + card.value() == 15:
            reward = 1.0
        
        # Get features and update weights
        features = self._getPeggingFeatures([self.lastAction], prev_count, opp_cards_left)
        
        # Perceptron update rule (simplified)
        if reward > 0:
            self.peggingWeights += self.alpha * reward * features
        
        self.lastState = None
        self.lastAction = None

    def explainThrow(self):
        """Explain throwing decision (optional)."""
        pass

    def explainPlay(self):
        """Explain play decision (optional)."""
        pass

    def removeCard(self, card):
        """Remove card from opponent's hand (tracking)."""
        if card in self.hand:
            self.hand.remove(card)


if __name__ == '__main__':
    # Simple test
    from Myrmidon import Myrmidon
    from Arena import Arena
    
    player1 = Perceptron(1, alpha=0.1, verboseFlag=False)
    player2 = Myrmidon(2, numSims=5, verboseFlag=False)
    
    arena = Arena([player1, player2], repeatDeck=False, verboseFlag=False)
    results = arena.playHands(10)
    
    print(f"Results: {results}")
