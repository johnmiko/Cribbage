#!/usr/bin/env python3
"""
SimplePerceptron: Raw card inputs, no manually engineered features.
Uses the raw 6 dealt cards + starter card as direct input for throwing decisions,
and visible cards during pegging for play decisions.
"""

import numpy as np
from crib_ai_trainer.Player import Player
from itertools import combinations


class SimplePerceptron(Player):
    """
    Perceptron-based player that learns directly from raw card data.
    No manually engineered features; inputs are card values/indices.
    Includes suit information for flush detection and opponent prediction.
    """

    def __init__(self, number, alpha=0.1, verboseFlag=False):
        """
        Initialize SimplePerceptron.
        
        Args:
            number: Player number (1 or 2)
            alpha: Learning rate (default 0.1)
            verboseFlag: Verbose output (default False)
        """
        super().__init__(number, verboseFlag)
        self.alpha = alpha
        self.name = "SimplePerceptron"
        
        # Weights for throwing (7 cards, each with 13 ranks + 4 suits = 17 features per card)
        # Total: 7 * 13 (ranks) + 7 * 4 (suits) = 119 features
        self.throwingWeights = np.zeros(7 * 13 + 7 * 4)
        
        # Weights for pegging (6 visible cards + count)
        # Total: 6 * 13 (ranks) + 6 * 4 (suits) + 1 (count) = 103 features
        self.peggingWeights = np.zeros(6 * 13 + 6 * 4 + 1)
        
        # State tracking for learning
        self.lastThrowCombo = None
        self.lastPlayedCard = None
        self.lastPlayState = None
        self.starter = None

    def _card_to_rank(self, card):
        """Extract rank index 0-12 from Deck.Card (Rank enum)."""
        return (card.rank.value - 1) % 13

    def _card_to_suit(self, card):
        """Extract suit index 0-3 from Deck.Card (Suit enum)."""
        return (card.suit.value - 1) % 4

    def _encode_cards_with_suit(self, cards, max_cards=7, include_suit=True):
        """
        Encode cards as feature vector with rank and suit information.
        Each card gets encoded as:
        - Rank: one-hot across 13 positions (0-12)
        - Suit: one-hot across 4 positions (0-3)
        Padded with zeros if fewer cards than max_cards.
        
        Args:
            cards: List of Card objects
            max_cards: Maximum cards to encode (pad with zeros if fewer)
            include_suit: Whether to include suit information
        
        Returns:
            Feature vector of length max_cards * (13 + 4) if include_suit else max_cards * 13
        """
        if include_suit:
            feature_size = max_cards * (13 + 4)
        else:
            feature_size = max_cards * 13
        
        feature_vector = np.zeros(feature_size)
        
        for i, card in enumerate(cards[:max_cards]):
            if card is None:
                continue
            rank = self._card_to_rank(card)
            if include_suit:
                suit = self._card_to_suit(card)
                # Rank: positions [i*17, i*17+12]
                feature_vector[i * 17 + rank] = 1.0
                # Suit: positions [max_cards*13 + i*4, max_cards*13 + i*4+3]
                feature_vector[max_cards * 13 + i * 4 + suit] = 1.0
            else:
                feature_vector[i * 13 + rank] = 1.0
        
        return feature_vector

    def getThrowCards(self):
        """
        Decide which cards to throw to the crib using perceptron weights.
        Scores each of the 7 cards (6 in hand + starter) independently using the
        per-slot weights, then throws the two highest-scoring hand cards.
        """
        if not self.hand or len(self.hand) < 2:
            return []

        # Build the 7-card view: 6 in hand + starter (if present), else pad with None
        cards_for_encoding = list(self.hand)
        if self.starter is not None:
            cards_for_encoding.append(self.starter)
        while len(cards_for_encoding) < 7:
            cards_for_encoding.append(None)

        # Encode all cards positionally (slot i gets its own 17-feature slice)
        features = self._encode_cards_with_suit(cards_for_encoding, max_cards=7, include_suit=True)

        # Compute per-slot scores: each card uses its 17-feature slice in throwingWeights
        slot_width = 17  # 13 rank + 4 suit
        scores = []
        for idx, card in enumerate(cards_for_encoding):
            if card is None:
                continue
            start = idx * slot_width
            end = start + slot_width
            slot_weights = self.throwingWeights[start:end]
            slot_features = features[start:end]
            score = float(np.dot(slot_weights, slot_features))
            scores.append((score, idx, card))

        # Choose the two highest scores among hand cards (exclude starter slot)
        # Hand occupies the first len(self.hand) slots; starter (if any) is at index len(self.hand)
        hand_len = len(self.hand)
        hand_scores = [(s, i, c) for (s, i, c) in scores if i < hand_len]
        hand_scores.sort(reverse=True, key=lambda t: (t[0], t[1]))
        throw_cards = [t[2] for t in hand_scores[:2]]

        # Track kept combo (needed for learning): keep the remaining hand cards (max 4 for consistency)
        kept = [c for c in self.hand if c not in throw_cards][:4]
        self.lastThrowCombo = kept

        return throw_cards

    def throwCribCards(self, numCards, gameState):
        """Throw numCards cards to crib (required by Player interface)."""
        return self.getThrowCards()

    def playCard(self, gameState):
        """Play a card during pegging (required by Player interface)."""
        legalCards = gameState.get('legalCards', self.playhand)
        count = gameState.get('count', 0)
        return self.getPlayCard(legalCards, count)

    def getPlayCard(self, legalCards, count):
        """
        Decide which card to play in pegging.
        Uses perceptron to score remaining hand cards.
        Includes both rank and suit information for better predictions.
        """
        if not legalCards:
            return None
        
        best_score = -float('inf')
        best_card = legalCards[0]
        
        for card in legalCards:
            # Don't play card if it would exceed 31
            card_value = card.rank if card.rank != 1 else 1
            if card_value > 10:
                card_value = 10
            
            if count + card_value > 31:
                continue
            
            # Encode: up to 6 visible cards + count
            visible_cards = [c for c in self.playhand if c != card] + [card]
            features = self._encode_cards_with_suit(visible_cards[:6], max_cards=6, include_suit=True)
            # Add count as normalized feature (append at end)
            features = np.append(features, [count / 31.0])
            
            # Compute score
            score = np.dot(self.peggingWeights, features)
            
            if score > best_score:
                best_score = score
                best_card = card
        
        # Track state for learning
        self.lastPlayedCard = best_card
        self.lastPlayState = (self.playhand.copy(), count)
        
        return best_card

    def learnFromHandScores(self, scores, gameState):
        """
        Learn from hand throwing outcome.
        scores: dict with 'player1', 'player2', 'crib' keys
        """
        if self.lastThrowCombo is None or self.starter is None:
            return
        
        # Determine outcome for this player
        my_score = scores.get(f'player{self.number}', 0)
        opponent_num = 3 - self.number
        opponent_score = scores.get(f'player{opponent_num}', 0)
        
        # Outcome: +1 if we scored well, -1 if opponent scored better
        if my_score > opponent_score:
            outcome = 1
        elif my_score < opponent_score:
            outcome = -1
        else:
            outcome = 0
        
        # Encode the throw combo
        cards_for_encoding = self.lastThrowCombo + [self.starter, None, None]
        features = self._encode_cards_with_suit(cards_for_encoding, max_cards=7, include_suit=True)
        
        # Update weights: positive outcome -> increase weights, negative -> decrease
        if outcome > 0:
            self.throwingWeights += self.alpha * features
        elif outcome < 0:
            self.throwingWeights -= self.alpha * features

    def learnFromPegging(self, gameState):
        """
        Learn from pegging phase outcome.
        gameState: dict with game information including scores
        """
        if self.lastPlayState is None or self.lastPlayedCard is None:
            return
        
        # Get pegging scores
        my_peg_score = gameState.get(f'player{self.number}_peg_score', 0)
        opponent_num = 3 - self.number
        opponent_peg_score = gameState.get(f'player{opponent_num}_peg_score', 0)
        
        # Encode the pegging state
        visible_cards = self.lastPlayState[0] + [self.lastPlayedCard]
        features = self._encode_cards_with_suit(visible_cards[:6], max_cards=6, include_suit=True)
        features = np.append(features, [self.lastPlayState[1] / 31.0])
        
        # Update weights based on outcome
        outcome = my_peg_score - opponent_peg_score
        if outcome > 0:
            self.peggingWeights += self.alpha * features
        elif outcome < 0:
            self.peggingWeights -= self.alpha * features

    def explainThrow(self):
        """Explain throwing decision (no interpretable features, just model info)."""
        return f"{self.name} threw based on raw card perceptron weights"

    def explainPlay(self):
        """Explain pegging decision (no interpretable features, just model info)."""
        return f"{self.name} played based on raw card perceptron weights"

    def save_weights(self, path):
        """Save weights to file."""
        np.save(path, {
            'throwing': self.throwingWeights,
            'pegging': self.peggingWeights
        })

    def load_weights(self, path):
        """Load weights from file."""
        try:
            data = np.load(path, allow_pickle=True).item()
            self.throwingWeights = data['throwing']
            self.peggingWeights = data['pegging']
        except Exception as e:
            raise ValueError(f"Failed to load weights from {path}: {e}")
