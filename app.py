"""FastAPI wrapper to expose cribbage agents via HTTP.

Endpoints
- GET /health
- GET /opponents
- POST /game/new - Start a new hand session
- POST /game/{session_id}/throw-crib - Player discards cards
- POST /game/{session_id}/play-card - Player plays a card during pegging
- GET /game/{session_id}/state - Get current game state

Interactive human-vs-agent play with stateful sessions.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Local imports
from Cribbage import Cribbage
from PlayerRandom import PlayerRandom
from Myrmidon import Myrmidon
from Deck import Card


@dataclass
class Opponent:
    key: str
    label: str
    description: str
    factory: Callable[[int], object]


@dataclass
class GameSession:
    """Represents an ongoing cribbage hand."""
    session_id: str
    game: Cribbage
    player_seat: int
    opponent_key: str
    phase: str  # "crib_throw", "play", "scoring", "complete"
    player_hand_display: List[str]  # For frontend: serialize cards to strings


def _build_opponents() -> Dict[str, Opponent]:
    return {
        "random": Opponent(
            key="random",
            label="Random",
            description="Baseline agent that selects legal moves at random.",
            factory=lambda seat: PlayerRandom(seat, False),
        ),
        "myrmidon": Opponent(
            key="myrmidon",
            label="Myrmidon",
            description="Heuristic rollouts (shallow search).",
            factory=lambda seat: Myrmidon(seat, 5, False),
        ),
    }


OPPONENTS = _build_opponents()
SESSIONS: Dict[str, GameSession] = {}

app = FastAPI(title="Cribbage API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for requests/responses

class CardModel(BaseModel):
    rank: int
    suit: int


class NewGameRequest(BaseModel):
    opponent: str = Field(default="random", description="Opponent key")


class GameStateResponse(BaseModel):
    session_id: str
    phase: str
    player_hand: List[CardModel]
    opponent_score: int
    player_score: int
    dealer: int
    starter_card: Optional[CardModel]
    message: str
    crib_owner: str  # "player" or "opponent"
    opponent_cards_left: int = 0
    cards_played: List[dict] = []  # List of {"player": CardModel, "opponent": CardModel or None}


class ThrowCribRequest(BaseModel):
    card_indices: List[int] = Field(description="Indices of cards to discard (0-based)")


class PlayCardRequest(BaseModel):
    card_index: int = Field(description="Index of card to play (0-based)")


class OpponentModel(BaseModel):
    key: str
    label: str
    description: str


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/opponents", response_model=List[OpponentModel])
def list_opponents() -> List[OpponentModel]:
    return [
        OpponentModel(
            key=opp.key,
            label=opp.label,
            description=opp.description,
        )
        for opp in OPPONENTS.values()
    ]


def _build_opponent(key: str, seat: int):
    opponent = OPPONENTS.get(key)
    if not opponent:
        raise HTTPException(status_code=404, detail=f"Unknown opponent '{key}'")
    return opponent.factory(seat)


def _card_to_model(card) -> CardModel:
    """Convert Card object to CardModel for JSON."""
    return CardModel(rank=card.rank.value, suit=card.suit.value)


def _serialize_hand(hand: List) -> List[CardModel]:
    """Serialize a hand to CardModels."""
    return [_card_to_model(card) for card in hand]


def _build_game_state_response(session_id: str, session: GameSession) -> GameStateResponse:
    """Helper to construct GameStateResponse with played cards and opponent hand count."""
    player = session.game.players[0]
    opponent = session.game.players[1]
    
    # Build cards played history (pairs of player/opponent plays)
    cards_played = []
    playorder = session.game.playorder
    for i in range(0, len(playorder), 2):
        player_card = _card_to_model(playorder[i]) if i < len(playorder) else None
        opponent_card = _card_to_model(playorder[i + 1]) if i + 1 < len(playorder) else None
        cards_played.append({"player": player_card, "opponent": opponent_card})
    
    # Opponent cards left depends on phase: before playhand is created use hand length, otherwise playhand length
    opponent_cards_left = len(opponent.playhand) if opponent.playhand else len(opponent.hand)

    crib_owner = "player" if session.game.dealer == 0 else "opponent"

    return GameStateResponse(
        session_id=session_id,
        phase=session.phase,
        player_hand=_serialize_hand(player.hand if session.phase == "crib_throw" else player.playhand),
        player_score=player.pips,
        opponent_score=opponent.pips,
        dealer=session.game.dealer,
        starter_card=_card_to_model(session.game.starter) if session.game.starter else None,
        message="",
        crib_owner=crib_owner,
        opponent_cards_left=opponent_cards_left,
        cards_played=cards_played,
    )


@app.post("/game/new", response_model=GameStateResponse)
def new_game(req: NewGameRequest) -> GameStateResponse:
    """Start a new cribbage hand. Player is seat 1, opponent is seat 2."""
    opponent = _build_opponent(req.opponent, 2)
    player = PlayerRandom(1, False)
    
    game = Cribbage([player, opponent], verboseFlag=False)
    game.deal()
    
    session_id = str(uuid4())
    session = GameSession(
        session_id=session_id,
        game=game,
        player_seat=1,
        opponent_key=req.opponent,
        phase="crib_throw",
        player_hand_display=[],
    )
    SESSIONS[session_id] = session
    
    state = _build_game_state_response(session_id, session)
    state.message = "Discard 2 cards to the crib."
    return state


@app.get("/game/{session_id}/state", response_model=GameStateResponse)
def get_game_state(session_id: str) -> GameStateResponse:
    """Get current game state."""
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = SESSIONS[session_id]
    state = _build_game_state_response(session_id, session)
    return state


@app.post("/game/{session_id}/throw-crib", response_model=GameStateResponse)
def throw_crib(session_id: str, req: ThrowCribRequest) -> GameStateResponse:
    """Player discards cards to the crib. Expects 2 card indices."""
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = SESSIONS[session_id]
    if session.phase != "crib_throw":
        raise HTTPException(status_code=400, detail=f"Not in crib_throw phase (current: {session.phase})")
    
    if len(req.card_indices) != 2:
        raise HTTPException(status_code=400, detail="Must discard exactly 2 cards")
    
    player = session.game.players[0]
    opponent = session.game.players[1]
    
    # Validate indices
    for idx in req.card_indices:
        if idx < 0 or idx >= len(player.hand):
            raise HTTPException(status_code=400, detail=f"Invalid card index: {idx}")
    
    # Remove cards from hand (remove in reverse order to avoid index shifting)
    cards_to_discard = [player.hand[i] for i in sorted(req.card_indices, reverse=True)]
    for idx in sorted(req.card_indices, reverse=True):
        session.game.crib.append(player.hand.pop(idx))
    
    # Opponent throws
    opp_thrown = opponent.throwCribCards(2, session.game.gameState())
    for card in opp_thrown:
        session.game.crib.append(card)
    
    # Cut the deck for starter
    session.game.cut()
    player.createPlayHand()
    opponent.createPlayHand()
    
    session.phase = "play"
    
    state = _build_game_state_response(session_id, session)
    state.message = "Play a card (must keep count <= 31)."
    return state


@app.post("/game/{session_id}/play-card", response_model=GameStateResponse)
def play_card(session_id: str, req: PlayCardRequest) -> GameStateResponse:
    """Player plays a card. Opponent plays in response. Returns updated state."""
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = SESSIONS[session_id]
    if session.phase != "play":
        raise HTTPException(status_code=400, detail=f"Not in play phase (current: {session.phase})")
    
    player = session.game.players[0]
    opponent = session.game.players[1]
    
    if req.card_index < 0 or req.card_index >= len(player.playhand):
        raise HTTPException(status_code=400, detail=f"Invalid card index: {req.card_index}")
    
    count = session.game.gameState()['count']
    card = player.playhand[req.card_index]
    
    if count + card.value() > 31:
        raise HTTPException(status_code=400, detail="Card would exceed count of 31")
    
    # Player plays card
    played = player.playhand.pop(req.card_index)
    count += played.value()
    session.game.inplay.append(played)
    session.game.playorder.append(played)
    player.pips += _score_cards(session.game.inplay)
    
    # Opponent plays (or goes)
    opp_card = opponent.playCard(session.game.gameState())
    if opp_card is None:
        # Opponent goes; player gets 1 pip
        player.pips += 1
        # TODO: implement full go/31 logic
    else:
        count += opp_card.value()
        session.game.inplay.append(opp_card)
        session.game.playorder.append(opp_card)
        opponent.pips += _score_cards(session.game.inplay)
    
    # Check if hand is complete
    if len(player.playhand) == 0 and len(opponent.playhand) == 0:
        session.phase = "scoring"
    
    state = _build_game_state_response(session_id, session)
    state.message = f"Count: {count}"
    return state


def _score_cards(count_cards: List) -> int:
    """Score cards in pegging phase (pairs, runs, 15, 31)."""
    # Import here to avoid circular deps
    from Scoring import scoreCards as crib_scoreCards
    return crib_scoreCards(count_cards, False)
