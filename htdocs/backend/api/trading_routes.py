from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field

from database import get_db
from data.trading_db import (
    create_trade, close_trade, get_trades, get_trade_by_id,
    record_trading_decision, get_trading_decisions,
    record_bot_thought, get_bot_thoughts,
    get_profit_summaries, get_total_profit_stats,
    add_joke, get_jokes, get_random_joke, update_joke, delete_joke, initialize_default_jokes
)
from models.trading_models import Trade, TradingDecision, BotThought, ProfitSummary, BotJoke

router = APIRouter(
    prefix="/trading",
    tags=["trading"],
    responses={404: {"description": "Not found"}},
)

# Pydantic models for request validation

class TradeCreate(BaseModel):
    symbol: str
    entry_price: float
    entry_quantity: float
    trade_type: str
    strategy_used: Optional[str] = None
    confidence_score: Optional[float] = None
    decision_id: Optional[int] = None

class TradeClose(BaseModel):
    exit_price: float
    exit_quantity: float

class DecisionCreate(BaseModel):
    symbol: str
    decision: str
    confidence_score: Optional[float] = None
    thought_process: Optional[str] = None
    indicators: Optional[dict] = None
    market_conditions: Optional[dict] = None
    model_used: Optional[str] = None
    model_version: Optional[str] = None

class ThoughtCreate(BaseModel):
    thought_type: str
    thought_content: str
    symbol: Optional[str] = None
    confidence: Optional[float] = None

class JokeCreate(BaseModel):
    joke_text: str
    category: Optional[str] = "general"

class JokeUpdate(BaseModel):
    joke_text: Optional[str] = None
    category: Optional[str] = None
    active: Optional[bool] = None
    metrics: Optional[dict] = None

# API Routes

@router.post("/trades/", response_model=dict)
def api_create_trade(trade: TradeCreate, db: Session = Depends(get_db)):
    """Create a new trade record"""
    try:
        db_trade = create_trade(
            db=db,
            symbol=trade.symbol,
            entry_price=trade.entry_price,
            entry_quantity=trade.entry_quantity,
            trade_type=trade.trade_type,
            strategy_used=trade.strategy_used,
            confidence_score=trade.confidence_score,
            decision_id=trade.decision_id
        )
        return {"status": "success", "data": db_trade.to_dict()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.put("/trades/{trade_id}/close", response_model=dict)
def api_close_trade(trade_id: int, close_data: TradeClose, db: Session = Depends(get_db)):
    """Close an existing trade and calculate profit/loss"""
    try:
        db_trade = close_trade(
            db=db,
            trade_id=trade_id,
            exit_price=close_data.exit_price,
            exit_quantity=close_data.exit_quantity
        )
        return {"status": "success", "data": db_trade.to_dict()}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/trades/", response_model=dict)
def api_get_trades(
    skip: int = 0,
    limit: int = 100,
    symbol: Optional[str] = None,
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get trades with optional filtering"""
    trades = get_trades(db=db, skip=skip, limit=limit, symbol=symbol, status=status)
    return {
        "status": "success",
        "count": len(trades),
        "data": [trade.to_dict() for trade in trades]
    }

@router.get("/trades/{trade_id}", response_model=dict)
def api_get_trade(trade_id: int, db: Session = Depends(get_db)):
    """Get a specific trade by ID"""
    trade = get_trade_by_id(db=db, trade_id=trade_id)
    if not trade:
        raise HTTPException(status_code=404, detail=f"Trade with ID {trade_id} not found")
    return {"status": "success", "data": trade.to_dict()}

@router.post("/decisions/", response_model=dict)
def api_record_decision(decision: DecisionCreate, db: Session = Depends(get_db)):
    """Record a trading decision with all its analytical context"""
    try:
        db_decision = record_trading_decision(
            db=db,
            symbol=decision.symbol,
            decision=decision.decision,
            confidence_score=decision.confidence_score,
            thought_process=decision.thought_process,
            indicators=decision.indicators,
            market_conditions=decision.market_conditions,
            model_used=decision.model_used,
            model_version=decision.model_version
        )
        return {"status": "success", "data": db_decision.to_dict()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/decisions/", response_model=dict)
def api_get_decisions(
    skip: int = 0,
    limit: int = 100,
    symbol: Optional[str] = None,
    decision: Optional[str] = None,
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None,
    db: Session = Depends(get_db)
):
    """Get trading decisions with optional filtering"""
    decisions = get_trading_decisions(
        db=db, skip=skip, limit=limit, symbol=symbol,
        decision=decision, from_date=from_date, to_date=to_date
    )
    return {
        "status": "success",
        "count": len(decisions),
        "data": [decision.to_dict() for decision in decisions]
    }

@router.post("/thoughts/", response_model=dict)
def api_record_thought(thought: ThoughtCreate, db: Session = Depends(get_db)):
    """Record a bot thought or analytical reasoning"""
    try:
        db_thought = record_bot_thought(
            db=db,
            thought_type=thought.thought_type,
            thought_content=thought.thought_content,
            symbol=thought.symbol,
            confidence=thought.confidence,
            metrics=thought.metrics
        )
        return {"status": "success", "data": db_thought.to_dict()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/thoughts/", response_model=dict)
def api_get_thoughts(
    skip: int = 0,
    limit: int = 100,
    thought_type: Optional[str] = None,
    symbol: Optional[str] = None,
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None,
    db: Session = Depends(get_db)
):
    """Get bot thoughts with optional filtering"""
    thoughts = get_bot_thoughts(
        db=db, skip=skip, limit=limit, thought_type=thought_type,
        symbol=symbol, from_date=from_date, to_date=to_date
    )
    return {
        "status": "success",
        "count": len(thoughts),
        "data": [thought.to_dict() for thought in thoughts]
    }

@router.get("/profit/summaries/", response_model=dict)
def api_get_profit_summaries(
    skip: int = 0,
    limit: int = 30,
    symbol: Optional[str] = None,
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None,
    db: Session = Depends(get_db)
):
    """Get profit summaries with optional filtering"""
    summaries = get_profit_summaries(
        db=db, skip=skip, limit=limit, symbol=symbol,
        from_date=from_date, to_date=to_date
    )
    return {
        "status": "success",
        "count": len(summaries),
        "data": [summary.to_dict() for summary in summaries]
    }

@router.get("/profit/stats/", response_model=dict)
def api_get_profit_stats(
    symbol: Optional[str] = None,
    days: int = Query(30, ge=1, le=365),
    db: Session = Depends(get_db)
):
    """Get aggregated profit statistics over a specified period"""
    stats = get_total_profit_stats(db=db, symbol=symbol, days=days)
    return {"status": "success", "data": stats}


# Bot Jokes API endpoints
@router.post("/jokes/", response_model=dict)
def api_add_joke(joke: JokeCreate, db: Session = Depends(get_db)):
    """Add a new joke to the database"""
    db_joke = add_joke(db, joke_text=joke.joke_text, category=joke.category)
    return {
        "status": "success",
        "data": db_joke.to_dict()
    }


@router.get("/jokes/", response_model=dict)
def api_get_jokes(
    skip: int = 0,
    limit: int = 100,
    category: Optional[str] = None,
    active_only: bool = True,
    db: Session = Depends(get_db)
):
    """Get jokes with optional filtering"""
    jokes = get_jokes(db, skip, limit, category, active_only)
    return {
        "status": "success",
        "count": len(jokes),
        "data": [joke.to_dict() for joke in jokes]
    }


@router.get("/jokes/random/", response_model=dict)
def api_get_random_joke(
    category: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get a random joke, optionally from a specific category"""
    joke = get_random_joke(db, category)
    if not joke:
        raise HTTPException(status_code=404, detail="No jokes found")
    return {
        "status": "success",
        "data": joke.to_dict()
    }


@router.put("/jokes/{joke_id}", response_model=dict)
def api_update_joke(
    joke_id: int, 
    joke_update: JokeUpdate, 
    db: Session = Depends(get_db)
):
    """Update an existing joke"""
    updated_joke = update_joke(
        db, 
        joke_id, 
        joke_text=joke_update.joke_text,
        category=joke_update.category,
        active=joke_update.active
    )
    if not updated_joke:
        raise HTTPException(status_code=404, detail=f"Joke with ID {joke_id} not found")
    return {
        "status": "success",
        "data": updated_joke.to_dict()
    }


@router.delete("/jokes/{joke_id}", response_model=dict)
def api_delete_joke(joke_id: int, db: Session = Depends(get_db)):
    """Delete a joke from the database"""
    success = delete_joke(db, joke_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Joke with ID {joke_id} not found")
    return {"status": "success", "message": f"Joke with ID {joke_id} deleted"}


@router.post("/jokes/initialize/", response_model=dict)
def api_initialize_jokes(db: Session = Depends(get_db)):
    """Initialize the database with default jokes if none exist"""
    count_before = db.query(BotJoke).count()
    initialize_default_jokes(db)
    count_after = db.query(BotJoke).count()
    return {
        "status": "success", 
        "jokes_added": count_after - count_before,
        "total_jokes": count_after
    }
