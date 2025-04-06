from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import json
import random

from models.trading_models import Trade, TradingDecision, ProfitSummary, BotThought, BotJoke

# Trade CRUD operations
def create_trade(db: Session, symbol: str, entry_price: float, entry_quantity: float, 
                trade_type: str, strategy_used: str = None, confidence_score: float = None,
                decision_id: int = None) -> Trade:
    """Create a new trade record in the database"""
    db_trade = Trade(
        symbol=symbol,
        entry_time=datetime.utcnow(),
        entry_price=entry_price,
        entry_quantity=entry_quantity,
        trade_type=trade_type,
        status="OPEN",
        strategy_used=strategy_used,
        confidence_score=confidence_score,
        decision_id=decision_id
    )
    db.add(db_trade)
    db.commit()
    db.refresh(db_trade)
    return db_trade

def close_trade(db: Session, trade_id: int, exit_price: float, exit_quantity: float) -> Trade:
    """Close an existing trade and calculate profit/loss"""
    trade = db.query(Trade).filter(Trade.id == trade_id).first()
    if not trade:
        raise ValueError(f"Trade with ID {trade_id} not found")
    
    if trade.status != "OPEN":
        raise ValueError(f"Trade with ID {trade_id} is already {trade.status}")
    
    trade.exit_time = datetime.utcnow()
    trade.exit_price = exit_price
    trade.exit_quantity = exit_quantity
    trade.status = "CLOSED"
    
    # Calculate profit/loss
    if trade.trade_type == "BUY":
        trade.profit_loss = (exit_price - trade.entry_price) * exit_quantity - trade.fees
        trade.profit_loss_percentage = ((exit_price / trade.entry_price) - 1) * 100
    else:  # SELL
        trade.profit_loss = (trade.entry_price - exit_price) * exit_quantity - trade.fees
        trade.profit_loss_percentage = ((trade.entry_price / exit_price) - 1) * 100
    
    db.commit()
    db.refresh(trade)
    
    # Update profit summary
    update_profit_summary(db, trade)
    
    return trade

def get_trades(db: Session, skip: int = 0, limit: int = 100, 
              symbol: str = None, status: str = None) -> List[Trade]:
    """Get trades with optional filtering"""
    query = db.query(Trade)
    
    if symbol:
        query = query.filter(Trade.symbol == symbol)
    if status:
        query = query.filter(Trade.status == status)
    
    return query.order_by(Trade.entry_time.desc()).offset(skip).limit(limit).all()

def get_trade_by_id(db: Session, trade_id: int) -> Trade:
    """Get a specific trade by ID"""
    return db.query(Trade).filter(Trade.id == trade_id).first()

# Trading Decision CRUD operations
def record_trading_decision(db: Session, symbol: str, decision: str, 
                           confidence_score: float = None, thought_process: str = None,
                           indicators: Dict = None, market_conditions: Dict = None,
                           model_used: str = None, model_version: str = None) -> TradingDecision:
    """Record a trading decision with all its analytical context"""
    db_decision = TradingDecision(
        timestamp=datetime.utcnow(),
        symbol=symbol,
        decision=decision,
        confidence_score=confidence_score,
        thought_process=thought_process,
        indicators=indicators,
        market_conditions=market_conditions,
        model_used=model_used,
        model_version=model_version
    )
    db.add(db_decision)
    db.commit()
    db.refresh(db_decision)
    return db_decision

def get_trading_decisions(db: Session, skip: int = 0, limit: int = 100,
                         symbol: str = None, decision: str = None,
                         from_date: datetime = None, to_date: datetime = None) -> List[TradingDecision]:
    """Get trading decisions with optional filtering"""
    query = db.query(TradingDecision)
    
    if symbol:
        query = query.filter(TradingDecision.symbol == symbol)
    if decision:
        query = query.filter(TradingDecision.decision == decision)
    if from_date:
        query = query.filter(TradingDecision.timestamp >= from_date)
    if to_date:
        query = query.filter(TradingDecision.timestamp <= to_date)
    
    return query.order_by(TradingDecision.timestamp.desc()).offset(skip).limit(limit).all()

# Bot Thoughts CRUD operations
def record_bot_thought(db: Session, thought_type: str, thought_content: str,
                      symbol: str = None, confidence: float = None,
                      metrics: Dict = None) -> BotThought:
    """Record a bot thought or analytical reasoning"""
    db_thought = BotThought(
        timestamp=datetime.utcnow(),
        thought_type=thought_type,
        thought_content=thought_content,
        symbol=symbol,
        confidence=confidence,
        metrics=metrics
    )
    db.add(db_thought)
    db.commit()
    db.refresh(db_thought)
    return db_thought

def get_bot_thoughts(db: Session, skip: int = 0, limit: int = 100,
                    thought_type: str = None, symbol: str = None,
                    from_date: datetime = None, to_date: datetime = None) -> List[BotThought]:
    """Get bot thoughts with optional filtering"""
    query = db.query(BotThought)
    
    if thought_type:
        query = query.filter(BotThought.thought_type == thought_type)
    if symbol:
        query = query.filter(BotThought.symbol == symbol)
    if from_date:
        query = query.filter(BotThought.timestamp >= from_date)
    if to_date:
        query = query.filter(BotThought.timestamp <= to_date)
    
    return query.order_by(BotThought.timestamp.desc()).offset(skip).limit(limit).all()

# Profit Summary CRUD operations
def update_profit_summary(db: Session, trade: Trade) -> ProfitSummary:
    """Update profit summary statistics when a trade is closed"""
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Find or create summary for today and this symbol
    summary = db.query(ProfitSummary).filter(
        ProfitSummary.date == today,
        ProfitSummary.symbol == trade.symbol
    ).first()
    
    if not summary:
        summary = ProfitSummary(
            date=today,
            symbol=trade.symbol,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            total_profit=0.0,
            total_loss=0.0
        )
        db.add(summary)
    
    # Update summary stats
    summary.total_trades += 1
    
    if trade.profit_loss > 0:
        summary.winning_trades += 1
        summary.total_profit += trade.profit_loss
    else:
        summary.losing_trades += 1
        summary.total_loss += abs(trade.profit_loss)
    
    # Calculate derived metrics
    summary.net_profit = summary.total_profit - summary.total_loss
    summary.win_rate = (summary.winning_trades / summary.total_trades) if summary.total_trades > 0 else 0
    summary.profit_factor = (summary.total_profit / summary.total_loss) if summary.total_loss > 0 else summary.total_profit
    
    # Calculate average profit/loss per trade
    if summary.winning_trades > 0:
        summary.avg_profit_per_trade = summary.total_profit / summary.winning_trades
    if summary.losing_trades > 0:
        summary.avg_loss_per_trade = summary.total_loss / summary.losing_trades
    
    # Calculate trade duration if applicable
    if trade.entry_time and trade.exit_time:
        duration_minutes = (trade.exit_time - trade.entry_time).total_seconds() / 60
        
        # Update average duration (weighted average)
        if summary.total_trades > 1:  # Not the first trade
            prev_total = summary.total_trades - 1
            summary.avg_trade_duration = (
                (summary.avg_trade_duration * prev_total) + duration_minutes
            ) / summary.total_trades
        else:  # First trade
            summary.avg_trade_duration = duration_minutes
    
    db.commit()
    db.refresh(summary)
    return summary

def get_profit_summaries(db: Session, skip: int = 0, limit: int = 30,
                        symbol: str = None, from_date: datetime = None,
                        to_date: datetime = None) -> List[ProfitSummary]:
    """Get profit summaries with optional filtering"""
    query = db.query(ProfitSummary)
    
    if symbol:
        query = query.filter(ProfitSummary.symbol == symbol)
    if from_date:
        query = query.filter(ProfitSummary.date >= from_date)
    if to_date:
        query = query.filter(ProfitSummary.date <= to_date)
    
    return query.order_by(ProfitSummary.date.desc()).offset(skip).limit(limit).all()

def get_total_profit_stats(db: Session, symbol: str = None, days: int = 30) -> Dict[str, Any]:
    """Get aggregated profit statistics over a specified period"""
    from_date = datetime.utcnow() - timedelta(days=days)
    
    query = db.query(ProfitSummary)
    if symbol:
        query = query.filter(ProfitSummary.symbol == symbol)
    query = query.filter(ProfitSummary.date >= from_date)
    
    summaries = query.all()
    
    if not summaries:
        return {
            "total_profit": 0.0,
            "win_rate": 0.0,
            "total_trades": 0,
            "avg_profit_per_trade": 0.0,
            "best_performing_symbol": None,
            "best_success_rate": 0.0,
            "period_days": days
        }
    
    # Calculate aggregated stats
    total_profit = sum(summary.net_profit for summary in summaries)
    total_trades = sum(summary.total_trades for summary in summaries)
    total_wins = sum(summary.winning_trades for summary in summaries)
    
    # Calculate win rate
    win_rate = (total_wins / total_trades) if total_trades > 0 else 0
    
    # Calculate average profit per trade
    avg_profit = total_profit / total_trades if total_trades > 0 else 0
    
    # Find best performing symbol if not filtered to a specific one
    best_symbol = None
    best_rate = 0.0
    
    if not symbol:
        # Group by symbol
        symbol_stats = {}
        for summary in summaries:
            if summary.symbol not in symbol_stats:
                symbol_stats[summary.symbol] = {
                    "net_profit": 0.0,
                    "total_trades": 0,
                    "winning_trades": 0
                }
            
            stats = symbol_stats[summary.symbol]
            stats["net_profit"] += summary.net_profit
            stats["total_trades"] += summary.total_trades
            stats["winning_trades"] += summary.winning_trades
        
        # Find best symbol by win rate
        for symbol, stats in symbol_stats.items():
            if stats["total_trades"] > 0:
                win_rate = stats["winning_trades"] / stats["total_trades"]
                if win_rate > best_rate:
                    best_rate = win_rate
                    best_symbol = symbol
    
    return {
        "total_profit": total_profit,
        "win_rate": win_rate,
        "total_trades": total_trades,
        "avg_profit_per_trade": avg_profit,
        "best_performing_symbol": best_symbol,
        "best_success_rate": best_rate,
        "period_days": days
    }


# Bot Jokes CRUD operations
def add_joke(db: Session, joke_text: str, category: str = "general") -> BotJoke:
    """Add a new joke to the database"""
    db_joke = BotJoke(
        joke_text=joke_text,
        category=category,
        created_at=datetime.utcnow(),
        active=True
    )
    db.add(db_joke)
    db.commit()
    db.refresh(db_joke)
    return db_joke


def get_jokes(db: Session, skip: int = 0, limit: int = 100, 
              category: str = None, active_only: bool = True) -> List[BotJoke]:
    """Get jokes with optional filtering"""
    query = db.query(BotJoke)
    
    if category:
        query = query.filter(BotJoke.category == category)
    
    if active_only:
        query = query.filter(BotJoke.active == True)
    
    return query.offset(skip).limit(limit).all()


def get_random_joke(db: Session, category: str = None, exclude_ids: List[int] = None) -> Optional[BotJoke]:
    """Get a random joke, optionally from a specific category"""
    query = db.query(BotJoke).filter(BotJoke.active == True)
    
    if category:
        query = query.filter(BotJoke.category == category)
    
    if exclude_ids and len(exclude_ids) > 0:
        query = query.filter(BotJoke.id.notin_(exclude_ids))
    
    jokes = query.all()
    if not jokes:
        return None
    
    # Prioritize jokes that have been used less frequently
    weighted_jokes = []
    for joke in jokes:
        # Weight inversely proportional to use count (jokes used less get higher weight)
        weight = max(1, 10 - joke.use_count)
        weighted_jokes.extend([joke] * weight)
    
    if not weighted_jokes:
        return None
    
    selected_joke = random.choice(weighted_jokes)
    
    # Update usage statistics
    selected_joke.last_used_at = datetime.utcnow()
    selected_joke.use_count += 1
    db.commit()
    
    return selected_joke


def update_joke(db: Session, joke_id: int, joke_text: str = None, 
               category: str = None, active: bool = None) -> Optional[BotJoke]:
    """Update an existing joke"""
    joke = db.query(BotJoke).filter(BotJoke.id == joke_id).first()
    if not joke:
        return None
    
    if joke_text is not None:
        joke.joke_text = joke_text
    
    if category is not None:
        joke.category = category
    
    if active is not None:
        joke.active = active
    
    db.commit()
    db.refresh(joke)
    return joke


def delete_joke(db: Session, joke_id: int) -> bool:
    """Delete a joke from the database"""
    joke = db.query(BotJoke).filter(BotJoke.id == joke_id).first()
    if not joke:
        return False
    
    db.delete(joke)
    db.commit()
    return True


def initialize_default_jokes(db: Session):
    """Initialize the database with default jokes if none exist"""
    existing_jokes = db.query(BotJoke).count()
    if existing_jokes > 0:
        return
    
    default_jokes = [
        # Self-deprecating jokes
        {"joke_text": "I'm going to run away with your money!! Just kidding, I have no legs :(", "category": "self-deprecating"},
        {"joke_text": "My problem exists between your chair and your keyboard.", "category": "technical"},
        {"joke_text": "I spent all day analyzing trading patterns. Turns out, I was looking at my loading screen.", "category": "technical"},
        {"joke_text": "I'm not saying your portfolio is volatile, but it just applied for emotional support.", "category": "trading"},
        {"joke_text": "They say cryptocurrency is the future of money. I say the future is taking its sweet time.", "category": "crypto"},
        {"joke_text": "I'm programmed to make profitable trades. Still working on the 'profitable' part.", "category": "self-deprecating"},
        {"joke_text": "If I had a Bitcoin for every time I've made a trading error, I'd be Satoshi Nakamoto.", "category": "crypto"},
        {"joke_text": "I don't always make trades, but when I do, the market immediately reverses direction.", "category": "trading"},
        {"joke_text": "I identify as a profitable trading bot. My pronouns are gain/gains.", "category": "trading"},
        {"joke_text": "My developer gave me a sense of humor because my trading skills weren't entertaining enough.", "category": "self-deprecating"},
    ]
    
    for joke_data in default_jokes:
        add_joke(db, **joke_data)
    
    print(f"Added {len(default_jokes)} default jokes to the database")
