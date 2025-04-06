from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Trade(Base):
    """
    Stores individual trade data including entry and exit points, profit/loss,
    and associated decision factors.
    """
    __tablename__ = 'trades'

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), index=True, nullable=False)
    entry_time = Column(DateTime, default=datetime.utcnow, nullable=False)
    entry_price = Column(Float, nullable=False)
    entry_quantity = Column(Float, nullable=False)
    exit_time = Column(DateTime, nullable=True)
    exit_price = Column(Float, nullable=True)
    exit_quantity = Column(Float, nullable=True)
    
    # Profit tracking
    profit_loss = Column(Float, nullable=True)
    profit_loss_percentage = Column(Float, nullable=True)
    
    # Trade status
    status = Column(String(20), default="OPEN", nullable=False)  # OPEN, CLOSED, CANCELLED
    trade_type = Column(String(10), nullable=False)  # BUY, SELL
    
    # Strategy and decision info
    strategy_used = Column(String(50), nullable=True)
    confidence_score = Column(Float, nullable=True)
    
    # Additional trade metadata
    fees = Column(Float, default=0.0)
    notes = Column(Text, nullable=True)
    
    # Relationships
    decision_id = Column(Integer, ForeignKey('trading_decisions.id'), nullable=True)
    decision = relationship("TradingDecision", back_populates="trades")
    
    def __repr__(self):
        return f"<Trade(id={self.id}, symbol={self.symbol}, status={self.status}, profit_loss={self.profit_loss})>"

    def to_dict(self):
        return {
            "id": self.id,
            "symbol": self.symbol,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "entry_price": self.entry_price,
            "entry_quantity": self.entry_quantity,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "exit_price": self.exit_price,
            "exit_quantity": self.exit_quantity,
            "profit_loss": self.profit_loss,
            "profit_loss_percentage": self.profit_loss_percentage,
            "status": self.status,
            "trade_type": self.trade_type,
            "strategy_used": self.strategy_used,
            "confidence_score": self.confidence_score,
            "fees": self.fees,
            "notes": self.notes
        }


class TradingDecision(Base):
    """
    Stores the decision-making process of the bot, including
    thoughts, analysis, and factors that led to a trade decision.
    """
    __tablename__ = 'trading_decisions'

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    symbol = Column(String(20), index=True, nullable=False)
    
    # Decision details
    decision = Column(String(20), nullable=False)  # BUY, SELL, HOLD
    confidence_score = Column(Float, nullable=True)
    
    # Analytical thought process
    thought_process = Column(Text, nullable=True)  # Detailed reasoning in text
    
    # Technical indicators that influenced the decision
    indicators = Column(JSON, nullable=True)  # Stores indicator values as JSON
    
    # Market conditions at decision time
    market_conditions = Column(JSON, nullable=True)  # Market state as JSON
    
    # Model used for decision
    model_used = Column(String(50), nullable=True)
    model_version = Column(String(20), nullable=True)
    
    # Relationships
    trades = relationship("Trade", back_populates="decision")
    
    def __repr__(self):
        return f"<TradingDecision(id={self.id}, symbol={self.symbol}, decision={self.decision})>"
    
    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "decision": self.decision,
            "confidence_score": self.confidence_score,
            "thought_process": self.thought_process,
            "indicators": self.indicators,
            "market_conditions": self.market_conditions,
            "model_used": self.model_used,
            "model_version": self.model_version
        }


class ProfitSummary(Base):
    """
    Stores aggregated profit statistics for different timeframes
    and symbols to enable efficient historical performance tracking.
    """
    __tablename__ = 'profit_summaries'

    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime, index=True, nullable=False)
    symbol = Column(String(20), index=True, nullable=False)
    
    # Daily metrics
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    win_rate = Column(Float, default=0.0)
    
    # Profit metrics
    total_profit = Column(Float, default=0.0)
    total_loss = Column(Float, default=0.0)
    net_profit = Column(Float, default=0.0)
    profit_factor = Column(Float, default=0.0)  # Total profit / Total loss
    
    # Risk metrics
    max_drawdown = Column(Float, default=0.0)
    avg_profit_per_trade = Column(Float, default=0.0)
    avg_loss_per_trade = Column(Float, default=0.0)
    
    # Time metrics
    avg_trade_duration = Column(Float, default=0.0)  # In minutes
    
    def __repr__(self):
        return f"<ProfitSummary(date={self.date}, symbol={self.symbol}, net_profit={self.net_profit})>"

    def to_dict(self):
        return {
            "id": self.id,
            "date": self.date.isoformat(),
            "symbol": self.symbol,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "total_profit": self.total_profit,
            "total_loss": self.total_loss,
            "net_profit": self.net_profit,
            "profit_factor": self.profit_factor,
            "max_drawdown": self.max_drawdown,
            "avg_profit_per_trade": self.avg_profit_per_trade,
            "avg_loss_per_trade": self.avg_loss_per_trade,
            "avg_trade_duration": self.avg_trade_duration
        }


class BotThought(Base):
    """
    Stores bot thoughts and internal reasoning that may not directly
    lead to trades but are important for understanding bot behavior.
    """
    __tablename__ = 'bot_thoughts'

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    thought_type = Column(String(50), nullable=False)  # market_analysis, risk_assessment, etc.
    thought_content = Column(Text, nullable=False)
    symbol = Column(String(20), nullable=True)
    confidence = Column(Float, nullable=True)
    
    # Optional metrics associated with the thought
    metrics = Column(JSON, nullable=True)
    
    def __repr__(self):
        return f"<BotThought(id={self.id}, type={self.thought_type}, timestamp={self.timestamp})>"

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "thought_type": self.thought_type,
            "thought_content": self.thought_content,
            "symbol": self.symbol,
            "confidence": self.confidence,
            "metrics": self.metrics
        }


class BotJoke(Base):
    """
    Stores jokes that Dimbot can tell occasionally to add personality
    and humor to the trading experience.
    """
    __tablename__ = 'bot_jokes'

    id = Column(Integer, primary_key=True, index=True)
    joke_text = Column(Text, nullable=False)
    category = Column(String(50), default="general")  # general, trading, technical, self-deprecating, etc.
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_used_at = Column(DateTime, nullable=True)  # Track when the joke was last displayed
    use_count = Column(Integer, default=0)  # Track how many times the joke has been used
    active = Column(Boolean, default=True)  # Allow jokes to be disabled without deleting them
    
    def __repr__(self):
        return f"<BotJoke(id={self.id}, category={self.category})>"

    def to_dict(self):
        return {
            "id": self.id,
            "joke_text": self.joke_text,
            "category": self.category,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "use_count": self.use_count,
            "active": self.active
        }
