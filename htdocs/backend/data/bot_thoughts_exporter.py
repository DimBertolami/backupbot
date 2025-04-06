import json
import os
import sys
import logging
from datetime import datetime
from sqlalchemy.orm import Session

# Set up proper path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import SessionLocal
from data.trading_db import record_bot_thought, record_trading_decision, get_total_profit_stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bot_thoughts_export.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def export_bot_thoughts_to_db(thoughts_file_path="/opt/lampp/htdocs/bot/frontend/public/trading_data/bot_thoughts.json"):
    """
    Reads the bot thoughts from the JSON file and exports them to the database.
    """
    try:
        # Check if file exists
        if not os.path.exists(thoughts_file_path):
            logger.warning(f"Bot thoughts file not found at: {thoughts_file_path}")
            return False
        
        # Read the JSON file
        with open(thoughts_file_path, 'r') as file:
            data = json.load(file)
            
        if 'thoughts' not in data or not isinstance(data['thoughts'], list):
            logger.error("Invalid bot thoughts JSON structure, 'thoughts' list not found")
            return False
            
        # Open database session
        db = SessionLocal()
        try:
            # Process each thought
            count = 0
            for thought in data['thoughts']:
                # Basic validation
                if not isinstance(thought, dict) or 'thought_content' not in thought:
                    continue
                
                # Get thought data
                thought_type = thought.get('thought_type', 'trading_analysis')
                thought_content = thought['thought_content']
                symbol = thought.get('symbol')
                confidence = thought.get('confidence')
                metrics = thought.get('metrics')
                
                # Record in database
                record_bot_thought(
                    db=db, 
                    thought_type=thought_type,
                    thought_content=thought_content,
                    symbol=symbol,
                    confidence=confidence,
                    metrics=metrics
                )
                count += 1
                
            logger.info(f"Successfully exported {count} bot thoughts to database")
            
            # If there are decisions to export, also export those
            if 'decisions' in data and isinstance(data['decisions'], list):
                count = 0
                for decision in data['decisions']:
                    if not isinstance(decision, dict) or 'decision' not in decision:
                        continue
                    
                    # Record decision
                    record_trading_decision(
                        db=db,
                        symbol=decision.get('symbol'),
                        decision=decision['decision'],
                        confidence_score=decision.get('confidence_score'),
                        thought_process=decision.get('thought_process'),
                        indicators=decision.get('indicators'),
                        market_conditions=decision.get('market_conditions'),
                        model_used=decision.get('model_used')
                    )
                    count += 1
                
                logger.info(f"Successfully exported {count} trading decisions to database")
                
            return True
        except Exception as e:
            logger.error(f"Error exporting bot thoughts to database: {str(e)}")
            return False
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error processing bot thoughts file: {str(e)}")
        return False

def export_trading_status_to_db(status_file_path="/opt/lampp/htdocs/bot/frontend/public/trading_data/live_trading_status.json"):
    """
    Exports trading status data from JSON to database, including thoughts and performance metrics.
    """
    try:
        # Check if file exists
        if not os.path.exists(status_file_path):
            logger.warning(f"Trading status file not found at: {status_file_path}")
            return False
        
        # Read the JSON file
        with open(status_file_path, 'r') as file:
            data = json.load(file)
            
        # Open database session
        db = SessionLocal()
        try:
            # First, export thoughts if they exist
            if 'thoughts' in data and isinstance(data['thoughts'], list):
                timestamp = data.get('timestamp', datetime.utcnow().isoformat())
                
                for thought in data['thoughts']:
                    if not isinstance(thought, str):
                        continue
                    
                    # Record the thought
                    record_bot_thought(
                        db=db, 
                        thought_type='live_trading',
                        thought_content=thought,
                        symbol=data.get('symbol'),
                        metrics=data.get('metrics')
                    )
                
                logger.info(f"Exported {len(data['thoughts'])} thoughts from trading status")
            
            # If there's a decision, export it
            if 'current_decision' in data and data['current_decision']:
                decision = data['current_decision']
                
                # Record the decision
                record_trading_decision(
                    db=db,
                    symbol=data.get('symbol'),
                    decision=decision['action'] if isinstance(decision, dict) and 'action' in decision else decision,
                    confidence_score=decision.get('confidence') if isinstance(decision, dict) else None,
                    thought_process=decision.get('reasoning') if isinstance(decision, dict) else None,
                    indicators=data.get('indicators'),
                    market_conditions=data.get('market_conditions')
                )
                
                logger.info(f"Exported current decision from trading status")
            
            return True
        except Exception as e:
            logger.error(f"Error exporting trading status to database: {str(e)}")
            return False
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error processing trading status file: {str(e)}")
        return False

def generate_performance_summary(db: Session = None):
    """
    Generates a performance summary from the database.
    """
    try:
        # Use provided session or create new one
        should_close_db = False
        if db is None:
            db = SessionLocal()
            should_close_db = True
            
        try:
            # Get overall stats for 30 days
            stats_30d = get_total_profit_stats(db=db, days=30)
            
            # Get stats for 7 days
            stats_7d = get_total_profit_stats(db=db, days=7)
            
            # Get stats for 24 hours
            stats_24h = get_total_profit_stats(db=db, days=1)
            
            # Return combined stats
            return {
                "last_24h": stats_24h,
                "last_7d": stats_7d,
                "last_30d": stats_30d
            }
        except Exception as e:
            logger.error(f"Error generating performance summary: {str(e)}")
            return None
        finally:
            if should_close_db:
                db.close()
                
    except Exception as e:
        logger.error(f"Database session error in performance summary: {str(e)}")
        return None

if __name__ == "__main__":
    # When run directly, export bot thoughts and trading status to database
    export_bot_thoughts_to_db()
    export_trading_status_to_db()
    
    # Generate and display performance summary
    summary = generate_performance_summary()
    if summary:
        print("\nPerformance Summary:")
        print(json.dumps(summary, indent=2))
