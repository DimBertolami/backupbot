from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database import Base
from models.trading_models import BotJoke

# Database URL for SQLite - make sure it matches the one in database.py
DATABASE_URL = "sqlite:///./new_test.db"

def init_joke_table():
    print("Creating BotJoke table in the database...")
    
    # Create engine
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
    
    # Create table if it doesn't exist
    BotJoke.__table__.create(bind=engine, checkfirst=True)
    
    # Create a session
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    
    try:
        # Check if jokes table exists and has data
        joke_count = db.query(BotJoke).count()
        print(f"Found {joke_count} existing jokes in the database.")
        
        # Initialize some default jokes if there are none
        if joke_count == 0:
            from data.trading_db import initialize_default_jokes
            initialize_default_jokes(db)
            joke_count = db.query(BotJoke).count()
            print(f"Added default jokes. Now there are {joke_count} jokes in the database.")
    finally:
        db.close()
    
    print("BotJoke table initialization complete!")

if __name__ == "__main__":
    init_joke_table()
