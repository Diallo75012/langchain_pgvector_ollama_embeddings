from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.orm import scoped_session, sessionmaker
import os
from dotenv import load_dotenv


load_dotenv()

# Database connection string
DATABASE_URI = f"postgresql://{os.getenv('USER')}:{os.getenv('PASSWORD')}@{os.getenv('HOST')}:{os.getenv('PORT')}/{os.getenv('DATABASE')}"

# Create the engine that will interface with the database
engine = create_engine(DATABASE_URI, echo=True)

# Create a configured "Session" class
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

# Create a scoped session
session = scoped_session(SessionLocal)

# Base class for declarative models
Base = declarative_base()
Base.query = session.query_property()

def init_db():
    # Import all modules here that might define models so that
    # they will be registered properly on the metadata. Otherwise
    # you will have to import them first before calling init_db()
    import models  # Assuming your models are defined in models.py
    Base.metadata.create_all(bind=engine)
