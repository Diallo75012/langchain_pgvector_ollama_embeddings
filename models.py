from sqlalchemy import create_engine, Column, Integer, ForeignKey, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class PgRecord(Base):
    __tablename__ = 'pgrecord'
    id_nbr = Column(Integer, primary_key=True) #will be unique and will autoincrement natively
    embedding_id = Column(Integer, nullable=True)
