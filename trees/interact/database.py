from sqlalchemy import create_engine, Column, Integer
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

class SqlSession(object):

    def __init__(self, session):
        self.session = session


    def __enter__(self):
        return self.session

    def __exit__(self, *args):
        self.session.close()

class Database(object):

    def __init__(self, location):
        self.engine_string = "sqlite:///%s.db" % location
        self.engine = create_engine(self.engine_string)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def session(self):
        return SqlSession(self.Session())

    def get_interactions(self):
        with self.session() as session:
            interactions= session.query(Interaction).all()
        return [
            (i.a, i.b, i.c, i.oou) for i in interactions
        ]

    def add_interaction(self, interaction):
        a, b, c, oou = interaction
        with self.session() as session:
            interaction = Interaction(a=a, b=b, c=c, oou=oou)
            session.add(interaction)
            session.commit()

Base = declarative_base()

class Interaction(Base):
    __tablename__ = "interaction"

    id = Column('id', Integer, primary_key=True)
    a = Column('a', Integer)
    b = Column('b', Integer)
    c = Column('c', Integer)
    oou = Column('oou', Integer)

    def __repr__(self):
        return "<Interaction(interaction=(%u, %u, %u), oou=%u)>" % (self.a, self.b, self.c, self.oou)
