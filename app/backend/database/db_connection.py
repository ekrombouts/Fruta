import os
from typing import Optional

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


class Database:
    def __init__(
        self,
        user: Optional[str] = None,
        password: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:

        load_dotenv()

        self.user: Optional[str] = user or os.getenv("DB_USER")
        self.password: Optional[str] = password or os.getenv("DB_PASSWORD")
        self.host: Optional[str] = host or os.getenv("DB_HOST")
        self.port: Optional[str] = port or os.getenv("DB_PORT")
        self.name: Optional[str] = name or os.getenv("DB_NAME")
        self.engine: Optional[Engine] = None

    def get_engine(self) -> Engine:
        # Create a SQLAlchemy engine if it doesn't exist
        if not self.engine:
            try:
                db_url = f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"
                self.engine = create_engine(db_url)
            except Exception as e:
                raise RuntimeError(f"Fout bij maken van engine: {e}") from e
        return self.engine

    def test_connection(self) -> bool:
        # Test the database connection
        engine = self.get_engine()
        if engine is None:
            print("Engine creation failed. Cannot test connection.")
            return
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                print("Connection successful")
                return True
        except Exception as e:
            print("Connection failed:", e)
        return False


# Example usage
if __name__ == "__main__":
    db = Database()
    db.test_connection()
