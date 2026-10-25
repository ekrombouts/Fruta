from typing import Any, Dict, Optional

import pandas as pd

# from database.db_connection import Database
from backend.database.db_connection import Database
from sqlalchemy import create_engine, text


class FrutaData:
    """Database loader for Fruta data."""

    # Create a database engine using the Database class
    _engine = Database().get_engine()

    @classmethod
    def _get_connection(cls):
        # Get a connection from the database engine
        return cls._engine.connect()

    @classmethod
    def run_query(
        cls, query: str, params: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Execute a SQL query and return the result as a pandas DataFrame.
        If an error occurs, return an empty DataFrame.
        """
        try:
            with cls._get_connection() as conn:
                return pd.read_sql(text(query), conn, params=params or {})
        except Exception:
            return pd.DataFrame()

    @classmethod
    def load_table(cls, table_name: str) -> pd.DataFrame:
        """
        Load all rows from a specific table in the database.
        """
        return cls.run_query(f"SELECT * FROM {table_name}")

    @classmethod
    def load_clients(cls) -> pd.DataFrame:
        """
        Load all rows from the 'profiles' table.
        """
        return cls.load_table("profiles")

    @classmethod
    def load_profiles(cls) -> pd.DataFrame:
        """
        Load all rows from the 'profiles' table.
        """
        return cls.load_table("profiles")

    @classmethod
    def load_scenarios(cls) -> pd.DataFrame:
        """
        Load all rows from the 'scenarios' table.
        """
        return cls.load_table("scenarios")

    @classmethod
    def load_records(cls) -> pd.DataFrame:
        """
        Load all rows from the 'records' table.
        """
        return cls.load_table("records")


class FrutaClients:
    """
    Class to manage and retrieve client data.
    """

    def __init__(self) -> None:
        # Load all clients into a DataFrame
        self.clients = FrutaData.load_clients()

    def get_client(self, client_id: str) -> "FrutaClient":
        """
        Retrieve a specific client by their client_id.
        Returns a FrutaClient object or None if the client is not found.
        """
        row = self.clients[self.clients["client_id"] == client_id]
        if row.empty:
            return None
        return FrutaClient(client_id, row.iloc[0].to_dict())

    def get_random_client(self) -> "FrutaClient":
        """
        Retrieve a random client from the loaded clients.
        Returns a FrutaClient object.
        """
        row = self.clients.sample().iloc[0]
        return FrutaClient(row["client_id"], row.to_dict())


class FrutaClient:
    """
    Class to represent a single client and their associated data.
    """

    # Use the same database engine as FrutaData
    _engine = FrutaData._engine

    def __init__(
        self, client_id: str, client_details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize a FrutaClient object with client_id and optional client details.
        If client details are not provided, fetch them from the database.
        """
        self.client_id = client_id
        self.client_details = client_details or self._fetch_client_details()

        # Basic client information
        self.geslacht = (
            self.client_details.get("geslacht") if self.client_details else None
        )
        self.voornaam = (
            self.client_details.get("voornaam") if self.client_details else None
        )
        self.achternaam = (
            self.client_details.get("achternaam") if self.client_details else None
        )
        # self.name = self.client_details.get("name") if self.client_details else None
        self.diagnose = (
            self.client_details.get("diagnose") if self.client_details else None
        )
        self.somatiek = (
            self.client_details.get("somatiek") if self.client_details else None
        )
        self.adl = self.client_details.get("adl") if self.client_details else None
        self.mobiliteit = (
            self.client_details.get("mobiliteit") if self.client_details else None
        )
        self.gedrag = self.client_details.get("gedrag") if self.client_details else None
        self.start_date = (
            self.client_details.get("start_date") if self.client_details else None
        )
        self.duration = (
            self.client_details.get("duration") if self.client_details else None
        )
        self.complications = (
            self.client_details.get("complications") if self.client_details else None
        )
        self.ward = self.client_details.get("ward") if self.client_details else None
        self.model = self.client_details.get("model") if self.client_details else None
        self.naam = self._format_naam()

    def __str__(self) -> str:
        """
        String representation of the client.
        """
        return f"{self.naam} ({self.client_id})"

    def _fetch_client_details(self) -> Optional[Dict[str, Any]]:
        """
        Fetch client details from the database using the client_id.
        Returns a dictionary of client details or None if not found.
        """
        query = "SELECT * FROM profiles WHERE client_id = :client_id"
        with self._engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params={"client_id": self.client_id})
        return df.iloc[0].to_dict() if not df.empty else None

    # Function to format client profile into a string
    def _format_naam(self) -> str:
        titel = "Mevrouw" if self.geslacht == "v" else "Meneer"
        return f"{titel} {self.voornaam} {self.achternaam}"

    def get_notes(
        self, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Retrieve notes (records) for the client within an optional date range.
        Returns a pandas DataFrame or None if no client details are available.
        """
        if not self.client_details:
            return None

        query = """
        SELECT * FROM records
        WHERE client_id = :client_id
        """
        params: Dict[str, Any] = {"client_id": self.client_id}

        # Add date filters to the query if provided
        if start_date and end_date:
            query += " AND date::date BETWEEN :start_date AND :end_date"
            params.update({"start_date": start_date, "end_date": end_date})
        elif start_date:
            query += " AND date::date >= :start_date"
            params.update({"start_date": start_date})
        elif end_date:
            query += " AND date::date <= :end_date"
            params.update({"end_date": end_date})

        query += " ORDER BY date ASC"

        with self._engine.connect() as conn:
            return pd.read_sql(text(query), conn, params=params)

    def get_scenario(self) -> Optional[pd.DataFrame]:
        """
        Retrieve the scenario associated with the client.
        Returns a pandas DataFrame or None if no client details are available.
        """
        if not self.client_details:
            return None

        query = "SELECT * FROM scenarios WHERE client_id = :client_id"
        with self._engine.connect() as conn:
            return pd.read_sql(text(query), conn, params={"client_id": self.client_id})


# Example usage:
if __name__ == "__main__":
    fruta_clients = FrutaClients()
    client = fruta_clients.get_random_client()
    print(client)
    notes = client.get_notes()
    print(notes.head(2))
    scenario = client.get_scenario()
    print(scenario.head(2))
