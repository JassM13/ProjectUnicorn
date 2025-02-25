import os
from psycopg2 import pool
from dotenv import load_dotenv

class PostgresManager:
    __instance = None

    @staticmethod
    def getInstance():
        if PostgresManager.__instance is None:
            PostgresManager.__instance = PostgresManager()
        return PostgresManager.__instance

    def __init__(self):
        if PostgresManager.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            load_dotenv()
            connection_string = os.getenv("POSTGRESQL_DB")

            if not connection_string:
                raise Exception("DATABASE_URL is not set!")

            self.pool = pool.SimpleConnectionPool(
                1,  # Min connections
                10,  # Max connections
                connection_string
            )

            if not self.pool:
                raise Exception("Failed to create connection pool!")

    def get_connection(self):
        """Get a connection from the pool."""
        return self.pool.getconn()

    def release_connection(self, conn):
        """Return a connection to the pool."""
        self.pool.putconn(conn)

    def close_pool(self):
        """Close all connections (Only call this when shutting down)."""
        self.pool.closeall()
