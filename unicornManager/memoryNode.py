import datetime

class MemoryNode:
    def __init__(self, data):
        self.data = data  # The information stored in this node
        self.connections = {}  # Connections to other nodes (concepts)
        self.timestamp = datetime.datetime.now()  # Initialize with current time

    def add_connection(self, node, relevance=1):
        """Connect this node to another memory node."""
        self.connections[node] = relevance
        self.update_timestamp()

    def update_timestamp(self):
        """Update the timestamp to indicate recent usage."""
        self.timestamp = datetime.datetime.now()

    def is_recent(self, threshold_minutes=60):
        """Check if the memory was accessed recently."""
        if self.timestamp is None:
            return False
        return (datetime.datetime.now() - self.timestamp).total_seconds() < threshold_minutes * 60